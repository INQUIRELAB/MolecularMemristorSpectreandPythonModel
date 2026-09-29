import os
import sys
import subprocess
import pickle
import numpy as np
import threading
import atexit
import weakref

_LIB_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_LIB_DIR = os.path.dirname(_LIB_DIR)

_mpl_initialized = False

def _init_matplotlib():
    global _mpl_initialized
    if not _mpl_initialized:
        import matplotlib as mpl
        
        # Force the main process to use the non-interactive 'Agg' backend in standalone terminal runs.
        # This completely avoids loading Tkinter on the main simulation thread and allows safe asynchronous background plot saving.
        # In interactive notebooks (Jupyter, Colab), preserving the inline backend allows figures to display in output cells.
        is_nb = False
        try:
            from .sys_utils import is_interactive_notebook
            is_nb = is_interactive_notebook()
        except Exception:
            pass

        if is_nb:
            try:
                from IPython import get_ipython
                _ipy = get_ipython()
                if _ipy is not None:
                    _ipy.run_line_magic('matplotlib', 'inline')
                else:
                    mpl.use('module://matplotlib_inline.backend_inline')
            except Exception:
                try:
                    mpl.use('module://matplotlib_inline.backend_inline')
                except Exception:
                    try:
                        mpl.use('inline')
                    except Exception:
                        pass
        else:
            mpl.use('Agg')
            
        mpl.rcParams['agg.path.chunksize'] = 10000
        _mpl_initialized = True

_BANNER_LINE = "-" * 62
_MID_LINE = f"--- MolmemPlotter: Figure Save Details ".ljust(59) + "---"
_HEADER_BLOCK = f"\n{_BANNER_LINE}\n{_MID_LINE}\n{_BANNER_LINE}"
_INTERACTIVE_BACKENDS = frozenset({'tkagg', 'gtk3agg', 'gtk4agg', 'qt5agg', 'qt6agg', 'macosx', 'wxagg'})



class MolmemPlotter:
    """
    Automated plotting command interface for MolMemristor simulations.
    Provides a safe, persistent way to build figures with multiple waveforms.
    Automatically scales time axes and extracts data natively from the simulator.
    """
    
    def __init__(self, sim):
        self._is_noop = getattr(sim, '_is_noop', False)
        if self._is_noop:
            self.sim = sim
            self.figures = {}
            self.axes = {}
            self._save_threads = []
            return
        _init_matplotlib()
        self.sim = sim
        self.figures = {}
        self.axes = {}
        self.metrics = {}
        self._subplot_scales = {}
        self._save_threads = []
        self._lock = threading.Lock()
        
        # Register atexit cleanup using a weak reference wrapper to avoid strong-reference memory leaks
        weak_self = weakref.ref(self)
        def wait_all_wrapper():
            ref = weak_self()
            if ref is not None:
                ref.wait_all()
        self._atexit_ref = wait_all_wrapper
        atexit.register(self._atexit_ref)

    def __del__(self):
        try:
            if hasattr(self, '_atexit_ref'):
                atexit.unregister(self._atexit_ref)
        except Exception:
            pass

    def _getTimeScaleAndUnit(self, tArr=None):
        if tArr is None:
            tArr = self.sim.getTime()
        t_len = len(tArr)
        maxT = float(tArr[-1]) if t_len > 0 else 0.0
        if maxT >= 1.0: return 1.0, "s"
        elif maxT >= 1e-3: return 1e3, "ms"
        elif maxT >= 1e-6: return 1e6, "us"
        else: return 1e9, "ns"

    def _apply_layout(self, fig):
        try:
            has_title = getattr(fig, '_suptitle', None) is not None or bool(fig.texts)
            top = 0.92 if has_title else 0.95
            fig.subplots_adjust(left=0.08, right=0.98, top=top, bottom=0.08, wspace=0.35, hspace=0.35)
        except Exception:
            try:
                fig.subplots_adjust(wspace=0.35, hspace=0.35)
            except Exception:
                pass

    def createFigure(self, name, title=None, figsize=(10, 4), subplots=1):
        """Creates a new named figure. 'subplots' can be an int (rows) or a tuple (rows, cols)."""
        if self._is_noop:
            return
        import matplotlib.pyplot as plt
        if name in self.figures:
            print(f"Warning: Overwriting existing figure '{name}'.")
            plt.close(self.figures[name])
            
        self.metrics[name] = []
            
        if isinstance(subplots, tuple):
            rows, cols = subplots
        else:
            rows, cols = subplots, 1
        fig, axs = plt.subplots(rows, cols, figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=12, fontweight='bold')
            
        self.figures[name] = fig
        
        # Squeeze 1D shapes (like (3, 1) or (1, 3)) down to a flat 1D array
        # This prevents `subplotIdx=0` from throwing on a `(3, 1)` grid because 
        # normally a (3, 1) grid returns a 2D array of [[ax], [ax], [ax]].
        axs = np.atleast_1d(axs)
        if axs.ndim == 2 and (axs.shape[0] == 1 or axs.shape[1] == 1):
            axs = axs.ravel()
            
        self.axes[name] = axs

    def addWaveform(self, name, dataType, row=None, col=None, node=None, subplotIdx=0, twinx=False, customData=None, **kwargs):
        """
        Adds a waveform to a previously created figure.
        dataType: 'Voltage', 'Current', 'Pulses', 'State', 'Conductance', or 'Custom'
        subplotIdx: 0-indexed int or (row, col) tuple for which subplot to draw on.
        twinx: If True, creates and plots on a twin y-axis.
        """
        if self._is_noop:
            return
        with self._lock:
            fig_axes = self.axes.get(name)
            if fig_axes is None:
                raise ValueError(f"Figure '{name}' does not exist. Call createFigure first.")
                
            try:
                if isinstance(subplotIdx, tuple):
                    ax = fig_axes[subplotIdx[0], subplotIdx[1]]
                elif isinstance(fig_axes, np.ndarray):
                    ax = fig_axes.flat[subplotIdx]
                else:
                    ax = fig_axes[subplotIdx]
            except (IndexError, TypeError):
                raise IndexError(f"subplotIdx {subplotIdx} is out of bounds or invalid for figure '{name}'.")
            
            customXData = kwargs.pop('customXData', None)
            logy = kwargs.pop('logy', False)
            
            if customXData is not None:
                xData = customXData
            else:
                # Auto-scale Time
                tArr = self.sim.getTime()
                tScale, _ = self._getTimeScaleAndUnit(tArr)
                xData = tArr * tScale if tScale != 1.0 else tArr
            
            # Resolve or establish scaling for the target axis to prevent unit mismatch on the same subplot
            target_scale_dict = self._subplot_scales.setdefault(name, {}).setdefault(subplotIdx, {})
            dt_type = dataType.lower()
            established_scale = target_scale_dict.get(dt_type)
            label = kwargs.get('label', None)
            yUnit = 'V'
            
            if dt_type == 'voltage':
                yData = self.sim.getVoltage(node=node, row=row, col=col)
                if not label:
                    if row is not None: label = f"WL{row} Voltage"
                    elif col is not None: label = f"BL{col} Voltage"
                    else: label = f"Node {node} Voltage"
            elif dt_type == 'current':
                yData = self.sim.getCurrent(row=row, col=col)
                if established_scale is not None:
                    scale, yUnit = established_scale
                else:
                    y_len = len(yData)
                    if y_len > 0:
                        abs_min = abs(float(yData.min()))
                        abs_max = abs(float(yData.max()))
                        maxVal = abs_min if abs_min > abs_max else abs_max
                    else:
                        maxVal = 0.0
                    if maxVal >= 1.0:
                        scale, yUnit = 1.0, "A"
                    elif maxVal >= 1e-3:
                        scale, yUnit = 1e3, "mA"
                    elif maxVal >= 1e-6:
                        scale, yUnit = 1e6, "\u03BCA"
                    else:
                        scale, yUnit = 1e9, "nA"
                    target_scale_dict[dt_type] = (scale, yUnit)
                if scale != 1.0:
                    yData = yData * scale
                if not label: label = f"Current (R{row}C{col}) [{yUnit}]"
            elif dt_type == 'pulses':
                yData = self.sim.getState(row=row, col=col)
                if not label: label = f"Pulses (R{row}C{col})"
            elif dt_type == 'state':
                yData = self.sim.getF22(row=row, col=col)
                if not label: label = f"State f22 (R{row}C{col})"
            elif dt_type == 'conductance':
                yData = self.sim.getConductance(row=row, col=col)
                if established_scale is not None:
                    scale, yUnit = established_scale
                else:
                    y_len = len(yData)
                    maxVal = float(yData.max()) if y_len > 0 else 0.0
                    if maxVal >= 1.0:
                        scale, yUnit = 1.0, "S"
                    elif maxVal >= 1e-3:
                        scale, yUnit = 1e3, "mS"
                    elif maxVal >= 1e-6:
                        scale, yUnit = 1e6, "\u03BCS"
                    else:
                        scale, yUnit = 1e9, "nS"
                    target_scale_dict[dt_type] = (scale, yUnit)
                if scale != 1.0:
                    yData = yData * scale
                if not label: label = f"Conductance (R{row}C{col}) [{yUnit}]"
            elif dt_type in ('temperature', 'temp'):
                yData = self.sim.getTemperature(row=row, col=col)
                if not label: label = f"Temperature (R{row}C{col}) [K]"
            elif dt_type == 'custom':
                if customData is None:
                    raise ValueError("Must provide 'customData' for dataType='Custom'")
                yData = customData
            else:
                raise ValueError(f"Unknown dataType: {dataType}")
                
            if 'label' not in kwargs and label is not None:
                kwargs['label'] = label
                
            # Handle twin axes
            targetAx = ax
            if twinx:
                if not hasattr(ax, 'twinx_ax'):
                    ax.twinx_ax = ax.twinx()
                targetAx = ax.twinx_ax
                
            # Auto-Label Y-Axis
            if targetAx.get_ylabel() == '':
                if dt_type == 'voltage': targetAx.set_ylabel('Voltage (V)')
                elif dt_type == 'current': targetAx.set_ylabel(f'Current ({yUnit})')
                elif dt_type == 'pulses': targetAx.set_ylabel('Number of Pulses (n)')
                elif dt_type == 'state': targetAx.set_ylabel('State (f22)')
                elif dt_type == 'conductance': targetAx.set_ylabel(f'Conductance ({yUnit})')
                elif dt_type in ('temperature', 'temp'): targetAx.set_ylabel('Temperature (K)')
                
            # Pure raw data mode: Direct plotting of raw ODE solver timesteps.
            # Record metric info for console/log printing
            target_str = "Global"
            if row is not None and col is not None:
                target_str = f"Device R{row}C{col}"
            elif row is not None:
                target_str = f"Row {row}"
            elif col is not None:
                target_str = f"Col {col}"
            elif node is not None:
                target_str = f"Node {node}"
                
            if dt_type == 'current' or dt_type == 'conductance':
                unit_str = yUnit
            elif dt_type == 'state':
                unit_str = 'f22 ratio'
            elif dt_type == 'pulses':
                unit_str = 'pulses'
            elif dt_type in ('temperature', 'temp'):
                unit_str = 'K'
            elif dt_type == 'custom':
                unit_str = 'custom'
            else:
                unit_str = 'V'
            
            self.metrics[name].append({
                'dataType': dataType.capitalize(),
                'target': target_str,
                'unit': unit_str,
                'label': label
            })
            
            if logy:
                targetAx.set_yscale('log')
                
            targetAx.plot(xData, yData, **kwargs)
            
    def formatAxis(self, name, subplotIdx=0, title=None, xlabel='Auto', ylabel=None, 
                   twinxYlabel=None, grid=True, legend=True, loc='best'):
        """Applies formatting to a specific subplot. xlabel='Auto' sets Time (SI)."""
        if self._is_noop:
            return
        with self._lock:
            fig_axes = self.axes.get(name)
            if fig_axes is None:
                raise ValueError(f"Figure '{name}' does not exist.")
                
            try:
                ax = fig_axes[subplotIdx]
                if isinstance(ax, np.ndarray):
                    return # Skip formatting if resolving to an array of axes
            except (IndexError, TypeError):
                raise IndexError(f"subplotIdx {subplotIdx} is out of bounds or invalid for figure '{name}'.")
            
            if title:
                ax.set_title(title, pad=10)
                
            if xlabel == 'Auto':
                tArr = self.sim.getTime()
                _, tUnit = self._getTimeScaleAndUnit(tArr)
                ax.set_xlabel(f"Time ({tUnit})")
            elif xlabel: 
                ax.set_xlabel(xlabel)
                
            if ylabel: ax.set_ylabel(ylabel)
            if grid: ax.grid(grid)
            if legend: 
                # Only add legend if there are labels defined
                handles, labels = ax.get_legend_handles_labels()
                if labels:
                    ax.legend(loc=loc)
                
            if hasattr(ax, 'twinx_ax'):
                if twinxYlabel:
                    ax.twinx_ax.set_ylabel(twinxYlabel)
                if legend:
                    # Place twinx legend somewhere else to avoid overlap
                    twinHandles, twinLabels = ax.twinx_ax.get_legend_handles_labels()
                    if twinLabels:
                        ax.twinx_ax.legend(loc='center right')

    def _print_save_summary(self, name, absTarget):
        try:
            metrics_list = self.metrics.get(name, [])
            if not getattr(self.sim, 'detailedPrint', True):
                print(f"[>] MolmemPlotter > Saved figure '{name}' to {absTarget} ({len(metrics_list)} metrics)")
                return
                
            metrics_summary = [
                f"    - {m['dataType'].ljust(15)} : {m['target'].ljust(15)} (scaled to {m['unit']})"
                for m in metrics_list
            ]
            
            print(_HEADER_BLOCK)
            print(f"  > Figure Name           : {name}")
            print(f"  > Target Path           : {absTarget}")
            if metrics_summary:
                print("  > Waveform Metrics Contained:")
                for line in metrics_summary:
                    print(line)
            print()
        except Exception:
            print(f"Saved figure '{name}' to {absTarget}")

    def saveFigure(self, name, savePath):
        """
        Safely saves the figure to the specified relative path.
        Creates any necessary subdirectories.
        Restricts saving to locations at or below the current working directory.
        """
        if self._is_noop:
            return
        fig = self.figures.get(name)
        if fig is None:
            raise ValueError(f"Figure '{name}' does not exist.")
            
        # Security Check
        baseDir = os.getcwd()
        absTarget = os.path.abspath(savePath)
        
        # Ensure the target directory is completely within the base directory
        # commonpath returns the longest common sub-path
        if os.path.commonpath([baseDir, absTarget]) != baseDir:
            raise PermissionError(f"Save path '{savePath}' is restricted. Cannot save above the current directory.")
            
        # Ensure it's not trying to save to a root directory directly (on Windows this would be C:\ or D:\)
        if os.path.dirname(absTarget) == absTarget:
            raise PermissionError("Cannot save directly to root directory.")
            
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(absTarget), exist_ok=True)
        
        import matplotlib
        backend = matplotlib.get_backend().lower()
        
        is_gpu = (os.environ.get('MOLMEM_GPU_AVAILABLE') == '1' and sys.platform == 'win32')
        if backend in _INTERACTIVE_BACKENDS and not is_gpu:
            # Save synchronously in the main thread to avoid GUI/Tkinter threading issues
            try:
                with self._lock:
                    self._apply_layout(fig)
                    fig.savefig(absTarget, pil_kwargs={'compress_level': 1})
                self._print_save_summary(name, absTarget)
            except Exception as e:
                print(f"Error saving figure '{name}': {e}")
        else:
            from .sys_utils import log_to_run_log_only
            log_to_run_log_only(f"Spawning Thread: Figure Saver ({name}) | Target=Background Plot Renderer")
            
            def _save_task():
                from .sys_utils import log_to_run_log_only
                import threading
                cur_t = threading.current_thread()
                log_to_run_log_only(f"Thread {cur_t.name} initialized: Target=Background Plot Renderer (Saving {name})")
                try:
                    with self._lock:
                        self._apply_layout(fig)
                    fig.savefig(absTarget, pil_kwargs={'compress_level': 1})
                    self._print_save_summary(name, absTarget)
                except Exception as e:
                    print(f"Error saving figure '{name}': {e}")
                finally:
                    try:
                        log_to_run_log_only(f"Thread {cur_t.name} terminated: Target=Background Plot Renderer (Saved {name})")
                    except Exception:
                        pass
                    
            thread = threading.Thread(target=_save_task, name=f"FigSaver_{name}")
            thread.daemon = True
            with self._lock:
                self._save_threads.append(thread)
            thread.start()

    def wait_all(self):
        """Wait for all background figure saving operations to complete."""
        if self._is_noop:
            return
        with self._lock:
            threads = list(self._save_threads)
        for t in threads:
            if t.is_alive():
                t.join()
        with self._lock:
            self._save_threads.clear()

    def renderFigure(self, name):
        """Displays the specified figure interactively in a live plot window."""
        if self._is_noop:
            return
        import matplotlib
        import matplotlib.pyplot as plt
        from .sys_utils import is_interactive_notebook
        is_nb = False
        try:
            is_nb = is_interactive_notebook()
        except Exception:
            pass

        backend_name = matplotlib.get_backend().lower()
        is_headless = False
        if not is_nb:
            if sys.platform.startswith('linux'):
                is_headless = not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
            elif os.environ.get('MOLMEM_HEADLESS') == '1':
                is_headless = True
        if is_headless:
            print(f"[*] MolmemPlotter > Headless/non-interactive display environment detected (backend: {backend_name}).")
            print(f"    Figure '{name}' cannot be shown interactively; please save it manually using the API:")
            print(f"    plotter.saveFigure('{name}', 'sim_results/{name}.png')")
            return
        print(f"[>] MolmemPlotter > Rendering figure '{name}'")
        fig = self.figures.get(name)
        if fig is None:
            raise ValueError(f"Figure '{name}' does not exist.")
            
        self._apply_layout(fig)
        
        if is_nb:
            try:
                from IPython.display import display
                display(fig)
            except Exception:
                plt.figure(fig.number)
                plt.show()
            finally:
                plt.close(fig)
            return

        # Use subprocess if backend is non-interactive (Agg) or if PyTorch GPU is active on Windows to bypass UI deadlocks
        is_gpu = (os.environ.get('MOLMEM_GPU_AVAILABLE') == '1' and sys.platform == 'win32')
        use_subprocess = (not is_nb) and (backend_name == 'agg' or is_gpu)

        if use_subprocess:
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.abspath(os.path.join(temp_dir, f".molmem_temp_{name}_{os.getpid()}.pkl"))
            try:
                with open(temp_path, 'wb') as f:
                    pickle.dump(fig, f)
                code = f"""
import os
import sys
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'{_PARENT_LIB_DIR}')
try:
    from molmem_lib.sys_utils import log_to_run_log_only
    log_to_run_log_only(f"Subprocess {{os.getpid()}} initialized: Platform=CPU (Matplotlib GUI renderer) | Threads=OS Managed | Core Affinity=OS Managed")
except Exception:
    pass

temp_path = r'{temp_path}'
with open(temp_path, 'rb') as f:
    fig = pickle.load(f)
try:
    os.remove(temp_path)
except Exception:
    pass

fig.canvas.mpl_connect('close_event', lambda e: os._exit(0))
plt.show()
"""
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = ''
                env['HIP_VISIBLE_DEVICES'] = ''
                env['MOLMEM_CPU_ONLY'] = '1'
                env['MOLMEM_FORCE_GPU'] = '0'
                env['MOLMEM_MP_CHILD'] = '1'
                try:
                    from .sys_utils import log_to_run_log_only
                    log_to_run_log_only("Spawning Subprocess (Popen): Matplotlib Interactive Figure Render Window | Platform=CPU (Matplotlib GUI renderer) | Threads=OS Managed | Core Affinity=OS Managed")
                except Exception:
                    pass
                    
                proc = subprocess.Popen(
                    [sys.executable, "-c", code],
                    env=env,
                    close_fds=True,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=0x08000000 if sys.platform == 'win32' else 0
                )
                molmem_lib = sys.modules.get('molmem_lib')
                if molmem_lib is not None:
                    getattr(molmem_lib, "_active_plot_processes", []).append(proc)
                try:
                    proc.wait()
                except KeyboardInterrupt:
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    raise
                finally:
                    try:
                        from .sys_utils import log_to_run_log_only
                        log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess (Popen) terminated: Matplotlib Interactive Figure Render Window | Platform=CPU (Matplotlib GUI renderer)")
                    except Exception:
                        pass
                    if molmem_lib is not None:
                        try:
                            getattr(molmem_lib, "_active_plot_processes", []).remove(proc)
                        except ValueError:
                            pass
            except Exception as e:
                print(f"Error launching subprocess plotter: {e}")
            finally:
                try:
                    from .sys_utils import safe_remove_file
                    safe_remove_file(temp_path)
                except Exception:
                    pass
            return

        # Standard non-GPU path
        plt.figure(fig.number)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            plt.close(fig)

    def showAll(self):
        """Displays all open figures interactively."""
        if self._is_noop:
            return
        self.wait_all()
        import matplotlib
        import matplotlib.pyplot as plt
        from .sys_utils import is_interactive_notebook
        is_nb = False
        try:
            is_nb = is_interactive_notebook()
        except Exception:
            pass

        fig_nums = plt.get_fignums()
        backend_name = matplotlib.get_backend().lower()
        is_headless = False
        if not is_nb:
            if sys.platform.startswith('linux'):
                is_headless = not (os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))
            elif os.environ.get('MOLMEM_HEADLESS') == '1':
                is_headless = True
        if is_headless:
            print(f"[*] MolmemPlotter > Headless/non-interactive display environment detected (backend: {backend_name}).")
            print(f"    Figures cannot be shown interactively; please save them manually using the API:")
            print(f"    e.g.: plotter.saveFigure('<figure_name>', 'sim_results/<filename>.png')")
            return
        print(f"[>] MolmemPlotter > Rendering all open figures ({len(fig_nums)} figures)")
        
        if is_nb:
            try:
                from IPython.display import display
                for num in fig_nums:
                    display(plt.figure(num))
            except Exception:
                plt.show()
            finally:
                plt.close('all')
            return

        # Spawn interactive GUI via subprocess to prevent main-process UI deadlocks and Agg conflicts
        use_subprocess = (not is_nb) and (backend_name == 'agg' or sys.platform == 'win32')

        if use_subprocess:
            temp_paths = []
            try:
                code = f"""
import os
import sys
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, r'{_PARENT_LIB_DIR}')
try:
    from molmem_lib.sys_utils import log_to_run_log_only
    log_to_run_log_only(f"Subprocess {{os.getpid()}} initialized: Platform=CPU (Matplotlib GUI renderer) | Threads=OS Managed | Core Affinity=OS Managed")
except Exception:
    pass

open_figs = {len(fig_nums)}
def on_close(e):
    global open_figs
    open_figs -= 1
    if open_figs <= 0:
        os._exit(0)
"""
                import tempfile
                temp_dir = tempfile.gettempdir()
                for i, num in enumerate(fig_nums):
                    fig = plt.figure(num)
                    temp_path = os.path.abspath(os.path.join(temp_dir, f".molmem_temp_all_{num}_{os.getpid()}.pkl"))
                    with open(temp_path, 'wb') as f:
                        pickle.dump(fig, f)
                    temp_paths.append(temp_path)
                    code += f"""
p{i} = r'{temp_path}'
with open(p{i}, 'rb') as f:
    fig{i} = pickle.load(f)
try:
    os.remove(p{i})
except Exception:
    pass
fig{i}.canvas.mpl_connect('close_event', on_close)
"""
                code += "\nplt.show()\n"
                if temp_paths:
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = ''
                    env['HIP_VISIBLE_DEVICES'] = ''
                    env['MOLMEM_CPU_ONLY'] = '1'
                    env['MOLMEM_FORCE_GPU'] = '0'
                    env['MOLMEM_MP_CHILD'] = '1'
                    try:
                        from .sys_utils import log_to_run_log_only
                        log_to_run_log_only("Spawning Subprocess (Popen): Matplotlib Interactive Multi-Figure Render Window | Platform=CPU (Matplotlib GUI renderer) | Threads=OS Managed | Core Affinity=OS Managed")
                    except Exception:
                        pass
                        
                    proc = subprocess.Popen(
                        [sys.executable, "-c", code],
                        env=env,
                        close_fds=True,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=0x08000000 if sys.platform == 'win32' else 0
                    )
                    molmem_lib = sys.modules.get('molmem_lib')
                    if molmem_lib is not None:
                        getattr(molmem_lib, "_active_plot_processes", []).append(proc)
                    # Pause the line profiler before waiting for user interaction on the Matplotlib figure
                    prof = getattr(molmem_lib, "_global_line_profiler", None)
                    if prof is not None:
                        try:
                            prof.stop()
                        except Exception:
                            pass
                            
                    try:
                        proc.wait()
                    except KeyboardInterrupt:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                        raise
                    finally:
                        try:
                            from .sys_utils import log_to_run_log_only
                            log_to_run_log_only(f"[WORKER_SPAWN] [PID {os.getpid()}] Subprocess (Popen) terminated: Matplotlib Interactive Multi-Figure Render Window | Platform=CPU (Matplotlib GUI renderer)")
                        except Exception:
                            pass
                        # Resume the line profiler once the wait completes
                        if prof is not None:
                            try:
                                prof.start()
                            except Exception:
                                pass
                        if molmem_lib is not None:
                            try:
                                getattr(molmem_lib, "_active_plot_processes", []).remove(proc)
                            except ValueError:
                                pass
            except Exception as e:
                print(f"Error launching subprocess plotter: {e}")
            finally:
                for path in temp_paths:
                    try:
                        from .sys_utils import safe_remove_file
                        safe_remove_file(path)
                    except Exception:
                        pass
            return

        # Standard non-GPU path
        try:
            plt.show()
        except KeyboardInterrupt:
            pass
        except Exception as e:
            err_msg = str(e).lower()
            if "display" in err_msg or "tcl" in err_msg or "tkinter" in err_msg:
                sys.stderr.write(f"\n[*] Headless/Non-GUI display environment detected ({e}). Saving figures to disk instead.\n")
                sys.stderr.flush()
                try:
                    out_dir = os.path.join(os.getcwd(), "sim_results")
                    os.makedirs(out_dir, exist_ok=True)
                    for i, fig_num in enumerate(plt.get_fignums(), 1):
                        fig = plt.figure(fig_num)
                        fig_path = os.path.join(out_dir, f"figure_{i}.png")
                        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
                    sys.stderr.write(f"[*] Saved {len(plt.get_fignums())} figure(s) to '{out_dir}'.\n")
                    sys.stderr.flush()
                except Exception:
                    pass
            else:
                sys.stderr.write(f"\n[!] Plotting display error: {e}\n")
                sys.stderr.flush()
        finally:
            plt.close('all')
