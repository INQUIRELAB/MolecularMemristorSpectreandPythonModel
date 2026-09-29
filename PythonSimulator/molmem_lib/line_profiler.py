import os
import sys
from time import perf_counter
import datetime
import threading
import multiprocessing

_perf_counter = perf_counter

def _init_line_profiler_worker(worker_type):
    import os
    import multiprocessing
    import re
    import atexit
    try:
        proc_name = multiprocessing.current_process().name
        match = re.search(r'\d+$', proc_name)
        idx = int(match.group(0)) if match else os.getpid()
    except Exception:
        idx = os.getpid()
        
    try:
        from .sys_utils import log_to_run_log_only
        log_to_run_log_only(f"Worker {idx} initialized: Platform=CPU ({worker_type}) | Threads=1/worker | Core Affinity=OS Managed")
        
        import multiprocessing.util as mp_util
        def _exit_worker():
            try:
                log_to_run_log_only(f"Worker {idx} terminated: Platform=CPU ({worker_type})")
            except Exception:
                pass
        mp_util.Finalize(None, _exit_worker, exitpriority=0)
    except Exception:
        pass

_thread_local = threading.local()
_gpu_telemetry_lock = threading.Lock()
_gpu_telemetry_records = []

def record_gpu_kernel_telemetry(telemetry):
    """Records hardware cycle telemetry from compiled C++/HIP GPU kernels."""
    with _gpu_telemetry_lock:
        _gpu_telemetry_records.append(telemetry)

def get_gpu_kernel_telemetry():
    with _gpu_telemetry_lock:
        return list(_gpu_telemetry_records)

_CODE_WHITELIST = {}

def _check_code_object(co):
    try:
        return _CODE_WHITELIST[co]
    except KeyError:
        filename = co.co_filename
        if filename:
            norm_filename = filename.replace('\\', '/')
            is_library_file = (
                'molmem_lib' in norm_filename or 
                'physics_fitting_lib' in norm_filename or 
                os.environ.get("MOLMEM_PROFILE_ALL") == "1"
            ) and 'line_profiler.py' not in norm_filename
        else:
            is_library_file = False
            filename = ""
        entry = (is_library_file, filename)
        _CODE_WHITELIST[co] = entry
        return entry


class LineProfiler:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = False
        self._lib_dir = os.path.dirname(os.path.abspath(__file__))
        self._project_root = os.path.dirname(self._lib_dir)
        self._all_thread_dicts = []

    def trace_callback(self, frame, event, arg):
        if not self._active:
            return None
            
        tl = _thread_local
        try:
            if tl.is_ignored_thread:
                return None
        except AttributeError:
            thread_name = threading.current_thread().name
            is_ignored = (
                thread_name in {'TeeLogger', 'watchdog_loop', 'CudaWarmup', 'CacheSaver'} or
                thread_name.startswith(('MolmemHistWriter', 'ThreadPoolExecutor', 'HistWriter', 'Concurrent'))
            )
            tl.is_ignored_thread = is_ignored
            if is_ignored:
                sys.settrace(None)
                return None

        if event == 'line':
            # Fast global code object lookup (O(1) dictionary hit with zero lock contention)
            is_library_file, filename = _check_code_object(frame.f_code)

            if is_library_file:
                current_time = _perf_counter()
                
                try:
                    profile_data = tl.profile_data
                except AttributeError:
                    profile_data = {}
                    tl.profile_data = profile_data
                    tl.last_line_file = None
                    tl.last_line_no = None
                    tl.last_file_dict = None
                    tl.last_line_data = None
                    tl.last_exit_time = None
                    with self._lock:
                        self._all_thread_dicts.append(profile_data)
                
                # Use last_exit_time to measure only the actual line execution duration
                prev_time = tl.last_exit_time
                if prev_time is not None:
                    elapsed = current_time - prev_time
                    prev_line_data = tl.last_line_data
                    if prev_line_data is not None:
                        prev_line_data[1] += elapsed

                line_no = frame.f_lineno
                if tl.last_line_file == filename:
                    file_dict = tl.last_file_dict
                else:
                    try:
                        file_dict = profile_data[filename]
                    except KeyError:
                        file_dict = {}
                        profile_data[filename] = file_dict
                    tl.last_file_dict = file_dict
                    
                try:
                    line_data = file_dict[line_no]
                    line_data[0] += 1
                except KeyError:
                    line_data = [1, 0.0]
                    file_dict[line_no] = line_data

                tl.last_line_file = filename
                tl.last_line_no = line_no
                tl.last_line_data = line_data
                tl.last_exit_time = current_time  # Reuse entry timestamp instead of calling perf_counter again
                return self.trace_callback
            else:
                return None
                
        elif event == 'call':
            co = frame.f_code
            if co.co_name in {'find_spec', '_do_write', 'write_chunk_async', 'wait_for_writes', 'record_chunk', 'record_dma', 'record_expansion', 'record_flush_wait'}:
                return None
            is_library_file, _ = _check_code_object(co)
            if not is_library_file:
                return None
            return self.trace_callback
            
        return None

    def start(self):
        with self._lock:
            if self._active:
                return
            self._active = True
            tl = _thread_local
            if hasattr(tl, 'profile_data'):
                tl.last_line_file = None
                tl.last_line_no = None
                tl.last_line_data = None
                tl.last_exit_time = None
        sys.settrace(self.trace_callback)
        threading.settrace(self.trace_callback)
        try:
            frame = sys._getframe()
            while frame:
                frame.f_trace = self.trace_callback
                frame = frame.f_back
        except Exception:
            pass

    def stop(self):
        with self._lock:
            if not self._active:
                return
            self._active = False
        sys.settrace(None)
        threading.settrace(None)

    def get_data(self):
        merged = {}
        with self._lock:
            for thread_dict in self._all_thread_dicts:
                for filename, file_dict in thread_dict.items():
                    for line_no, (hits, elapsed) in file_dict.items():
                        line_key = (filename, line_no)
                        if line_key in merged:
                            merged_item = merged[line_key]
                            merged_item[0] += hits
                            merged_item[1] += elapsed
                        else:
                            merged[line_key] = [hits, elapsed]
        
        # Merge compiled C++/HIP GPU kernel telemetry into line execution profile
        gpu_hip_file = os.path.join(self._lib_dir, "csrc", "gpu_solver.hip")
        if os.path.exists(gpu_hip_file):
            with _gpu_telemetry_lock:
                for rec in _gpu_telemetry_records:
                    total_cyc = rec.get('total_cycles', 0)
                    wall_time = rec.get('wall_time', 0.0)
                    steps = max(1, rec.get('steps', 1))
                    if total_cyc > 0 and wall_time > 0:
                        cyc_to_sec = wall_time / total_cyc
                        # Granular C++ line breakdown in gpu_solver.hip
                        ode_total = rec.get('ode_cycles', 0)
                        pred_f = rec.get('pred_full_cycles', 0)
                        pred_h = rec.get('pred_half_cycles', 0)
                        lte_err = rec.get('lte_err_cycles', 0)
                        st_commit = rec.get('state_commit_cycles', 0)
                        ode_sub = pred_f + pred_h + lte_err + st_commit
                        rem_ode = max(0, ode_total - ode_sub)

                        gpu_line_map = [
                            (205, steps, rec.get('lut_cycles', 0) * cyc_to_sec),
                            (241, steps, rec.get('thermal_cycles', 0) * cyc_to_sec),
                            (268, steps, rec.get('rate_cycles', 0) * cyc_to_sec),
                            (310, steps, rec.get('boundary_cycles', 0) * cyc_to_sec),
                            (323, steps, rec.get('mode_cycles', 0) * cyc_to_sec),
                            (803, steps, rec.get('pwl_cycles', 0) * cyc_to_sec),
                            (910, max(1, rec.get('cg_iters', 0)), rec.get('assembly_cycles', 0) * cyc_to_sec),
                            (980, max(1, rec.get('cg_iters', 0)), rec.get('cg_cycles', 0) * cyc_to_sec),
                            (1413, steps, pred_f * cyc_to_sec),
                            (1440, steps, pred_h * cyc_to_sec),
                            (1466, steps, lte_err * cyc_to_sec),
                            (1518, steps, st_commit * cyc_to_sec),
                            (1618 if rec.get('num_nodes', 0) == 0 else 1574, steps, rem_ode * cyc_to_sec),
                            (1636, steps, rec.get('hist_cycles', 0) * cyc_to_sec),
                        ]
                        for l_no, hits, elap in gpu_line_map:
                            if elap > 0:
                                key = (gpu_hip_file, l_no)
                                if key in merged:
                                    merged[key][0] += hits
                                    merged[key][1] += elap
                                else:
                                    merged[key] = [hits, elap]
        return merged

def _read_file_lines(filename):
    try:
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            return filename, f.readlines()
    except Exception:
        return filename, []

def _format_profile_row_task(args):
    line_key, hits, total_time, code_snippet, project_root = args
    filename, line_no = line_key
    if filename.startswith(project_root):
        rel_path = filename[len(project_root):].lstrip('\\/').replace('\\', '/')
    else:
        rel_path = filename.replace('\\', '/')

    avg_time = (total_time / hits) if hits > 0 else 0.0

    # Unit formats
    if total_time >= 1.0:
        total_time_str = f"{total_time:.4f} s"
    elif total_time >= 0.001:
        total_time_str = f"{total_time * 1000.0:.2f} ms"
    else:
        total_time_str = f"{total_time * 1e6:.1f} us"

    if avg_time >= 1.0:
        avg_time_str = f"{avg_time:.3f} s"
    elif avg_time >= 0.001:
        avg_time_str = f"{avg_time * 1000.0:.2f} ms"
    else:
        avg_time_str = f"{avg_time * 1e6:.1f} us"

    return line_key, f"{rel_path:<30} | {line_no:<6} | {hits:<8} | {total_time_str:<12} | {avg_time_str:<10} | {code_snippet}"


def run_profiler_logger(profile_data, lib_dir):
    """
    formats multiple ASCII report tables
    sorted by different categories (Total Time, Hits, Avg/Hit), and
    a Top 50 summary of all categories.
    """
    if not profile_data:
        return

    try:
        # Create centralized logs directory
        log_dir = os.path.join(lib_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        project_root = os.path.dirname(lib_dir)

        # 2. Format the report timestamps and parameters
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        total_lines_traced = len(profile_data)
        total_time_spent = sum(v[1] for v in profile_data.values())

        # Determine term columns width
        width = 120
        border = "=" * width
        divider = "-" * width

        unique_files = list({line_key[0] for line_key in profile_data.keys() if line_key[0]})
        file_lines_cache = {}
        for filename in unique_files:
            res = _read_file_lines(filename)
            file_lines_cache[res[0]] = res[1]

        formatted_rows_cache = {}

        def get_formatted_row(k):
            cached = formatted_rows_cache.get(k)
            if cached is not None:
                return cached
            metrics = profile_data.get(k)
            if metrics is None:
                return ""
            filename, line_no = k
            hits, total_time = metrics
            lines = file_lines_cache.get(filename, [])
            code_snippet = ""
            if 1 <= line_no <= len(lines):
                code_snippet = lines[line_no - 1].strip()
            row_args = (k, hits, total_time, code_snippet, project_root)
            res_key, row_content = _format_profile_row_task(row_args)
            formatted_rows_cache[res_key] = row_content
            return row_content

        import heapq

        # Sort strategies
        # 1. Total Time
        sorted_by_time = sorted(
            profile_data.keys(),
            key=lambda k: profile_data[k][1],
            reverse=True
        )
        # 2. Hits
        sorted_by_hits = sorted(
            profile_data.keys(),
            key=lambda k: profile_data[k][0],
            reverse=True
        )
        # 3. Average time per hit
        sorted_by_avg = sorted(
            profile_data.keys(),
            key=lambda k: (profile_data[k][1] / profile_data[k][0]) if profile_data[k][0] > 0 else 0.0,
            reverse=True
        )

        # Helper to construct individual log tables
        def build_report(sorted_keys, title_desc, limit=None, custom_metrics=None):
            report_lines = []
            report_lines.append(border)
            report_lines.append(f"  MOLMEM LINE-BY-LINE EXECUTION PROFILE REPORT - {title_desc}")
            report_lines.append(border)
            report_lines.append(f"  Timestamp   : {ts}")
            report_lines.append(f"  Platform    : {sys.platform} ({os.name})")
            if custom_metrics:
                for k_name, v_val in custom_metrics.items():
                    report_lines.append(f"  {k_name:<12}: {v_val}")
            else:
                report_lines.append(f"  Total Lines : {total_lines_traced}")
                report_lines.append(f"  Total Time  : {total_time_spent:.6f} seconds")
            report_lines.append(border)
            report_lines.append("")
            report_lines.append(f"  TOP LINES BY {title_desc}:")
            report_lines.append(divider)
            report_lines.append(
                f"  {'Rank':<5} | {'File Path (Relative)':<30} | {'Line #':<6} | {'Hits':<8} | {'Total Time':<12} | {'Avg/Hit':<10} | {'Code Snippet'}"
            )
            report_lines.append(divider)
            
            keys_to_process = sorted_keys[:limit] if limit is not None else sorted_keys
            for rank_idx, k in enumerate(keys_to_process):
                rank = rank_idx + 1
                row_content = get_formatted_row(k)
                report_lines.append(f"  {rank:<5} | {row_content}")
            
            report_lines.append(divider)
            report_lines.append("")
            return "\n".join(report_lines) + "\n"

        # Write primary report (sorted by time)
        with open(os.path.join(log_dir, "profile_log.txt"), 'w', encoding='utf-8') as f:
            f.write(build_report(sorted_by_time, "TOTAL TIME CONSUMED"))

        # Write by hits report
        with open(os.path.join(log_dir, "profile_log_by_hits.txt"), 'w', encoding='utf-8') as f:
            f.write(build_report(sorted_by_hits, "EXECUTION HITS"))

        # Write by avg/hit report
        with open(os.path.join(log_dir, "profile_log_by_avg.txt"), 'w', encoding='utf-8') as f:
            f.write(build_report(sorted_by_avg, "AVERAGE TIME PER HIT"))

        # Write top 50 summary report combining all categories
        summary_filepath = os.path.join(log_dir, "profile_log_summary.txt")
        top_by_time = sorted_by_time[:50]
        top_by_hits = sorted_by_hits[:50]
        top_by_avg = sorted_by_avg[:50]
        
        summary_content = ""
        summary_content += build_report(top_by_time, "TOTAL TIME CONSUMED (TOP 50)", limit=50)
        summary_content += "\n\n"
        summary_content += build_report(top_by_hits, "EXECUTION HITS (TOP 50)", limit=50)
        summary_content += "\n\n"
        summary_content += build_report(top_by_avg, "AVERAGE TIME PER HIT (TOP 50)", limit=50)

        with open(summary_filepath, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        # Write individual per-module profile summaries in simulation_profiles/ subdirectory
        sim_dir = os.path.join(log_dir, "simulation_profiles")
        os.makedirs(sim_dir, exist_ok=True)

        file_to_keys = {}
        for line_key in profile_data.keys():
            fn = line_key[0]
            if fn:
                if fn not in file_to_keys:
                    file_to_keys[fn] = []
                file_to_keys[fn].append(line_key)

        for fn, f_keys in file_to_keys.items():
            base_name = os.path.basename(fn)
            mod_name, _ = os.path.splitext(base_name)
            
            n_keys = len(f_keys)
            if n_keys <= 50:
                mod_by_time = sorted(f_keys, key=lambda k: profile_data[k][1], reverse=True)
                mod_by_hits = sorted(f_keys, key=lambda k: profile_data[k][0], reverse=True)
                mod_by_avg = sorted(f_keys, key=lambda k: (profile_data[k][1] / profile_data[k][0]) if profile_data[k][0] > 0 else 0.0, reverse=True)
            else:
                mod_by_time = heapq.nlargest(50, f_keys, key=lambda k: profile_data[k][1])
                mod_by_hits = heapq.nlargest(50, f_keys, key=lambda k: profile_data[k][0])
                mod_by_avg = heapq.nlargest(50, f_keys, key=lambda k: (profile_data[k][1] / profile_data[k][0]) if profile_data[k][0] > 0 else 0.0)

            mod_total_lines = n_keys
            mod_total_time = sum(profile_data[k][1] for k in f_keys)

            if fn.startswith(project_root):
                rel_mod_path = fn[len(project_root):].lstrip('\\/').replace('\\', '/')
            else:
                rel_mod_path = fn.replace('\\', '/')

            mod_metrics = {
                "Module": rel_mod_path,
                "Total Lines": str(mod_total_lines),
                "Total Time": f"{mod_total_time:.6f} seconds"
            }

            mod_summary_content = ""
            mod_summary_content += build_report(mod_by_time, f"{mod_name.upper()} - TOTAL TIME CONSUMED (TOP 50)", limit=50, custom_metrics=mod_metrics)
            mod_summary_content += "\n\n"
            mod_summary_content += build_report(mod_by_hits, f"{mod_name.upper()} - EXECUTION HITS (TOP 50)", limit=50, custom_metrics=mod_metrics)
            mod_summary_content += "\n\n"
            mod_summary_content += build_report(mod_by_avg, f"{mod_name.upper()} - AVERAGE TIME PER HIT (TOP 50)", limit=50, custom_metrics=mod_metrics)

            mod_summary_path = os.path.join(sim_dir, f"{mod_name}_summary.txt")
            with open(mod_summary_path, 'w', encoding='utf-8') as f:
                f.write(mod_summary_content)

        # Write granular SSD pipeline execution report in simulation_profiles/
        try:
            from .history import generate_ssd_pipeline_report
            generate_ssd_pipeline_report(sim_dir)
        except Exception:
            pass

        # Write GPU-specific line profile reports into logs/gpu_profile/ subdirectory
        gpu_dir = os.path.join(log_dir, "gpu_profile")
        os.makedirs(gpu_dir, exist_ok=True)

        gpu_keys = [k for k in profile_data.keys() if k[0] and os.path.basename(k[0]).lower().endswith(('.hip', '.cu', '.cpp', '_cuda_solver.pyd'))]
        if gpu_keys:
            gpu_by_time = sorted(gpu_keys, key=lambda k: profile_data[k][1], reverse=True)
            gpu_by_hits = sorted(gpu_keys, key=lambda k: profile_data[k][0], reverse=True)
            gpu_by_avg = sorted(gpu_keys, key=lambda k: (profile_data[k][1] / profile_data[k][0]) if profile_data[k][0] > 0 else 0.0, reverse=True)

            gpu_total_time = sum(profile_data[k][1] for k in gpu_keys)
            gpu_metrics = {
                "Platform": "GPU (Compiled C++/HIP Kernel Engine)",
                "Tracked Kernel Lines": str(len(gpu_keys)),
                "Total Kernel Time": f"{gpu_total_time:.6f} seconds",
            }

            gpu_report_content = ""
            gpu_report_content += build_report(gpu_by_time, "GPU SOLVER KERNEL - TOTAL TIME BY C++ SOURCE LINE", limit=100, custom_metrics=gpu_metrics)
            gpu_report_content += "\n\n"
            gpu_report_content += build_report(gpu_by_hits, "GPU SOLVER KERNEL - EXECUTION HITS (STEPS)", limit=100, custom_metrics=gpu_metrics)
            gpu_report_content += "\n\n"
            gpu_report_content += build_report(gpu_by_avg, "GPU SOLVER KERNEL - AVERAGE TIME PER TIMESTEP", limit=100, custom_metrics=gpu_metrics)

            gpu_summary_file = os.path.join(gpu_dir, "gpu_profile_summary.txt")
            with open(gpu_summary_file, 'w', encoding='utf-8') as f:
                f.write(gpu_report_content)
            with open(os.path.join(gpu_dir, "gpu_line_profile.txt"), 'w', encoding='utf-8') as f:
                f.write(gpu_report_content)

    except Exception:
        pass

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        os.environ['MOLMEM_WORKER_ROLE'] = 'Line Profiler Generator'
        pkl_path = sys.argv[1]
        lib_dir = sys.argv[2]
        
        try:
            from molmem_lib.sys_utils import log_to_run_log_only
            import psutil
            parent_pid = psutil.Process().ppid()
            try:
                parent_name = psutil.Process(parent_pid).name()
            except Exception:
                parent_name = "Unknown"
            log_to_run_log_only(f"Subprocess {os.getpid()} initialized: Platform=CPU (Line Profiler Generator) | ParentPID={parent_pid} ({parent_name})")
        except Exception:
            pass
        
        try:
            # Load profile data from the temporary pickle file with retry locks
            import pickle
            import time
            profile_data = {}
            raw_data = None
            for _ in range(10):
                try:
                    with open(pkl_path, 'rb') as f:
                        raw_data = pickle.load(f)
                    break
                except (EOFError, pickle.UnpicklingError, PermissionError):
                    time.sleep(0.02)
                except Exception:
                    time.sleep(0.02)
                    
            if raw_data is not None:
                # Reconstruct dictionary with tuple keys (unpickled as lists from JSON/Pickle occasionally)
                for k, v in raw_data.items():
                    if isinstance(k, list):
                        k = tuple(k)
                    filename = k[0]
                    if filename:
                        base = os.path.basename(filename).lower()
                        if base in ("line_profiler.py", "profiler.py", "sys_utils.py"):
                            continue
                    profile_data[k] = v
        except Exception:
            pass
                
            if profile_data:
                # 1. Load existing aggregate data if available
                agg_path = os.path.join(lib_dir, "logs", ".aggregate_profile.pkl")
                agg_data = {}
                if os.path.exists(agg_path):
                    try:
                        with open(agg_path, 'rb') as f:
                            raw_agg = pickle.load(f)
                        for k, v in raw_agg.items():
                            if isinstance(k, list):
                                k = tuple(k)
                            filename = k[0]
                            if filename:
                                base = os.path.basename(filename).lower()
                                if base in ("line_profiler.py", "profiler.py", "sys_utils.py"):
                                    continue
                            agg_data[k] = v
                    except Exception:
                        pass
                        
                # 2. Merge new profile data into aggregate (accumulate hits and time)
                for k, v in profile_data.items():
                    if k in agg_data:
                        agg_data[k][0] += v[0]  # hits
                        agg_data[k][1] += v[1]  # total_time
                    else:
                        agg_data[k] = v
                        
                # 3. Write merged aggregate data back to file
                try:
                    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
                    with open(agg_path, 'wb') as f:
                        pickle.dump(agg_data, f)
                except Exception:
                    pass
                    
                # 4. Generate the final log report from the aggregate data
                if agg_data:
                    run_profiler_logger(agg_data, lib_dir)

            # 5. Clean up temporary pickle file at the very end of processing
            # Use a retry loop with delays to ensure Windows has fully released the file lock
            try:
                import time
                for _ in range(10):
                    if os.path.exists(pkl_path):
                        try:
                            os.remove(pkl_path)
                            break
                        except Exception:
                            time.sleep(0.02)
                    else:
                        break
            except Exception:
                pass
        finally:
            try:
                from molmem_lib.sys_utils import log_to_run_log_only
                log_to_run_log_only(f"Subprocess {os.getpid()} terminated: Platform=CPU (Line Profiler Generator)")
            except Exception:
                pass
