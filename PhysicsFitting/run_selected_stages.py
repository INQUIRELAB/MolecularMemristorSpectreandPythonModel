import sys
import builtins
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import runpy

# Ensure matplotlib uses a non-interactive backend so it doesn't try to open windows
matplotlib.use('Agg')

# =============================================================================
# Logger Setup
# =============================================================================
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.__stdout__
        self.log = open(filename, "w", encoding='utf-8')
   
    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode('ascii', 'replace').decode('ascii'))
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

base_dir = Path(__file__).parent
log_file_path = base_dir / "pipeline_results_selected.txt"

# =============================================================================
# Monkey Patch plt.show (To save figures automatically)
# =============================================================================
figures_dir = base_dir / "Figures"
figures_dir.mkdir(exist_ok=True)
fig_counter = 1

def mocked_show(*args, **kwargs):
    global fig_counter
    # Get all active figures
    fig_managers = plt._pylab_helpers.Gcf.get_all_fig_managers()
    
    plot_data_dir = base_dir / "PlotData"
    plot_data_dir.mkdir(exist_ok=True)
    
    for manager in fig_managers:
        fig = manager.canvas.figure
        
        # Try to use the figure title as filename if possible
        title = "figure"
        if fig._suptitle:
            title = fig._suptitle.get_text()
        elif fig.axes:
            ax_title = fig.axes[0].get_title()
            if ax_title:
                title = ax_title
                
        title_clean = "".join([c if c.isalnum() else "_" for c in title]).strip("_")
        if not title_clean:
            title_clean = "figure"
            
        # 1. Save PNG
        png_filename = figures_dir / f"{fig_counter:02d}_{title_clean}.png"
        fig.savefig(png_filename, dpi=300, bbox_inches='tight')
        print(f"\n[Pipeline] Saved figure to {png_filename}")
        
        # 2. Extract Data to JSON
        fig_data = {'title': title, 'axes': []}
        import json
        import numpy as np
        
        def sanitize_val(v):
            if isinstance(v, np.ndarray): return sanitize_val(v.tolist())
            if isinstance(v, list): return [sanitize_val(x) for x in v]
            if isinstance(v, (np.float32, np.float64)): return float(v)
            if isinstance(v, (np.int32, np.int64)): return int(v)
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)): return None
            return v
            
        for ax in fig.axes:
            ax_data = {
                'title': ax.get_title(),
                'xlabel': ax.get_xlabel(),
                'ylabel': ax.get_ylabel(),
                'xlim': ax.get_xlim(),
                'ylim': ax.get_ylim(),
                'xscale': ax.get_xscale(),
                'yscale': ax.get_yscale(),
                'lines': [],
                'texts': [],
                'legend': None
            }
            
            # Extract lines
            for line in ax.get_lines():
                xd = line.get_xdata()
                yd = line.get_ydata()
                if len(xd) == 0: continue
                ax_data['lines'].append({
                    'label': line.get_label(),
                    'x': sanitize_val(xd),
                    'y': sanitize_val(yd),
                    'color': str(line.get_color()) if not isinstance(line.get_color(), np.ndarray) else "tuple",
                    'linestyle': line.get_linestyle(),
                    'linewidth': line.get_linewidth(),
                    'marker': line.get_marker(),
                    'alpha': line.get_alpha()
                })
                
            # Extract floating texts
            for t in ax.texts:
                ax_data['texts'].append({
                    'text': t.get_text(),
                    'x': t.get_position()[0],
                    'y': t.get_position()[1],
                    'fontsize': t.get_fontsize(),
                    'ha': t.get_ha(),
                    'va': t.get_va(),
                    'has_bbox': True if t.get_bbox_patch() else False
                })
                
            # Extract custom legend
            leg = ax.get_legend()
            if leg:
                handles_data = []
                try:
                    for h, t in zip(leg.legend_handles, leg.get_texts()):
                        color = h.get_color() if hasattr(h, 'get_color') else 'black'
                        ls = h.get_linestyle() if hasattr(h, 'get_linestyle') else '-'
                        lw = h.get_linewidth() if hasattr(h, 'get_linewidth') else 1.0
                        handles_data.append({
                            'label': t.get_text(),
                            'color': str(color) if not isinstance(color, np.ndarray) else "tuple",
                            'linestyle': ls,
                            'linewidth': lw
                        })
                except Exception:
                    pass
                    
                ax_data['legend'] = {
                    'texts': [t.get_text() for t in leg.get_texts()],
                    'custom_handles': handles_data
                }
                
            fig_data['axes'].append(ax_data)
            
        json_filename = plot_data_dir / f"{fig_counter:02d}_{title_clean}.json"
        try:
            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(fig_data, f, indent=2)
            print(f"[Pipeline] Saved raw plot data to {json_filename}")
        except Exception as e:
            print(f"[Error] Failed to save JSON plot data: {e}")
            
        fig_counter += 1
        
    plt.close('all')

plt.show = mocked_show

# =============================================================================
# Interactive Menu and Input Parsing
# =============================================================================
def get_selected_stages():
    stages = [
        ("STAGE 1: phi_from_probability.py", "physics_fitting_lib/phi_from_probability.py"),
        ("STAGE 2: RegularizedFitting.py", "physics_fitting_lib/RegularizedFitting.py"),
        ("STAGE 3: UniversalModel.py", "physics_fitting_lib/UniversalModel.py"),
        ("STAGE 4: ConductanceFitting.py", "physics_fitting_lib/ConductanceFitting.py")
    ]
    
    print("\n" + "="*64)
    print("          SELECT PHYSICS FITTING STAGES TO RUN")
    print("="*64)
    for idx, (name, _) in enumerate(stages):
        print(f" [{idx + 1}] {name}")
    print(" [5] Run All Stages")
    print(" [6] Exit")
    print("="*64)
    
    try:
        user_input = input("\nEnter stage number(s) (e.g., '1', '1,3', '2-4', or '5' for all): ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        return []
        
    if not user_input or user_input in ['6', 'exit', 'q', 'quit']:
        return []
        
    if user_input == '5' or user_input == 'all':
        return stages
        
    selected = []
    # Parse ranges and list
    parts = user_input.replace(' ', '').split(',')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                for i in range(start, end + 1):
                    if 1 <= i <= len(stages):
                        selected.append(stages[i - 1])
            except ValueError:
                pass
        else:
            try:
                i = int(part)
                if 1 <= i <= len(stages):
                    selected.append(stages[i - 1])
            except ValueError:
                pass
                
    # Remove duplicates preserving order
    unique_selected = []
    for s in selected:
        if s not in unique_selected:
            unique_selected.append(s)
            
    return unique_selected

if __name__ == "__main__":
    selected_stages = get_selected_stages()
    if not selected_stages:
        print("No stages selected. Exiting.")
        sys.exit(0)
        
    auto_answer = input("Auto-answer 'y' to script internal prompts? [Y/n]: ").strip().lower()
    if auto_answer != 'n':
        # Monkey Patch Input
        original_input = builtins.input
        def mocked_input(prompt=""):
            print(prompt + "y (auto-answered)")
            return "y"
        builtins.input = mocked_input
        print("[Pipeline] Auto-answer enabled.")
    else:
        print("[Pipeline] Fully interactive mode enabled. You will be prompted by internal scripts.")
        
    # Start logging after interactive prompt is done
    sys.stdout = Logger(log_file_path)
    
    start_time = time.time()
    print("\n================================================================")
    print("          SELECTED PHYSICS MODELING PIPELINE EXECUTION          ")
    print("================================================================")
    print(f"Logging output to {log_file_path}")
    
    for stage_name, script_file in selected_stages:
        script_path = base_dir / script_file
        print(f"\n\n{'='*64}")
        print(f"  {stage_name}")
        print(f"{'='*64}\n")
        
        if not script_path.exists():
            print(f"[Error] Script {script_file} not found in {base_dir}")
            continue
            
        stage_start = time.time()
        try:
            # Temporarily change argv so the script thinks it's being run directly
            old_argv = sys.argv
            sys.argv = [str(script_path)]
            
            runpy.run_path(str(script_path), run_name='__main__')
            
            sys.argv = old_argv
        except Exception as e:
            if isinstance(e, FileNotFoundError) and "Probability Matrix" in str(e):
                print(f"\n[Warning] {stage_name} skipped. 'Probability Matrix.xlsx' not found.\nThis is expected if you are relying on the pre-generated 'Phi_from_probability.csv'.")
            else:
                print(f"\n[Error] {stage_name} failed with exception: {e}")
                import traceback
                traceback.print_exc(file=sys.stdout)
            
        stage_elapsed = time.time() - stage_start
        print(f"\n[Pipeline] {stage_name} completed in {stage_elapsed:.2f} seconds.")
        
    total_elapsed = time.time() - start_time
    print(f"\n\n{'='*64}")
    print(f"  PIPELINE FINISHED IN {total_elapsed:.2f} SECONDS")
    print(f"  All logs written to {log_file_path}")
    print(f"  All figures saved to {figures_dir}")
    print(f"{'='*64}\n")
