import os
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import math
import csv
from PIL import Image, ImageTk

class PlotDigitizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Molecular Memristor Plot Digitizer")
        self.root.geometry("1100x750")
        
        # Directories
        self.fitting_dir = os.path.dirname(os.path.abspath(__file__))
        screenshots_path = os.path.join(self.fitting_dir, "Screenshots")
        self.screenshot_dir = screenshots_path if os.path.exists(screenshots_path) else self.fitting_dir
        
        # Discover available device subdirectories (e.g. Ru_azo)
        self.device_list = self._get_device_dirs()
        self.selected_device = tk.StringVar(value=self.device_list[0] if self.device_list else "Ru_azo")
        self.output_dir = os.path.join(self.fitting_dir, self.selected_device.get())
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
            
        # Variables
        self.selected_file = tk.StringVar()
        self.status_text = tk.StringVar(value="Select an image file to begin.")
        self.x_min_val = tk.DoubleVar(value=0.0)
        self.x_max_val = tk.DoubleVar(value=100.0)
        self.y_min_val = tk.DoubleVar(value=0.0)
        self.y_max_val = tk.DoubleVar(value=100.0)
        self.log_x = tk.BooleanVar(value=False)
        self.log_y = tk.BooleanVar(value=False)
        self.dataset_name = tk.StringVar(value="extracted_data")
        
        # Auto-trace variables (initialized early for trace binding)
        self.tolerance_val = tk.DoubleVar(value=25.0)
        self.search_win_val = tk.IntVar(value=15)
        self.max_gap_val = tk.IntVar(value=20)
        self.smooth_win_val = tk.IntVar(value=1)
        self.drag_points_val = tk.IntVar(value=30)
        self.handle_size_val = tk.IntVar(value=3)
        self.start_click_x = None
        self.start_click_y = None
        self.dragged_point_idx = None
        
        # Zooming & Panning variables
        self.zoom_factor = 1.0
        self.pil_image = None
        
        # Bind traces for parameter autoupdates
        self.tolerance_val.trace_add("write", self._on_parameter_change)
        self.search_win_val.trace_add("write", self._on_parameter_change)
        self.max_gap_val.trace_add("write", self._on_parameter_change)
        self.smooth_win_val.trace_add("write", self._on_parameter_change)
        self.drag_points_val.trace_add("write", self._on_parameter_change)
        self.handle_size_val.trace_add("write", self._on_handle_size_change)
        self.x_min_val.trace_add("write", self._on_calibration_param_change)
        self.x_max_val.trace_add("write", self._on_calibration_param_change)
        self.y_min_val.trace_add("write", self._on_calibration_param_change)
        self.y_max_val.trace_add("write", self._on_calibration_param_change)
        self.log_x.trace_add("write", self._on_calibration_param_change)
        self.log_y.trace_add("write", self._on_calibration_param_change)
        
        # State: 'select', 'cal_box', 'digitize'
        self.state = 'select'
        self.loading_csv = False
        
        # Calibration Pixel Coordinates
        self.px_x_min = None
        self.px_x_max = None
        self.px_y_min = None
        self.px_y_max = None
        self.y_ticks = []  # List of dicts: {'pixel_y': float, 'value': float}
        
        # Extracted Points
        self.points = []
        self.target_color = None
        
        # UI Setup
        self._create_widgets()
        self._scan_images()
        self._scan_csvs()

    def _get_device_dirs(self):
        ignore_dirs = {"Figures", "__pycache__", "PlotData", "Screenshots", "TargetVsRawPlots"}
        devices = []
        fitting_dir = getattr(self, "fitting_dir", os.path.dirname(os.path.abspath(__file__)))
        if os.path.exists(fitting_dir):
            for entry in os.scandir(fitting_dir):
                if entry.is_dir() and entry.name not in ignore_dirs:
                    devices.append(entry.name)
        devices.sort()
        if not devices:
            devices = ["Ru_azo"]
        return devices

    def _on_device_change(self, event=None):
        dev = self.selected_device.get().strip()
        if not dev:
            return
        self.output_dir = os.path.join(self.fitting_dir, dev)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self._scan_csvs()

    def _create_widgets(self):
        # Main Layout: Left Control Panel (Scrollable), Right Scrollable Canvas
        left_container = ttk.Frame(self.root, width=305)
        left_container.pack(side=tk.LEFT, fill=tk.Y)
        left_container.pack_propagate(False)
        
        # Get standard background color for Canvas to match ttk style
        style = ttk.Style()
        bg_color = style.lookup("TFrame", "background")
        if not bg_color:
            bg_color = "#f0f0f0"
            
        left_canvas = tk.Canvas(left_container, borderwidth=0, highlightthickness=0, width=285, bg=bg_color)
        left_scrollbar = ttk.Scrollbar(left_container, orient=tk.VERTICAL, command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        left_panel = ttk.Frame(left_canvas, padding=10)
        left_canvas.create_window((0, 0), window=left_panel, anchor=tk.NW, width=285)
        
        def _configure_left_panel(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left_panel.bind("<Configure>", _configure_left_panel)
        
        # Enable mousewheel scrolling on hover
        def _on_left_mousewheel(event):
            if event.num == 4 or event.delta > 0:
                left_canvas.yview_scroll(-2, "units")
            elif event.num == 5 or event.delta < 0:
                left_canvas.yview_scroll(2, "units")

        def _bind_left_mousewheel(event):
            left_container.bind_all("<MouseWheel>", _on_left_mousewheel)
            left_container.bind_all("<Button-4>", _on_left_mousewheel)
            left_container.bind_all("<Button-5>", _on_left_mousewheel)
            
        def _unbind_left_mousewheel(event):
            left_container.unbind_all("<MouseWheel>")
            left_container.unbind_all("<Button-4>")
            left_container.unbind_all("<Button-5>")
            
        left_container.bind("<Enter>", _bind_left_mousewheel)
        left_container.bind("<Leave>", _unbind_left_mousewheel)
        
        right_panel = ttk.Frame(self.root, padding=10)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Left Panel Widgets
        # 1. File Selection
        ttk.Label(left_panel, text="1. Select Screenshot Image:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.file_cb = ttk.Combobox(left_panel, textvariable=self.selected_file, state="readonly")
        self.file_cb.pack(fill=tk.X, pady=5)
        self.file_cb.bind("<<ComboboxSelected>>", self._load_image)
        
        # 2. Calibration values
        ttk.Label(left_panel, text="2. Axis Calibration Ranges:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=15)
        
        grid_frame = ttk.Frame(left_panel)
        grid_frame.pack(fill=tk.X)
        
        ttk.Label(grid_frame, text="X Min:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid_frame, textvariable=self.x_min_val, width=12).grid(row=0, column=1, pady=2, padx=5)
        
        ttk.Label(grid_frame, text="X Max:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid_frame, textvariable=self.x_max_val, width=12).grid(row=1, column=1, pady=2, padx=5)
        
        ttk.Label(grid_frame, text="Y Min:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid_frame, textvariable=self.y_min_val, width=12).grid(row=2, column=1, pady=2, padx=5)
        
        ttk.Label(grid_frame, text="Y Max:").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(grid_frame, textvariable=self.y_max_val, width=12).grid(row=3, column=1, pady=2, padx=5)
        
        # Log scales
        ttk.Checkbutton(left_panel, text="Logarithmic X-Axis", variable=self.log_x).pack(anchor=tk.W, pady=2)
        ttk.Checkbutton(left_panel, text="Logarithmic Y-Axis", variable=self.log_y).pack(anchor=tk.W, pady=2)
        
        # 3. Calibration buttons
        self.cal_btn = ttk.Button(left_panel, text="Calibrate Axes", command=self._start_calibration)
        self.cal_btn.pack(fill=tk.X, pady=10)
        
        # Y-Axis Tick Calibration frame
        ytick_frame = ttk.LabelFrame(left_panel, text="Y-Axis Tick Calibration (Optional)")
        ytick_frame.pack(fill=tk.X, pady=5)
        
        ytick_btn_frame = ttk.Frame(ytick_frame)
        ytick_btn_frame.pack(fill=tk.X, pady=2, padx=5)
        
        self.add_ytick_btn = ttk.Button(ytick_btn_frame, text="Add Y-Tick", command=self._start_add_ytick)
        self.add_ytick_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.clear_ytick_btn = ttk.Button(ytick_btn_frame, text="Clear Ticks", command=self._clear_yticks)
        self.clear_ytick_btn.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        self.y_ticks_list = tk.Listbox(ytick_frame, height=3, font=("Courier", 8))
        self.y_ticks_list.pack(fill=tk.X, pady=5, padx=5)
        
        # 4. Point controls
        ttk.Label(left_panel, text="3. Data Point Extraction:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=15)
        
        self.auto_trace = tk.BooleanVar(value=False)
        ttk.Checkbutton(left_panel, text="Auto-Trace Curve by Color", variable=self.auto_trace).pack(anchor=tk.W, pady=2)
        
        # Sub-frame for Auto-Trace options
        self.trace_frame = ttk.LabelFrame(left_panel, text="Auto-Trace Options")
        self.trace_frame.pack(fill=tk.X, pady=5)
        
        # Color swatch and Pick button
        swatch_frame = ttk.Frame(self.trace_frame)
        swatch_frame.pack(fill=tk.X, pady=5, padx=5)
        
        self.pick_color_btn = ttk.Button(swatch_frame, text="Pick Color", command=self._start_pick_color)
        self.pick_color_btn.pack(side=tk.LEFT, padx=2)
        
        self.color_swatch = tk.Label(swatch_frame, text="None", width=12, relief=tk.SUNKEN, bg="white", fg="black")
        self.color_swatch.pack(side=tk.LEFT, padx=5, fill=tk.Y, expand=True)
        
        # Tolerance setting
        tol_frame = ttk.Frame(self.trace_frame)
        tol_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(tol_frame, text="Tolerance:").pack(side=tk.LEFT)
        self.tolerance_entry = ttk.Spinbox(tol_frame, from_=1.0, to=255.0, increment=5.0, textvariable=self.tolerance_val, width=8)
        self.tolerance_entry.pack(side=tk.RIGHT, padx=2)
        
        # Vertical search window setting
        win_frame = ttk.Frame(self.trace_frame)
        win_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(win_frame, text="V-Search Window:").pack(side=tk.LEFT)
        self.search_win_entry = ttk.Spinbox(win_frame, from_=5, to=100, increment=5, textvariable=self.search_win_val, width=8)
        self.search_win_entry.pack(side=tk.RIGHT, padx=2)
        
        # Max Gap setting
        gap_frame = ttk.Frame(self.trace_frame)
        gap_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(gap_frame, text="Max Gap (px):").pack(side=tk.LEFT)
        self.max_gap_entry = ttk.Spinbox(gap_frame, from_=5, to=100, increment=5, textvariable=self.max_gap_val, width=8)
        self.max_gap_entry.pack(side=tk.RIGHT, padx=2)
        
        # Smooth Window setting
        smooth_frame = ttk.Frame(self.trace_frame)
        smooth_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(smooth_frame, text="Smooth Window (px):").pack(side=tk.LEFT)
        self.smooth_entry = ttk.Spinbox(smooth_frame, from_=1, to=51, increment=2, textvariable=self.smooth_win_val, width=8)
        self.smooth_entry.pack(side=tk.RIGHT, padx=2)
        
        # Draggable Points setting
        drag_pts_frame = ttk.Frame(self.trace_frame)
        drag_pts_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(drag_pts_frame, text="Draggable Points:").pack(side=tk.LEFT)
        self.drag_pts_entry = ttk.Spinbox(drag_pts_frame, from_=5, to=200, increment=5, textvariable=self.drag_points_val, width=8)
        self.drag_pts_entry.pack(side=tk.RIGHT, padx=2)
        
        # Point Size scale slider
        size_frame = ttk.Frame(self.trace_frame)
        size_frame.pack(fill=tk.X, pady=2, padx=5)
        ttk.Label(size_frame, text="Point Size:").pack(side=tk.LEFT)
        self.size_scale = ttk.Scale(size_frame, from_=2, to=10, variable=self.handle_size_val, orient=tk.HORIZONTAL)
        self.size_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        self.show_markers = tk.BooleanVar(value=True)
        ttk.Checkbutton(left_panel, text="Show Extracted Points & Line", variable=self.show_markers, command=self._toggle_markers_visibility).pack(anchor=tk.W, pady=2)
        
        self.points_list = tk.Listbox(left_panel, height=8, font=("Courier", 9))
        self.points_list.pack(fill=tk.BOTH, expand=True, pady=5)
        
        btn_frame = ttk.Frame(left_panel)
        btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(btn_frame, text="Delete Last", command=self._delete_last_point).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btn_frame, text="Clear All", command=self._clear_all_points).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # 4. Export details
        ttk.Label(left_panel, text="4. Save Extracted Dataset:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(15, 2))
        
        ttk.Label(left_panel, text="Target Device Subdirectory:").pack(anchor=tk.W)
        self.device_cb = ttk.Combobox(left_panel, textvariable=self.selected_device, values=self.device_list)
        self.device_cb.pack(fill=tk.X, pady=2)
        self.device_cb.bind("<<ComboboxSelected>>", self._on_device_change)
        self.device_cb.bind("<FocusOut>", self._on_device_change)
        
        ttk.Label(left_panel, text="File Name (e.g. SaturationCurrent):").pack(anchor=tk.W, pady=(4, 0))
        ttk.Entry(left_panel, textvariable=self.dataset_name).pack(fill=tk.X, pady=2)
        
        # Presets frame for standard DeviceFitting names
        preset_frame = ttk.LabelFrame(left_panel, text="Device Fitting Presets", padding=4)
        preset_frame.pack(fill=tk.X, pady=4)
        ttk.Button(preset_frame, text="NonSaturationCurrent", command=lambda: self.dataset_name.set("NonSaturationCurrent")).pack(fill=tk.X, pady=1)
        ttk.Button(preset_frame, text="SaturationCurrent", command=lambda: self.dataset_name.set("SaturationCurrent")).pack(fill=tk.X, pady=1)
        ttk.Button(preset_frame, text="1.22VSaturation", command=lambda: self.dataset_name.set("1.22VSaturation")).pack(fill=tk.X, pady=1)
        
        self.save_btn = ttk.Button(left_panel, text="Export to CSV", command=self._save_to_csv, state=tk.DISABLED)
        self.save_btn.pack(fill=tk.X, pady=8)
        
        # 5. Import/Load details
        ttk.Label(left_panel, text="5. Load Extracted Dataset:", font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=10)
        self.selected_csv = tk.StringVar()
        self.csv_cb = ttk.Combobox(left_panel, textvariable=self.selected_csv, state="readonly")
        self.csv_cb.pack(fill=tk.X, pady=5)
        
        load_btn_frame = ttk.Frame(left_panel)
        load_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(load_btn_frame, text="Load CSV", command=self._load_from_csv).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(load_btn_frame, text="Refresh", command=self._scan_csvs).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
        
        # Right Panel Widgets (Canvas + Scrollbars)
        self.status_bar = ttk.Label(right_panel, textvariable=self.status_text, font=("Helvetica", 10, "italic"), background="#dddddd", anchor=tk.W, padding=5)
        self.status_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))
        
        canvas_container = ttk.Frame(right_panel)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.h_scroll = ttk.Scrollbar(canvas_container, orient=tk.HORIZONTAL)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.v_scroll = ttk.Scrollbar(canvas_container, orient=tk.VERTICAL)
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.canvas = tk.Canvas(canvas_container, bg="white", xscrollcommand=self.h_scroll.set, yscrollcommand=self.v_scroll.set)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.h_scroll.config(command=self.canvas.xview)
        self.v_scroll.config(command=self.canvas.yview)
        
        self.canvas.bind("<ButtonPress-1>", self._on_button_press)
        self.canvas.bind("<B1-Motion>", self._on_button_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_button_release)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        
        # Panning bindings (Middle click or Right click drag)
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_release)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_release)
        
        # Zooming bindings when mouse is in range of the canvas
        self.canvas.bind("<Enter>", self._on_mouse_enter)
        self.canvas.bind("<Leave>", self._on_mouse_leave)

    def _scan_images(self):
        if not os.path.exists(self.screenshot_dir):
            os.makedirs(self.screenshot_dir, exist_ok=True)
            
        valid_exts = (".png", ".jpg", ".jpeg")
        files = [f for f in os.listdir(self.screenshot_dir) if f.lower().endswith(valid_exts)]
        files.sort()
        self.file_cb['values'] = files
        if files:
            self.file_cb.current(0)
            self._load_image()
        else:
            self.status_text.set("No image files found in Screenshots directory. Place your screenshots there.")

    def _load_image(self, event=None):
        filename = self.selected_file.get()
        if not filename:
            return
        
        path = os.path.join(self.screenshot_dir, filename)
        try:
            self.pil_image = Image.open(path).convert("RGB")
            self.zoom_factor = 1.0
            
            # Reset application state
            self.state = 'select'
            self.px_x_min = self.px_x_max = self.px_y_min = self.px_y_max = None
            self.points = []
            self.start_click_x = self.start_click_y = None
            self.target_color = None
            if hasattr(self, 'color_swatch'):
                self.color_swatch.config(bg="white", fg="black", text="None")
            self.points_list.delete(0, tk.END)
            self._display_image()
            
            self.save_btn.config(state=tk.DISABLED)
            self.status_text.set(f"Successfully loaded {filename}. Press 'Calibrate Axes' to define calibration coordinates.")
        except Exception as e:
            messagebox.showerror("Error Loading Image", f"Could not load image:\n{e}")

    def _start_calibration(self):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            messagebox.showwarning("Warning", "Please select and load an image first.")
            return
            
        # Clear existing markers
        self.canvas.delete("marker_cal")
        self.canvas.delete("marker_data")
        self.canvas.delete("marker_line")
        self.points = []
        self.points_list.delete(0, tk.END)
        self.save_btn.config(state=tk.DISABLED)
        
        # Start calibration workflow (Click-and-drag Bounding Box)
        self.state = 'cal_box'
        self.status_text.set("Click and drag a bounding box covering the plot axes area.")

    def _start_add_ytick(self):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            messagebox.showwarning("Warning", "Please select and load an image first.")
            return
        self.state = 'add_y_tick'
        self.canvas.config(cursor="crosshair")
        self.status_text.set("Click on the plot area to place a Y-tick reference line.")

    def _clear_yticks(self):
        self.y_ticks = []
        if hasattr(self, 'y_ticks_list'):
            self.y_ticks_list.delete(0, tk.END)
        self._recalculate_all_points()
        self._display_image()
        self.status_text.set("Cleared all custom Y-ticks.")

    def _start_pick_color(self):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            messagebox.showwarning("Warning", "Please select and load an image first.")
            return
        self.state = 'pick_color'
        self.canvas.config(cursor="crosshair")
        self.status_text.set("Click a pixel on the plot image to pick the target curve color.")

    def _on_parameter_change(self, *args):
        if hasattr(self, 'loading_csv') and self.loading_csv:
            return
        try:
            self.tolerance_val.get()
            self.search_win_val.get()
            self.max_gap_val.get()
            self.smooth_win_val.get()
            self.drag_points_val.get()
        except tk.TclError:
            return
            
        if hasattr(self, 'start_click_x') and self.start_click_x is not None:
            if self.auto_trace.get() and self.state == 'digitize':
                # Preserve manual points, clear auto-traced ones
                self.points = [pt for pt in self.points if not pt.get('is_auto', False)]
                self.points_list.delete(0, tk.END)
                self._display_image()
                for i, pt in enumerate(self.points):
                    self.points_list.insert(tk.END, f"{i+1:02d}: X={pt['phys_x']:0.4g}, Y={pt['phys_y']:0.4g}")
                self._auto_trace_curve(self.start_click_x, self.start_click_y)

    def _on_calibration_param_change(self, *args):
        if hasattr(self, 'loading_csv') and self.loading_csv:
            return
        try:
            self.x_min_val.get()
            self.x_max_val.get()
            self.y_min_val.get()
            self.y_max_val.get()
        except tk.TclError:
            return
        self._recalculate_all_points()

    def _on_handle_size_change(self, *args):
        try:
            self.handle_size_val.get()
        except tk.TclError:
            return
        self._redraw_markers_and_lines()

    def _on_button_press(self, event):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            return
        px_x = self.canvas.canvasx(event.x)
        px_y = self.canvas.canvasy(event.y)
        
        if self.state == 'cal_box':
            self.box_start_x = px_x / self.zoom_factor
            self.box_start_y = px_y / self.zoom_factor
            self.canvas.delete("cal_box")
            return
            
        # Check if click is near any existing point for dragging
        click_radius = 8.0  # tolerance in canvas pixels
        closest_idx = None
        min_dist = float('inf')
        for idx, pt in enumerate(self.points):
            pt_canvas_x = pt['orig_x'] * self.zoom_factor
            pt_canvas_y = pt['orig_y'] * self.zoom_factor
            dist = math.sqrt((px_x - pt_canvas_x)**2 + (px_y - pt_canvas_y)**2)
            if dist < click_radius and dist < min_dist:
                min_dist = dist
                closest_idx = idx
                
        if closest_idx is not None:
            self.dragged_point_idx = closest_idx
            self.canvas.config(cursor="hand2")
            self.status_text.set(f"Dragging point #{closest_idx+1}")
            return
            
        self.dragged_point_idx = None
        self._on_canvas_click(event)

    def _on_button_drag(self, event):
        if self.state == 'cal_box':
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.delete("cal_box")
            self.canvas.create_rectangle(self.box_start_x * self.zoom_factor, self.box_start_y * self.zoom_factor, cur_x, cur_y, outline="green", width=2, dash=(4, 4), tags="cal_box")
        elif hasattr(self, 'dragged_point_idx') and self.dragged_point_idx is not None:
            px_x = self.canvas.canvasx(event.x)
            px_y = self.canvas.canvasy(event.y)
            orig_x = px_x / self.zoom_factor
            orig_y = px_y / self.zoom_factor
            
            # Constrain to image boundaries
            orig_x = max(0.0, min(float(self.pil_image.width - 1), orig_x))
            orig_y = max(0.0, min(float(self.pil_image.height - 1), orig_y))
            
            phys_x, phys_y = self._pixel_to_physical(orig_x, orig_y)
            
            # Update coordinate values in the list
            self.points[self.dragged_point_idx]['orig_x'] = orig_x
            self.points[self.dragged_point_idx]['orig_y'] = orig_y
            self.points[self.dragged_point_idx]['phys_x'] = phys_x
            self.points[self.dragged_point_idx]['phys_y'] = phys_y
            
            # Update Listbox item text
            self.points_list.delete(self.dragged_point_idx)
            self.points_list.insert(self.dragged_point_idx, f"{self.dragged_point_idx+1:02d}: X={phys_x:0.4g}, Y={phys_y:0.4g}")
            
            # Redraw only the markers and line to avoid lagging Lanczos image resize
            self._redraw_markers_and_lines()

    def _on_button_release(self, event):
        if self.state == 'cal_box':
            cur_x = self.canvas.canvasx(event.x)
            cur_y = self.canvas.canvasy(event.y)
            self.canvas.delete("cal_box")
            
            x1, x2 = min(self.box_start_x, cur_x / self.zoom_factor), max(self.box_start_x, cur_x / self.zoom_factor)
            y1, y2 = min(self.box_start_y, cur_y / self.zoom_factor), max(self.box_start_y, cur_y / self.zoom_factor)
            
            if x2 - x1 < 5 or y2 - y1 < 5:
                return
                
            self.px_x_min = (x1, y2)
            self.px_x_max = (x2, y2)
            self.px_y_min = (x1, y2)
            self.px_y_max = (x1, y1)
            
            self._recalculate_all_points()
            self._display_image()
            self.state = 'digitize'
            self.save_btn.config(state=tk.NORMAL)
            self.status_text.set("Calibration Bounding Box defined! Click on the plot curve to auto-trace.")
        elif hasattr(self, 'dragged_point_idx') and self.dragged_point_idx is not None:
            self.canvas.config(cursor="")
            self.status_text.set(f"Finished dragging point #{self.dragged_point_idx+1}")
            self.dragged_point_idx = None

    def _on_canvas_click(self, event):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            return
            
        px_x = self.canvas.canvasx(event.x)
        px_y = self.canvas.canvasy(event.y)
        
        orig_x = px_x / self.zoom_factor
        orig_y = px_y / self.zoom_factor
        
        if self.state == 'add_y_tick':
            self.state = 'digitize'
            self.canvas.config(cursor="")
            import tkinter.simpledialog as sd
            val = sd.askfloat("Y-Tick Value", "Enter the physical value for this tick mark (e.g., 5.0 for 5 mA):")
            if val is not None:
                self.y_ticks.append({'pixel_y': orig_y, 'value': val})
                self.y_ticks.sort(key=lambda t: t['pixel_y'])
                self.y_ticks_list.delete(0, tk.END)
                for idx, tick in enumerate(self.y_ticks):
                    self.y_ticks_list.insert(tk.END, f"{idx+1:02d}: {tick['value']} @ {tick['pixel_y']:.1f}px")
                self._recalculate_all_points()
                self.save_btn.config(state=tk.NORMAL)
                self._display_image()
                self.status_text.set(f"Added Y-Tick: {val} at {orig_y:.1f}px")
            else:
                self.status_text.set("Y-Tick addition cancelled.")
            return
            
        if self.state == 'pick_color':
            self.state = 'digitize'
            self.canvas.config(cursor="")
            orig_x_int = int(orig_x)
            orig_y_int = int(orig_y)
            img_w = self.pil_image.width
            img_h = self.pil_image.height
            if 0 <= orig_x_int < img_w and 0 <= orig_y_int < img_h:
                try:
                    color = self.pil_image.getpixel((orig_x_int, orig_y_int))
                    self.target_color = color[:3]
                    r, g, b = self.target_color
                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    fg = "white" if (r*0.299 + g*0.587 + b*0.114) < 128 else "black"
                    self.color_swatch.config(bg=hex_color, fg=fg, text=f"RGB({r},{g},{b})")
                    
                    # Ensure calibration is initialized to image bounds if not already calibrated
                    if not (self.px_x_min and self.px_x_max and self.px_y_min and self.px_y_max):
                        self.px_x_min = (0.0, float(img_h - 1))
                        self.px_x_max = (float(img_w - 1), float(img_h - 1))
                        self.px_y_min = (0.0, float(img_h - 1))
                        self.px_y_max = (0.0, 0.0)
                        self.x_min_val.set(0.0)
                        self.x_max_val.set(float(img_w - 1))
                        self.y_min_val.set(0.0)
                        self.y_max_val.set(float(img_h - 1))
                        self.save_btn.config(state=tk.NORMAL)
                        
                    self.status_text.set(f"Picked color RGB({r},{g},{b}). Traced curve!")
                    
                    if self.auto_trace.get():
                        self.start_click_x = orig_x
                        self.start_click_y = orig_y
                        # Clear previous auto-traced points to avoid stacking/duplication
                        self.points = [pt for pt in self.points if not pt.get('is_auto', False)]
                        self.points_list.delete(0, tk.END)
                        for i, pt in enumerate(self.points):
                            self.points_list.insert(tk.END, f"{i+1:02d}: X={pt['phys_x']:0.4g}, Y={pt['phys_y']:0.4g}")
                        self._auto_trace_curve(orig_x, orig_y)
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to get pixel color: {e}")
            return

        elif self.state == 'digitize' or self.state == 'select':
            # Ensure calibration is initialized to image bounds if not already calibrated
            if not (self.px_x_min and self.px_x_max and self.px_y_min and self.px_y_max):
                img_w = self.pil_image.width
                img_h = self.pil_image.height
                self.px_x_min = (0.0, float(img_h - 1))
                self.px_x_max = (float(img_w - 1), float(img_h - 1))
                self.px_y_min = (0.0, float(img_h - 1))
                self.px_y_max = (0.0, 0.0)
                self.x_min_val.set(0.0)
                self.x_max_val.set(float(img_w - 1))
                self.y_min_val.set(0.0)
                self.y_max_val.set(float(img_h - 1))
                self.state = 'digitize'
                self.save_btn.config(state=tk.NORMAL)
                
            if self.auto_trace.get():
                self.start_click_x = orig_x
                self.start_click_y = orig_y
                
                # Always capture the target color from the clicked pixel first (forces single-click tracing!)
                start_x_int = max(0, min(self.pil_image.width - 1, int(orig_x)))
                start_y_int = max(0, min(self.pil_image.height - 1, int(orig_y)))
                try:
                    c = self.pil_image.getpixel((start_x_int, start_y_int))
                    self.target_color = c[:3]
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to get pixel color: {e}")
                    return
                
                # Clear previous auto-traced points to avoid stacking/duplication
                self.points = [pt for pt in self.points if not pt.get('is_auto', False)]
                self.points_list.delete(0, tk.END)
                for i, pt in enumerate(self.points):
                    self.points_list.insert(tk.END, f"{i+1:02d}: X={pt['phys_x']:0.4g}, Y={pt['phys_y']:0.4g}")
                self._auto_trace_curve(orig_x, orig_y)
            else:
                phys_x, phys_y = self._pixel_to_physical(orig_x, orig_y)
                
                pt = {
                    'orig_x': orig_x,
                    'orig_y': orig_y,
                    'phys_x': phys_x,
                    'phys_y': phys_y,
                    'is_auto': False
                }
                self.points.append(pt)
                self.points_list.insert(tk.END, f"{len(self.points):02d}: X={phys_x:0.4g}, Y={phys_y:0.4g}")
                self.points_list.see(tk.END)
                self._display_image()

    def _on_canvas_motion(self, event):
        if self.state == 'digitize':
            px_x = self.canvas.canvasx(event.x)
            px_y = self.canvas.canvasy(event.y)
            orig_x = px_x / self.zoom_factor
            orig_y = px_y / self.zoom_factor
            phys_x, phys_y = self._pixel_to_physical(orig_x, orig_y)
            self.status_text.set(f"Digitizing - Current coordinates: X = {phys_x:0.4g}, Y = {phys_y:0.4g}")

    def _auto_trace_curve(self, start_x, start_y):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            return
            
        img_w = self.pil_image.width
        img_h = self.pil_image.height
        
        # Determine bounds from calibration box, defaulting to image size if uncalibrated
        x_min = int(self.px_x_min[0]) if self.px_x_min else 0
        x_max = int(self.px_x_max[0]) if self.px_x_max else img_w - 1
        x_min = max(0, x_min)
        x_max = min(img_w - 1, x_max)
        
        y_min_bound = int(self.px_y_max[1]) if self.px_y_max else 0
        y_max_bound = int(self.px_y_min[1]) if self.px_y_min else img_h - 1
        y_min_bound = max(0, y_min_bound)
        y_max_bound = min(img_h - 1, y_max_bound)
        
        start_x_int = max(0, min(img_w - 1, int(start_x)))
        start_y_int = max(0, min(img_h - 1, int(start_y)))
        
        # Get target color from mouse position if not set
        if self.target_color is None:
            try:
                c = self.pil_image.getpixel((start_x_int, start_y_int))
                self.target_color = c[:3]
            except:
                return
                
        r_target, g_target, b_target = self.target_color
        
        # Update visual swatch
        hex_color = f"#{r_target:02x}{g_target:02x}{b_target:02x}"
        fg = "white" if (r_target*0.299 + g_target*0.587 + b_target*0.114) < 128 else "black"
        self.color_swatch.config(bg=hex_color, fg=fg, text=f"RGB({r_target},{g_target},{b_target})")
        
        # Don't trace white or near-white background
        if r_target > 240 and g_target > 240 and b_target > 240:
            messagebox.showwarning("Warning", "The target color is white/background. Please click on a colored curve.")
            return
            
        tolerance = self.tolerance_val.get()
        search_window = self.search_win_val.get()
        max_gap = self.max_gap_val.get()
        max_misses = max_gap
        
        pixels = self.pil_image.load()
        
        # Helper to check color match
        def color_matches(px, py):
            try:
                c = pixels[px, py]
                r, g, b = c[:3]
                dist = math.sqrt((r - r_target)**2 + (g - g_target)**2 + (b - b_target)**2)
                return dist <= tolerance
            except:
                return False
                
        # Snap start to best matching color in a 5x5 window around the click coordinate
        best_snap_x, best_snap_y = start_x_int, start_y_int
        min_dist = float('inf')
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                tx = start_x_int + dx
                ty = start_y_int + dy
                if 0 <= tx < img_w and 0 <= ty < img_h:
                    c = pixels[tx, ty]
                    r, g, b = c[:3]
                    dist = math.sqrt((r - r_target)**2 + (g - g_target)**2 + (b - b_target)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_snap_x, best_snap_y = tx, ty
                        
        start_x_int = best_snap_x
        start_y_int = best_snap_y
        
        # Estimate initial direction by testing 16 angles around the click coordinate
        matching_angles = []
        for angle_idx in range(16):
            angle = angle_idx * (2 * math.pi / 16)
            test_x = int(round(start_x_int + 4.0 * math.cos(angle)))
            test_y = int(round(start_y_int + 4.0 * math.sin(angle)))
            if 0 <= test_x < img_w and 0 <= test_y < img_h:
                if color_matches(test_x, test_y):
                    c = pixels[test_x, test_y]
                    r, g, b = c[:3]
                    dist = math.sqrt((r - r_target)**2 + (g - g_target)**2 + (b - b_target)**2)
                    matching_angles.append((angle, dist))
                    
        if matching_angles:
            matching_angles.sort(key=lambda item: item[1])
            best_angle = matching_angles[0][0]
        else:
            best_angle = 0.0 # Default to horizontal right
            
        initial_vx = math.cos(best_angle)
        initial_vy = math.sin(best_angle)
        
        # Helper for vector tracing in a given starting direction (predictor-corrector step along tangents)
        def trace_vector(init_vx, init_vy):
            step_size = 2.0
            pts = []
            curr_x, curr_y = float(start_x_int), float(start_y_int)
            vx, vy = init_vx, init_vy
            consecutive_misses = 0
            
            while len(pts) < 1000:
                pred_x = curr_x + step_size * vx
                pred_y = curr_y + step_size * vy
                
                # Check bounds
                if not (x_min <= pred_x <= x_max and y_min_bound <= pred_y <= y_max_bound):
                    break
                    
                # Search perpendicular to the direction vector
                nx, ny = -vy, vx
                matching_s = []
                local_search_window = min(7.0, float(search_window))
                s = -local_search_window
                while s <= local_search_window:
                    nx_x = pred_x + s * nx
                    ny_y = pred_y + s * ny
                    if 0 <= nx_x < img_w and 0 <= ny_y < img_h:
                        if color_matches(int(round(nx_x)), int(round(ny_y))):
                            matching_s.append(s)
                    s += 0.5
                    
                if matching_s:
                    best_s = min(matching_s, key=lambda s_val: abs(s_val))
                    best_idx = matching_s.index(best_s)
                    
                    # Group contiguous matching pixels
                    left_idx = best_idx
                    while left_idx > 0:
                        if abs(matching_s[left_idx] - matching_s[left_idx - 1] - 0.5) < 1e-3:
                            left_idx -= 1
                        else:
                            break
                            
                    right_idx = best_idx
                    while right_idx < len(matching_s) - 1:
                        if abs(matching_s[right_idx + 1] - matching_s[right_idx] - 0.5) < 1e-3:
                            right_idx += 1
                        else:
                            break
                            
                    s_mid = (matching_s[left_idx] + matching_s[right_idx]) / 2.0
                    
                    next_x = pred_x + s_mid * nx
                    next_y = pred_y + s_mid * ny
                    
                    if not (x_min <= next_x <= x_max and y_min_bound <= next_y <= y_max_bound):
                        break
                        
                    # Update local direction vector with inertia (damping high-frequency wiggles)
                    new_vx = next_x - curr_x
                    new_vy = next_y - curr_y
                    length = math.sqrt(new_vx*new_vx + new_vy*new_vy)
                    if length > 0:
                        new_vx /= length
                        new_vy /= length
                        
                        beta = 0.2  # Blending factor (inertia)
                        vx = (1 - beta) * vx + beta * new_vx
                        vy = (1 - beta) * vy + beta * new_vy
                        
                        # Re-normalize direction vector
                        v_len = math.sqrt(vx*vx + vy*vy)
                        if v_len > 0:
                            vx /= v_len
                            vy /= v_len
                        
                    pts.append((next_x, next_y))
                    curr_x, curr_y = next_x, next_y
                    consecutive_misses = 0
                else:
                    # Sharp turn recovery: scan angles around the current direction to find where the curve went
                    curr_angle = math.atan2(vy, vx)
                    best_turn_angle = None
                    min_turn_dist = float('inf')
                    
                    # Scan angles from -170 to +170 degrees in steps of 5 degrees
                    # Exclude the direct backward direction (around 180 degrees) to prevent backtracking
                    for angle_offset_deg in range(-170, 171, 5):
                        angle_offset = math.radians(angle_offset_deg)
                        test_angle = curr_angle + angle_offset
                        
                        # Scan multiple radii to ensure we cross a narrow line
                        for test_radius in [3.0, 4.0, 5.0, 6.0]:
                            tx = int(round(curr_x + test_radius * math.cos(test_angle)))
                            ty = int(round(curr_y + test_radius * math.sin(test_angle)))
                            
                            if 0 <= tx < img_w and 0 <= ty < img_h:
                                if color_matches(tx, ty):
                                    c = pixels[tx, ty]
                                    r, g, b = c[:3]
                                    dist = math.sqrt((r - r_target)**2 + (g - g_target)**2 + (b - b_target)**2)
                                    
                                    # Verify we are not looping back to recently visited points
                                    already_traced = False
                                    # Reduce the history window and threshold to allow very sharp turns
                                    for past_pt in pts[-4:]:
                                        if math.sqrt((tx - past_pt[0])**2 + (ty - past_pt[1])**2) < 2.0:
                                            already_traced = True
                                            break
                                    if math.sqrt((tx - start_x_int)**2 + (ty - start_y_int)**2) < 2.0:
                                        already_traced = True
                                            
                                    if not already_traced and dist < min_turn_dist:
                                        min_turn_dist = dist
                                        best_turn_angle = test_angle
                                    
                    if best_turn_angle is not None:
                        # Update direction vector to follow the sharp turn
                        vx = math.cos(best_turn_angle)
                        vy = math.sin(best_turn_angle)
                        curr_x = curr_x + 2.0 * vx
                        curr_y = curr_y + 2.0 * vy
                        pts.append((curr_x, curr_y))
                        consecutive_misses = 0
                    else:
                        consecutive_misses += 1
                        if consecutive_misses > max_misses:
                            break
                        curr_x = pred_x
                        curr_y = pred_y
            return pts

        # Run vector tracing in both directions (forward and backward)
        forward_pts = trace_vector(initial_vx, initial_vy)
        backward_pts = trace_vector(-initial_vx, -initial_vy)
        
        # Combine stitched paths
        traced_points = backward_pts[::-1] + [(float(start_x_int), float(start_y_int))] + forward_pts
        
        # Combine and sort points by X for PCHIP interpolation compatibility
        traced_points = sorted(traced_points, key=lambda pt: pt[0])
        
        # 1. Centering step: Refine the line center by searching along the normal direction of the curve tangent.
        # This is run first on the raw traced points to find the exact center of the band.
        refined_points = []
        if len(traced_points) >= 2:
            n_pts = len(traced_points)
            for i in range(n_pts):
                px, py = traced_points[i]
                
                # Estimate tangent vector using a wider neighbor window to filter noise and wiggles
                k = min(5, n_pts // 4)
                k = max(1, k)
                idx_prev = max(0, i - k)
                idx_next = min(n_pts - 1, i + k)
                
                dx = traced_points[idx_next][0] - traced_points[idx_prev][0]
                dy = traced_points[idx_next][1] - traced_points[idx_prev][1]
                
                length = math.sqrt(dx*dx + dy*dy)
                if length == 0:
                    nx, ny = 0.0, 1.0
                else:
                    # Normal vector is (-dy, dx)
                    nx = -dy / length
                    ny = dx / length
                    
                # Find the closest matching pixel near s=0
                max_start_d = min(7.0, float(search_window))
                step = 0.5
                start_s = None
                
                # Scan outward from s=0 in alternating directions to find the nearest matching pixel
                steps_limit = int(max_start_d / step)
                for step_idx in range(steps_limit + 1):
                    s_abs = step_idx * step
                    found = False
                    for sign in [1, -1] if s_abs > 0 else [1]:
                        s_val = sign * s_abs
                        tx = int(round(px + s_val * nx))
                        ty = int(round(py + s_val * ny))
                        if 0 <= tx < img_w and 0 <= ty < img_h:
                            if color_matches(tx, ty):
                                start_s = s_val
                                found = True
                                break
                    if found:
                        break
                        
                if start_s is not None:
                    # Expand positive along normal to find the upper bound
                    s_pos = start_s
                    while True:
                        s_next = s_pos + step
                        if abs(s_next) > search_window:
                            break
                        tx = int(round(px + s_next * nx))
                        ty = int(round(py + s_next * ny))
                        if 0 <= tx < img_w and 0 <= ty < img_h:
                            if color_matches(tx, ty):
                                s_pos = s_next
                            else:
                                break
                        else:
                            break
                            
                    # Expand negative along normal to find the lower bound
                    s_neg = start_s
                    while True:
                        s_next = s_neg - step
                        if abs(s_next) > search_window:
                            break
                        tx = int(round(px + s_next * nx))
                        ty = int(round(py + s_next * ny))
                        if 0 <= tx < img_w and 0 <= ty < img_h:
                            if color_matches(tx, ty):
                                s_neg = s_next
                            else:
                                break
                        else:
                            break
                            
                    s_mid = (s_pos + s_neg) / 2.0
                    refined_x = px + s_mid * nx
                    refined_y = py + s_mid * ny
                    refined_points.append((refined_x, refined_y))
                else:
                    # Fallback to raw point if no matching pixel is found nearby
                    refined_points.append((px, py))
            traced_points = refined_points

        # 2. Shape-Preserving Gaussian Smoothing: Apply Gaussian-weighted moving average to centered points.
        # This is run second to smooth out sub-pixel discretization wiggles without changing the overall line shape.
        smooth_window = self.smooth_win_val.get()
        if smooth_window > 1 and len(traced_points) > 1:
            half = smooth_window // 2
            # Use sigma = half / 2.0 to give standard Gaussian decay
            sigma = max(1.0, half / 2.0)
            weights = []
            for j in range(-half, half + 1):
                w = math.exp(- (j ** 2) / (2 * sigma ** 2))
                weights.append(w)
            
            x_arr = [pt[0] for pt in traced_points]
            y_arr = [pt[1] for pt in traced_points]
            x_smoothed = []
            y_smoothed = []
            n = len(y_arr)
            for i in range(n):
                start = max(0, i - half)
                end = min(n, i + half + 1)
                
                # Align weights slice for boundary handling
                slice_weights = weights[start - (i - half) : end - (i - half)]
                sum_weights = sum(slice_weights)
                
                window_x = x_arr[start:end]
                window_y = y_arr[start:end]
                
                wx = sum(x * w for x, w in zip(window_x, slice_weights)) / sum_weights
                wy = sum(y * w for y, w in zip(window_y, slice_weights)) / sum_weights
                
                x_smoothed.append(wx)
                y_smoothed.append(wy)
            
            traced_points = [(x_smoothed[i], y_smoothed[i]) for i in range(n)]

        # Downsample the traced points to the user-specified number of Control Points
        n_traced = len(traced_points)
        try:
            target_k = self.drag_points_val.get()
        except:
            target_k = 30
            
        if n_traced > target_k and target_k >= 2:
            downsampled_pts = []
            for k_idx in range(target_k):
                idx = int(round(k_idx * (n_traced - 1) / (target_k - 1)))
                downsampled_pts.append(traced_points[idx])
            traced_points = downsampled_pts

        if len(traced_points) < 2:
            messagebox.showwarning("Warning", "No curve traced. Try adjusting target color, tolerance, or bounding box.")
            return
            
        added_count = 0
        for px_x, px_y in traced_points:
            phys_x, phys_y = self._pixel_to_physical(px_x, px_y)
            pt = {
                'orig_x': px_x,
                'orig_y': px_y,
                'phys_x': phys_x,
                'phys_y': phys_y,
                'is_auto': True
            }
            self.points.append(pt)
            self.points_list.insert(tk.END, f"{len(self.points):02d}: X={phys_x:0.4g}, Y={phys_y:0.4g}")
            added_count += 1
            
        self._display_image()
        self.points_list.see(tk.END)
        self.status_text.set(f"Auto-traced {added_count} points using Center of Mass sweep.")

    def _toggle_markers_visibility(self):
        self._display_image()

    def _display_image(self):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            return
        
        w = int(self.pil_image.width * self.zoom_factor)
        h = int(self.pil_image.height * self.zoom_factor)
        
        resized_img = self.pil_image.resize((w, h), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(resized_img)
        
        self.canvas.config(scrollregion=(0, 0, w, h))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
        
        self._redraw_all_items()

    def _redraw_markers_and_lines(self):
        self.canvas.delete("marker_data")
        self.canvas.delete("marker_line")
        
        # 2. Points & Lines
        if self.show_markers.get():
            manual_pts = [pt for pt in self.points if not pt.get('is_auto', False)]
            auto_pts = [pt for pt in self.points if pt.get('is_auto', False)]
            
            # Draw manual points as yellow dots with dynamic size
            handle_size = self.handle_size_val.get()
            for pt in manual_pts:
                px_x = pt['orig_x'] * self.zoom_factor
                px_y = pt['orig_y'] * self.zoom_factor
                self.canvas.create_oval(px_x - handle_size, px_y - handle_size, px_x + handle_size, px_y + handle_size, fill="yellow", outline="black", tags="marker_data")
                
            # Draw auto-traced points as small cyan dots for fine tuning drag with dynamic size
            auto_handle_size = max(1, handle_size - 1)
            for pt in auto_pts:
                px_x = pt['orig_x'] * self.zoom_factor
                px_y = pt['orig_y'] * self.zoom_factor
                self.canvas.create_oval(px_x - auto_handle_size, px_y - auto_handle_size, px_x + auto_handle_size, px_y + auto_handle_size, fill="cyan", outline="black", tags="marker_data")
                
            # Interpolate and draw auto-traced points as a continuous PCHIP line
            if len(auto_pts) >= 2:
                auto_pts = sorted(auto_pts, key=lambda pt: pt['orig_x'])
                
                # Remove duplicate or very close x coordinates to satisfy PCHIP uniqueness constraint
                unique_pts = []
                last_x = -float('inf')
                for pt in auto_pts:
                    x_val = pt['orig_x']
                    if x_val > last_x + 1e-5:
                        unique_pts.append(pt)
                        last_x = x_val
                        
                if len(unique_pts) >= 2:
                    try:
                        from scipy.interpolate import PchipInterpolator
                        import numpy as np
                        
                        x_arr = np.array([pt['orig_x'] for pt in unique_pts])
                        y_arr = np.array([pt['orig_y'] for pt in unique_pts])
                        
                        pchip = PchipInterpolator(x_arr, y_arr)
                        
                        # Generate dense points along the x range
                        x_dense = np.arange(x_arr[0], x_arr[-1] + 0.5, 1.0)
                        y_dense = pchip(x_dense)
                        
                        coords = []
                        for x_val, y_val in zip(x_dense, y_dense):
                            coords.append(x_val * self.zoom_factor)
                            coords.append(y_val * self.zoom_factor)
                            
                        if len(coords) >= 4:
                            self.canvas.create_line(*coords, fill="black", width=2, tags="marker_line")
                    except Exception as e:
                        # Fallback to simple segment connections if interpolation fails
                        coords = []
                        for pt in unique_pts:
                            coords.append(pt['orig_x'] * self.zoom_factor)
                            coords.append(pt['orig_y'] * self.zoom_factor)
                        if len(coords) >= 4:
                            self.canvas.create_line(*coords, fill="black", width=2, tags="marker_line")

    def _redraw_all_items(self):
        # 1. Bounding Box
        if self.px_x_min and self.px_x_max and self.px_y_min and self.px_y_max:
            x1 = self.px_x_min[0] * self.zoom_factor
            x2 = self.px_x_max[0] * self.zoom_factor
            y1 = self.px_y_max[1] * self.zoom_factor
            y2 = self.px_y_min[1] * self.zoom_factor
            
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, dash=(2, 2), tags="marker_cal")
            self.canvas.create_text(x1, y2 + 10, text=f"X Min ({self.x_min_val.get()})", fill="blue", font=("Helvetica", 9, "bold"), tags="marker_cal")
            self.canvas.create_text(x2, y2 + 10, text=f"X Max ({self.x_max_val.get()})", fill="blue", font=("Helvetica", 9, "bold"), tags="marker_cal")
            self.canvas.create_text(x1 - 25, y2, text=f"Y Min ({self.y_min_val.get()})", fill="red", font=("Helvetica", 9, "bold"), tags="marker_cal")
            self.canvas.create_text(x1 - 25, y1, text=f"Y Max ({self.y_max_val.get()})", fill="red", font=("Helvetica", 9, "bold"), tags="marker_cal")
            
        # Draw custom Y-ticks if present
        if hasattr(self, 'y_ticks') and self.y_ticks:
            img_w = self.pil_image.width * self.zoom_factor
            for idx, tick in enumerate(self.y_ticks):
                py = tick['pixel_y'] * self.zoom_factor
                self.canvas.create_line(0, py, img_w, py, fill="red", width=1.5, dash=(4, 2), tags="marker_cal")
                self.canvas.create_text(15, py - 8, text=f"Y Tick {idx+1}: {tick['value']}", fill="red", font=("Helvetica", 8, "bold"), tags="marker_cal", anchor=tk.W)
                
        # 2. Points & Lines
        self._redraw_markers_and_lines()

    def _on_mouse_enter(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_zoom)
        self.canvas.bind_all("<Button-4>", self._on_zoom)
        self.canvas.bind_all("<Button-5>", self._on_zoom)

    def _on_mouse_leave(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_zoom(self, event):
        if not hasattr(self, 'pil_image') or self.pil_image is None:
            return
            
        # Get mouse position relative to canvas widget
        x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
        y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()
        
        # Verify mouse is actually inside canvas bounds
        if x < 0 or x > self.canvas.winfo_width() or y < 0 or y > self.canvas.winfo_height():
            return
            
        # Mouse position on canvas coordinate space (accounting for scrolling)
        mouse_x = self.canvas.canvasx(x)
        mouse_y = self.canvas.canvasy(y)
        
        # event.delta works on Windows, event.num on Linux
        if event.num == 4 or event.delta > 0:
            factor = 1.15
        elif event.num == 5 or event.delta < 0:
            factor = 1.0 / 1.15
        else:
            return
            
        new_zoom = self.zoom_factor * factor
        if new_zoom < 0.25 or new_zoom > 8.0:
            return
            
        self.zoom_factor = new_zoom
        self._display_image()
        
        # Adjust scroll so position under mouse remains constant
        new_mouse_x = mouse_x * factor
        new_mouse_y = mouse_y * factor
        self.canvas.xview_moveto((new_mouse_x - x) / (self.pil_image.width * self.zoom_factor))
        self.canvas.yview_moveto((new_mouse_y - y) / (self.pil_image.height * self.zoom_factor))

    def _on_pan_start(self, event):
        self.canvas.config(cursor="fleur")
        self.canvas.scan_mark(event.x, event.y)

    def _on_pan_drag(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_pan_release(self, event):
        self.canvas.config(cursor="")

    def _pixel_to_physical(self, px_x, px_y):
        # X-Axis Math
        ref_x_min = self.x_min_val.get()
        ref_x_max = self.x_max_val.get()
        dx_pixel = self.px_x_max[0] - self.px_x_min[0]
        
        if dx_pixel == 0:
            phys_x = 0.0
        else:
            fraction_x = (px_x - self.px_x_min[0]) / dx_pixel
            if self.log_x.get():
                # Avoid math error on <= 0 log limit
                val_min_log = math.log10(max(1e-15, ref_x_min))
                val_max_log = math.log10(max(1e-15, ref_x_max))
                phys_x = 10 ** (val_min_log + fraction_x * (val_max_log - val_min_log))
            else:
                phys_x = ref_x_min + fraction_x * (ref_x_max - ref_x_min)
                
        # Y-Axis Math
        if hasattr(self, 'y_ticks') and len(self.y_ticks) >= 2:
            n_ticks = len(self.y_ticks)
            sum_p = 0.0
            sum_v = 0.0
            for tick in self.y_ticks:
                sum_p += tick['pixel_y']
                if self.log_y.get():
                    sum_v += math.log10(max(1e-15, tick['value']))
                else:
                    sum_v += tick['value']
            
            mean_p = sum_p / n_ticks
            mean_v = sum_v / n_ticks
            
            num = 0.0
            denom = 0.0
            for tick in self.y_ticks:
                diff_p = tick['pixel_y'] - mean_p
                if self.log_y.get():
                    diff_v = math.log10(max(1e-15, tick['value'])) - mean_v
                else:
                    diff_v = tick['value'] - mean_v
                num += diff_p * diff_v
                denom += diff_p * diff_p
                
            if denom == 0:
                phys_y = 0.0
            else:
                slope = num / denom
                intercept = mean_v - slope * mean_p
                if self.log_y.get():
                    phys_y = 10 ** (slope * px_y + intercept)
                else:
                    phys_y = slope * px_y + intercept
        else:
            # Fallback to Bounding Box calibration math
            ref_y_min = self.y_min_val.get()
            ref_y_max = self.y_max_val.get()
            if not self.px_y_max or not self.px_y_min:
                phys_y = 0.0
            else:
                dy_pixel = self.px_y_max[1] - self.px_y_min[1]
                if dy_pixel == 0:
                    phys_y = 0.0
                else:
                    fraction_y = (px_y - self.px_y_min[1]) / dy_pixel
                    if self.log_y.get():
                        val_min_log = math.log10(max(1e-15, ref_y_min))
                        val_max_log = math.log10(max(1e-15, ref_y_max))
                        phys_y = 10 ** (val_min_log + fraction_y * (val_max_log - val_min_log))
                    else:
                        phys_y = ref_y_min + fraction_y * (ref_y_max - ref_y_min)
                
        return phys_x, phys_y

    def _recalculate_all_points(self):
        if not hasattr(self, 'points') or not self.points:
            return
        if not (self.px_x_min and self.px_x_max):
            return
            
        for pt in self.points:
            phys_x, phys_y = self._pixel_to_physical(pt['orig_x'], pt['orig_y'])
            pt['phys_x'] = phys_x
            pt['phys_y'] = phys_y
            
        if hasattr(self, 'points_list'):
            self.points_list.delete(0, tk.END)
            for i, pt in enumerate(self.points):
                self.points_list.insert(tk.END, f"{i+1:02d}: X={pt['phys_x']:0.4g}, Y={pt['phys_y']:0.4g}")

    def _delete_last_point(self):
        if self.points:
            self.points.pop()
            self.points_list.delete(tk.END)
            self._display_image()
            self.status_text.set(f"Deleted point. Remaining points: {len(self.points)}")

    def _clear_all_points(self):
        self.points = []
        self.points_list.delete(0, tk.END)
        self._display_image()
        self.status_text.set("Cleared all extracted points.")

    def _save_to_csv(self):
        if not self.points:
            messagebox.showwarning("Warning", "No data points to save.")
            return
            
        name = self.dataset_name.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a name for the dataset.")
            return
            
        filename = f"{name}.csv"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, filename)
        
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                # Write calibration metadata
                f.write(f"# CALIBRATION_METADATA\n")
                f.write(f"# x_min={self.x_min_val.get()}\n")
                f.write(f"# x_max={self.x_max_val.get()}\n")
                f.write(f"# y_min={self.y_min_val.get()}\n")
                f.write(f"# y_max={self.y_max_val.get()}\n")
                f.write(f"# log_x={self.log_x.get()}\n")
                f.write(f"# log_y={self.log_y.get()}\n")
                if self.px_x_min:
                     f.write(f"# px_x_min={self.px_x_min[0]},{self.px_x_min[1]}\n")
                if self.px_x_max:
                     f.write(f"# px_x_max={self.px_x_max[0]},{self.px_x_max[1]}\n")
                if self.px_y_min:
                     f.write(f"# px_y_min={self.px_y_min[0]},{self.px_y_min[1]}\n")
                if self.px_y_max:
                     f.write(f"# px_y_max={self.px_y_max[0]},{self.px_y_max[1]}\n")
                if self.target_color:
                     f.write(f"# target_color={self.target_color[0]},{self.target_color[1]},{self.target_color[2]}\n")
                f.write(f"# tolerance={self.tolerance_val.get()}\n")
                f.write(f"# search_window={self.search_win_val.get()}\n")
                f.write(f"# max_gap={self.max_gap_val.get()}\n")
                f.write(f"# smooth_window={self.smooth_win_val.get()}\n")
                f.write(f"# drag_points={self.drag_points_val.get()}\n")
                f.write(f"# handle_size={self.handle_size_val.get()}\n")
                if self.start_click_x is not None:
                     f.write(f"# start_click_x={self.start_click_x}\n")
                if self.start_click_y is not None:
                     f.write(f"# start_click_y={self.start_click_y}\n")
                if hasattr(self, 'y_ticks') and self.y_ticks:
                    for tick in self.y_ticks:
                        f.write(f"# y_tick={tick['pixel_y']},{tick['value']}\n")
                
                writer = csv.writer(f)
                writer.writerow(["X_value", "Y_value", "Pixel_X", "Pixel_Y", "Is_Auto"])
                for pt in self.points:
                    writer.writerow([pt['phys_x'], pt['phys_y'], pt['orig_x'], pt['orig_y'], pt.get('is_auto', False)])
            
            messagebox.showinfo("Export Successful", f"Saved {len(self.points)} points to:\n{path}")
            self._scan_csvs()  # Refresh the CSV dropdown
        except Exception as e:
            messagebox.showerror("Error Saving CSV", f"Could not save dataset:\n{e}")

    def _scan_csvs(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        files = [f for f in os.listdir(self.output_dir) if f.lower().endswith(".csv")]
        files.sort()
        self.csv_cb['values'] = files
        if files:
            # Look for a file that matches the selected image file name
            img_file = self.selected_file.get()
            if img_file:
                base_img_name = os.path.splitext(img_file)[0]
                matching = [f for f in files if os.path.splitext(f)[0] == base_img_name]
                if matching:
                    self.csv_cb.set(matching[0])
                    return
            self.csv_cb.current(0)
        else:
            self.selected_csv.set("")

    def _load_from_csv(self):
        filename = self.selected_csv.get()
        if not filename:
            messagebox.showwarning("Warning", "Please select a CSV file to load.")
            return
            
        path = os.path.join(self.output_dir, filename)
        if not os.path.exists(path):
            messagebox.showerror("Error", f"File not found: {path}")
            return
            
        self.loading_csv = True
        try:
            self.canvas.delete("marker_cal")
            self.canvas.delete("marker_data")
            self.points = []
            self.points_list.delete(0, tk.END)
            
            self.px_x_min = self.px_x_max = self.px_y_min = self.px_y_max = None
            self.y_ticks = []
            if hasattr(self, 'y_ticks_list'):
                self.y_ticks_list.delete(0, tk.END)
            metadata = {}
            points_data = []
            
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    content = line[1:].strip()
                    if "=" in content:
                        parts = content.split("=", 1)
                        if len(parts) == 2:
                            key = parts[0].strip()
                            val_str = parts[1].strip()
                            if key == "y_tick":
                                try:
                                    y_p, y_v = val_str.split(",")
                                    self.y_ticks.append({'pixel_y': float(y_p), 'value': float(y_v)})
                                except ValueError:
                                    pass
                            else:
                                metadata[key] = val_str
                elif line and not line.startswith("X_value"):
                    parts = line.split(",")
                    if len(parts) >= 4:
                        try:
                            is_auto = False
                            if len(parts) >= 5:
                                is_auto = (parts[4].strip() == "True")
                            points_data.append({
                                'phys_x': float(parts[0]),
                                'phys_y': float(parts[1]),
                                'orig_x': float(parts[2]),
                                'orig_y': float(parts[3]),
                                'is_auto': is_auto
                            })
                        except ValueError:
                            pass
            
            # Apply metadata if present
            if "x_min" in metadata:
                self.x_min_val.set(float(metadata["x_min"]))
            if "x_max" in metadata:
                self.x_max_val.set(float(metadata["x_max"]))
            if "y_min" in metadata:
                self.y_min_val.set(float(metadata["y_min"]))
            if "y_max" in metadata:
                self.y_max_val.set(float(metadata["y_max"]))
            if "log_x" in metadata:
                self.log_x.set(metadata["log_x"] == "True")
            if "log_y" in metadata:
                self.log_y.set(metadata["log_y"] == "True")
            if "target_color" in metadata:
                parts = metadata["target_color"].split(",")
                self.target_color = (int(parts[0]), int(parts[1]), int(parts[2]))
                r, g, b = self.target_color
                hex_color = f"#{r:02x}{g:02x}{b:02x}"
                fg = "white" if (r*0.299 + g*0.587 + b*0.114) < 128 else "black"
                self.color_swatch.config(bg=hex_color, fg=fg, text=f"RGB({r},{g},{b})")
            else:
                self.target_color = None
                self.color_swatch.config(bg="white", fg="black", text="None")
            if "tolerance" in metadata:
                self.tolerance_val.set(float(metadata["tolerance"]))
            if "search_window" in metadata:
                self.search_win_val.set(int(metadata["search_window"]))
            if "max_gap" in metadata:
                self.max_gap_val.set(int(metadata["max_gap"]))
            if "smooth_window" in metadata:
                self.smooth_win_val.set(int(metadata["smooth_window"]))
            if "drag_points" in metadata:
                self.drag_points_val.set(int(metadata["drag_points"]))
            if "handle_size" in metadata:
                self.handle_size_val.set(int(metadata["handle_size"]))
            if "start_click_x" in metadata:
                self.start_click_x = float(metadata["start_click_x"])
            else:
                self.start_click_x = None
            if "start_click_y" in metadata:
                self.start_click_y = float(metadata["start_click_y"])
            else:
                self.start_click_y = None
                
            # Restore calibration pixels and draw markers
            if "px_x_min" in metadata:
                parts = metadata["px_x_min"].split(",")
                self.px_x_min = (float(parts[0]), float(parts[1]))
            if "px_x_max" in metadata:
                parts = metadata["px_x_max"].split(",")
                self.px_x_max = (float(parts[0]), float(parts[1]))
            if "px_y_min" in metadata:
                parts = metadata["px_y_min"].split(",")
                self.px_y_min = (float(parts[0]), float(parts[1]))
            if "px_y_max" in metadata:
                parts = metadata["px_y_max"].split(",")
                self.px_y_max = (float(parts[0]), float(parts[1]))
                
            # Restore state to digitize if we have full calibration
            if self.px_x_min and self.px_x_max and self.px_y_min and self.px_y_max:
                self.state = 'digitize'
                self.save_btn.config(state=tk.NORMAL)
                self.status_text.set("Calibration metadata loaded! You can continue digitizing.")
            else:
                self.state = 'select'
                self.status_text.set("Points loaded without calibration. Please Calibrate Axes before adding new points.")
                
            # Populate Y-ticks listbox
            if hasattr(self, 'y_ticks') and self.y_ticks:
                self.y_ticks.sort(key=lambda t: t['pixel_y'])
                if hasattr(self, 'y_ticks_list'):
                    for idx, tick in enumerate(self.y_ticks):
                        self.y_ticks_list.insert(tk.END, f"{idx+1:02d}: {tick['value']} @ {tick['pixel_y']:.1f}px")
                        
            # Populate points array
            for pt in points_data:
                self.points.append(pt)
            
            # Automatically recalculate loaded points to apply Y-tick regressions or bounding-box adjustments
            self._recalculate_all_points()
                
            # Redraw everything on canvas (this draws the bounding box and points correctly scaled)
            if self.pil_image:
                self._display_image()
                
            # Set dataset name to match loaded CSV name
            loaded_name = filename[:-4] if filename.lower().endswith(".csv") else filename
            self.dataset_name.set(loaded_name)
            
            messagebox.showinfo("Success", f"Successfully loaded {len(points_data)} points from {filename}.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV file:\n{e}")
        finally:
            self.loading_csv = False

if __name__ == "__main__":
    root = tk.Tk()
    app = PlotDigitizerApp(root)
    root.mainloop()
