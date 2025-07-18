import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
from datetime import datetime
import os
from pathlib import Path

class VisualManualExtractionInterface:
    """
    Enhanced v4 Visual Manual Extraction Interface
    Implements ALL features from the fork documentation
    """
    
    def __init__(self):
        # CURSOR AI v4 - Complete implementation with all fork features
        self.root = tk.Tk()
        self.root.title("Visual Manual Extraction Interface v4 - Full Fork Features")
        
        # Get screen dimensions for scaling
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        
        # Adjust window size based on screen size and scaling
        if screen_width <= 1366:  # Small laptop
            window_width = min(1100, int(screen_width * 0.85))
            window_height = min(650, int(screen_height * 0.8))
        elif screen_width <= 1920:  # Standard desktop
            window_width = 1300
            window_height = 800
        else:  # Large/high-DPI screens
            window_width = 1400
            window_height = 900
            
        self.root.geometry(f"{window_width}x{window_height}")
        
        # Store screen info for responsive design
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Core state variables
        self.image = None
        self.display_image_obj = None
        self.current_file = None
        self.show_processed = False
        self.threshold = 0.5
        
        # CRITICAL: Dictionary-based grid structure from fork
        self.grid_points = {}  # (row, col) -> (x, y) coordinates
        self.extracted_values = {}  # (row, col) -> 0/1 values
        self.grid_cell_values = {}  # (row, col) -> 0/1 values for display in grid cells
        
        # Grid management state (fork structure)
        self.grid_rows = 111  # Default 111 rows
        self.grid_cols = 111  # Default 111 columns
        self.grid_mode = "custom"  # Start in custom mode for immediate draggable functionality
        
        # Grid positioning parameters - extend to edges
        self.origin_x = 0  # Start at left edge
        self.origin_y = 0  # Start at top edge
        self.row_pitch = 10
        self.col_pitch = 10
        
        # Mouse interaction state
        self.dragging_point = None  # (row, col) tuple
        self.drag_offset = (0, 0)
        
        # CRITICAL: Group move functionality - always enabled
        self.group_move_mode = True
        self.group_move_active = True
        
        # CRITICAL: Click-to-move functionality
        self.click_to_move_mode = False
        self.selected_point = None
        
        # CRITICAL: Deletion selection functionality
        self.deletion_selected_point = None  # Point selected for deletion
        
        # Undo system
        self.grid_history = []  # For undo functionality
        self.max_history = 50
        
        # Position indicator
        self.position_indicator_visible = True
        self.current_row = 0
        self.current_col = 0
        
        # Binary values display toggle
        self.show_binary_values = False  # Start with values hidden for clarity
        print(f"[DEBUG] __init__: initial self.show_binary_values = {self.show_binary_values}")
        
        # Zoom and display (match fork naming)
        self.zoom_level = 1.0
        self.zoom_enabled = True  # Enable zoom by default for trackpad gestures
        # Debug logging
        self.debug_log_enabled = True
        self.debug_log_file = "visual_extraction_debug.md"
        
        # Manual extraction state variables
        self.current_line_bits = []  # Current line being built
        self.current_bit = 0
        self.extracted_data = []  # List of binary lines
        
        self.setup_ui()
        self.setup_keyboard_shortcuts()
        self.log_debug("Interface initialized", {"version": "v4", "grid_structure": "dictionary"})
        
        # Auto-load Satoshi image and initialize grid
        self.auto_load_satoshi_image()
        
        # Auto-load most recent session if available
        self.auto_load_recent_session()
        
    def setup_ui(self):
        """Create the enhanced user interface"""
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Image display
        left_panel = ttk.Frame(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Canvas for image with scrollbars
        canvas_frame = ttk.Frame(left_panel)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(canvas_frame, bg='gray20', highlightthickness=0)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Right panel - Controls (responsive design with scrollbar)
        # Adjust width based on screen size - WIDER for manual extraction controls
        if hasattr(self, 'screen_width') and self.screen_width <= 1366:
            panel_width = 420  # Wider for small screens to fit manual controls
        elif hasattr(self, 'screen_width') and self.screen_width <= 1920:
            panel_width = 450  # Wider for standard screens
        else:
            panel_width = 480  # Full width for large screens
            
        right_panel = ttk.Frame(main_container, width=panel_width)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(3, 0))
        right_panel.pack_propagate(False)
        
        # Add scrollbar to right panel
        canvas_right = tk.Canvas(right_panel, highlightthickness=0)
        scrollbar_right = ttk.Scrollbar(right_panel, orient="vertical", command=canvas_right.yview)
        scrollable_frame = ttk.Frame(canvas_right)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas_right.configure(scrollregion=canvas_right.bbox("all"))
        )
        
        canvas_right.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas_right.configure(yscrollcommand=scrollbar_right.set)
        
        canvas_right.pack(side="left", fill="both", expand=True)
        scrollbar_right.pack(side="right", fill="y")
        
        # Bind mouse wheel to control panel scrolling
        def _on_control_mousewheel(event):
            canvas_right.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas_right.bind("<MouseWheel>", _on_control_mousewheel)
        
        # Create organized sections
        self.create_file_section(scrollable_frame)
        self.create_grid_section(scrollable_frame)
        self.create_extraction_section(scrollable_frame)
        self.create_visualization_section(scrollable_frame)
        self.create_export_section(scrollable_frame)
        self.create_debug_section(scrollable_frame)
        
        # Status bar
        self.create_status_bar()
        
        # Bind canvas events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<Motion>", self.on_canvas_motion)
        self.canvas.bind("<Button-3>", self.on_canvas_right_click)
        
        # Bind zoom events for trackpad pinch gestures
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        
    def create_file_section(self, parent):
        """File loading section with presets"""
        section = ttk.LabelFrame(parent, text="File Operations", padding=(5, 5))
        section.pack(fill=tk.X, pady=(0, 5))
        
        # File buttons
        file_buttons = ttk.Frame(section)
        file_buttons.pack(fill=tk.X, pady=1)
        ttk.Button(file_buttons, text="Load Image", command=self.load_image, width=12).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(file_buttons, text="Load Session", command=self.load_grid, width=12).pack(side=tk.LEFT, padx=(0,2))
        ttk.Button(file_buttons, text="Load Recent", command=self.load_recent_session, width=12).pack(side=tk.LEFT)
        
        # Image presets
        presets_frame = ttk.Frame(section)
        presets_frame.pack(fill=tk.X, pady=2)
        ttk.Label(presets_frame, text="Presets:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        
        preset_buttons = ttk.Frame(presets_frame)
        preset_buttons.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(preset_buttons, text="Top4", command=lambda: self.load_preset("top4"), width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(preset_buttons, text="Full", command=lambda: self.load_preset("full"), width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(preset_buttons, text="Grid", command=lambda: self.load_preset("grid"), width=6).pack(side=tk.LEFT, padx=1)
        
        # Current file display
        self.file_label = ttk.Label(section, text="No file loaded", wraplength=260, font=('Arial', 8))
        self.file_label.pack(fill=tk.X, pady=(3, 0))
        
    def create_grid_section(self, parent):
        """Enhanced grid management section"""
        section = ttk.LabelFrame(parent, text="Grid Management", padding=(5, 5))
        section.pack(fill=tk.X, pady=(0, 5))
        
        # Grid mode selection
        mode_frame = ttk.Frame(section)
        mode_frame.pack(fill=tk.X, pady=(0, 3))
        
        ttk.Label(mode_frame, text="Mode:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        self.grid_mode_var = tk.StringVar(value=self.grid_mode)
        
        ttk.Radiobutton(mode_frame, text="Uniform", variable=self.grid_mode_var, 
                       value="uniform", command=self.update_grid_mode).pack(side=tk.LEFT, padx=1)
        ttk.Radiobutton(mode_frame, text="Custom", variable=self.grid_mode_var, 
                       value="custom", command=self.update_grid_mode).pack(side=tk.LEFT, padx=1)
        
        # Grid size controls (ultra compact)
        size_frame = ttk.Frame(section)
        size_frame.pack(fill=tk.X, pady=2)
        
        # Single row layout
        ttk.Label(size_frame, text="R:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 2))
        self.grid_rows_var = tk.StringVar(value=str(self.grid_rows))
        ttk.Entry(size_frame, textvariable=self.grid_rows_var, width=6).pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Label(size_frame, text="C:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 2))
        self.grid_cols_var = tk.StringVar(value=str(self.grid_cols))
        ttk.Entry(size_frame, textvariable=self.grid_cols_var, width=6).pack(side=tk.LEFT, padx=(0, 8))
        
        ttk.Button(size_frame, text="Update", command=self.update_grid_size, width=8).pack(side=tk.LEFT, padx=(5, 0))
        
        # Line adding controls
        line_frame = ttk.Frame(section)
        line_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(line_frame, text="Add Line:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 5))
        self.add_line_mode = None  # Track which line type to add
        ttk.Button(line_frame, text="+ X Line", command=self.start_add_x_line, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(line_frame, text="+ Y Line", command=self.start_add_y_line, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(line_frame, text="Cancel", command=self.cancel_add_line, width=8).pack(side=tk.LEFT, padx=1)
        
        # Control options (compact layout)
        control_frame = ttk.Frame(section)
        control_frame.pack(fill=tk.X, pady=2)
        
        # Single row with shorter labels
        self.click_to_move_var = tk.BooleanVar(value=self.click_to_move_mode)
        ttk.Checkbutton(control_frame, text="Click-Move", 
                       variable=self.click_to_move_var, 
                       command=self.toggle_click_to_move).pack(side=tk.LEFT)
        
        self.position_indicator_var = tk.BooleanVar(value=self.position_indicator_visible)
        ttk.Checkbutton(control_frame, text="Position", 
                       variable=self.position_indicator_var, 
                       command=self.toggle_position_indicator).pack(side=tk.LEFT, padx=(10, 0))
        
        self.show_binary_values_var = tk.BooleanVar(value=self.show_binary_values)
        print(f"[DEBUG] show_binary_values_var id={id(self.show_binary_values_var)}, value={self.show_binary_values_var.get()}")
        self.show_binary_values_cb = ttk.Checkbutton(
            control_frame,
            text="Show Values",
            variable=self.show_binary_values_var,
            command=self.toggle_binary_values
        )
        self.show_binary_values_cb.pack(side=tk.LEFT, padx=(10, 0))
        
        # Grid management buttons (compact single row)
        button_frame = ttk.Frame(section)
        button_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="Reset", command=self.reset_grid, width=8).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(button_frame, text="Undo", command=self.undo_grid_change, width=8).pack(side=tk.LEFT, padx=3)
        ttk.Button(button_frame, text="Save", command=self.save_grid, width=8).pack(side=tk.LEFT, padx=3)
        
        # Line deletion controls (new row)
        delete_frame = ttk.Frame(section)
        delete_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(delete_frame, text="Delete:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(delete_frame, text="Row", command=self.delete_current_row, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(delete_frame, text="Col", command=self.delete_current_column, width=6).pack(side=tk.LEFT, padx=1)
        ttk.Button(delete_frame, text="All Rows", command=self.delete_all_rows, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(delete_frame, text="All Cols", command=self.delete_all_columns, width=8).pack(side=tk.LEFT, padx=1)
        
        # Grid info
        self.grid_info_label = ttk.Label(section, text="Grid: 0 points", font=('Arial', 8))
        self.grid_info_label.pack(fill=tk.X, pady=(3, 0))
        
    def create_extraction_section(self, parent):
        """Extraction controls"""
        section = ttk.LabelFrame(parent, text="Extraction", padding=(5, 5))
        section.pack(fill=tk.X, pady=(0, 5))
        
        # Threshold control
        threshold_frame = ttk.Frame(section)
        threshold_frame.pack(fill=tk.X, pady=(0, 3))
        
        ttk.Label(threshold_frame, text="Threshold:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        self.threshold_var = tk.DoubleVar(value=self.threshold)
        threshold_slider = ttk.Scale(threshold_frame, from_=0, to=1, 
                                   variable=self.threshold_var, 
                                   command=self.update_threshold)
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        
        self.threshold_label = ttk.Label(threshold_frame, text=f"{self.threshold:.2f}", font=('Arial', 8))
        self.threshold_label.pack(side=tk.LEFT)
        
        # Extraction method
        method_frame = ttk.Frame(section)
        method_frame.pack(fill=tk.X, pady=3)
        
        ttk.Label(method_frame, text="Method:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        self.extraction_method = tk.StringVar(value="threshold")
        method_menu = ttk.Combobox(method_frame, textvariable=self.extraction_method, 
                                  values=["threshold", "adaptive", "edge", "manual"], 
                                  state="readonly", width=12, font=('Arial', 8))
        method_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Extract button
        ttk.Button(section, text="Extract All Points", 
                  command=self.extract_all_points).pack(fill=tk.X, pady=3)
        
        # Stats
        self.extraction_stats = ttk.Label(section, text="Extracted: 0/0 points", font=('Arial', 8))
        self.extraction_stats.pack(fill=tk.X)
        
        # CRITICAL: Manual extraction section
        manual_frame = ttk.LabelFrame(section, text="Manual Input", padding=(5, 5))
        manual_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Position tracking and navigation (combined)
        pos_nav_frame = ttk.Frame(manual_frame)
        pos_nav_frame.pack(fill=tk.X, pady=2)
        
        # Position label on left
        self.position_label = ttk.Label(pos_nav_frame, text="Row: 0, Col: 0, Bit: 0", font=('Arial', 8))
        self.position_label.pack(side=tk.LEFT, anchor=tk.W)
        
        # Navigation controls on right
        nav_frame = ttk.Frame(pos_nav_frame)
        nav_frame.pack(side=tk.RIGHT)
        ttk.Button(nav_frame, text="←", command=self.prev_bit, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="→", command=self.next_bit, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="↑", command=self.prev_row, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(nav_frame, text="↓", command=self.next_row, width=3).pack(side=tk.LEFT, padx=1)
        
        # Binary input and controls (combined)
        input_frame = ttk.Frame(manual_frame)
        input_frame.pack(fill=tk.X, pady=2)
        
        # Current line display and binary input field (top row)
        line_frame = ttk.Frame(input_frame)
        line_frame.pack(fill=tk.X, pady=1)
        
        self.current_line_label = ttk.Label(line_frame, text="Line: 1", font=('Arial', 8))
        self.current_line_label.pack(side=tk.LEFT, anchor=tk.W)
        
        # Current line bits display
        self.current_bits_label = ttk.Label(line_frame, text="Bits: ", font=('Arial', 8))
        self.current_bits_label.pack(side=tk.LEFT, anchor=tk.W, padx=(10, 0))
        
        self.binary_input = tk.StringVar()
        self.binary_entry = ttk.Entry(line_frame, textvariable=self.binary_input, width=15, font=('Arial', 8))
        self.binary_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Binary input buttons and completion (bottom row)
        control_frame = ttk.Frame(input_frame)
        control_frame.pack(fill=tk.X, pady=1)
        
        # Left side: bit buttons
        bit_frame = ttk.Frame(control_frame)
        bit_frame.pack(side=tk.LEFT)
        ttk.Button(bit_frame, text="0", command=lambda: self.add_bit("0"), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(bit_frame, text="1", command=lambda: self.add_bit("1"), width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(bit_frame, text="Del", command=self.delete_bit, width=5).pack(side=tk.LEFT, padx=1)
        ttk.Button(bit_frame, text="Clear", command=self.clear_line, width=6).pack(side=tk.LEFT, padx=1)
        
        # Right side: completion buttons
        complete_frame = ttk.Frame(control_frame)
        complete_frame.pack(side=tk.RIGHT)
        ttk.Button(complete_frame, text="Complete", command=self.complete_line, width=8).pack(side=tk.LEFT, padx=1)
        ttk.Button(complete_frame, text="Reset", command=self.reset_position, width=6).pack(side=tk.LEFT, padx=1)
        
        # Keyboard shortcuts help
        help_frame = ttk.Frame(manual_frame)
        help_frame.pack(fill=tk.X, pady=(3, 0))
        help_text = "Keys: 0/1/Space=add bit & advance, Enter=advance only, F1=complete line, ←→↑↓=navigate"
        ttk.Label(help_frame, text=help_text, font=('Arial', 7), foreground='gray').pack(anchor=tk.W)
        
    def create_visualization_section(self, parent):
        """Visualization controls"""
        section = ttk.LabelFrame(parent, text="View", padding=(5, 5))
        section.pack(fill=tk.X, pady=(0, 5))
        
        # View options
        view_frame = ttk.Frame(section)
        view_frame.pack(fill=tk.X, pady=(0, 2))
        
        self.show_grid_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_frame, text="Grid", 
                       variable=self.show_grid_var, 
                       command=self.display_image).pack(side=tk.LEFT, padx=(0, 5))
        
        # FIX: Wire the "Values" checkbox to the same variable as "Show Values"
        self.show_values_var = self.show_binary_values_var  # <- share the same BooleanVar
        ttk.Checkbutton(view_frame, text="Values", 
                       variable=self.show_values_var, 
                       command=self.toggle_binary_values).pack(side=tk.LEFT, padx=(0, 5))  # <- call the real toggle
        
        self.show_processed_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(view_frame, text="Process", 
                       variable=self.show_processed_var, 
                       command=self.toggle_processed_view).pack(side=tk.LEFT)
        
        # Color scheme
        color_frame = ttk.Frame(section)
        color_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(color_frame, text="Colors:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        self.color_scheme = tk.StringVar(value="classic")
        color_menu = ttk.Combobox(color_frame, textvariable=self.color_scheme, 
                                 values=["classic", "contrast", "heatmap"], 
                                 state="readonly", width=10, font=('Arial', 8))
        color_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        color_menu.bind("<<ComboboxSelected>>", lambda e: self.display_image())
        
        # Zoom controls
        zoom_frame = ttk.Frame(section)
        zoom_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(zoom_frame, text="Zoom:", font=('Arial', 8)).pack(side=tk.LEFT, padx=(0, 3))
        self.zoom_var = tk.BooleanVar(value=self.zoom_enabled)
        ttk.Checkbutton(zoom_frame, text="Ctrl+Scroll", 
                       variable=self.zoom_var, 
                       command=self.toggle_zoom).pack(side=tk.LEFT)
        
        # Zoom info
        self.zoom_label = ttk.Label(zoom_frame, text=f"{self.zoom_level:.1f}x", font=('Arial', 8))
        self.zoom_label.pack(side=tk.RIGHT)
        
    def create_export_section(self, parent):
        """Export controls"""
        section = ttk.LabelFrame(parent, text="Export", padding=(5, 5))
        section.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(section, text="Export Text", command=self.export_as_text).pack(fill=tk.X, pady=1)
        ttk.Button(section, text="Export Binary", command=self.export_as_binary).pack(fill=tk.X, pady=1)
        ttk.Button(section, text="Save Session", command=self.save_grid).pack(fill=tk.X, pady=1)
        
    def create_debug_section(self, parent):
        """Debug controls"""
        section = ttk.LabelFrame(parent, text="Debug", padding=(5, 5))
        section.pack(fill=tk.X)
        
        self.debug_var = tk.BooleanVar(value=self.debug_log_enabled)
        ttk.Checkbutton(section, text="Debug Logging", 
                       variable=self.debug_var, 
                       command=self.toggle_debug_logging).pack(anchor=tk.W)
        
        ttk.Button(section, text="Clear Log", 
                  command=self.clear_debug_log).pack(fill=tk.X, pady=(3, 0))
        
    def create_status_bar(self):
        """Create status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.coords_label = ttk.Label(status_frame, text="", relief=tk.SUNKEN, width=25)
        self.coords_label.pack(side=tk.RIGHT)
        
    def setup_keyboard_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.root.bind("<Control-z>", lambda e: self.undo_grid_change())
        self.root.bind("<Control-s>", lambda e: self.save_grid())
        self.root.bind("<Control-o>", lambda e: self.load_image())
        self.root.bind("<Escape>", self.cancel_operations)
        
        # Line deletion shortcuts
        self.root.bind("<Control-r>", lambda e: self.delete_current_row())
        self.root.bind("<Control-c>", lambda e: self.delete_current_column())
        self.root.bind("<Control-Shift-r>", lambda e: self.delete_all_rows())
        self.root.bind("<Control-Shift-c>", lambda e: self.delete_all_columns())
        
        # CRITICAL: Delete key for selected control points
        self.root.bind("<Delete>", lambda e: self.delete_selected_control_point())
        
        # CRITICAL: Manual extraction keyboard shortcuts
        self.root.bind("<Key>", self.on_key_press)
        
    def auto_load_satoshi_image(self):
        """Auto-load the Satoshi image if found"""
        satoshi_paths = [
            "data/mibera_satoshi_4K.png",
            "data/mibera_satoshi.png", 
            "mibera_satoshi_4K.png",
            "mibera_satoshi.png",
            "satoshi.png"
        ]
        
        for path in satoshi_paths:
            if os.path.exists(path):
                try:
                    self.current_file = path
                    self.image = Image.open(path)
                    
                    if self.image.mode not in ['RGB', 'L']:
                        self.image = self.image.convert('RGB')
                    
                    self.file_label.config(text=f"Auto-loaded: {os.path.basename(path)}")
                    
                    # Initialize grid
                    self.initialize_grid()
                    self.display_image()
                    
                    self.status_label.config(text=f"Auto-loaded Satoshi: {self.image.size[0]}x{self.image.size[1]} pixels")
                    self.log_debug("Auto-loaded Satoshi image", {
                        "file": path,
                        "size": self.image.size,
                        "grid_points": len(self.grid_points)
                    })
                    return
                    
                except Exception as e:
                    self.log_debug("Auto-load failed", {"path": path, "error": str(e)})
    
    def auto_load_recent_session(self):
        """Auto-load the most recently used session file"""
        print(f"[DEBUG] auto_load_recent_session called")
        try:
            # Look for session files in current directory
            session_files = []
            for file in os.listdir('.'):
                if file.endswith('.json') and (file.startswith('grid') or 'complete' in file.lower()):
                    file_path = os.path.join('.', file)
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    session_files.append((file_path, mtime))
            
            print(f"[DEBUG] Found {len(session_files)} session files")
            
            if session_files:
                # Sort by modification time (most recent first)
                session_files.sort(key=lambda x: x[1], reverse=True)
                most_recent_file = session_files[0][0]
                
                print(f"[DEBUG] Loading most recent session: {most_recent_file}")
                # Load the most recent session
                self.log_debug("Auto-loading recent session", {"file": most_recent_file})
                self.load_session_file(most_recent_file, auto_load=True)
                
                # Verify the load worked
                print(f"[DEBUG] After auto-load: grid_cell_values count = {len(self.grid_cell_values)}")
                print(f"[DEBUG] After auto-load: grid_cell_values = {dict(list(self.grid_cell_values.items())[:3])}")
            else:
                print(f"[DEBUG] No session files found for auto-load")
                self.log_debug("No session files found for auto-load")
                
        except Exception as e:
            print(f"[DEBUG] Auto-load session failed: {e}")
            import traceback
            traceback.print_exc()
            self.log_debug("Auto-load session failed", {"error": str(e)})
            # Continue with default initialization if auto-load fails
    
    def load_recent_session(self):
        """Manually load the most recent session file"""
        try:
            # Look for session files in current directory
            session_files = []
            for file in os.listdir('.'):
                if file.endswith('.json') and (file.startswith('grid') or 'complete' in file.lower()):
                    file_path = os.path.join('.', file)
                    # Get file modification time
                    mtime = os.path.getmtime(file_path)
                    session_files.append((file_path, mtime))
            
            if session_files:
                # Sort by modification time (most recent first)
                session_files.sort(key=lambda x: x[1], reverse=True)
                most_recent_file = session_files[0][0]
                
                # Load the most recent session
                self.log_debug("Loading recent session", {"file": most_recent_file})
                self.load_session_file(most_recent_file, auto_load=False)
            else:
                messagebox.showinfo("No Sessions", "No session files found in current directory")
                
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load recent session: {e}")
            self.log_debug("Load recent session failed", {"error": str(e)})
        
    def load_session_file(self, filename, auto_load=False):
        """Load session from file (internal method)"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                grid_config = json.load(f)
            
            # Load image if specified and exists
            if grid_config.get('image_file') and os.path.exists(grid_config['image_file']):
                self.current_file = grid_config['image_file']
                self.image = Image.open(self.current_file)
                if self.image.mode not in ['RGB', 'L']:
                    self.image = self.image.convert('RGB')
                self.file_label.config(text=f"File: {os.path.basename(self.current_file)}")
            
            # Load grid parameters
            params = grid_config.get('parameters', {})
            self.row_pitch = params.get('row_pitch', self.row_pitch)
            self.col_pitch = params.get('col_pitch', self.col_pitch)
            self.origin_y = params.get('origin_y', self.origin_y)
            self.origin_x = params.get('origin_x', self.origin_x)
            self.threshold = params.get('threshold', self.threshold)
            self.zoom_level = params.get('zoom_level', 1.0)
            
            # Load grid structure
            self.grid_mode = grid_config.get('grid_mode', 'custom')
            self.grid_rows = grid_config.get('grid_rows', 111)
            self.grid_cols = grid_config.get('grid_cols', 111)
            
            # Update UI
            self.grid_mode_var.set(self.grid_mode)
            self.grid_rows_var.set(str(self.grid_rows))
            self.grid_cols_var.set(str(self.grid_cols))
            self.threshold_var.set(self.threshold)
            
            # Load grid points with dictionary structure
            if grid_config.get('grid_points'):
                self.grid_points = {}
                for key_str, pos in grid_config['grid_points'].items():
                    row, col = map(int, key_str.split(','))
                    self.grid_points[(row, col)] = tuple(pos)
            
            # Load extracted values
            if grid_config.get('extracted_values'):
                self.extracted_values = {}
                for key_str, value in grid_config['extracted_values'].items():
                    row, col = map(int, key_str.split(','))
                    self.extracted_values[(row, col)] = value
            
            # Load grid cell values for manual extraction
            if grid_config.get('grid_cell_values'):
                self.grid_cell_values = {}
                for key_str, value in grid_config['grid_cell_values'].items():
                    row, col = map(int, key_str.split(','))
                    self.grid_cell_values[(row, col)] = value
            
            self.display_image()
            self.update_grid_info()
            
            if auto_load:
                self.status_label.config(text=f"Auto-loaded recent session: {os.path.basename(filename)}")
                self.log_debug("Session auto-loaded", {"filename": filename, "points": len(self.grid_points)})
            else:
                self.status_label.config(text=f"Loaded grid from {os.path.basename(filename)}")
                self.log_debug("Grid loaded", {"filename": filename, "points": len(self.grid_points)})
                
        except Exception as e:
            if not auto_load:
                messagebox.showerror("Load Error", f"Failed to load grid: {e}")
            self.log_debug("Grid load failed", {"error": str(e), "auto_load": auto_load})
    
    def initialize_grid(self):
        """Initialize grid with dictionary structure - CRITICAL FUNCTION"""
        if not self.image:
            return
            
        # Save state BEFORE clearing (if grid exists)
        if self.grid_points:
            self.save_grid_state()
        
        # Clear existing grid
        self.grid_points = {}
        self.extracted_values = {}
        
        # Calculate grid parameters based on image size
        img_width, img_height = self.image.size
        
        # Calculate spacing to fit grid within image
        if self.grid_cols > 1:
            self.col_pitch = (img_width - 2 * self.origin_x) / (self.grid_cols - 1)
        else:
            self.col_pitch = 0
            
        if self.grid_rows > 1:
            self.row_pitch = (img_height - 2 * self.origin_y) / (self.grid_rows - 1)
        else:
            self.row_pitch = 0
        
        # Generate grid points with dictionary structure
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                x = self.origin_x + col * self.col_pitch
                y = self.origin_y + row * self.row_pitch
                self.grid_points[(row, col)] = (x, y)
        
        self.update_grid_info()
        self.log_debug("Grid initialized", {
            "rows": self.grid_rows,
            "cols": self.grid_cols,
            "points": len(self.grid_points),
            "structure": "dictionary"
        })
    
    def load_preset(self, preset_type):
        """Load predefined grid presets"""
        if not self.image:
            messagebox.showwarning("No Image", "Load an image first")
            return
            
        if preset_type == "top4":
            self.grid_rows = 4
            self.grid_cols = 40
        elif preset_type == "full":
            self.grid_rows = 111
            self.grid_cols = 111
        elif preset_type == "grid":
            self.grid_rows = 50
            self.grid_cols = 50
        
        self.grid_rows_var.set(str(self.grid_rows))
        self.grid_cols_var.set(str(self.grid_cols))
        self.grid_mode = "uniform"
        self.grid_mode_var.set("uniform")
        
        self.initialize_grid()
        self.display_image()
        self.log_debug("Loaded preset", {"type": preset_type, "rows": self.grid_rows, "cols": self.grid_cols})
    
    def update_grid_mode(self):
        """Update grid mode"""
        self.grid_mode = self.grid_mode_var.get()
        if self.grid_mode == "uniform" and self.image:
            self.initialize_grid()
            self.display_image()
        self.log_debug("Grid mode changed", {"mode": self.grid_mode})
    
    def update_grid_size(self):
        """Update grid size from UI controls"""
        try:
            new_rows = int(self.grid_rows_var.get())
            new_cols = int(self.grid_cols_var.get())
            
            if new_rows > 0 and new_cols > 0 and new_rows <= 1000 and new_cols <= 1000:
                self.grid_rows = new_rows
                self.grid_cols = new_cols
                
                if self.image:
                    self.initialize_grid()
                    self.display_image()
                    
                self.log_debug("Grid size updated", {"rows": self.grid_rows, "cols": self.grid_cols})
            else:
                messagebox.showerror("Invalid Size", "Grid size must be between 1 and 1000")
                self.grid_rows_var.set(str(self.grid_rows))
                self.grid_cols_var.set(str(self.grid_cols))
                
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers")
            self.grid_rows_var.set(str(self.grid_rows))
            self.grid_cols_var.set(str(self.grid_cols))
    
    # Group move is now always enabled - no toggle needed
    
    def toggle_click_to_move(self):
        """Toggle click-to-move mode - CRITICAL FUNCTION"""
        self.click_to_move_mode = self.click_to_move_var.get()
        
        # Clear selected point when disabling
        if not self.click_to_move_mode:
            self.selected_point = None
            
        status = "enabled" if self.click_to_move_mode else "disabled"
        self.status_label.config(text=f"Click-to-move mode {status}")
        
        # CRITICAL: Update display to show/hide selected point
        self.display_image()
        
        self.log_debug("Click-to-move toggled", {"enabled": self.click_to_move_mode, "group_move": self.group_move_mode})
    
    def toggle_position_indicator(self):
        """Toggle position indicator"""
        self.position_indicator_visible = self.position_indicator_var.get()
        self.display_image()
        self.log_debug("Position indicator toggled", {"visible": self.position_indicator_visible})
    
    def toggle_binary_values(self):
        """Toggle binary values display - OPTIMIZED"""
        self.show_binary_values = self.show_binary_values_var.get()
        
        # Only refresh if we have binary values to show/hide
        if len(self.grid_cell_values) > 0:
            self.display_image()
        
        # Update status
        self.status_label.config(text=f"Binary values {'shown' if self.show_binary_values else 'hidden'}")
        self.log_debug("Binary values toggled", {"visible": self.show_binary_values})
    
    def save_grid_state(self):
        """Save current grid state for undo"""
        if len(self.grid_history) >= self.max_history:
            self.grid_history.pop(0)
        
        state = {
            'grid_points': self.grid_points.copy(),
            'extracted_values': self.extracted_values.copy(),
            'timestamp': datetime.now().isoformat()
        }
        self.grid_history.append(state)
    
    def undo_grid_change(self):
        """Undo last grid change"""
        if len(self.grid_history) > 1:
            self.grid_history.pop()  # Remove current state
            previous_state = self.grid_history[-1]
            self.grid_points = previous_state['grid_points'].copy()
            self.extracted_values = previous_state['extracted_values'].copy()
            self.display_image()
            self.update_grid_info()
            self.log_debug("Undo performed", {"restored_points": len(self.grid_points)})
        else:
            self.status_label.config(text="No more undo history")
    
    def update_grid_info(self):
        """Update grid information display"""
        total = len(self.grid_points)
        extracted = len(self.extracted_values)
        self.grid_info_label.config(text=f"Grid: {total} points")
        self.extraction_stats.config(text=f"Extracted: {extracted}/{total} points")
    
    def toggle_zoom(self):
        """Toggle zoom functionality"""
        self.zoom_enabled = self.zoom_var.get()
        status = "enabled" if self.zoom_enabled else "disabled"
        self.status_label.config(text=f"Ctrl+Scroll zoom {status} - Regular scrolling always works")
        self.log_debug("Zoom toggled", {"enabled": self.zoom_enabled})
    
    def on_mouse_wheel(self, event):
        """Handle mouse wheel - FIXED scrolling approach"""
        if not self.image:
            return
            
        # ONLY handle zoom when explicitly requested with Ctrl key
        # This fixes the scrolling issue by being less aggressive about capturing events
        if hasattr(event, 'state') and event.state & 0x4:  # Ctrl key held
            if self.zoom_enabled:
                # Handle zoom only when Ctrl is held
                if hasattr(event, 'delta') and event.delta:
                    delta = event.delta / 120
                elif hasattr(event, 'num'):
                    if event.num == 4:
                        delta = 1
                    elif event.num == 5:
                        delta = -1
                    else:
                        return
                else:
                    return
        
                # Apply zoom
                zoom_speed = 0.1
                old_zoom = self.zoom_level
                self.zoom_level *= (1 + delta * zoom_speed)
                self.zoom_level = max(0.1, min(5.0, self.zoom_level))
                
                if abs(self.zoom_level - old_zoom) > 0.01:
                    self.display_image()
                    self.zoom_label.config(text=f"{self.zoom_level:.1f}x")
                    self.status_label.config(text=f"Zoom: {self.zoom_level:.1f}x")
                    
                return "break"  # Only prevent scrolling when we actually zoom
        
        # For ALL other mouse wheel events, let them pass through normally
        # This should fix the scrolling issue by not intercepting regular scroll events
        return
    

    
    def on_canvas_click(self, event):
        """Handle canvas click - CRITICAL for group move and click-to-move"""
        if not self.image or not self.grid_points:
            return
        
        # Convert canvas coordinates to image coordinates (match fork)
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = canvas_x / self.zoom_level
        image_y = canvas_y / self.zoom_level
        
        # Find closest grid point
        min_dist = float('inf')
        closest_point = None
        
        for (row, col), (gx, gy) in self.grid_points.items():
            dist = ((image_x - gx) ** 2 + (image_y - gy) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_point = (row, col)
        
        # Check if click is close enough (match fork threshold)
        if min_dist <= 10 and closest_point:
            # CRITICAL: Check if this is a control point (top row or left column) for deletion
            row, col = closest_point
            is_control_point = (row == 0 or col == 0)
            
            if self.click_to_move_mode:
                # Click-to-move mode: select point first, then move on second click
                if self.selected_point is None:
                    # First click: select the point
                    # CRITICAL: Ensure the point exists in grid before selecting
                    if closest_point in self.grid_points:
                        self.selected_point = closest_point
                        self.status_label.config(text=f"Selected point {closest_point} - click destination to move")
                        self.log_debug("Selected point for click-to-move", {"point": closest_point})
                        # CRITICAL: Update display to show selected point
                        self.display_image()
                    else:
                        self.status_label.config(text="Cannot select invalid point")
                        self.log_debug("Attempted to select invalid point", {"point": closest_point})
                else:
                    # Second click: move selected point to this location
                    # CRITICAL: Check if selected point still exists in grid
                    if self.selected_point not in self.grid_points:
                        self.selected_point = None
                        self.status_label.config(text="Selected point no longer exists")
                        self.display_image()
                        return
                    
                    old_x, old_y = self.grid_points[self.selected_point]
                    
                    # CRITICAL: Always use group move for control points
                    row, col = self.selected_point
                    if row == 0:  # Top row - move entire column
                        offset_x = image_x - old_x
                        # Apply same offset to all points in this column
                        for r in range(self.grid_rows):
                            if (r, col) in self.grid_points:
                                old_x_col, old_y_col = self.grid_points[(r, col)]
                                self.grid_points[(r, col)] = (old_x_col + offset_x, old_y_col)
                    elif col == 0:  # Leftmost column - move entire row
                        offset_y = image_y - old_y
                        # Apply same offset to all points in this row
                        for c in range(self.grid_cols):
                            if (row, c) in self.grid_points:
                                old_x_row, old_y_row = self.grid_points[(row, c)]
                                self.grid_points[(row, c)] = (old_x_row, old_y_row + offset_y)
                    else:
                        # Regular point move
                        self.grid_points[self.selected_point] = (image_x, image_y)
                    
                    # Save state for undo
                    self.save_grid_state()
                    
                    # Determine move type based on point position
                    if row == 0 or col == 0:
                        move_type = "group"
                    else:
                        move_type = "single"
                    self.status_label.config(text=f"Moved point {self.selected_point} ({move_type} move) to ({int(image_x)}, {int(image_y)})")
                    self.log_debug("Moved point via click-to-move", {
                        "point": self.selected_point,
                        "from": (old_x, old_y),
                        "to": (image_x, image_y),
                        "group_move": self.group_move_mode
                    })
                    
                    # Clear selection and update display
                    self.selected_point = None
                    self.display_image()
                    
            elif self.grid_mode_var.get() == "custom":
                # Regular drag mode: start dragging this point
                self.dragging_point = closest_point
                gx, gy = self.grid_points[closest_point]
                self.drag_offset = (image_x - gx, image_y - gy)
                
                self.log_debug("Started dragging point", {
                    "point": closest_point, 
                    "group_mode": self.group_move_mode
                })
            
            # CRITICAL: Handle deletion selection for control points
            if is_control_point:
                # Select this point for deletion
                self.deletion_selected_point = closest_point
                point_type = "column" if row == 0 else "row"
                self.status_label.config(text=f"Selected {point_type} {col if row == 0 else row} for deletion - press Delete key")
                self.log_debug("Selected point for deletion", {"point": closest_point, "type": point_type})
                # Update display to show deletion selection
                self.display_image()
        else:
            # Click not near any grid point - handle add line mode or jump to cell
            
            # Check if we're in add line mode
            if self.add_line_mode == "x":
                # Add vertical line at clicked position
                self.add_x_line_at_position(image_x, image_y)
                self.add_line_mode = None  # Exit add line mode
                return
            elif self.add_line_mode == "y":
                # Add horizontal line at clicked position
                self.add_y_line_at_position(image_x, image_y)
                self.add_line_mode = None  # Exit add line mode
                return
            
            # First handle existing click-to-move clearing
            if self.click_to_move_mode and self.selected_point:
                # Clear selection if clicking away from grid
                self.selected_point = None
                self.status_label.config(text="Selection cleared")
                # CRITICAL: Update display to remove selected point highlight
                self.display_image()
            
            # Clear deletion selection if clicking away from grid
            if self.deletion_selected_point:
                self.deletion_selected_point = None
                self.status_label.config(text="Deletion selection cleared")
                self.display_image()
            
            # Click-to-jump: Find which cell was clicked and jump cursor there
            target_row, target_col = self.find_cell_at_position(image_x, image_y)
            if target_row is not None and target_col is not None:
                # Jump to the clicked cell
                self.current_row = target_row
                self.current_col = target_col
                self.current_bit = 0  # Reset bit position
                self.update_position_info()
                self.status_label.config(text=f"Jumped to cell ({target_row}, {target_col})")
                self.display_image()
                self.log_debug("Jumped to cell", {"row": target_row, "col": target_col})
    
    def start_add_x_line(self):
        """Start adding a vertical line (X line)"""
        self.add_line_mode = "x"
        self.status_label.config(text="Click where you want to add a vertical line (X line)")
        print(f"[DEBUG] Started add X line mode")
        self.log_debug("Started add X line mode")
    
    def start_add_y_line(self):
        """Start adding a horizontal line (Y line)"""
        self.add_line_mode = "y"
        self.status_label.config(text="Click where you want to add a horizontal line (Y line)")
        self.log_debug("Started add Y line mode")
    
    def cancel_add_line(self):
        """Cancel adding a line"""
        self.add_line_mode = None
        self.status_label.config(text="Add line mode cancelled")
        self.log_debug("Cancelled add line mode")
    
    def add_x_line_at_position(self, x, y):
        """Add a vertical line (X line) at the specified position"""
        if not self.grid_points:
            return
        
        # Find the column index where this line should be inserted
        # Look for the rightmost column that has x-coordinate less than the click position
        target_col = 0
        for (row, col), (gx, gy) in self.grid_points.items():
            if gx < x and col >= target_col:
                target_col = col + 1
        
        # Don't add if it would be too close to existing lines
        min_distance = 10
        for (row, col), (gx, gy) in self.grid_points.items():
            if abs(gx - x) < min_distance:
                self.status_label.config(text=f"Too close to existing line at column {col}")
                return
        
        # Shift all existing columns to the right
        new_grid_points = {}
        for (row, col), (gx, gy) in self.grid_points.items():
            if col >= target_col:
                new_grid_points[(row, col + 1)] = (gx, gy)
            else:
                new_grid_points[(row, col)] = (gx, gy)
        
        # Add the new column at the target position
        for row in range(self.grid_rows):
            # Find the Y coordinate for this row
            y_coord = y  # Default to click position
            if (row, target_col - 1) in new_grid_points:
                # Use Y coordinate from adjacent point
                _, y_coord = new_grid_points[(row, target_col - 1)]
            elif (row, target_col + 1) in new_grid_points:
                # Use Y coordinate from adjacent point
                _, y_coord = new_grid_points[(row, target_col + 1)]
            
            new_grid_points[(row, target_col)] = (x, y_coord)
        
        self.grid_points = new_grid_points
        self.grid_cols += 1
        self.grid_cols_var.set(str(self.grid_cols))
        
        self.save_grid_state()
        self.display_image()
        self.status_label.config(text=f"Added vertical line at column {target_col}")
        self.log_debug("Added X line", {"x": x, "col": target_col})
    
    def add_y_line_at_position(self, x, y):
        """Add a horizontal line (Y line) at the specified position"""
        if not self.grid_points:
            return
        
        # Find the row index where this line should be inserted
        # Look for the bottom row that has y-coordinate less than the click position
        target_row = 0
        for (row, col), (gx, gy) in self.grid_points.items():
            if gy < y and row >= target_row:
                target_row = row + 1
        
        # Don't add if it would be too close to existing lines
        min_distance = 10
        for (row, col), (gx, gy) in self.grid_points.items():
            if abs(gy - y) < min_distance:
                self.status_label.config(text=f"Too close to existing line at row {row}")
                return
        
        # Shift all existing rows down
        new_grid_points = {}
        for (row, col), (gx, gy) in self.grid_points.items():
            if row >= target_row:
                new_grid_points[(row + 1, col)] = (gx, gy)
            else:
                new_grid_points[(row, col)] = (gx, gy)
        
        # Add the new row at the target position
        for col in range(self.grid_cols):
            # Find the X coordinate for this column
            x_coord = x  # Default to click position
            if (target_row - 1, col) in new_grid_points:
                # Use X coordinate from adjacent point
                x_coord, _ = new_grid_points[(target_row - 1, col)]
            elif (target_row + 1, col) in new_grid_points:
                # Use X coordinate from adjacent point
                x_coord, _ = new_grid_points[(target_row + 1, col)]
            
            new_grid_points[(target_row, col)] = (x_coord, y)
        
        self.grid_points = new_grid_points
        self.grid_rows += 1
        self.grid_rows_var.set(str(self.grid_rows))
        
        self.save_grid_state()
        self.display_image()
        self.status_label.config(text=f"Added horizontal line at row {target_row}")
        self.log_debug("Added Y line", {"y": y, "row": target_row})

    def find_cell_at_position(self, x, y):
        """Find which cell contains the given position"""
        if not self.grid_points:
            return None, None
        
        # For custom grid, we need to find which cell the position falls into
        # by checking which cell boundaries contain the point
        
        # Check all grid points to find the cell that contains this position
        for (row, col) in self.grid_points:
            # Calculate cell boundaries for this grid point
            gx, gy = self.grid_points[(row, col)]
            
            # Cell starts at the grid point
            x1 = gx
            y1 = gy
            
            # Calculate right boundary (to next column's grid point or default)
            if (row, col + 1) in self.grid_points:
                next_x, _ = self.grid_points[(row, col + 1)]
                x2 = next_x
            else:
                x2 = gx + 40  # Default cell width
            
            # Calculate bottom boundary (to next row's grid point or default)
            if (row + 1, col) in self.grid_points:
                _, next_y = self.grid_points[(row + 1, col)]
                y2 = next_y
            else:
                y2 = gy + 40  # Default cell height
            
            # Check if the position is within this cell
            if x1 <= x <= x2 and y1 <= y <= y2:
                return row, col
        
        return None, None

    def on_canvas_drag(self, event):
        """Handle canvas drag - CRITICAL GROUP MOVE IMPLEMENTATION"""
        if not self.image or self.dragging_point is None:
            return
        
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        image_x = canvas_x / self.zoom_level
        image_y = canvas_y / self.zoom_level
        
        new_x = image_x - self.drag_offset[0]
        new_y = image_y - self.drag_offset[1]
        
        row, col = self.dragging_point
        
        # CRITICAL: Always use group move for control points (top row and left column)
        if row == 0:  # Top row - move entire column
            current_x, current_y = self.grid_points[self.dragging_point]
            offset_x = new_x - current_x
            
            # Apply same offset to all points in this column
            for r in range(self.grid_rows):
                if (r, col) in self.grid_points:
                    old_x, old_y = self.grid_points[(r, col)]
                    self.grid_points[(r, col)] = (old_x + offset_x, old_y)
                    
        elif col == 0:  # Leftmost column - move entire row
            current_x, current_y = self.grid_points[self.dragging_point]
            offset_y = new_y - current_y
            
            # Apply same offset to all points in this row
            for c in range(self.grid_cols):
                if (row, c) in self.grid_points:
                    old_x, old_y = self.grid_points[(row, c)]
                    self.grid_points[(row, c)] = (old_x, old_y + offset_y)
        else:
            # Regular point move
            self.grid_points[self.dragging_point] = (new_x, new_y)
        
        # Update display
        self.display_image()
    
    def on_canvas_release(self, event):
        """Handle canvas release - CRITICAL: Save state AFTER the move"""
        if self.dragging_point is not None:
            # CRITICAL: Save grid state for undo AFTER the move is complete
            self.save_grid_state()
            
            self.log_debug("Finished dragging point", {
                "point": self.dragging_point,
                "group_mode": self.group_move_mode
            })
            self.dragging_point = None
            self.drag_offset = (0, 0)
            self.update_grid_info()
    
    def on_canvas_motion(self, event):
        """Handle mouse motion for coordinate display"""
        if self.image:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            image_x = canvas_x / self.zoom_level
            image_y = canvas_y / self.zoom_level
            
            if 0 <= image_x < self.image.width and 0 <= image_y < self.image.height:
                pixel = self.image.getpixel((int(image_x), int(image_y)))
                if isinstance(pixel, int):
                    pixel_str = f"Gray: {pixel}"
                else:
                    pixel_str = f"RGB: {pixel[:3]}"
                    
                # Show nearest grid point info
                nearest_info = ""
                if self.grid_points:
                    min_dist = float('inf')
                    nearest_point = None
                    for (row, col), (gx, gy) in self.grid_points.items():
                        dist = ((image_x - gx) ** 2 + (image_y - gy) ** 2) ** 0.5
                        if dist < min_dist:
                            min_dist = dist
                            nearest_point = (row, col)
                    
                    if nearest_point and min_dist <= 20:
                        nearest_info = f" | Grid({nearest_point[0]},{nearest_point[1]})"
                
                self.coords_label.config(text=f"({int(image_x)}, {int(image_y)}) {pixel_str}{nearest_info}")
    
    def on_canvas_right_click(self, event):
        """Handle right-click for context menu"""
        pass  # Placeholder for context menu
    
    def cancel_operations(self, event=None):
        """Cancel current operations"""
        self.dragging_point = None
        self.drag_offset = (0, 0)
        self.status_label.config(text="Operations cancelled")
    
    def display_image(self):
        """Display image with grid overlay - CRITICAL DISPLAY FUNCTION"""
        if not self.image:
            return
        print(f"[DEBUG] display_image called. show_binary_values = {self.show_binary_values}")
        print(f"[DEBUG] display_image: id(self.image)={id(self.image)}, type(self.image)={type(self.image)}")
        display_img = self.image.copy()
        print(f"[DEBUG] display_image: id(display_img)={id(display_img)}, type(display_img)={type(display_img)}")
        
        # Apply zoom (match fork naming)
        if self.zoom_level != 1.0:
            new_width = int(display_img.width * self.zoom_level)
            new_height = int(display_img.height * self.zoom_level)
            display_img = display_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to OpenCV format for grid drawing
        cv_image = cv2.cvtColor(np.array(display_img), cv2.COLOR_RGB2BGR)
        
        # Draw grid if enabled
        if self.show_grid_var.get() and self.grid_points:
            self.draw_grid_on_image(cv_image)
        
        # Draw position indicator if enabled
        if self.position_indicator_visible:
            self.draw_current_position(cv_image)
        
        # Convert back to PIL and display
        pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        self.display_image_obj = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.display_image_obj)
        
        # Update scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def draw_grid_on_image(self, image):
        """Draw grid on OpenCV image - CRITICAL GRID DRAWING"""
        height, width = image.shape[:2]
        
        # Remove debug output for better performance
        pass
        
        # Draw grid points and connections
        if self.grid_points:
            # Using custom grid path - optimize by limiting scope
            cell_boundaries = {}
            
            # Draw ALL grid cells to avoid corruption
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    if (row, col) in self.grid_points:
                        x, y = self.grid_points[(row, col)]
                        x_scaled = int(x * self.zoom_level)
                        y_scaled = int(y * self.zoom_level)
                        
                        # Calculate cell boundaries where grid point is at TOP-LEFT corner
                        # This makes the cell extend from the grid point to the next grid point
                        
                        # Cell starts at the grid point
                        x1 = x_scaled
                        y1 = y_scaled
                        
                        # Calculate right boundary (to next column's grid point or default)
                        if (row, col + 1) in self.grid_points:
                            next_x, _ = self.grid_points[(row, col + 1)]
                            next_x_scaled = int(next_x * self.zoom_level)
                            x2 = next_x_scaled
                        else:
                            x2 = min(width, x_scaled + 40)  # Default cell width
                        
                        # Calculate bottom boundary (to next row's grid point or default)
                        if (row + 1, col) in self.grid_points:
                            _, next_y = self.grid_points[(row + 1, col)]
                            next_y_scaled = int(next_y * self.zoom_level)
                            y2 = next_y_scaled
                        else:
                            y2 = min(height, y_scaled + 40)  # Default cell height
                        
                        # Ensure minimum cell size
                        if x2 - x1 < 20:
                            x2 = x1 + 20
                        if y2 - y1 < 20:
                            y2 = y1 + 20
                        
                        cell_boundaries[(row, col)] = (x1, y1, x2, y2)
                        
                        # Draw grid point with different colors based on state
                        if self.dragging_point == (row, col):
                            # Currently dragging: red circle
                            cv2.circle(image, (x_scaled, y_scaled), 5, (0, 0, 255), -1)
                        elif (row == 0 or col == 0):
                            # Group control points: magenta dots (always visible)
                            cv2.circle(image, (x_scaled, y_scaled), 4, (255, 0, 255), -1)
                        elif self.selected_point == (row, col) and self.click_to_move_mode:
                            # Selected point in click-to-move mode: blue circle
                            cv2.circle(image, (x_scaled, y_scaled), 5, (255, 0, 0), -1)
                        else:
                            # Regular points: green circles
                            cv2.circle(image, (x_scaled, y_scaled), 3, (0, 255, 0), -1)
                        
                        # Draw connections to all adjacent points (all 4 directions)
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # Right, down, left, up
                            next_row, next_col = row + dr, col + dc
                            if (next_row, next_col) in self.grid_points:
                                next_x, next_y = self.grid_points[(next_row, next_col)]
                                next_x_scaled = int(next_x * self.zoom_level)
                                next_y_scaled = int(next_y * self.zoom_level)
                                cv2.line(image, (x_scaled, y_scaled), (next_x_scaled, next_y_scaled), (0, 255, 0), 1)
                        
                        # Draw binary value if present and toggle is on
                        if (row, col) in self.grid_cell_values:
                            self.draw_binary_value_in_cell(image, row, col, x1, y1, x2, y2)
                        
                        # Highlight current position
                        if row == self.current_row and col == self.current_col:
                            # Draw a bright border around current cell
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
                        
                        # Highlight selected point (for click-to-move)
                        if self.selected_point and (row, col) == self.selected_point:
                            cv2.circle(image, (x_scaled, y_scaled), 6, (255, 0, 255), 2)
        else:
            # Using uniform grid path
            cell_width = width / self.grid_cols
            cell_height = height / self.grid_rows
            
            # Draw ALL grid cells to avoid corruption
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    # Calculate cell coordinates
                    x1 = int(col * cell_width)
                    y1 = int(row * cell_height)
                    x2 = int((col + 1) * cell_width)
                    y2 = int((row + 1) * cell_height)
                    
                    # Draw cell border
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    # Draw binary value if present and toggle is on
                    if (row, col) in self.grid_cell_values:
                        self.draw_binary_value_in_cell(image, row, col, x1, y1, x2, y2)
                    
                    # Highlight current position
                    if row == self.current_row and col == self.current_col:
                        # Draw a bright border around current cell
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
    
    def draw_binary_value_in_cell(self, image, row, col, x1, y1, x2, y2):
        """Draw binary value in a specific cell - OPTIMIZED"""
        # CRITICAL: Only draw if toggle is enabled
        if not self.show_binary_values:
            return
        
        value = str(self.grid_cell_values[(row, col)])
        
        # Determine color based on value (cached)
        if value == "1":
            bg_color = (0, 100, 0)  # Dark green background
            text_color = (0, 255, 0)  # Bright green text
        else:
            bg_color = (0, 0, 100)  # Dark blue background
            text_color = (255, 255, 255)  # White text
        
        # Fill cell with background color
        cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
        
        # Draw cell border
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        
        # Calculate text position and size - properly centered within cell content area
        # Account for 1-pixel border by using inner cell dimensions
        border_offset = 1
        inner_x1 = x1 + border_offset
        inner_y1 = y1 + border_offset
        inner_x2 = x2 - border_offset
        inner_y2 = y2 - border_offset
        
        inner_width = inner_x2 - inner_x1
        inner_height = inner_y2 - inner_y1
        
        # Use better font scale calculation based on inner dimensions
        font_scale = min(inner_width, inner_height) / 25.0
        font_scale = max(0.4, min(1.2, font_scale))
        thickness = max(1, int(font_scale * 1.5))
        
        # Get text size for proper centering
        text_size = cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_width, text_height = text_size
        
        # Calculate centered position within inner cell area
        text_x = inner_x1 + (inner_width - text_width) // 2
        text_y = inner_y1 + (inner_height + text_height) // 2
        
        cv2.putText(image, value, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
    
    def draw_current_position(self, image):
        """Draw current position indicator"""
        if not self.position_indicator_visible:
            return
        
        height, width = image.shape[:2]
        
        # Calculate position based on grid parameters, even if point doesn't exist
        if self.grid_points:
            # Try to find the actual grid point first
            if (self.current_row, self.current_col) in self.grid_points:
                x, y = self.grid_points[(self.current_row, self.current_col)]
            else:
                # Calculate theoretical position based on grid parameters
                x = self.origin_x + self.current_col * self.col_pitch
                y = self.origin_y + self.current_row * self.row_pitch
            
            x_scaled = int(x * self.zoom_level)
            y_scaled = int(y * self.zoom_level)
            
            if 0 <= x_scaled < width and 0 <= y_scaled < height:
                # Draw crosshair (larger and brighter)
                cv2.line(image, (x_scaled-12, y_scaled), (x_scaled+12, y_scaled), (0, 255, 255), 2)
                cv2.line(image, (x_scaled, y_scaled-12), (x_scaled, y_scaled+12), (0, 255, 255), 2)
                
                # Draw circle around current position (larger and brighter)
                cv2.circle(image, (x_scaled, y_scaled), 8, (0, 255, 255), 2)
                
                # Draw position text (larger and brighter)
                cv2.putText(image, f"({self.current_row},{self.current_col})", 
                           (x_scaled+15, y_scaled-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    def load_image(self):
        """Load image file"""
        initial_dir = os.path.dirname(self.current_file) if self.current_file else os.getcwd()
        
        filename = filedialog.askopenfilename(
            title="Select Image",
            initialdir=initial_dir,
            filetypes=[
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All images", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            try:
                self.current_file = filename
                self.image = Image.open(filename)
                
                if self.image.mode not in ['RGB', 'L']:
                    self.image = self.image.convert('RGB')
                
                self.file_label.config(text=f"File: {os.path.basename(filename)}")
                
                # Initialize grid
                self.initialize_grid()
                self.display_image()
                
                self.status_label.config(text=f"Loaded: {self.image.size[0]}x{self.image.size[1]} pixels")
                self.log_debug("Image loaded", {
                    "file": filename,
                    "size": self.image.size,
                    "grid_points": len(self.grid_points)
                })
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.log_debug("Image load failed", {"error": str(e)})
    
    def reset_grid(self):
        """Reset grid to default state"""
        if self.image:
            # Save current state before reset
            if self.grid_points:
                self.save_grid_state()
            # CRITICAL: Clear selected point before resetting grid
            self.selected_point = None
            self.initialize_grid()
            self.display_image()
            self.log_debug("Grid reset")
    
    def delete_current_row(self):
        """Delete the current row (based on manual extraction position)"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to delete from")
            return
        
        row_to_delete = self.current_row
        if row_to_delete >= self.grid_rows:
            messagebox.showwarning("Invalid Row", f"Row {row_to_delete} is out of range")
            return
        
        # Save state before deletion
        self.save_grid_state()
        
        # Delete all points in the current row
        points_deleted = 0
        for col in range(self.grid_cols):
            if (row_to_delete, col) in self.grid_points:
                del self.grid_points[(row_to_delete, col)]
                points_deleted += 1
            if (row_to_delete, col) in self.extracted_values:
                del self.extracted_values[(row_to_delete, col)]
        
        # Update grid dimensions
        self.grid_rows -= 1
        self.grid_rows_var.set(str(self.grid_rows))
        
        # CRITICAL: Clear selected point if it was in the deleted row
        if self.selected_point and self.selected_point[0] == row_to_delete:
            self.selected_point = None
            self.status_label.config(text=f"Deleted row {row_to_delete} ({points_deleted} points) - selection cleared")
        else:
            self.status_label.config(text=f"Deleted row {row_to_delete} ({points_deleted} points)")
        
        # Update display
        self.display_image()
        self.update_grid_info()
        
        self.log_debug("Row deleted", {"row": row_to_delete, "points_deleted": points_deleted})
    
    def delete_current_column(self):
        """Delete the current column (based on manual extraction position)"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to delete from")
            return
        
        col_to_delete = self.current_col
        if col_to_delete >= self.grid_cols:
            messagebox.showwarning("Invalid Column", f"Column {col_to_delete} is out of range")
            return
        
        # Save state before deletion
        self.save_grid_state()
        
        # Delete all points in the current column
        points_deleted = 0
        for row in range(self.grid_rows):
            if (row, col_to_delete) in self.grid_points:
                del self.grid_points[(row, col_to_delete)]
                points_deleted += 1
            if (row, col_to_delete) in self.extracted_values:
                del self.extracted_values[(row, col_to_delete)]
        
        # Update grid dimensions
        self.grid_cols -= 1
        self.grid_cols_var.set(str(self.grid_cols))
        
        # CRITICAL: Clear selected point if it was in the deleted column
        if self.selected_point and self.selected_point[1] == col_to_delete:
            self.selected_point = None
            self.status_label.config(text=f"Deleted column {col_to_delete} ({points_deleted} points) - selection cleared")
        else:
            self.status_label.config(text=f"Deleted column {col_to_delete} ({points_deleted} points)")
        
        # Update display
        self.display_image()
        self.update_grid_info()
        
        self.log_debug("Column deleted", {"column": col_to_delete, "points_deleted": points_deleted})
    
    def delete_all_rows(self):
        """Delete all rows from the grid"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to delete from")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete ALL rows? This cannot be undone."):
            return
        
        # Save state before deletion
        self.save_grid_state()
        
        # Count points to be deleted
        points_deleted = len(self.grid_points)
        
        # Clear all grid points and extracted values
        self.grid_points.clear()
        self.extracted_values.clear()
        
        # Reset grid dimensions
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_rows_var.set("0")
        self.grid_cols_var.set("0")
        
        # CRITICAL: Clear selected point since all rows were deleted
        self.selected_point = None
        
        # Update display
        self.display_image()
        self.update_grid_info()
        
        self.status_label.config(text=f"Deleted all rows ({points_deleted} points)")
        self.log_debug("All rows deleted", {"points_deleted": points_deleted})
    
    def delete_all_columns(self):
        """Delete all columns from the grid"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to delete from")
            return
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", "Are you sure you want to delete ALL columns? This cannot be undone."):
            return
        
        # Save state before deletion
        self.save_grid_state()
        
        # Count points to be deleted
        points_deleted = len(self.grid_points)
        
        # Clear all grid points and extracted values
        self.grid_points.clear()
        self.extracted_values.clear()
        
        # Reset grid dimensions
        self.grid_rows = 0
        self.grid_cols = 0
        self.grid_rows_var.set("0")
        self.grid_cols_var.set("0")
        
        # CRITICAL: Clear selected point since all columns were deleted
        self.selected_point = None
        
        # Update display
        self.display_image()
        self.update_grid_info()
        
        self.status_label.config(text=f"Deleted all columns ({points_deleted} points)")
        self.log_debug("All columns deleted", {"points_deleted": points_deleted})
    
    def delete_selected_control_point(self):
        """Delete the row or column corresponding to the selected control point"""
        if not self.deletion_selected_point:
            self.status_label.config(text="No control point selected for deletion")
            return
        
        row, col = self.deletion_selected_point
        
        # Determine what to delete based on the control point
        if row == 0:  # Top row point - delete entire column
            self.delete_column_by_index(col)
        elif col == 0:  # Left column point - delete entire row
            self.delete_row_by_index(row)
        else:
            self.status_label.config(text="Selected point is not a control point")
            return
    
    def delete_row_by_index(self, row_index):
        """Delete a specific row by index"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to delete from")
            return
        
        if row_index >= self.grid_rows:
            messagebox.showwarning("Invalid Row", f"Row {row_index} is out of range")
            return
        
        # Save state before deletion
        self.save_grid_state()
        
        # Delete all points in the specified row
        points_deleted = 0
        for col in range(self.grid_cols):
            if (row_index, col) in self.grid_points:
                del self.grid_points[(row_index, col)]
                points_deleted += 1
            if (row_index, col) in self.extracted_values:
                del self.extracted_values[(row_index, col)]
            if (row_index, col) in self.grid_cell_values:
                del self.grid_cell_values[(row_index, col)]
        
        # Renumber all rows after the deleted row
        new_grid_points = {}
        new_extracted_values = {}
        new_grid_cell_values = {}
        
        for (row, col), coords in self.grid_points.items():
            if row > row_index:
                new_grid_points[(row - 1, col)] = coords
            else:
                new_grid_points[(row, col)] = coords
        
        for (row, col), value in self.extracted_values.items():
            if row > row_index:
                new_extracted_values[(row - 1, col)] = value
            else:
                new_extracted_values[(row, col)] = value
        
        for (row, col), value in self.grid_cell_values.items():
            if row > row_index:
                new_grid_cell_values[(row - 1, col)] = value
            else:
                new_grid_cell_values[(row, col)] = value
        
        self.grid_points = new_grid_points
        self.extracted_values = new_extracted_values
        self.grid_cell_values = new_grid_cell_values
        
        # Update grid dimensions
        self.grid_rows -= 1
        self.grid_rows_var.set(str(self.grid_rows))
        
        # Clear deletion selection
        self.deletion_selected_point = None
        
        # Update display
        self.display_image()
        self.update_grid_info()
        
        self.status_label.config(text=f"Deleted row {row_index} ({points_deleted} points)")
        self.log_debug("Row deleted by index", {"row": row_index, "points_deleted": points_deleted})
    
    def delete_column_by_index(self, col_index):
        """Delete a specific column by index"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to delete from")
            return
        
        if col_index >= self.grid_cols:
            messagebox.showwarning("Invalid Column", f"Column {col_index} is out of range")
            return
        
        # Save state before deletion
        self.save_grid_state()
        
        # Delete all points in the specified column
        points_deleted = 0
        for row in range(self.grid_rows):
            if (row, col_index) in self.grid_points:
                del self.grid_points[(row, col_index)]
                points_deleted += 1
            if (row, col_index) in self.extracted_values:
                del self.extracted_values[(row, col_index)]
            if (row, col_index) in self.grid_cell_values:
                del self.grid_cell_values[(row, col_index)]
        
        # Renumber all columns after the deleted column
        new_grid_points = {}
        new_extracted_values = {}
        new_grid_cell_values = {}
        
        for (row, col), coords in self.grid_points.items():
            if col > col_index:
                new_grid_points[(row, col - 1)] = coords
            else:
                new_grid_points[(row, col)] = coords
        
        for (row, col), value in self.extracted_values.items():
            if col > col_index:
                new_extracted_values[(row, col - 1)] = value
            else:
                new_extracted_values[(row, col)] = value
        
        for (row, col), value in self.grid_cell_values.items():
            if col > col_index:
                new_grid_cell_values[(row, col - 1)] = value
            else:
                new_grid_cell_values[(row, col)] = value
        
        self.grid_points = new_grid_points
        self.extracted_values = new_extracted_values
        self.grid_cell_values = new_grid_cell_values
        
        # Update grid dimensions
        self.grid_cols -= 1
        self.grid_cols_var.set(str(self.grid_cols))
        
        # Clear deletion selection
        self.deletion_selected_point = None
        
        # Update display
        self.display_image()
        self.update_grid_info()
        
        self.status_label.config(text=f"Deleted column {col_index} ({points_deleted} points)")
        self.log_debug("Column deleted by index", {"column": col_index, "points_deleted": points_deleted})
    
    def save_grid(self):
        """Save grid configuration to file"""
        if not self.grid_points:
            messagebox.showwarning("No Grid", "No grid to save")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                grid_config = {
                    'timestamp': datetime.now().isoformat(),
                    'image_file': self.current_file,
                    'image_size': self.image.size if self.image else None,
                    'grid_mode': self.grid_mode,
                    'grid_rows': self.grid_rows,
                    'grid_cols': self.grid_cols,
                    'grid_points': {f"{k[0]},{k[1]}": list(v) for k, v in self.grid_points.items()},
                    'extracted_values': {f"{k[0]},{k[1]}": v for k, v in self.extracted_values.items()},
                    'grid_cell_values': {f"{k[0]},{k[1]}": v for k, v in self.grid_cell_values.items()},
                    'parameters': {
                        'row_pitch': self.row_pitch,
                        'col_pitch': self.col_pitch,
                        'origin_y': self.origin_y,
                        'origin_x': self.origin_x,
                        'threshold': self.threshold,
                        'zoom_level': self.zoom_level
                    }
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(grid_config, f, indent=2)
                
                self.status_label.config(text=f"Saved grid to {os.path.basename(filename)}")
                self.log_debug("Grid saved", {"filename": filename, "points": len(self.grid_points)})
                
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save grid: {e}")
    
    def load_grid(self):
        """Load grid configuration from file"""
        filename = filedialog.askopenfilename(
            title="Load Grid Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            self.load_session_file(filename, auto_load=False)
    
    def update_threshold(self, value):
        """Update threshold value"""
        self.threshold = float(value)
        self.threshold_label.config(text=f"{self.threshold:.2f}")
    
    def extract_all_points(self):
        """Extract values for all grid points"""
        if not self.image or not self.grid_points:
            messagebox.showwarning("No Data", "No image or grid points to extract from")
            return
        
        self.extracted_values = {}
        method = self.extraction_method.get()
        
        for (row, col), (x, y) in self.grid_points.items():
            try:
                # Ensure coordinates are within image bounds
                x_int = int(max(0, min(x, self.image.width - 1)))
                y_int = int(max(0, min(y, self.image.height - 1)))
                
                pixel = self.image.getpixel((x_int, y_int))
                
                if method == "threshold":
                    if isinstance(pixel, int):
                        value = 1 if pixel > self.threshold * 255 else 0
                    else:
                        gray = sum(pixel[:3]) / 3
                        value = 1 if gray > self.threshold * 255 else 0
                elif method == "adaptive":
                    value = self.adaptive_threshold(x_int, y_int)
                elif method == "edge":
                    value = self.edge_detection(x_int, y_int)
                elif method == "manual":
                    continue  # Skip automatic extraction for manual mode
                else:
                    value = 0
                
                self.extracted_values[(row, col)] = value
                
            except Exception as e:
                self.log_debug("Extraction error", {"point": (row, col), "error": str(e)})
                self.extracted_values[(row, col)] = 0
        
        self.display_image()
        self.update_grid_info()
        self.status_label.config(text=f"Extracted {len(self.extracted_values)} values")
        self.log_debug("All points extracted", {
            "method": method,
            "threshold": self.threshold,
            "extracted": len(self.extracted_values)
        })
    
    def adaptive_threshold(self, x, y, window_size=15):
        """Adaptive threshold extraction"""
        half_window = window_size // 2
        x_start = max(0, x - half_window)
        x_end = min(self.image.width, x + half_window + 1)
        y_start = max(0, y - half_window)
        y_end = min(self.image.height, y + half_window + 1)
        
        total = 0
        count = 0
        
        for yi in range(y_start, y_end):
            for xi in range(x_start, x_end):
                pixel = self.image.getpixel((xi, yi))
                if isinstance(pixel, int):
                    total += pixel
                else:
                    total += sum(pixel[:3]) / 3
                count += 1
        
        local_mean = total / count if count > 0 else 128
        pixel = self.image.getpixel((x, y))
        
        if isinstance(pixel, int):
            gray = pixel
        else:
            gray = sum(pixel[:3]) / 3
            
        return 1 if gray > local_mean * self.threshold else 0
    
    def edge_detection(self, x, y):
        """Simple edge detection"""
        if x == 0 or x >= self.image.width - 1 or y == 0 or y >= self.image.height - 1:
            return 0
        
        # Get surrounding pixels
        pixels = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                pixel = self.image.getpixel((x + dx, y + dy))
                if isinstance(pixel, int):
                    pixels.append(pixel)
                else:
                    pixels.append(sum(pixel[:3]) / 3)
        
        # Sobel operators
        gx = (pixels[2] + 2*pixels[5] + pixels[8]) - (pixels[0] + 2*pixels[3] + pixels[6])
        gy = (pixels[6] + 2*pixels[7] + pixels[8]) - (pixels[0] + 2*pixels[1] + pixels[2])
        
        magnitude = (gx**2 + gy**2)**0.5
        return 1 if magnitude > self.threshold * 255 else 0
    
    def toggle_processed_view(self):
        """Toggle processed view"""
        self.show_processed = self.show_processed_var.get()
        self.display_image()
    
    def export_as_text(self):
        """Export extracted values as text"""
        if not self.extracted_values:
            messagebox.showwarning("No Data", "No extracted values to export")
            return
        
        # Create binary string in row-major order
        binary_string = ""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if (row, col) in self.extracted_values:
                    binary_string += str(self.extracted_values[(row, col)])
                else:
                    binary_string += "0"
        
        # Convert to ASCII
        text_output = ""
        for i in range(0, len(binary_string), 8):
            byte = binary_string[i:i+8]
            if len(byte) == 8:
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:
                    text_output += chr(char_code)
                else:
                    text_output += f"[{char_code:02x}]"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Binary: {binary_string}\n\n")
                    f.write(f"Text: {text_output}\n\n")
                    f.write(f"Grid: {self.grid_rows}x{self.grid_cols}\n")
                    f.write(f"Method: {self.extraction_method.get()}\n")
                    f.write(f"Threshold: {self.threshold}\n")
                
                self.status_label.config(text=f"Exported to {os.path.basename(filename)}")
                self.log_debug("Exported as text", {"filename": filename, "length": len(binary_string)})
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def export_as_binary(self):
        """Export as binary file"""
        if not self.extracted_values:
            messagebox.showwarning("No Data", "No extracted values to export")
            return
        
        # Create binary string
        binary_string = ""
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                if (row, col) in self.extracted_values:
                    binary_string += str(self.extracted_values[(row, col)])
                else:
                    binary_string += "0"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".bin",
            filetypes=[("Binary files", "*.bin"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                byte_data = bytearray()
                for i in range(0, len(binary_string), 8):
                    byte = binary_string[i:i+8]
                    if len(byte) == 8:
                        byte_data.append(int(byte, 2))
                
                with open(filename, 'wb') as f:
                    f.write(byte_data)
                
                self.status_label.config(text=f"Exported {len(byte_data)} bytes")
                self.log_debug("Exported as binary", {"filename": filename, "bytes": len(byte_data)})
                
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export: {e}")
    
    def log_debug(self, action, details=None):
        """Enhanced debug logging"""
        if not self.debug_log_enabled:
            return
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        log_entry = f"\n## {timestamp} - {action}\n"
        
        if details:
            if isinstance(details, dict):
                for key, value in details.items():
                    log_entry += f"- **{key}**: {value}\n"
            else:
                log_entry += f"- {details}\n"
        
        log_entry += "\n"
        
        try:
            with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"Debug log error: {e}")
    
    def toggle_debug_logging(self):
        """Toggle debug logging"""
        self.debug_log_enabled = self.debug_var.get()
        self.log_debug("Debug logging toggled", {"enabled": self.debug_log_enabled})
    
    def clear_debug_log(self):
        """Clear debug log"""
        try:
            with open(self.debug_log_file, 'w') as f:
                f.write(f"# Visual Manual Extraction Debug Log\n")
                f.write(f"# Cleared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            self.status_label.config(text="Debug log cleared")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear debug log: {e}")
    
    # CRITICAL: Manual extraction functions
    def add_bit(self, bit):
        """Add a bit to the current line and advance position - OPTIMIZED"""
        if len(self.current_line_bits) < 96:  # Limit to 96 bits (12 bytes)
            self.current_line_bits.append(bit)
            
            # Store the bit value in the CURRENT grid cell (where cursor currently is)
            self.grid_cell_values[(self.current_row, self.current_col)] = int(bit)
            
            # Clear the input field after adding the bit
            self.binary_input.set('')
            
            # Clear the bits display to prevent accumulation to the left of input
            self.current_bits_label.config(text="Bits: ")
            
            # Only update display if binary values are visible
            if self.show_binary_values:
                self.display_image()
            
            # Skip debug logging for performance
            # self.log_debug("Added bit", {"bit": bit, "position": (self.current_row, self.current_col, self.current_bit)})
    
    def add_bit_and_advance(self, bit):
        """Add a bit and advance to next position"""
        self.add_bit(bit)
        self.next_bit()
    
    def get_previous_position(self):
        """Get the previous position before the current one"""
        # Since we use one bit per cell, previous position is just previous column
        prev_col = self.current_col - 1
        prev_row = self.current_row
        
        if prev_col < 0:
            prev_col = self.grid_cols - 1
            prev_row -= 1
            if prev_row < 0:
                prev_row = self.grid_rows - 1
        
        # Keep track of bit position for line building
        prev_bit = self.current_bit - 1
        if prev_bit < 0:
            prev_bit = 7
        
        return prev_row, prev_col, prev_bit
    
    def delete_bit(self):
        """Delete the last bit from the current line and move cursor back"""
        if self.current_line_bits:
            # First move cursor back to the previous position
            self.prev_bit()
            
            # Remove the bit from the current line
            self.current_line_bits.pop()
            
            # Remove the value from the current grid cell (after moving back)
            if (self.current_row, self.current_col) in self.grid_cell_values:
                del self.grid_cell_values[(self.current_row, self.current_col)]
            
            # Clear the input field after deleting the bit
            self.binary_input.set('')
            
            # Only update display if binary values are visible
            if self.show_binary_values:
                self.display_image()
            
            # Skip debug logging for performance
            # self.log_debug("Deleted bit", {"remaining": len(self.current_line_bits), "position": (self.current_row, self.current_col)})
    
    def next_bit(self):
        """Move to the next bit position - OPTIMIZED"""
        # Move to next column for each bit (one bit per cell)
        self.current_col += 1
        if self.current_col >= self.grid_cols:
            self.current_col = 0
            self.current_row += 1
            if self.current_row >= self.grid_rows:
                self.current_row = 0
        
        # Keep track of bit position within the binary line
        self.current_bit += 1
        if self.current_bit >= 8:
            self.current_bit = 0
        
        self.update_position_info()
        # Only redraw if we have binary values to show, or if position indicator is visible
        if self.show_binary_values or self.position_indicator_visible:
            self.display_image()
    
    def prev_bit(self):
        """Move to the previous bit position - OPTIMIZED"""
        # Move to previous column for each bit (one bit per cell)
        self.current_col -= 1
        if self.current_col < 0:
            self.current_col = self.grid_cols - 1
            self.current_row -= 1
            if self.current_row < 0:
                self.current_row = self.grid_rows - 1
        
        # Keep track of bit position within the binary line
        self.current_bit -= 1
        if self.current_bit < 0:
            self.current_bit = 7
        
        self.update_position_info()
        # Only redraw if we have binary values to show, or if position indicator is visible
        if self.show_binary_values or self.position_indicator_visible:
            self.display_image()
    
    def next_row(self):
        """Move to the next row"""
        self.current_row += 1
        if self.current_row >= self.grid_rows:
            self.current_row = 0
        self.current_col = 0
        self.current_bit = 0
        
        self.update_position_info()
        self.display_image()
    
    def prev_row(self):
        """Move to the previous row"""
        self.current_row -= 1
        if self.current_row < 0:
            self.current_row = self.grid_rows - 1
        self.current_col = 0
        self.current_bit = 0
        
        self.update_position_info()
        self.display_image()
    
    def complete_line(self):
        """Complete the current line and add to extracted data"""
        if not self.current_line_bits:
            messagebox.showwarning("Warning", "No bits to complete")
            return
        
        # Pad to 96 bits if necessary
        while len(self.current_line_bits) < 96:
            self.current_line_bits.append('0')
        
        line = ''.join(self.current_line_bits)
        self.extracted_data.append(line)
        
        # Clear current line
        self.current_line_bits = []
        self.binary_input.set('')
        
        # Move to next row
        self.next_row()
        
        # Update display
        self.update_data_display()
        
        self.log_debug("Completed line", {"line": line, "total_lines": len(self.extracted_data)})
    
    def clear_line(self):
        """Clear the current line"""
        self.current_line_bits = []
        self.binary_input.set('')
        # Clear all grid cell values
        self.grid_cell_values.clear()
        self.display_image()
        self.log_debug("Cleared line")
    
    def reset_position(self):
        """Reset position to start"""
        self.current_row = 0
        self.current_col = 0
        self.current_bit = 0
        # Clear all grid cell values
        self.grid_cell_values.clear()
        self.update_position_info()
        self.display_image()
        self.log_debug("Reset position")
    
    def update_position_info(self):
        """Update position display"""
        self.position_label.config(text=f"Row: {self.current_row}, Col: {self.current_col}, Bit: {self.current_bit}")
        self.current_line_label.config(text=f"Current Line: {len(self.extracted_data) + 1}")
        # Keep bits display clear to avoid clutter next to input field
        self.current_bits_label.config(text="Bits: ")
    
    def update_data_display(self):
        """Update the data display (placeholder for now)"""
        # This would update a text widget showing extracted data
        # For now, just log the data
        self.log_debug("Data updated", {"total_lines": len(self.extracted_data)})
    
    def on_key_press(self, event):
        """Handle keyboard events for manual extraction"""
        # Debug: log all key events
        self.log_debug("Key pressed", {"char": event.char, "keysym": event.keysym, "keycode": event.keycode})
        
        if event.char in ['0', '1']:
            self.add_bit_and_advance(event.char)
        elif event.char == ' ':  # Space key - add 0 and progress
            self.add_bit_and_advance('0')
        elif event.keysym == 'BackSpace':
            self.delete_bit()
        elif event.keysym == 'Right':
            self.next_bit()
        elif event.keysym == 'Left':
            self.prev_bit()
        elif event.keysym == 'Up':
            self.prev_row()
        elif event.keysym == 'Down':
            self.next_row()
        elif event.keysym == 'Return':
            # Enter key - progress to next position (horizontal progression)
            self.next_bit()
            self.status_label.config(text=f"Enter pressed - moved to Row: {self.current_row}, Col: {self.current_col}, Bit: {self.current_bit}")
            self.log_debug("Enter key pressed - moved to next position")
        elif event.keysym == 'F1':
            # F1 key - complete current line and move to next row
            self.complete_line()
    
    def run(self):
        """Run the interface"""
        # Ensure root window has focus for keyboard events
        self.root.focus_set()
        self.root.mainloop()


if __name__ == "__main__":
    # CURSOR AI v4 - Complete implementation with all fork features
    app = VisualManualExtractionInterface()
    app.run()