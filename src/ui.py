import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import time
from src import config

class LivingCanvasUI:
    """
    Tkinter User Interface for The Living Canvas.
    Renders the main viewport, sidebar controls, and telemetry dashboard.
    """
    def __init__(self, root, stream_a, stream_b, engine, dashboard, cap):
        self.root = root
        self.stream_a = stream_a
        self.stream_b = stream_b
        self.engine = engine
        self.dashboard = dashboard
        self.cap = cap
        
        # UI State Variables
        self.show_diagnostics = tk.BooleanVar(value=False)
        self.active_style = tk.StringVar(value="Starry Night")
        self.particle_count = tk.IntVar(value=config.DEFAULT_PARTICLE_COUNT)
        self.alpha_val = tk.DoubleVar(value=config.DEFAULT_ALPHA)
        self.edge_influence_val = tk.DoubleVar(value=config.DEFAULT_EDGE_INFLUENCE)
        self.fade_rate_val = tk.DoubleVar(value=0.08)
        
        # Configure root window styles
        self.root.title("The Living Canvas - Interactive Generative Vector Art")
        self.root.geometry("1150x660")
        self.root.configure(bg="#121212")
        
        # Set dark theme styles
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure(".", background="#121212", foreground="#E0E0E0")
        self.style.configure("TFrame", background="#121212")
        self.style.configure("TLabel", background="#121212", foreground="#E0E0E0", font=("Segoe UI", 10))
        self.style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"), foreground="#FFFFFF")
        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#BB86FC")
        
        # Style sliders and buttons
        self.style.configure("Horizontal.TScale", background="#121212")
        self.style.configure("TButton", background="#333333", foreground="#FFFFFF", borderwidth=0, font=("Segoe UI", 9, "bold"))
        self.style.map("TButton", background=[("active", "#BB86FC"), ("pressed", "#3700B3")])
        self.style.configure("TCombobox", fieldbackground="#1e1e1e", background="#333333", foreground="#FFFFFF")
        
        # Create Layout Frames
        self._create_layout()

    def _create_layout(self):
        # 1. Sidebar Control Panel
        sidebar = ttk.Frame(self.root, padding=15, width=320, style="TFrame")
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)
        
        # App Title
        title_lbl = ttk.Label(sidebar, text="LIVING CANVAS", style="Title.TLabel")
        title_lbl.pack(anchor=tk.W, pady=(0, 20))
        
        # Section 1: Styles
        style_frame = ttk.LabelFrame(sidebar, text=" Artistic Style ", padding=10)
        style_frame.pack(fill=tk.X, pady=(0, 15))
        
        style_cb = ttk.Combobox(
            style_frame, 
            textvariable=self.active_style, 
            values=config.ALL_STYLES,
            state="readonly"
        )
        style_cb.pack(fill=tk.X, pady=5)
        style_cb.bind("<<ComboboxSelected>>", self._on_style_changed)
        
        # Section 2: Particle Controls
        controls_frame = ttk.LabelFrame(sidebar, text=" Brush Simulation ", padding=10)
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Brush Count Slider
        ttk.Label(controls_frame, text="Brush Density:").pack(anchor=tk.W)
        brush_slider = ttk.Scale(
            controls_frame, 
            from_=config.MIN_PARTICLE_COUNT, 
            to=config.MAX_PARTICLE_COUNT, 
            variable=self.particle_count,
            orient=tk.HORIZONTAL,
            command=self._on_particles_changed
        )
        brush_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Fluid Viscosity (Alpha) Slider
        ttk.Label(controls_frame, text="Viscosity (Motion Speed):").pack(anchor=tk.W)
        alpha_slider = ttk.Scale(
            controls_frame, 
            from_=config.MIN_ALPHA, 
            to=config.MAX_ALPHA, 
            variable=self.alpha_val,
            orient=tk.HORIZONTAL,
            command=self._on_alpha_changed
        )
        alpha_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Edge Alignment Slider
        ttk.Label(controls_frame, text="Edge Contour Alignment:").pack(anchor=tk.W)
        edge_slider = ttk.Scale(
            controls_frame, 
            from_=0.0, 
            to=1.0, 
            variable=self.edge_influence_val,
            orient=tk.HORIZONTAL,
            command=self._on_edge_changed
        )
        edge_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Paint Persistence (Fading Rate) Slider
        ttk.Label(controls_frame, text="Paint Fade Rate (Wetness):").pack(anchor=tk.W)
        fade_slider = ttk.Scale(
            controls_frame, 
            from_=0.01, 
            to=0.30, 
            variable=self.fade_rate_val,
            orient=tk.HORIZONTAL,
            command=self._on_fade_changed
        )
        fade_slider.pack(fill=tk.X, pady=(0, 10))
        
        # Section 3: Diagnostic Modes
        diag_frame = ttk.LabelFrame(sidebar, text=" View Mode ", padding=10)
        diag_frame.pack(fill=tk.X, pady=(0, 15))
        
        art_btn = ttk.Radiobutton(diag_frame, text="Artistic Canvas View", variable=self.show_diagnostics, value=False)
        art_btn.pack(anchor=tk.W, pady=5)
        
        diag_btn = ttk.Radiobutton(diag_frame, text="Engineering Diagnostics", variable=self.show_diagnostics, value=True)
        diag_btn.pack(anchor=tk.W, pady=5)
        
        # Reset Button
        reset_btn = ttk.Button(sidebar, text="CLEAR CANVAS", command=self._on_clear_canvas)
        reset_btn.pack(fill=tk.X, pady=(10, 5))
        
        # 2. Main Viewport (Right Panel)
        viewport_frame = ttk.Frame(self.root, padding=10, style="TFrame")
        viewport_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Canvas Viewport
        self.canvas_widget = tk.Canvas(
            viewport_frame, 
            width=config.RENDER_WIDTH, 
            height=config.RENDER_HEIGHT, 
            bg="#000000",
            highlightthickness=0
        )
        self.canvas_widget.pack(anchor=tk.CENTER, expand=True)
        
        # 3. Telemetry Footer
        self.telemetry_frame = ttk.Frame(viewport_frame, padding=5, style="TFrame")
        self.telemetry_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.fps_lbl = ttk.Label(self.telemetry_frame, text="FPS: --", padding=(10, 0))
        self.fps_lbl.pack(side=tk.LEFT)
        
        self.latency_a_lbl = ttk.Label(self.telemetry_frame, text="Stream A (AI): -- ms", padding=(10, 0))
        self.latency_a_lbl.pack(side=tk.LEFT)
        
        self.latency_b_lbl = ttk.Label(self.telemetry_frame, text="Stream B (CV): -- ms", padding=(10, 0))
        self.latency_b_lbl.pack(side=tk.LEFT)
        
        self.ram_lbl = ttk.Label(self.telemetry_frame, text="RAM: -- MB", padding=(10, 0))
        self.ram_lbl.pack(side=tk.LEFT)

    def _on_style_changed(self, event):
        style = self.active_style.get()
        # Thread-safely load the model in Stream A
        self.stream_a.set_style(style)

    def _on_particles_changed(self, event):
        self.engine.set_particle_count(self.particle_count.get())

    def _on_alpha_changed(self, event):
        self.engine.alpha = self.alpha_val.get()

    def _on_edge_changed(self, event):
        self.engine.edge_influence = self.edge_influence_val.get()

    def _on_fade_changed(self, event):
        self.engine.fade_rate = self.fade_rate_val.get()

    def _on_clear_canvas(self):
        self.engine.reseed_all()

    def update_frame(self, buffer_a, buffer_b):
        """Webcam capture loop running inside the Tkinter main loop thread."""
        ret, frame = self.cap.read()
        if ret:
            # Write camera frame into the input double-buffers of Stream A and Stream B
            buffer_a.write(frame)
            buffer_b.write(frame)

    def render_loop(self):
        """Periodic rendering and UI update scheduled on Tkinter event loop."""
        # 1. Read the processed matrices
        stylized_texture = self.stream_a.output_buffer.read()
        control_matrices = self.stream_b.output_buffer.read()
        
        render_img = None
        
        # 2. Check if we are running in Diagnostics or Artistic mode
        if self.show_diagnostics.get():
            # Composite Diagnostic view
            if control_matrices is not None:
                render_img = self.dashboard.get_composite_diagnostics(control_matrices)
        else:
            # Artistic simulation view
            if control_matrices is not None:
                self.engine.update(control_matrices, stylized_texture, self.active_style.get())
                render_img = self.engine.draw()
                
        # 3. Draw image onto Tkinter canvas
        if render_img is not None:
            # Convert BGR (OpenCV) to RGB (PIL)
            rgb_img = cv2.cvtColor(render_img, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(rgb_img)
            self.photo = PIL.ImageTk.PhotoImage(image=pil_img)
            self.canvas_widget.create_image(0, 0, image=self.photo, anchor=tk.NW)
            
        # 4. Update Telemetry Panel
        fps = self.dashboard.tick_fps()
        ram = self.dashboard.get_ram_usage_mb()
        
        self.fps_lbl.config(text=f"FPS: {fps:.1f}")
        self.latency_a_lbl.config(text=f"Stream A (AI): {self.stream_a.running_latency_ms:.1f} ms")
        self.latency_b_lbl.config(text=f"Stream B (CV): {self.stream_b.running_latency_ms:.1f} ms")
        self.ram_lbl.config(text=f"RAM: {ram:.1f} MB" if ram > 0.0 else "RAM: N/A")
        
        # Schedule next update (target ~60 updates/sec, or 16ms interval)
        self.root.after(16, self.render_loop)
