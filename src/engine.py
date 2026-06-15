import numpy as np
import cv2
import random
from src import config

class VectorRenderingEngine:
    """
    Vector Rendering Engine.
    Simulates a pool of virtual brushstrokes (particles) and renders them onto
    a canvas, incorporating color, motion vectors, edges, and saliency.
    """
    def __init__(self):
        self.max_particles = config.MAX_PARTICLE_COUNT
        self.num_particles = config.DEFAULT_PARTICLE_COUNT
        
        # Pre-allocate particle property arrays
        self.x = np.zeros(self.max_particles, dtype=np.float32)
        self.y = np.zeros(self.max_particles, dtype=np.float32)
        self.length = np.zeros(self.max_particles, dtype=np.float32)
        self.angle = np.zeros(self.max_particles, dtype=np.float32)
        self.color = np.zeros((self.max_particles, 3), dtype=np.uint8)
        self.thickness = np.zeros(self.max_particles, dtype=np.float32)
        self.age = np.zeros(self.max_particles, dtype=np.int32)
        self.ttl = np.zeros(self.max_particles, dtype=np.int32)
        
        # Accumulator canvas
        self.canvas = np.ones((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8) * 255
        self.white_background = np.ones((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8) * 255
        # Dark background for styles like The Scream
        self.dark_background = np.ones((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8) * 15
        
        # Simulation parameters (controlled via UI sliders)
        self.alpha = config.DEFAULT_ALPHA
        self.edge_influence = config.DEFAULT_EDGE_INFLUENCE
        self.fade_rate = 0.08  # Accumulation fade rate for paint persistence
        
        # Pre-compute saliency gradient maps for attraction force
        self._saliency_gx = None
        self._saliency_gy = None
        
        # Initialize particles randomly across the screen
        self.reseed_all()

    def set_particle_count(self, count):
        """Change the active particle count."""
        old_count = self.num_particles
        self.num_particles = int(np.clip(count, config.MIN_PARTICLE_COUNT, self.max_particles))
        if self.num_particles > old_count:
            # Reseed the newly added particles
            self._reseed_indices(np.arange(old_count, self.num_particles))

    def reseed_all(self):
        """Randomly initialize all particles across the canvas."""
        indices = np.arange(self.max_particles)
        self.x[indices] = np.random.uniform(0, config.RENDER_WIDTH - 1, size=self.max_particles)
        self.y[indices] = np.random.uniform(0, config.RENDER_HEIGHT - 1, size=self.max_particles)
        self.angle[indices] = np.random.uniform(0, 2 * np.pi, size=self.max_particles)
        self.length[indices] = config.BRUSH_COARSE_LENGTH
        self.thickness[indices] = config.BRUSH_COARSE_SIZE
        self.color[indices] = [128, 128, 128]
        self.age[indices] = 0
        self.ttl[indices] = np.random.randint(config.PARTICLE_MIN_TTL, config.PARTICLE_MAX_TTL, size=self.max_particles)
        # Clear canvas
        self.canvas.fill(255)

    def _reseed_indices(self, indices, saliency_mask=None):
        """Reseed specific particles, optionally using saliency tournament selection."""
        n = len(indices)
        if n == 0:
            return
            
        if saliency_mask is not None:
            # Saliency-biased tournament selection: generate two candidate coordinates, keep the one with higher saliency
            cand1_x = np.random.uniform(0, config.RENDER_WIDTH - 1, size=n).astype(np.float32)
            cand1_y = np.random.uniform(0, config.RENDER_HEIGHT - 1, size=n).astype(np.float32)
            cand2_x = np.random.uniform(0, config.RENDER_WIDTH - 1, size=n).astype(np.float32)
            cand2_y = np.random.uniform(0, config.RENDER_HEIGHT - 1, size=n).astype(np.float32)
            
            c1_y_idx = np.clip(cand1_y.astype(np.int32), 0, config.RENDER_HEIGHT - 1)
            c1_x_idx = np.clip(cand1_x.astype(np.int32), 0, config.RENDER_WIDTH - 1)
            c2_y_idx = np.clip(cand2_y.astype(np.int32), 0, config.RENDER_HEIGHT - 1)
            c2_x_idx = np.clip(cand2_x.astype(np.int32), 0, config.RENDER_WIDTH - 1)
            
            sal1 = saliency_mask[c1_y_idx, c1_x_idx]
            sal2 = saliency_mask[c2_y_idx, c2_x_idx]
            
            better = sal1 >= sal2
            self.x[indices] = np.where(better, cand1_x, cand2_x)
            self.y[indices] = np.where(better, cand1_y, cand2_y)
        else:
            self.x[indices] = np.random.uniform(0, config.RENDER_WIDTH - 1, size=n)
            self.y[indices] = np.random.uniform(0, config.RENDER_HEIGHT - 1, size=n)
            
        self.angle[indices] = np.random.uniform(0, 2 * np.pi, size=n)
        self.age[indices] = 0
        self.ttl[indices] = np.random.randint(config.PARTICLE_MIN_TTL, config.PARTICLE_MAX_TTL, size=n)

    def update(self, control_matrices, stylized_texture, style_name="Starry Night"):
        """
        Update particle positions, orientation, sizing, and colors based on 
        the fusion of Stream A (texture) and Stream B (CV controls).
        """
        self.current_style = style_name
        self.last_control_matrices = control_matrices
        self.last_stylized_texture = stylized_texture
        
        # Read matrices
        flow = control_matrices["flow"]
        edge_mag = control_matrices["edge_magnitude"]
        edge_angle = control_matrices["edge_angle"]
        saliency = control_matrices["saliency"]
        
        # Define the subset of active particles
        active = np.arange(self.num_particles)
        
        # 1. Coordinate Clipping for Indexing (pre-advection)
        x_idx = np.clip(self.x[active].astype(np.int32), 0, config.RENDER_WIDTH - 1)
        y_idx = np.clip(self.y[active].astype(np.int32), 0, config.RENDER_HEIGHT - 1)
        
        # 2. Viscous Advection (Optical Flow) with magnitude clamping
        u = flow[y_idx, x_idx, 0]
        v = flow[y_idx, x_idx, 1]
        
        # Clamp flow magnitude to prevent particle teleportation
        flow_mag = np.sqrt(u**2 + v**2) + 1e-6
        max_displacement = 5.0  # Max pixels per frame
        scale = np.minimum(1.0, max_displacement / flow_mag)
        u_clamped = u * scale
        v_clamped = v * scale
        
        self.x[active] += self.alpha * u_clamped
        self.y[active] += self.alpha * v_clamped
        
        # 3. Saliency Attraction Force
        # Compute saliency gradient to nudge particles toward high-saliency regions
        if self._saliency_gx is None or self._saliency_gx.shape != saliency.shape:
            self._saliency_gx = np.zeros_like(saliency)
            self._saliency_gy = np.zeros_like(saliency)
        
        # Compute gradient of saliency map (points toward increasing saliency)
        self._saliency_gx = cv2.Sobel(saliency, cv2.CV_32F, 1, 0, ksize=5)
        self._saliency_gy = cv2.Sobel(saliency, cv2.CV_32F, 0, 1, ksize=5)
        
        # Re-index after advection for attraction sampling
        x_post = np.clip(self.x[active].astype(np.int32), 0, config.RENDER_WIDTH - 1)
        y_post = np.clip(self.y[active].astype(np.int32), 0, config.RENDER_HEIGHT - 1)
        
        # Apply gentle attraction toward salient regions (strength = 0.3)
        attraction_strength = 0.3
        sgx = self._saliency_gx[y_post, x_post]
        sgy = self._saliency_gy[y_post, x_post]
        self.x[active] += attraction_strength * sgx
        self.y[active] += attraction_strength * sgy
        
        # 4. Re-clip coordinates after all position updates
        x_idx = np.clip(self.x[active].astype(np.int32), 0, config.RENDER_WIDTH - 1)
        y_idx = np.clip(self.y[active].astype(np.int32), 0, config.RENDER_HEIGHT - 1)
        
        # 5. Edge Alignment
        # Orientation perpendicular to gradient (Sobel angle + pi/2)
        target_angle = edge_angle[y_idx, x_idx] + np.pi/2
        mag = edge_mag[y_idx, x_idx]
        
        # Robust angular interpolation handling wrap-around
        d_theta = (target_angle - self.angle[active] + np.pi) % (2 * np.pi) - np.pi
        self.angle[active] += self.edge_influence * mag * d_theta
        
        # 6. Saliency-Based Brush Sizing
        sal = saliency[y_idx, x_idx]
        is_foreground = sal >= config.SALIENCY_THRESHOLD
        
        # Update thickness and length based on saliency mask and style
        if self.current_style == "Oil Pastel Painting":
            # Broad blocky brush strokes
            self.thickness[active] = np.where(is_foreground, config.BRUSH_FINE_SIZE * 2.0, config.BRUSH_COARSE_SIZE * 2.2)
            self.length[active] = np.where(is_foreground, config.BRUSH_FINE_LENGTH * 1.5, config.BRUSH_COARSE_LENGTH * 1.5)
        elif self.current_style == "Line Drawing":
            # Thin sketch lines
            self.thickness[active] = np.where(is_foreground, 1.0, 1.5)
            self.length[active] = np.where(is_foreground, 6.0, 15.0)
        elif self.current_style == "Starry Night":
            # Tighter impasto strokes — finer in foreground for detail preservation
            self.thickness[active] = np.where(is_foreground, 3, 8)
            self.length[active] = np.where(is_foreground, 7, 16)
        elif self.current_style == "The Scream":
            # Long flowing lines
            self.thickness[active] = np.where(is_foreground, 3, 6)
            self.length[active] = np.where(is_foreground, 18, 35)
        elif self.current_style == "Modernist Geometric":
            # Geometric shapes
            self.thickness[active] = np.where(is_foreground, 6, 16)
            self.length[active] = np.where(is_foreground, 10, 24)
        else:
            self.thickness[active] = np.where(is_foreground, config.BRUSH_FINE_SIZE, config.BRUSH_COARSE_SIZE)
            self.length[active] = np.where(is_foreground, config.BRUSH_FINE_LENGTH, config.BRUSH_COARSE_LENGTH)
        
        # Add a small random jitter to brush thickness and length for natural looks
        self.thickness[active] += np.random.uniform(-1, 1, size=self.num_particles)
        if self.current_style == "Oil Pastel Painting":
            self.thickness[active] = np.clip(self.thickness[active], 4, 45)
        elif self.current_style == "Line Drawing":
            self.thickness[active] = np.clip(self.thickness[active], 0.5, 3)
        elif self.current_style == "Starry Night":
            self.thickness[active] = np.clip(self.thickness[active], 2, 14)
        elif self.current_style == "The Scream":
            self.thickness[active] = np.clip(self.thickness[active], 1, 10)
        else:
            self.thickness[active] = np.clip(self.thickness[active], 1, 24)
            
        # 7. Color Sampling — AFTER advection so particles carry the color of where they ARE
        if self.current_style == "Line Drawing":
            # Vectorized: ink/graphite color with edge-based thickness
            self.color[active] = [30, 30, 30]
            mag_vals = edge_mag[y_idx, x_idx]
            low_edge = mag_vals < 0.15
            self.thickness[active] = np.where(low_edge, 0, 1.0 + mag_vals * 1.5)
        else:
            if stylized_texture is not None:
                # Sample RGB color from the stylized Stream A texture at CURRENT position
                self.color[active] = stylized_texture[y_idx, x_idx]
            elif "color" in control_matrices:
                # Fallback to the original camera feed colors (resized)
                self.color[active] = control_matrices["color"][y_idx, x_idx]
            else:
                # Solid gray fallback
                self.color[active] = [128, 128, 128]
                
            # Style-specific color palette enhancements
            if self.current_style == "Starry Night":
                colors = self.color[active].astype(np.float32)
                colors[:, 0] = np.clip(colors[:, 0] * 1.35, 0, 255)  # Boost blue
                # Check for yellow pixels and boost them
                is_yellow = (colors[:, 2] > 90) & (colors[:, 1] > 90) & (colors[:, 0] < 130)
                colors[is_yellow, 2] = np.clip(colors[is_yellow, 2] * 1.4, 0, 255)
                colors[is_yellow, 1] = np.clip(colors[is_yellow, 1] * 1.4, 0, 255)
                self.color[active] = colors.astype(np.uint8)
                
            elif self.current_style == "The Scream":
                colors = self.color[active].astype(np.float32)
                # Boost fiery orange/red colors
                colors[:, 2] = np.clip(colors[:, 2] * 1.45, 0, 255)  # Boost red
                is_warm = colors[:, 2] > 70
                colors[is_warm, 1] = np.clip(colors[is_warm, 1] * 1.2, 0, 255)  # Boost green slightly for orange
                self.color[active] = colors.astype(np.uint8)
                
            elif self.current_style == "Modernist Geometric":
                colors = self.color[active].astype(np.float32)
                # High contrast saturation boost
                mean_color = np.mean(colors, axis=1, keepdims=True)
                colors = np.where(colors > mean_color, np.clip(colors * 1.3, 0, 255), colors * 0.7)
                self.color[active] = colors.astype(np.uint8)
            
        # 8. Age Increment & Reseeding
        self.age[active] += 1
        
        # Find dead particles: either aged out or moved out of bounds
        out_of_bounds = (
            (self.x[active] < 0) | (self.x[active] >= config.RENDER_WIDTH - 1) |
            (self.y[active] < 0) | (self.y[active] >= config.RENDER_HEIGHT - 1)
        )
        aged_out = self.age[active] >= self.ttl[active]
        dead_mask = out_of_bounds | aged_out
        dead_indices = active[dead_mask]
        
        # Reseed dead particles using saliency-biased seeding
        self._reseed_indices(dead_indices, saliency)
        
    def draw(self):
        """Render virtual brushstrokes onto the canvas accumulator using batched drawing."""
        style = self.current_style if hasattr(self, "current_style") else "Starry Night"
        
        # 1. Handle ASCII Character Vision separately (no particle drawing)
        if style == "ASCII Character Vision":
            ascii_canvas = np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8)
            
            # Select source image
            if hasattr(self, "last_stylized_texture") and self.last_stylized_texture is not None:
                img_grid = self.last_stylized_texture
            elif hasattr(self, "last_control_matrices") and self.last_control_matrices is not None and "color" in self.last_control_matrices:
                img_grid = self.last_control_matrices["color"]
            else:
                return ascii_canvas
                
            char_w = 8
            char_h = 10
            grid_w = config.RENDER_WIDTH // char_w
            grid_h = config.RENDER_HEIGHT // char_h
            
            img_small = cv2.resize(img_grid, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
            gray_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
            
            ascii_chars = " .:-=+*#%@"
            num_chars = len(ascii_chars)
            
            for y in range(grid_h):
                for x in range(grid_w):
                    brightness = gray_small[y, x]
                    char_idx = int(brightness / 255.0 * (num_chars - 1))
                    char = ascii_chars[char_idx]
                    color = tuple(int(c) for c in img_small[y, x])
                    pos_x = x * char_w
                    pos_y = y * char_h + char_h - 2
                    
                    cv2.putText(
                        ascii_canvas, char, (pos_x, pos_y),
                        cv2.FONT_HERSHEY_PLAIN, 0.7,
                        color, 1, cv2.LINE_AA
                    )
            return ascii_canvas

        # 2. Handle Cartoon style — render the filtered image directly (no particles)
        if style == "Cartoon":
            if hasattr(self, "last_stylized_texture") and self.last_stylized_texture is not None:
                cartoon_frame = self.last_stylized_texture.copy()
                
                # Enhance edges with thick black outlines for that classic cartoon look
                if hasattr(self, "last_control_matrices") and self.last_control_matrices is not None:
                    edge_mag = self.last_control_matrices["edge_magnitude"]
                    # Create strong black edge overlay
                    edge_mask = (edge_mag > 0.25).astype(np.uint8) * 255
                    # Dilate edges slightly for bolder outlines
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                    edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)
                    # Darken edges on the cartoon frame
                    edge_3ch = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
                    cartoon_frame = np.where(edge_3ch > 0, 
                                           np.clip(cartoon_frame.astype(np.int16) - 180, 0, 255).astype(np.uint8),
                                           cartoon_frame)
                
                return cartoon_frame
            else:
                return self.canvas

        # 3. Fade canvas — use dark background for intense styles, white for others
        fade_rate = self.fade_rate
        if style == "Line Drawing":
            fade_rate = 0.02  # Slower fade so pencil sketch details accumulate beautifully
        elif style == "The Scream":
            fade_rate = 0.04  # Slower fade for moody paint accumulation
        
        if style == "The Scream":
            # Fade toward dark canvas for The Scream's moody atmosphere
            self.canvas = cv2.addWeighted(self.canvas, 1.0 - fade_rate, self.dark_background, fade_rate, 0)
        else:
            self.canvas = cv2.addWeighted(self.canvas, 1.0 - fade_rate, self.white_background, fade_rate, 0)
        
        # 3. Vectorized math for stroke endpoints (FPS Optimization)
        num_p = self.num_particles
        xs = self.x[:num_p]
        ys = self.y[:num_p]
        angles = self.angle[:num_p]
        lengths = self.length[:num_p]
        thicknesses = self.thickness[:num_p]
        colors = self.color[:num_p]
        
        # Precompute endpoints in vector math
        x0s = xs.astype(np.int32)
        y0s = ys.astype(np.int32)
        cos_vals = np.cos(angles)
        sin_vals = np.sin(angles)
        
        x1s = np.clip((xs + lengths * cos_vals).astype(np.int32), 0, config.RENDER_WIDTH - 1)
        y1s = np.clip((ys + lengths * sin_vals).astype(np.int32), 0, config.RENDER_HEIGHT - 1)
        
        # 4. Draw brush strokes — use batched drawing for simple styles, per-particle for complex
        style = self.current_style if hasattr(self, "current_style") else "Starry Night"
        
        # Filter out zero-thickness particles
        valid_mask = thicknesses > 0
        if not np.any(valid_mask):
            return self.canvas
        
        if style in ("Starry Night", "The Scream", "Modernist Geometric", "Oil Pastel Painting"):
            # Complex styles need per-particle drawing but we skip invalid particles early
            valid_indices = np.where(valid_mask)[0]
            self._draw_complex_style(style, valid_indices, x0s, y0s, x1s, y1s, 
                                      cos_vals, sin_vals, lengths, thicknesses, colors)
        else:
            # Simple line styles (Line Drawing, default) — use batched color-bucketed drawing
            self._draw_batched(valid_mask, x0s, y0s, x1s, y1s, thicknesses, colors, style)
        
        return self.canvas

    def _draw_batched(self, valid_mask, x0s, y0s, x1s, y1s, thicknesses, colors, style):
        """Batch draw particles by quantizing colors into buckets for cv2.polylines."""
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return
            
        # Quantize colors to reduce unique values (shift right 4 bits = 16 levels per channel)
        quantized = colors[valid_idx] >> 4
        
        # Group by (quantized_color, thickness_int) tuple
        thick_int = thicknesses[valid_idx].astype(np.int32)
        thick_int = np.clip(thick_int, 1, 30)
        
        # Build a hash key for grouping: encode as single int
        keys = (quantized[:, 0].astype(np.int64) * 17 * 17 * 31 + 
                quantized[:, 1].astype(np.int64) * 17 * 31 + 
                quantized[:, 2].astype(np.int64) * 31 + 
                thick_int.astype(np.int64))
        
        unique_keys = np.unique(keys)
        
        for key in unique_keys:
            mask = keys == key
            batch_idx = valid_idx[mask]
            
            if len(batch_idx) == 0:
                continue
            
            # Get representative color (mean of the group, or just first element)
            rep_color = tuple(int(c) for c in colors[batch_idx[0]])
            rep_thick = int(thicknesses[batch_idx[0]])
            if rep_thick <= 0:
                rep_thick = 1
            
            # Build line segments as polylines: each segment is [pt0, pt1]
            # For disconnected segments we draw each as a 2-point polyline
            segments = []
            for i in batch_idx:
                segments.append(np.array([[x0s[i], y0s[i]], [x1s[i], y1s[i]]], dtype=np.int32))
            
            if style == "Line Drawing":
                cv2.polylines(self.canvas, segments, isClosed=False, color=(30, 30, 30), 
                             thickness=rep_thick, lineType=cv2.LINE_AA)
            else:
                cv2.polylines(self.canvas, segments, isClosed=False, color=rep_color, 
                             thickness=rep_thick, lineType=cv2.LINE_AA)

    def _draw_complex_style(self, style, valid_indices, x0s, y0s, x1s, y1s, 
                            cos_vals, sin_vals, lengths, thicknesses, colors):
        """Per-particle drawing for styles that require complex geometry."""
        for i in valid_indices:
            thickness = int(thicknesses[i])
            if thickness <= 0:
                continue
                
            color = tuple(int(c) for c in colors[i])  # BGR order
            
            if style == "Starry Night":
                # Draw thick, curved impasto swirl stroke (Vincent van Gogh style)
                curve_amp = lengths[i] * 0.22
                xm = int(x0s[i] + 0.5 * lengths[i] * cos_vals[i] - curve_amp * sin_vals[i])
                ym = int(y0s[i] + 0.5 * lengths[i] * sin_vals[i] + curve_amp * cos_vals[i])
                cv2.line(self.canvas, (x0s[i], y0s[i]), (xm, ym), color, thickness, lineType=cv2.LINE_AA)
                cv2.line(self.canvas, (xm, ym), (x1s[i], y1s[i]), color, thickness, lineType=cv2.LINE_AA)
                
            elif style == "The Scream":
                # Draw wavy flowing line (Edvard Munch style) — more steps for smoother waves
                num_steps = 5
                prev_pt = (x0s[i], y0s[i])
                for step in range(1, num_steps + 1):
                    t = step / num_steps
                    xl = x0s[i] + t * lengths[i] * cos_vals[i]
                    yl = y0s[i] + t * lengths[i] * sin_vals[i]
                    wave_offset = 5.0 * np.sin(t * np.pi * 2.0)
                    xp = int(xl - wave_offset * sin_vals[i])
                    yp = int(yl + wave_offset * cos_vals[i])
                    
                    cv2.line(self.canvas, prev_pt, (xp, yp), color, thickness, lineType=cv2.LINE_AA)
                    prev_pt = (xp, yp)
                    
            elif style == "Modernist Geometric":
                # Draw clean geometric shapes (Bauhaus/Kandinsky style)
                half_l = max(4, lengths[i] / 2)
                if i % 3 == 0:
                    # Rotated rectangle
                    half_w = max(2, thickness / 1.5)
                    c = cos_vals[i]
                    s = sin_vals[i]
                    p1 = (int(x0s[i] - half_l * c - half_w * s), int(y0s[i] - half_l * s + half_w * c))
                    p2 = (int(x0s[i] + half_l * c - half_w * s), int(y0s[i] + half_l * s + half_w * c))
                    p3 = (int(x0s[i] + half_l * c + half_w * s), int(y0s[i] + half_l * s - half_w * c))
                    p4 = (int(x0s[i] - half_l * c + half_w * s), int(y0s[i] - half_l * s - half_w * c))
                    pts = np.array([p1, p2, p3, p4], dtype=np.int32)
                    cv2.fillPoly(self.canvas, [pts], color)
                elif i % 3 == 1:
                    # Solid circle
                    cv2.circle(self.canvas, (x0s[i], y0s[i]), int(half_l * 0.8), color, -1, lineType=cv2.LINE_AA)
                else:
                    # Solid triangle
                    half_w = max(3, thickness)
                    p1 = (x1s[i], y1s[i])
                    p2 = (int(x0s[i] - half_w * sin_vals[i]), int(y0s[i] + half_w * cos_vals[i]))
                    p3 = (int(x0s[i] + half_w * sin_vals[i]), int(y0s[i] - half_w * cos_vals[i]))
                    pts = np.array([p1, p2, p3], dtype=np.int32)
                    cv2.fillPoly(self.canvas, [pts], color)
                    
            elif style == "Cartoon":
                # Cartoon is handled by direct rendering above, this is a fallback
                cv2.line(self.canvas, (x0s[i], y0s[i]), (x1s[i], y1s[i]), color, thickness, lineType=cv2.LINE_AA)
                
            elif style == "Oil Pastel Painting":
                # Thick, waxy overlapping strokes with circular caps and texture offsets
                cv2.line(self.canvas, (x0s[i], y0s[i]), (x1s[i], y1s[i]), color, thickness, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, (x0s[i], y0s[i]), thickness // 2, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(self.canvas, (x1s[i], y1s[i]), thickness // 2, color, -1, lineType=cv2.LINE_AA)
                
                # Textured pastel edge bleed
                offset_x = int(np.random.randint(-2, 3))
                offset_y = int(np.random.randint(-2, 3))
                cv2.line(self.canvas, (x0s[i] + offset_x, y0s[i] + offset_y), (x1s[i] + offset_x, y1s[i] + offset_y), color, max(1, thickness - 2), lineType=cv2.LINE_AA)
