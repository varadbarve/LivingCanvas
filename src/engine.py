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
        
        # Simulation parameters (controlled via UI sliders)
        self.alpha = config.DEFAULT_ALPHA
        self.edge_influence = config.DEFAULT_EDGE_INFLUENCE
        self.fade_rate = 0.08  # Accumulation fade rate for paint persistence
        
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

    def update(self, control_matrices, stylized_texture):
        """
        Update particle positions, orientation, sizing, and colors based on 
        the fusion of Stream A (texture) and Stream B (CV controls).
        """
        # Read matrices
        flow = control_matrices["flow"]
        edge_mag = control_matrices["edge_magnitude"]
        edge_angle = control_matrices["edge_angle"]
        saliency = control_matrices["saliency"]
        
        # Define the subset of active particles
        active = np.arange(self.num_particles)
        
        # 1. Coordinate Clipping for Indexing
        x_idx = np.clip(self.x[active].astype(np.int32), 0, config.RENDER_WIDTH - 1)
        y_idx = np.clip(self.y[active].astype(np.int32), 0, config.RENDER_HEIGHT - 1)
        
        # 2. Viscous Advection (Optical Flow)
        u = flow[y_idx, x_idx, 0]
        v = flow[y_idx, x_idx, 1]
        self.x[active] += self.alpha * u
        self.y[active] += self.alpha * v
        
        # 3. Edge Alignment
        # Orientation perpendicular to gradient (Sobel angle + pi/2)
        target_angle = edge_angle[y_idx, x_idx] + np.pi/2
        mag = edge_mag[y_idx, x_idx]
        
        # Robust angular interpolation handling wrap-around
        d_theta = (target_angle - self.angle[active] + np.pi) % (2 * np.pi) - np.pi
        self.angle[active] += self.edge_influence * mag * d_theta
        
        # 4. Saliency-Based Brush Sizing
        sal = saliency[y_idx, x_idx]
        is_foreground = sal >= config.SALIENCY_THRESHOLD
        
        # Update thickness and length based on saliency mask
        self.thickness[active] = np.where(is_foreground, config.BRUSH_FINE_SIZE, config.BRUSH_COARSE_SIZE)
        self.length[active] = np.where(is_foreground, config.BRUSH_FINE_LENGTH, config.BRUSH_COARSE_LENGTH)
        
        # Add a small random jitter to brush thickness and length for natural looks
        self.thickness[active] += np.random.uniform(-1, 1, size=self.num_particles)
        self.thickness[active] = np.clip(self.thickness[active], 1, 30)
        
        # 5. Color Sampling
        if stylized_texture is not None:
            # Sample RGB color from the stylized Stream A texture
            self.color[active] = stylized_texture[y_idx, x_idx]
        elif "color" in control_matrices:
            # Fallback to the original camera feed colors (resized)
            self.color[active] = control_matrices["color"][y_idx, x_idx]
        else:
            # Solid gray fallback
            self.color[active] = [128, 128, 128]
            
        # 6. Age Increment & Reseeding
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
        """Render virtual brushstrokes onto the canvas accumulator."""
        # 1. Fade canvas slightly towards white to create wet paint smearing
        self.canvas = cv2.addWeighted(self.canvas, 1.0 - self.fade_rate, self.white_background, self.fade_rate, 0)
        
        # 2. Draw brush strokes (lines) for active particles
        # Iterate in Python - cv2.line in C++ is fast enough for ~1500 particles
        for i in range(self.num_particles):
            x0 = int(self.x[i])
            y0 = int(self.y[i])
            
            # Calculate end coordinates based on stroke orientation and length
            theta = self.angle[i]
            l = self.length[i]
            x1 = int(x0 + l * np.cos(theta))
            y1 = int(y0 + l * np.sin(theta))
            
            # Clip drawing endpoints to canvas size
            x1 = np.clip(x1, 0, config.RENDER_WIDTH - 1)
            y1 = np.clip(y1, 0, config.RENDER_HEIGHT - 1)
            
            # Draw the line
            color = tuple(int(c) for c in self.color[i])  # BGR order
            thickness = max(1, int(self.thickness[i]))
            
            cv2.line(self.canvas, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)
            
        return self.canvas
