import time
import os
import cv2
import numpy as np
from src import config

# Try to import psutil for telemetry memory usage, fallback if missing
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

class DiagnosticsDashboard:
    """
    Dashboard utility to compile real-time telemetry metrics and 
    render diagnostic visualizations (optical flow, edges, saliency).
    """
    def __init__(self):
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.current_fps = 0.0
        
        # Keep track of active process for RAM usage
        if HAS_PSUTIL:
            self.process = psutil.Process(os.getpid())
        else:
            self.process = None

    def tick_fps(self):
        """Update and calculate running FPS."""
        self.fps_frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.current_fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = time.time()
        return self.current_fps

    def get_ram_usage_mb(self):
        """Get RSS memory footprint of the current Python process in MB."""
        if HAS_PSUTIL and self.process is not None:
            try:
                return self.process.memory_info().rss / (1024 * 1024)
            except Exception:
                return 0.0
        return 0.0

    @staticmethod
    def visualize_optical_flow(flow):
        """Render dense optical flow field into a colorized BGR image using HSV mapping."""
        h, w = flow.shape[:2]
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 1] = 255
        
        # Convert Cartesian coordinates to Polar (magnitude and angle in radians)
        u, v = flow[..., 0], flow[..., 1]
        mag, ang = cv2.cartToPolar(u, v)
        
        # Map angle to HSV hue [0, 180]
        hsv[..., 0] = ang * 180 / np.pi / 2
        # Normalize and map magnitude to HSV value [0, 255]
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert HSV representation back to BGR
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return flow_bgr

    @staticmethod
    def visualize_edges(edge_magnitude):
        """Render Sobel edge magnitudes as a high-contrast grayscale BGR image."""
        # edge_magnitude is normalized [0.0, 1.0]
        edge_uint8 = (edge_magnitude * 255).astype(np.uint8)
        
        # Apply a subtle color map for premium feel, e.g., cv2.COLORMAP_JET or just clean grayscale
        edge_bgr = cv2.cvtColor(edge_uint8, cv2.COLOR_GRAY2BGR)
        return edge_bgr

    @staticmethod
    def visualize_saliency(saliency_mask, frame):
        """Overlay the saliency mask onto the original camera feed for visual comparison."""
        # saliency_mask is [0.0, 1.0]
        mask_uint8 = (saliency_mask * 255).astype(np.uint8)
        
        # Colorize the mask: make foreground reddish/magenta
        mask_colored = np.zeros_like(frame)
        mask_colored[:, :] = [180, 0, 180]  # Magenta overlay
        
        # Alpha blend mask with the grayscale/dimmed frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        
        # Perform blending where saliency is high
        alpha = np.expand_dims(saliency_mask, axis=2)
        blended = (gray_bgr * (1.0 - 0.5 * alpha) + mask_colored * (0.5 * alpha)).astype(np.uint8)
        
        return blended

    def get_composite_diagnostics(self, control_matrices):
        """Compile a single 2x2 grid containing the source frame and three diagnostic views."""
        # Check if control matrices exist
        if control_matrices is None:
            blank = np.zeros((config.RENDER_HEIGHT, config.RENDER_WIDTH, 3), dtype=np.uint8)
            return blank
            
        gray_frame = control_matrices["gray"]
        gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR) if len(gray_frame.shape) == 2 else gray_frame
        
        flow_view = self.visualize_optical_flow(control_matrices["flow"])
        edge_view = self.visualize_edges(control_matrices["edge_magnitude"])
        saliency_view = self.visualize_saliency(control_matrices["saliency"], gray_bgr)
        
        # Resize all views to half of Render Resolution
        half_w = config.RENDER_WIDTH // 2
        half_h = config.RENDER_HEIGHT // 2
        
        top_left = cv2.resize(gray_bgr, (half_w, half_h))
        top_right = cv2.resize(flow_view, (half_w, half_h))
        bottom_left = cv2.resize(edge_view, (half_w, half_h))
        bottom_right = cv2.resize(saliency_view, (half_w, half_h))
        
        # Add titles overlay on each viewport
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255)
        thickness = 1
        
        cv2.putText(top_left, "Source (Grayscale)", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(top_right, "Dense Optical Flow (HSV)", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(bottom_left, "Sobel Edge Direction", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(bottom_right, "Saliency Segment (MediaPipe)", (10, 20), font, font_scale, color, thickness, cv2.LINE_AA)
        
        # Assemble 2x2 grid
        top_row = np.hstack((top_left, top_right))
        bottom_row = np.hstack((bottom_left, bottom_right))
        grid = np.vstack((top_row, bottom_row))
        
        # Draw divider lines
        cv2.line(grid, (half_w, 0), (half_w, config.RENDER_HEIGHT), (50, 50, 50), 2)
        cv2.line(grid, (0, half_h), (config.RENDER_WIDTH, half_h), (50, 50, 50), 2)
        
        return grid
