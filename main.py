import tkinter as tk
import cv2
import numpy as np
import sys
import os
import threading

# Add workspace directory to sys.path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src import config
from src import model_manager
from src.utils import DoubleBuffer
from src.stream_a import StreamA
from src.stream_b import StreamB
from src.engine import VectorRenderingEngine
from src.dashboard import DiagnosticsDashboard
from src.ui import LivingCanvasUI

class MockVideoCapture:
    """Fallback mock camera capture when no physical webcam is available."""
    def __init__(self):
        self.width = config.CAMERA_WIDTH
        self.height = config.CAMERA_HEIGHT
        self.frame_idx = 0
        
        # Pre-generate a textured backdrop
        self.backdrop = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Vertical gradient
        for y in range(self.height):
            self.backdrop[y, :, 0] = int(y / self.height * 180) # B
            self.backdrop[y, :, 1] = 100                        # G
            self.backdrop[y, :, 2] = int((1 - y / self.height) * 120) # R

    def read(self):
        # Create a frame copying the backdrop
        frame = self.backdrop.copy()
        
        # Draw moving shapes to simulate optical flow and saliency
        self.frame_idx += 1
        
        # Circle 1 (foreground person simulation)
        cx1 = int(self.width / 2 + 150 * np.cos(self.frame_idx * 0.03))
        cy1 = int(self.height / 2 + 100 * np.sin(self.frame_idx * 0.02))
        cv2.circle(frame, (cx1, cy1), 90, (100, 230, 100), -1)
        
        # Circle 2
        cx2 = int(self.width / 2 + 100 * np.sin(self.frame_idx * 0.04))
        cy2 = int(self.height / 2 + 80 * np.cos(self.frame_idx * 0.05))
        cv2.circle(frame, (cx2, cy2), 60, (230, 100, 100), -1)
        
        # Add soft moving text overlay
        cv2.putText(
            frame, 
            "DEMO MODE (No Camera)", 
            (30, self.height - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2, 
            cv2.LINE_AA
        )
        
        # Introduce a slight noise to stimulate edges
        noise = np.random.normal(0, 3, frame.shape).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        return True, frame

    def isOpened(self):
        return True

    def release(self):
        pass


def pre_download_models():
    """Trigger background downloading of all configured style transfer models."""
    print("Checking / downloading pre-trained ONNX style models...")
    for style in config.STYLE_MODELS.keys():
        try:
            model_manager.get_model_path(style)
        except Exception as e:
            print(f"Error downloading '{style}' style: {e}")
            print("Please ensure you have an active internet connection on the first run.")

def main():
    # 1. Download/Verify models
    # This block ensures models are downloaded before booting the UI
    pre_download_models()

    # 2. Initialize Video Capture
    print("Initializing Video Capture...")
    cap = cv2.VideoCapture(0)
    
    # Configure capture parameters
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
    
    # Check if camera opened correctly
    if not cap.isOpened():
        print("WARNING: Could not open webcam. Initializing Mock Camera Feed.")
        cap = MockVideoCapture()
    else:
        # Dry run read
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Camera failed to capture first frame. Using Mock Camera Feed.")
            cap = MockVideoCapture()

    # 3. Create Shared Double-Buffers
    camera_buffer_a = DoubleBuffer()
    camera_buffer_b = DoubleBuffer()
    
    stylized_texture_buffer = DoubleBuffer()
    control_matrices_buffer = DoubleBuffer()

    # 4. Instantiate Workers and Engines
    stream_a = StreamA(camera_buffer_a, stylized_texture_buffer)
    stream_b = StreamB(camera_buffer_b, control_matrices_buffer)
    
    engine = VectorRenderingEngine()
    dashboard = DiagnosticsDashboard()

    # 5. Boot UI
    root = tk.Tk()
    ui = LivingCanvasUI(root, stream_a, stream_b, engine, dashboard, cap)

    # 6. Start Processing Threads
    stream_a.start()
    stream_b.start()

    # 7. Orchestrate Webcam Thread Frame Grabbing
    def webcam_loop():
        ui.update_frame(camera_buffer_a, camera_buffer_b)
        root.after(15, webcam_loop) # Grabbing frame at ~60fps target

    # Start loop timers
    webcam_loop()
    ui.render_loop()

    # 8. Define Graceful Shutdown Sequence
    def on_closing():
        print("Shutting down Living Canvas pipeline...")
        # Signal stop events to worker threads
        stream_a.stop()
        stream_b.stop()
        
        # Close camera capture
        cap.release()
        
        # Give threads a moment to finish gracefully
        stream_a.join(timeout=1.0)
        stream_b.join(timeout=1.0)
        
        root.destroy()
        print("Pipeline shut down successfully.")
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run Tkinter Mainloop
    root.mainloop()

if __name__ == "__main__":
    main()
