import threading
import time
import cv2
import numpy as np
from src import config
from src.utils import DoubleBuffer, FrameTimer

class StreamB(threading.Thread):
    """
    Geometric Processing Stream.
    Computes Optical Flow, Edges, and Saliency maps in a background thread.
    """
    def __init__(self, input_buffer: DoubleBuffer, output_buffer: DoubleBuffer):
        super().__init__(daemon=True, name="StreamB-GeometricCV")
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        
        self._stop_event = threading.Event()
        self.latency_timer = FrameTimer()
        self.running_latency_ms = 0.0
        
        # State variables for CV algorithms
        self.prev_gray = None
        
        # Saliency segmentation components (MediaPipe vs MOG2 Fallback)
        self.selfie_segmentation = None
        self.mog2_subtractor = None
        self.use_fallback = False

    def _init_segmentation(self):
        """Try to initialize MediaPipe segmentation; fall back to MOG2 if import/init fails."""
        try:
            import mediapipe as mp
            mp_selfie = mp.solutions.selfie_segmentation
            # model_selection=0 is general, model_selection=1 is landscape (faster/lower res)
            self.selfie_segmentation = mp_selfie.SelfieSegmentation(model_selection=0)
            print("StreamB: MediaPipe Selfie Segmentation initialized successfully.")
        except Exception as e:
            print(f"StreamB: MediaPipe failed to load ({e}). Using OpenCV MOG2 fallback.")
            self.use_fallback = True
            # history=500, varThreshold=16, detectShadows=False
            self.mog2_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500, varThreshold=24, detectShadows=False
            )

    def stop(self):
        self._stop_event.set()

    def run(self):
        # Initialize segmentation inside the background thread context to avoid thread affinity issues
        self._init_segmentation()
        
        while not self._stop_event.is_set():
            # Check for new input frame
            if not self.input_buffer.has_new():
                time.sleep(0.005)
                continue
                
            frame = self.input_buffer.get_latest()
            if frame is None:
                continue
                
            start_time = time.time()
            try:
                # Downsample frame for geometric processing
                small_frame = cv2.resize(frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
                curr_gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                
                # Pre-allocate previous frame if not set
                if self.prev_gray is None:
                    self.prev_gray = curr_gray.copy()
                    
                # 1. Optical Flow (Farneback)
                flow = cv2.calcOpticalFlowFarneback(
                    self.prev_gray,
                    curr_gray,
                    None,
                    pyr_scale=config.FLOW_PYR_SCALE,
                    levels=config.FLOW_LEVELS,
                    winsize=config.FLOW_WINSIZE,
                    iterations=config.FLOW_ITERATIONS,
                    poly_n=config.FLOW_POLY_N,
                    poly_sigma=config.FLOW_POLY_SIGMA,
                    flags=config.FLOW_FLAGS
                )
                
                # Update prev frame
                self.prev_gray = curr_gray.copy()
                
                # 2. Edge Gradient (Sobel)
                # Compute gradients Gx, Gy at low-res
                gx = cv2.Sobel(curr_gray, cv2.CV_32F, 1, 0, ksize=3)
                gy = cv2.Sobel(curr_gray, cv2.CV_32F, 0, 1, ksize=3)
                
                # 3. Saliency Segmentation
                saliency_mask = None
                if not self.use_fallback and self.selfie_segmentation is not None:
                    try:
                        # Convert to RGB for MediaPipe
                        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                        results = self.selfie_segmentation.process(rgb_small)
                        if results.segmentation_mask is not None:
                            saliency_mask = results.segmentation_mask
                    except Exception as e:
                        print(f"StreamB: MediaPipe runtime error: {e}. Falling back to MOG2.")
                        self.use_fallback = True
                        self.mog2_subtractor = cv2.createBackgroundSubtractorMOG2(
                            history=500, varThreshold=24, detectShadows=False
                        )
                
                if self.use_fallback:
                    # MOG2 background subtractor on the small frame
                    fg_mask = self.mog2_subtractor.apply(small_frame)
                    
                    # Apply morphological opening to clean noise
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
                    
                    # Normalize mask to float [0.0, 1.0]
                    saliency_mask = fg_mask.astype(np.float32) / 255.0
                    
                if saliency_mask is None:
                    # Absolute fallback: uniform mask
                    saliency_mask = np.zeros((config.PROCESSING_HEIGHT, config.PROCESSING_WIDTH), dtype=np.float32)
                
                # 4. Upscale control matrices to Render Resolution
                # Resize Flow field and scale the vectors accordingly
                flow_resized = cv2.resize(flow, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
                scale_x = config.RENDER_WIDTH / config.PROCESSING_WIDTH
                scale_y = config.RENDER_HEIGHT / config.PROCESSING_HEIGHT
                flow_resized[:, :, 0] *= scale_x
                flow_resized[:, :, 1] *= scale_y
                
                # Resize Sobel components and compute magnitude and direction at high-res
                gx_resized = cv2.resize(gx, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
                gy_resized = cv2.resize(gy, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
                
                edge_magnitude = np.sqrt(gx_resized**2 + gy_resized**2)
                # Normalize edge magnitude to [0, 1] range
                max_mag = np.max(edge_magnitude)
                if max_mag > 0:
                    edge_magnitude /= max_mag
                
                edge_angle = np.arctan2(gy_resized, gx_resized)
                
                # Resize Saliency mask
                saliency_resized = cv2.resize(saliency_mask, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
                # Smooth out the mask slightly
                saliency_resized = cv2.GaussianBlur(saliency_resized, (5, 5), 0)
                
                # Resize original frame to Render Resolution as a fallback texture
                color_resized = cv2.resize(frame, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
                
                # Pack the results
                control_matrices = {
                    "flow": flow_resized,
                    "edge_magnitude": edge_magnitude,
                    "edge_angle": edge_angle,
                    "saliency": saliency_resized,
                    "gray": cv2.resize(curr_gray, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR),
                    "color": color_resized
                }
                
                # Record performance metrics
                latency = (time.time() - start_time) * 1000.0
                self.latency_timer.record_latency(latency)
                self.running_latency_ms = self.latency_timer.get_average_latency()
                
                # Write to double buffer
                self.output_buffer.write(control_matrices)
                
            except Exception as e:
                print(f"StreamB CV error: {e}")
                time.sleep(0.01)
                
            time.sleep(0.001)
