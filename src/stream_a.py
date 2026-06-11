import threading
import time
import cv2
import numpy as np
from src import config
from src import model_manager
from src.utils import DoubleBuffer, FrameTimer

class StreamA(threading.Thread):
    """
    Semantic Processing Stream.
    Runs ONNX fast style transfer models on downsampled webcam frames in a separate thread.
    """
    def __init__(self, input_buffer: DoubleBuffer, output_buffer: DoubleBuffer):
        super().__init__(daemon=True, name="StreamA-SemanticAI")
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        
        self.current_style = "Starry Night"
        self.session = None
        self.session_lock = threading.Lock()
        
        self._stop_event = threading.Event()
        self.latency_timer = FrameTimer()
        self.running_latency_ms = 0.0

    def set_style(self, style_name):
        """Update the style model being run, thread-safely reloading the session."""
        if style_name not in config.ALL_STYLES:
            print(f"StreamA: Style {style_name} not available.")
            return False
            
        with self.session_lock:
            try:
                print(f"StreamA: Switching style to '{style_name}'...")
                if style_name in config.STYLE_MODELS:
                    self.session = model_manager.load_style_model(style_name)
                else:
                    self.session = None  # Custom CV style, no ONNX session needed
                self.current_style = style_name
                print(f"StreamA: Style '{style_name}' successfully loaded.")
                return True
            except Exception as e:
                print(f"StreamA: Error switching style: {e}")
                return False

    def stop(self):
        self._stop_event.set()

    def run(self):
        # Initial model loading
        self.set_style(self.current_style)
        
        last_frame_time = time.time()
        
        while not self._stop_event.is_set():
            # Check for new input frame
            if not self.input_buffer.has_new():
                # Avoid spinning, wait for a new frame
                time.sleep(0.005)
                continue
                
            frame = self.input_buffer.get_latest()
            if frame is None:
                continue
                
            # Perform Inference under session lock (in case of style changes)
            with self.session_lock:
                if self.session is None and self.current_style in config.STYLE_MODELS:
                    time.sleep(0.01)
                    continue
                
                start_time = time.time()
                try:
                    if self.session is not None:
                        # 1. Preprocess
                        # Convert BGR (OpenCV default) to RGB
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Query model input dimensions dynamically (expected format is [batch, channels, height, width])
                        input_shape = self.session.get_inputs()[0].shape
                        model_h = input_shape[2] if (len(input_shape) > 2 and isinstance(input_shape[2], int)) else 224
                        model_w = input_shape[3] if (len(input_shape) > 3 and isinstance(input_shape[3], int)) else 224
                        
                        # Resize to model expected input size (OpenCV takes (width, height))
                        resized = cv2.resize(rgb_frame, (model_w, model_h))
                        # Convert to float32 [H, W, C] in range [0, 255]
                        in_data = resized.astype(np.float32)
                        # Transpose to channel-first [C, H, W] and expand batch size -> [1, C, H, W]
                        in_tensor = np.transpose(in_data, (2, 0, 1))
                        in_tensor = np.expand_dims(in_tensor, axis=0)
                        
                        # 2. ONNX Run
                        model_inputs = {self.session.get_inputs()[0].name: in_tensor}
                        model_outputs = self.session.run(None, model_inputs)
                        out_tensor = model_outputs[0]
                        
                        # 3. Postprocess
                        # Output shape is [1, 3, H, W]
                        out_tensor = np.squeeze(out_tensor, axis=0)
                        out_tensor = np.transpose(out_tensor, (1, 2, 0))
                        # Clip values to [0, 255] and cast to uint8
                        out_tensor = np.clip(out_tensor, 0, 255).astype(np.uint8)
                        # Convert back to BGR for OpenCV
                        stylized_bgr = cv2.cvtColor(out_tensor, cv2.COLOR_RGB2BGR)
                        
                        # Upscale back to Render Resolution for smooth color sampling
                        stylized_texture = cv2.resize(
                            stylized_bgr, 
                            (config.RENDER_WIDTH, config.RENDER_HEIGHT), 
                            interpolation=cv2.INTER_LINEAR
                        )
                    else:
                        # 1. Custom CV Style
                        stylized_texture = self.apply_cv_style(frame, self.current_style)
                    
                    # Record performance metrics
                    latency = (time.time() - start_time) * 1000.0
                    self.latency_timer.record_latency(latency)
                    self.running_latency_ms = self.latency_timer.get_average_latency()
                    
                    # Write to double buffer
                    self.output_buffer.write(stylized_texture)
                    
                except Exception as e:
                    print(f"StreamA inference error: {e}")
                    time.sleep(0.03) # Cooldown on failure
                    
            # Yield control to prevent CPU starvation
            time.sleep(0.001)

    def apply_cv_style(self, frame, style_name):
        """Apply a classic computer vision style filter to the frame."""
        if style_name == "Cartoon":
            # Downsample for performance
            small = cv2.resize(frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
            color = small.copy()
            for _ in range(2):
                color = cv2.bilateralFilter(color, d=9, sigmaColor=75, sigmaSpace=75)
            
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.medianBlur(gray, 5)
            edges = cv2.adaptiveThreshold(
                gray, 255, 
                cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, 
                blockSize=9, C=9
            )
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            cartoon = cv2.bitwise_and(color, edges_bgr)
            return cv2.resize(cartoon, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
        elif style_name == "Oil Pastel Painting":
            small = cv2.resize(frame, (config.PROCESSING_WIDTH, config.PROCESSING_HEIGHT))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            pastel = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
            pastel = cv2.morphologyEx(pastel, cv2.MORPH_OPEN, kernel)
            pastel = cv2.bilateralFilter(pastel, d=9, sigmaColor=120, sigmaSpace=100)
            
            # Increase color saturation slightly
            hsv = cv2.cvtColor(pastel, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.3
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            pastel = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            return cv2.resize(pastel, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
            
        else:
            # For Line Drawing or ASCII Character Vision, return resized original frame for color/luminance sampling
            return cv2.resize(frame, (config.RENDER_WIDTH, config.RENDER_HEIGHT), interpolation=cv2.INTER_LINEAR)
