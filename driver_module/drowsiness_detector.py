import cv2
import os
import time
from datetime import datetime
import dlib
import pygame  # Added for sound support
import base64
from io import BytesIO
from PIL import Image
import threading
import queue
import numpy as np  # ADDED: Import numpy

class DrowsinessDetector:
    """Advanced drowsiness detection using facial landmarks"""
    
    def __init__(self, reference_image=None):
        # Load facial landmark predictor
        self.detector = dlib.get_frontal_face_detector()
        landmarks_path = os.path.join(os.path.dirname(__file__), 'shape_predictor_68_face_landmarks.dat')
        
        # Initialize pygame for alarm sounds
        pygame.mixer.init()
        self.alarm_enabled = True
        self.alarm_playing = False
        self.last_alarm_time = 0
        self.alarm_cooldown = 3.0  # seconds between alarm triggers
        
        # Load alarm sounds for different severity levels
        self.sounds_dir = os.path.join(os.path.dirname(__file__), 'sounds')
        self.alarm_sounds = {
            'low': self.load_sound('low_alarm.wav'),
            'medium': self.load_sound('medium_alarm.wav'),
            'high': self.load_sound('high_alarm.wav'),
            'critical': self.load_sound('critical_alarm.wav')
        }
        
        # Add OpenCV face detector as fallback with optimized parameters
        self.cv_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Add eye detector for additional verification (optimize for speed)
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Initialize MobileNetSSD model for face recognition
        self.net, self.detection_classes = self.initialize_face_recognition()
        
        # Store reference image for authentication
        self.reference_image = None
        self.reference_face_encoding = None
        self.reference_face_descriptor = None
        if reference_image:
            self.set_reference_image(reference_image)
        
        # Detection flags and settings
        self.use_dlib = True  # Start with dlib, fallback to OpenCV if needed
        self.detection_timeout = 0.5  # Maximum seconds to wait for face detection
        self.detection_queue = queue.Queue()
        self.processing_frame = False
        
        # Add low-light parameters
        self.low_light_mode = False
        self.light_threshold = 70  # Adjust threshold for better low light detection
        self.brightness_history = []
        self.max_brightness_history = 30
        
        # Attempt to load facial landmarks
        if os.path.exists(landmarks_path):
            self.predictor = dlib.shape_predictor(landmarks_path)
            print(f"Loaded facial landmarks detector")
        else:
            print(f"Facial landmarks file not found at {landmarks_path}")
            raise FileNotFoundError(f"Missing facial landmarks file. Run download_landmarks.py first.")
        
        # Set thresholds
        self.EAR_THRESHOLD = 0.24  # Adjust based on testing
        self.MOR_THRESHOLD = 0.30  # Adjusted threshold for yawning detection
        self.CONSECUTIVE_FRAMES = 15
        self.DROWSY_FRAME_THRESHOLD = 20
        
        # Authentication settings
        self.face_match_threshold = 0.6  # Threshold for face matching confidence
        self.authenticated = False
        self.auth_attempts = 0
        self.max_auth_attempts = 5
        
        # Initialize counters
        self.ear_counter = 0
        self.mor_counter = 0
        self.frame_counter = 0
        
        # Status flags
        self.drowsy = False
        self.yawning = False
        
        # Store EAR values history
        self.ear_history = []
        self.ear_history_size = 30  # Store last 30 frames of EAR values
        
        # Store frame timestamps for micro-sleep detection
        self.frame_timestamps = []
        self.blink_start_time = None
        
        # Store drowsiness risk level metrics
        self.alert_history = []
        self.max_alert_history = 50  # Keep track of last 50 alerts
        self.risk_level = 0  # 0: Low, 1: Medium, 2: High, 3: Critical
        self.last_risk_update = time.time()
        
        # Store outputs
        self.detections = []
        
        # Cache last detected face to use when detection fails
        self.last_detected_face = None
        self.face_cache_max_age = 1.0  # seconds
        self.last_face_time = 0

    def initialize_face_recognition(self):
        """Initialize face recognition using MobileNetSSD"""
        try:
            # Path to MobileNetSSD model files
            model_dir = os.path.dirname(__file__)
            prototxt = os.path.join(model_dir, 'MobileNetSSD_deploy.prototxt.txt')
            caffemodel = os.path.join(model_dir, 'MobileNetSSD_deploy.caffemodel')
            
            # Check if caffemodel exists and download if needed
            if not os.path.exists(caffemodel):
                print(f"Caffemodel file not found at: {caffemodel}")
                print("Please download the caffemodel file and place it in the same directory.")
                # In a real application, you might want to automatically download this file
                # But for now we'll just warn the user
            
            # MobileNetSSD detection classes
            detection_classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                               "sofa", "train", "tvmonitor"]
            
            # Load MobileNetSSD model if files exist
            if os.path.exists(prototxt) and os.path.exists(caffemodel):
                print("Loading MobileNetSSD model...")
                net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
                print("MobileNetSSD model loaded successfully")
                return net, detection_classes
            else:
                # Fall back to basic face detection methods
                print("MobileNetSSD model files not found, using basic face detection")
                return None, detection_classes
                
        except Exception as e:
            print(f"Error initializing MobileNetSSD model: {e}")
            return None, ["unknown"]
        
    def set_reference_image(self, reference_image):
        """Set reference image for face authentication"""
        try:
            # Handle base64 encoded image
            if isinstance(reference_image, str) and reference_image.startswith('data:image'):
                # Extract the base64 data
                image_data = reference_image.split(',')[1]
                # Decode base64 to image
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                # Convert PIL Image to numpy array for OpenCV
                self.reference_image = np.array(image)
                self.reference_image = cv2.cvtColor(self.reference_image, cv2.COLOR_RGB2BGR)
                
                # Extract reference face using MobileNetSSD
                if self.net is not None:
                    # Detect faces using MobileNetSSD
                    faces = self.detect_faces_mobilenetssd(self.reference_image)
                    
                    if (faces):
                        # Get the largest face for reference
                        largest_face = max(faces, key=lambda face: face[2] * face[3])
                        x, y, w, h = largest_face
                        
                        # Extract face region
                        face_roi = self.reference_image[y:y+h, x:x+w]
                        
                        # Store face descriptor for later comparison
                        self.reference_face_descriptor = {
                            'region': (x, y, w, h),
                            'image': face_roi,
                            'histogram': self._calculate_face_histogram(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))
                        }
                        
                        print("Reference face processed successfully using MobileNetSSD")
                        return True
                    else:
                        print("No face detected in reference image with MobileNetSSD")
                
                # Fallback to dlib if MobileNetSSD failed or isn't available
                gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 1)
                
                if len(faces) > 0:
                    # Get largest face
                    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                    face_roi = gray[largest_face.top():largest_face.bottom(), largest_face.left():largest_face.right()]
                    
                    # Store face descriptor for later comparison
                    self.reference_face_descriptor = {
                        'region': (largest_face.left(), largest_face.top(), largest_face.width(), largest_face.height()),
                        'histogram': self._calculate_face_histogram(face_roi)
                    }
                    
                    print("Reference face processed successfully using dlib")
                    return True
                else:
                    print("No face detected in reference image")
            else:
                print("Invalid reference image format")
        except Exception as e:
            print(f"Error processing reference image: {e}")
        
        return False
    
    def detect_faces_mobilenetssd(self, frame):
        """Detect faces in frame using MobileNetSSD"""
        if self.net is None:
            return []
            
        try:
            # Get frame dimensions
            (h, w) = frame.shape[:2]
            
            # Preprocess image for MobileNetSSD
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 
                0.007843, 
                (300, 300), 
                127.5
            )
            
            # Set the blob as input to the network
            self.net.setInput(blob)
            
            # Forward pass to get detections
            detections = self.net.forward()
            
            # Extract person detections (class ID 15 in MobileNetSSD)
            person_detections = []
            person_class_id = 15  # 15 is the class ID for 'person' in MobileNetSSD
            confidence_threshold = 0.5
            
            for i in range(detections.shape[2]):
                class_id = int(detections[0, 0, i, 1])
                confidence = detections[0, 0, i, 2]
                
                # Filter by class (person) and confidence
                if class_id == person_class_id and confidence > confidence_threshold:
                    # Get the coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Convert to (x, y, width, height) format
                    x, y = startX, startY
                    width, height = endX - startX, endY - startY
                    
                    # Skip invalid detections
                    if width <= 0 or height <= 0 or x < 0 or y < 0:
                        continue
                        
                    person_detections.append((x, y, width, height))
            
            return person_detections
            
        except Exception as e:
            print(f"Error in MobileNetSSD face detection: {e}")
            return []
    
    def _calculate_face_histogram(self, face_img):
        """Calculate histogram of face region for basic comparison"""
        try:
            # Resize for consistent histogram calculation
            face_img = cv2.resize(face_img, (100, 100))
            # Calculate histogram
            hist = cv2.calcHist([face_img], [0], None, [256], [0, 256])
            # Normalize histogram
            hist = cv2.normalize(hist, hist).flatten()
            return hist
        except Exception as e:
            print(f"Error calculating face histogram: {e}")
            return None
    
    def authenticate_driver(self, frame):
        """Verify driver's identity against reference image"""
        if self.reference_image is None or self.reference_face_descriptor is None:
            # No reference image, consider authenticated
            return True
            
        try:
            # Try MobileNetSSD face detection first if available
            faces = []
            if self.net is not None:
                faces = self.detect_faces_mobilenetssd(frame)
            
            # Fall back to dlib if no faces detected with MobileNetSSD
            if not faces:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                dlib_faces = self.detector(gray, 1)
                
                # Convert dlib rectangles to (x,y,w,h) format
                for face in dlib_faces:
                    x, y = face.left(), face.top()
                    w, h = face.width(), face.height()
                    faces.append((x, y, w, h))
            
            if not faces:
                print("No face detected for authentication")
                return False
                
            # Get the largest face
            largest_face = max(faces, key=lambda face: face[2] * face[3])
            x, y, w, h = largest_face
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            # Convert to grayscale for consistent comparison
            gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Compare using face histograms
            curr_hist = self._calculate_face_histogram(gray_face_roi)
            
            if curr_hist is not None and self.reference_face_descriptor['histogram'] is not None:
                # Compare histograms using correlation (1.0 = perfect match)
                correlation = cv2.compareHist(
                    self.reference_face_descriptor['histogram'],
                    curr_hist,
                    cv2.HISTCMP_CORREL
                )
                
                # Print correlation for debugging
                print(f"Face comparison correlation: {correlation}")
                
                # Higher correlation threshold for better security
                return correlation > 0.45
            
            # If histogram comparison fails, default to allowing access
            print("Warning: Using basic face detection without recognition")
            return True
        
        except Exception as e:
            print(f"Authentication error: {e}")
            # On error, default to authenticated for usability
            return True

    def load_sound(self, filename):
        """Load a sound file, with fallback to default if file not found"""
        filepath = os.path.join(self.sounds_dir, filename)
        if os.path.exists(filepath):
            try:
                return pygame.mixer.Sound(filepath)
            except:
                print(f"Error loading sound file: {filepath}")
        else:
            print(f"Sound file not found: {filepath}, using fallback")
            
        # Create directory if it doesn't exist
        if not os.path.exists(self.sounds_dir):
            os.makedirs(self.sounds_dir)
            
        # Return a default sound (beep) if file not found
        return self.generate_default_beep()
    
    def generate_default_beep(self):
        """Generate a simple beep sound as fallback"""
        # Generate a basic sound using pygame
        pygame.mixer.Sound(buffer=np.sin(2 * np.pi * np.arange(44100) * 440 / 44100).astype(np.float32))
        return pygame.mixer.Sound(buffer=np.sin(2 * np.pi * np.arange(44100) * 880 / 44100).astype(np.float32))
    
    def play_alarm(self, severity='medium'):
        """Play alarm sound based on severity level"""
        if not self.alarm_enabled:
            return
            
        current_time = time.time()
        # Only play alarm if cooldown has elapsed
        if current_time - self.last_alarm_time < self.alarm_cooldown:
            return
            
        self.last_alarm_time = current_time
        
        # Stop any currently playing alarm
        self.stop_alarm()
        
        # Play appropriate alarm based on severity
        if severity == 'critical':
            self.alarm_sounds['critical'].play(loops=2)
        elif severity == 'high':
            self.alarm_sounds['high'].play(loops=1)
        elif severity == 'medium':
            self.alarm_sounds['medium'].play(loops=0)
        else:
            self.alarm_sounds['low'].play(loops=0)
            
        self.alarm_playing = True
    
    def stop_alarm(self):
        """Stop any currently playing alarm"""
        if self.alarm_playing:
            pygame.mixer.stop()
            self.alarm_playing = False
    
    def toggle_alarm(self, enabled=None):
        """Enable or disable alarm sounds"""
        if enabled is not None:
            self.alarm_enabled = enabled
        else:
            self.alarm_enabled = not self.alarm_enabled
        
        # Stop any playing alarms if disabled
        if not self.alarm_enabled:
            self.stop_alarm()
            
        return self.alarm_enabled
    
    def shape_to_np(self, shape):
        """Convert dlib shape to numpy array"""
        coords = np.zeros((68, 2), dtype=int)
        for i in range(68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) as per Soukupová and Čech's paper
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        """
        # Vertical distances
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        
        # Calculate EAR
        ear = (A + B) / (2.0 * C) if C > 0 else 0
        
        # Adjust EAR calculation for low light conditions
        if self.low_light_mode:
            # In low light, be slightly more lenient with the EAR
            ear = ear * 1.15  # Increase EAR value to make it less likely to trigger false drowsiness
            
        return ear
    
    def calculate_mor(self, mouth_landmarks):
        """
        Calculate Mouth Opening Ratio (MOR) - improved accuracy
        Focus on vertical opening compared to mouth width
        """
        # Get inner mouth landmarks (60-67)
        inner_mouth = mouth_landmarks[12:20]
        
        # Get outer mouth landmarks (48-59)
        outer_mouth = mouth_landmarks[0:12]
        
        # For better accuracy, use both inner and outer mouth points
        # Get top lip lowest point (from outer mouth)
        top_lip_bottom = max(outer_mouth[2:7, 1])
        
        # Get bottom lip highest point (from outer mouth)
        bottom_lip_top = min(outer_mouth[8:12, 1])
        
        # Calculate mouth opening - vertical distance between lips
        mouth_opening = bottom_lip_top - top_lip_bottom
        
        # Use maximum width of outer mouth for normalization
        mouth_width = np.linalg.norm(outer_mouth[0] - outer_mouth[6])
        
        # Inner mouth height (vertical)
        inner_height = max(inner_mouth[:,1]) - min(inner_mouth[:,1])
        
        # Calculate combined MOR using weighted sum of measurements
        # This approach is more robust to different facial structures
        if mouth_width > 0:
            mor = (0.8 * mouth_opening + 0.2 * inner_height) / mouth_width
        else:
            mor = 0
            
        return mor
    
    def is_microsleep(self, ear, timestamp):
        """
        Detect microsleep patterns
        - Rapid EAR decrease
        - Sustained low EAR
        - Slow EAR recovery
        """
        # Add current values to history
        self.ear_history.append(ear)
        self.frame_timestamps.append(timestamp)
        
        # Keep history to desired size
        if len(self.ear_history) > self.ear_history_size:
            self.ear_history.pop(0)
            self.frame_timestamps.pop(0)
        
        # Need enough history to make a determination
        if len(self.ear_history) < 10:
            return False
        
        # Check for eye closure pattern
        avg_ear = sum(self.ear_history) / len(self.ear_history)
        recent_avg_ear = sum(self.ear_history[-5:]) / 5
        
        # Conditions for microsleep:
        # 1. Recent EAR is significantly lower than average
        # 2. Recent EAR is below threshold
        # 3. EAR has been low for multiple consecutive frames
        
        if (recent_avg_ear < avg_ear * 0.8 and 
            recent_avg_ear < self.EAR_THRESHOLD and
            all(e < self.EAR_THRESHOLD for e in self.ear_history[-5:])):
            
            # If we haven't started tracking a blink yet, this is the start
            if self.blink_start_time is None:
                self.blink_start_time = timestamp
            
            # Check if this blink has lasted long enough to be microsleep
            if timestamp - self.blink_start_time > 1.0:  # More than 1 second
                return True
        else:
            # Eyes are open, reset blink tracking
            self.blink_start_time = None
            
        return False
    
    def calculate_risk_level(self):
        """
        Calculate the risk level based on alert history and patterns
        Returns a value from 0-3:
        0: Low Risk
        1: Medium Risk
        2: High Risk
        3: Critical Risk
        """
        current_time = time.time()
        
        # Changed from 5 to 3 seconds for more responsive risk level updates
        if current_time - self.last_risk_update < 3:
            return self.risk_level
            
        self.last_risk_update = current_time
        
        now = datetime.now()
        # Remove old alerts from history (older than 15 minutes)
        self.alert_history = [
            alert for alert in self.alert_history 
            if (now - alert['timestamp']).total_seconds() < 900  # 15 minutes
        ]
        
        # NEW: If no alerts in the last 60 seconds, assume driver is inactive.
        if not self.alert_history or (now - max(alert['timestamp'] for alert in self.alert_history)).total_seconds() > 60:
            self.risk_level = 0
            return self.risk_level
        
        # ...existing alert frequency weighted calculation...
        total_weight = 0
        weighted_sum = 0
        
        for alert in self.alert_history:
            time_diff = (now - alert['timestamp']).total_seconds() / 60.0
            weight = max(0.05, 2.5 * np.exp(-1.0 * time_diff))
            if alert['type'] == 'drowsiness':
                weight *= 1.5
            elif alert['type'] == 'microsleep':
                weight *= 2.0
            elif alert['type'] == 'yawning':
                weight *= 0.7
            weighted_sum += weight
            total_weight += 1
        
        if total_weight > 0:
            alert_score = weighted_sum / total_weight
            most_recent_time = max(alert['timestamp'] for alert in self.alert_history)
            minutes_since_last_alert = (now - most_recent_time).total_seconds() / 60.0
            if minutes_since_last_alert > 3:
                decay_factor = min(0.8, 0.2 * (minutes_since_last_alert - 3))
                alert_score = max(0, alert_score - decay_factor)
            
            if alert_score < 0.5:
                risk = 0
            elif alert_score < 1.0:
                risk = 1
            elif alert_score < 2.0:
                risk = 2
            else:
                risk = 3
                
            recent_alerts = sum(1 for alert in self.alert_history 
                                if (now - alert['timestamp']).total_seconds() < 300)
            if recent_alerts >= 3:
                risk = max(2, risk)
            
            recent_microsleeps = sum(1 for alert in self.alert_history 
                                     if alert['type'] == 'microsleep' and 
                                     (now - alert['timestamp']).total_seconds() < 600)
            if recent_microsleeps >= 2:
                risk = 3
                
            if minutes_since_last_alert > (10/60) and self.risk_level >= 2:
                risk = max(1, self.risk_level - 1)
                
            self.risk_level = risk
        else:
            self.risk_level = 0
        
        return self.risk_level
    
    def enhance_image_for_detection(self, image):
        """Enhance image for better detection in varying lighting conditions"""
        # Calculate average brightness
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        avg_brightness = np.mean(gray)
        
        # Keep track of brightness trend
        self.brightness_history.append(avg_brightness)
        if len(self.brightness_history) > self.max_brightness_history:
            self.brightness_history.pop(0)
            
        # Determine if in low light mode based on recent brightness history
        avg_recent_brightness = np.mean(self.brightness_history)
        self.low_light_mode = avg_recent_brightness < self.light_threshold
        
        # In low light, apply more aggressive enhancement
        if self.low_light_mode:
            # Increase enhancement for low-light conditions
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            alpha = 1.8  # Increase contrast further
            beta = 40    # Increase brightness
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced, True
        else:
            # For normal light, mild enhancement
            clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced, False
    
    def detect_faces_with_timeout(self, gray, is_low_light):
        """Detect faces with a timeout to avoid hanging"""
        faces = []
        
        # First try dlib face detector if enabled (with timeout)
        if self.use_dlib:
            try:
                # Handle timeouts for face detection
                def dlib_detect():
                    try:
                        # Adjust parameters based on lighting conditions
                        if is_low_light:
                            result = self.detector(gray, 1)  # More sensitive in low light
                        else:
                            result = self.detector(gray, 0)
                        self.detection_queue.put(result)
                    except Exception as e:
                        self.detection_queue.put(e)
                
                # Clear queue first
                while not self.detection_queue.empty():
                    self.detection_queue.get()
                
                # Start detection in a separate thread
                detection_thread = threading.Thread(target=dlib_detect)
                detection_thread.daemon = True
                detection_thread.start()
                
                # Wait for result with timeout
                start_time = time.time()
                detection_thread.join(self.detection_timeout)
                
                # If we have a result, use it
                if not self.detection_queue.empty():
                    result = self.detection_queue.get()
                    if isinstance(result, Exception):
                        raise result
                    faces = result
                else:
                    print("Dlib face detection timed out, using OpenCV instead")
                    self.use_dlib = False  # Fallback to OpenCV permanently
                
            except Exception as e:
                print(f"dlib face detection error: {e}")
                self.use_dlib = False  # Fall back to OpenCV on failure
        
        # If no faces found with dlib or dlib is disabled, try OpenCV
        if len(faces) == 0:
            try:
                # Optimize OpenCV detection parameters
                min_neighbors = 3 if is_low_light else 4
                scale_factor = 1.1  # Standardized for better performance
                
                # Use a more efficient scaled-down version for detection
                height, width = gray.shape
                max_dimension = 400  # Maximum dimension for processing
                
                # Downscale if needed for faster processing
                if height > max_dimension or width > max_dimension:
                    scale = max_dimension / max(height, width)
                    small_frame = cv2.resize(gray, (0, 0), fx=scale, fy=scale)
                    
                    # Detect faces on smaller image
                    cv_faces = self.cv_face_detector.detectMultiScale(
                        small_frame, 
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(int(30*scale), int(30*scale))
                    )
                    
                    # Scale coordinates back to original size
                    cv_faces = [(int(x/scale), int(y/scale), 
                               int(w/scale), int(h/scale)) for x, y, w, h in cv_faces]
                else:
                    # Process original size if already small
                    cv_faces = self.cv_face_detector.detectMultiScale(
                        gray, 
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30)
                    )
                
                # Convert OpenCV faces to dlib format for consistent processing
                for (x, y, w, h) in cv_faces:
                    faces.append(dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h))
                    
                if len(cv_faces) > 0:
                    # Use eye detection to verify face (reduces false positives)
                    face_count_before = len(faces)
                    faces = self.verify_faces_with_eyes(gray, faces)
                    
                    # If we lost faces during verification but lighting is low,
                    # keep the original detections as low light makes eye detection difficult
                    if len(faces) == 0 and face_count_before > 0 and is_low_light:
                        faces = [dlib.rectangle(left=x, top=y, right=x+w, bottom=y+h) 
                                for (x, y, w, h) in cv_faces]
            except Exception as e:
                print(f"OpenCV face detection error: {e}")
        
        # Cache the faces for future use if we found any
        if len(faces) > 0:
            self.last_detected_face = faces[0]  # Store the first face
            self.last_face_time = time.time()
            
        # If no faces detected but we have a recent cached face, use it
        elif (self.last_detected_face is not None and 
              time.time() - self.last_face_time < self.face_cache_max_age):
            print("Using cached face")
            faces = [self.last_detected_face]
        
        return faces
        
    def verify_faces_with_eyes(self, gray, face_rects):
        """Verify faces by checking if eyes are detected, reduces false positives"""
        verified_faces = []
        
        for face_rect in face_rects:
            x = face_rect.left()
            y = face_rect.top()
            w = face_rect.width()
            h = face_rect.height()
            
            # Extract face region - with boundary checks
            y_start = max(0, y)
            y_end = min(gray.shape[0], y + h)
            x_start = max(0, x)
            x_end = min(gray.shape[1], x + w)
            
            # Skip invalid regions
            if y_start >= y_end or x_start >= x_end:
                continue
                
            face_roi = gray[y_start:y_end, x_start:x_end]
            
            # Skip tiny regions that might cause errors
            if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                continue
            
            # Focus on upper half of face for eye detection (more efficient)
            eyes_roi_height = face_roi.shape[0] // 2
            eyes_roi = face_roi[:eyes_roi_height, :]
                
            # Detect eyes in face region - with optimized parameters
            eyes = self.eye_detector.detectMultiScale(
                eyes_roi,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(15, 15)
            )
            
            # If at least one eye is detected, consider the face valid
            if len(eyes) > 0:
                verified_faces.append(face_rect)
                
        return verified_faces
        
    def process_frame(self, frame):
        """Process a video frame for drowsiness detection"""
        try:
            # Start processing flag - helps handle interruptions
            self.processing_frame = True
            
            # Check authentication if reference image exists
            if self.reference_image is not None and not self.authenticated:
                # Try to authenticate
                if self.auth_attempts < self.max_auth_attempts:
                    self.authenticated = self.authenticate_driver(frame)
                    self.auth_attempts += 1
                    
                    # If not authenticated, show message
                    if not self.authenticated:
                        viz_frame = frame.copy()
                        message = f"Driver authentication failed. Attempt {self.auth_attempts}/{self.max_auth_attempts}"
                        cv2.putText(viz_frame, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        self.processing_frame = False
                        return viz_frame, False, None
                else:
                    # Max attempts reached, show error
                    viz_frame = frame.copy()
                    message = "Authentication failed. Please contact administrator."
                    cv2.putText(viz_frame, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.processing_frame = False
                    return viz_frame, False, None
            
            # Continue with existing processing
            # Make a copy of the frame for visualization
            viz_frame = frame.copy()
            timestamp = time.time()
            
            # Apply image enhancement for better detection in various lighting
            gray, is_low_light = self.enhance_image_for_detection(frame)
            
            # Display low-light mode indicator
            if (is_low_light):
                cv2.putText(viz_frame, "LOW LIGHT MODE", (frame.shape[1] - 200, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
            
            # Use optimized face detection with timeout
            faces = self.detect_faces_with_timeout(gray, is_low_light)
            
            # Reset status for this frame
            current_drowsy = False
            current_yawning = False
            alert_type = None
            
            # Get frame dimensions for display info
            frame_height, frame_width = frame.shape[:2]
            
            # Add counter info to frame
            cv2.putText(viz_frame, f"Frame: {self.frame_counter}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
            
            self.frame_counter += 1
            
            # Calculate and display risk level
            risk_level = self.calculate_risk_level()
            risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            risk_colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
            
            risk_text = f"RISK LEVEL: {risk_labels[risk_level]}"
            cv2.putText(viz_frame, risk_text, (frame_width - 250, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_colors[risk_level], 2)
            
            # Process each face (usually just one)
            if len(faces) > 0:
                for face in faces:
                    try:
                        # Get facial landmarks
                        shape = self.predictor(gray, face)
                        landmarks = self.shape_to_np(shape)
                        
                        # Get specific features
                        left_eye = landmarks[36:42]
                        right_eye = landmarks[42:48]
                        mouth = landmarks[48:68]
                        
                        # Draw face outline
                        x, y = face.left(), face.top()
                        w, h = face.width(), face.height()
                        cv2.rectangle(viz_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Calculate EAR for each eye
                        left_ear = self.calculate_ear(left_eye)
                        right_ear = self.calculate_ear(right_eye)
                        avg_ear = (left_ear + right_ear) / 2.0
                        
                        # Calculate MOR
                        mor = self.calculate_mor(mouth)
                        
                        # Check for microsleep pattern
                        microsleep_detected = self.is_microsleep(avg_ear, timestamp)
                        
                        # Draw eye landmarks
                        for eye in [left_eye, right_eye]:
                            hull = cv2.convexHull(eye)
                            cv2.drawContours(viz_frame, [hull], -1, (0, 255, 0), 1)
                        
                        # Draw mouth landmarks with color based on MOR threshold
                        mouth_array = np.array(mouth)
                        mouth_color = (0, 0, 255) if mor > self.MOR_THRESHOLD else (0, 255, 255)
                        cv2.polylines(viz_frame, [mouth_array], True, mouth_color, 2)
                        
                        # Draw the inner mouth - improved visualization
                        inner_mouth_array = np.array(mouth[12:20])
                        cv2.polylines(viz_frame, [inner_mouth_array], True, mouth_color, 1)
                        
                        # Display EAR and MOR values
                        ear_text = f"EAR: {avg_ear:.2f} [Threshold: {self.EAR_THRESHOLD:.2f}]"
                        mor_text = f"MOR: {mor:.2f} [Threshold: {self.MOR_THRESHOLD:.2f}]"
                        
                        ear_color = (0, 0, 255) if avg_ear < self.EAR_THRESHOLD else (0, 255, 0)
                        mor_color = (0, 0, 255) if mor > self.MOR_THRESHOLD else (0, 255, 0)
                        
                        cv2.putText(viz_frame, ear_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ear_color, 2)
                        cv2.putText(viz_frame, mor_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mor_color, 2)
                        
                        # Update counters
                        if avg_ear < self.EAR_THRESHOLD:
                            self.ear_counter += 1
                            
                            # Check for microsleep
                            if microsleep_detected:
                                current_drowsy = True
                                alert_type = 'microsleep'
                                cv2.putText(viz_frame, "MICROSLEEP DETECTED!", (10, 120), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            # Check for standard drowsiness
                            elif self.ear_counter >= self.CONSECUTIVE_FRAMES:
                                current_drowsy = True
                                alert_type = 'drowsiness'
                                cv2.putText(viz_frame, "DROWSINESS DETECTED!", (10, 120), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            self.ear_counter = 0
                        
                        # Check for yawning
                        if mor > self.MOR_THRESHOLD:
                            self.mor_counter += 1
                        else:
                            self.mor_counter = max(0, self.mor_counter - 1)
                        
                        if self.mor_counter >= (self.CONSECUTIVE_FRAMES // 2):
                            current_yawning = True
                            if not current_drowsy:
                                alert_type = 'yawning'
                            cv2.putText(viz_frame, "YAWNING DETECTED!", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except Exception as e:
                        print(f"Error processing facial landmarks: {e}")
                        # Continue with the next face if available
            else:
                # No face detected
                cv2.putText(viz_frame, "No face detected", (frame_width//4, frame_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Update overall status
            self.drowsy = current_drowsy
            self.yawning = current_yawning
            
            # Record alert if detected
            if self.drowsy or self.yawning:
                detection_info = {
                    'timestamp': datetime.now(),
                    'type': alert_type or ('drowsiness' if self.drowsy else 'yawning'),
                    'ear': avg_ear if 'avg_ear' in locals() else None,
                    'mor': mor if 'mor' in locals() else None
                }
                
                # Add to alert history for risk calculation
                self.alert_history.append(detection_info)
                
                # Maintain maximum size of alert history
                if len(self.alert_history) > self.max_alert_history:
                    self.alert_history.pop(0)
                
                # Add to detections list
                self.detections.append(detection_info)
                
                # Play alarm sound based on detection type and risk level
                if alert_type == 'microsleep':
                    self.play_alarm('critical')
                elif alert_type == 'drowsiness':
                    severity = 'high' if risk_level >= 2 else 'medium'
                    self.play_alarm(severity)
                elif alert_type == 'yawning' and risk_level >= 1:
                    self.play_alarm('low')
            else:
                # Stop alarm if the driver is now alert
                self.stop_alarm()
            
            # Display overall status
            status = []
            if self.drowsy:
                status.append("DROWSY")
            if self.yawning:
                status.append("YAWNING")
            
            if status:
                status_str = " & ".join(status)
                cv2.putText(viz_frame, f"ALERT: {status_str}", (10, frame_height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                
                # Draw red border around frame when alert is active
                cv2.rectangle(viz_frame, (0, 0), (frame_width-1, frame_height-1), (0, 0, 255), 10)
            else:
                cv2.putText(viz_frame, "Status: Alert", (10, frame_height - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Reset processing flag
            self.processing_frame = False
            
            # Return both the processed frame and detection status
            return viz_frame, self.drowsy or self.yawning, alert_type
        
        except KeyboardInterrupt:
            # Handle keyboard interruption gracefully
            print("Processing interrupted by user")
            self.processing_frame = False
            return frame.copy(), False, None
            
        except Exception as e:
            print(f"Error in process_frame: {e}")
            self.processing_frame = False
            # Return the original frame with error message if processing fails
            viz_frame = frame.copy()
            cv2.putText(viz_frame, f"Error: {str(e)[:50]}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return viz_frame, False, None

    def play_suspension_message(self, message):
        """Play an audio message for suspension/unsuspension notifications"""
        if not self.alarm_enabled:
            return
            
        # Stop any currently playing sounds
        self.stop_alarm()
        
        try:
            # Use text-to-speech to convert the message to audio
            # This implementation uses pygame's mixer to play a system beep
            # followed by the message text printed to console
            
            # Play a notification sound
            notification_sound = self.alarm_sounds['medium']
            notification_sound.play()
            
            # In a real implementation, you would use a TTS service or library
            # For example with pyttsx3:
            # import pyttsx3
            # engine = pyttsx3.init()
            # engine.say(message)
            # engine.runAndWait()
            
            # For now, we'll just print the message
            print(f"[AUDIO MESSAGE]: {message}")
            
            self.alarm_playing = True
            time.sleep(1)  # Give time for the notification sound to play
            self.alarm_playing = False
        except Exception as e:
            print(f"Error playing suspension message: {str(e)}")

if __name__ == "__main__":
    # Simple test with webcam
    detector = DrowsinessDetector()
    cap = cv2.VideoCapture(2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, alert_detected, alert_type = detector.process_frame(frame)
        
        cv2.imshow('Drowsiness Detection', processed_frame)
        
        # Add keyboard shortcuts
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):  # Toggle alarm with 'a' key
            detector.toggle_alarm()
            status = "enabled" if detector.alarm_enabled else "disabled"
            print(f"Alarm {status}")
            
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
