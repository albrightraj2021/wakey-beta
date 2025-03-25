import cv2
import numpy as np
import os
import time
from datetime import datetime
import dlib
import pygame  # Added for sound support
import base64
from io import BytesIO
from PIL import Image

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
        
        # Add OpenCV face detector as fallback
        self.cv_face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Add eye detector for additional verification
        self.eye_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye_tree_eyeglasses.xml')
        
        # Initialize MobileNetSSD model for face recognition
        self.face_recognizer = self.initialize_face_recognition()
        
        # Store reference image for authentication
        self.reference_image = None
        self.reference_face_encoding = None
        if reference_image:
            self.set_reference_image(reference_image)
        
        self.use_dlib = True  # Start with dlib, fallback to OpenCV if needed
        
        # Add low-light parameters
        self.low_light_mode = False
        self.light_threshold = 60  # Threshold to determine low light conditions
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

    def initialize_face_recognition(self):
        """Initialize face recognition using OpenCV's face recognizer"""
        # For simplicity, we'll use OpenCV's built-in face recognition
        # In production, you might want to use a more sophisticated method like FaceNet
        face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        return face_recognizer
        
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
                
                # Extract face encoding from reference image
                gray = cv2.cvtColor(self.reference_image, cv2.COLOR_BGR2GRAY)
                faces = self.detector(gray, 1)
                
                if len(faces) > 0:
                    # Get largest face
                    largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
                    face_roi = gray[largest_face.top():largest_face.bottom(), largest_face.left():largest_face.right()]
                    
                    # Train recognizer with this face
                    self.face_recognizer.train([face_roi], np.array([1]))
                    
                    print("Reference face processed successfully")
                    return True
                else:
                    print("No face detected in reference image")
            else:
                print("Invalid reference image format")
        except Exception as e:
            print(f"Error processing reference image: {e}")
        
        return False
        
    def authenticate_driver(self, frame):
        """Verify driver's identity against reference image"""
        if self.reference_image is None:
            # No reference image, consider authenticated
            return True
            
        # Authentication logic
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector(gray, 1)
            
            if len(faces) == 0:
                return False
                
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            face_roi = gray[largest_face.top():largest_face.bottom(), largest_face.left():largest_face.right()]
            
            # Resize to match training data if necessary
            face_roi = cv2.resize(face_roi, (100, 100))
            
            # Predict using face recognizer
            label, confidence = self.face_recognizer.predict(face_roi)
            
            # Lower confidence means better match in LBPH
            is_match = confidence < 70  # Threshold for confidence
            
            return is_match
        except Exception as e:
            print(f"Authentication error: {e}")
            return False

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
        
        # Remove old alerts from history (older than 15 minutes)
        now = datetime.now()
        self.alert_history = [
            alert for alert in self.alert_history 
            if (now - alert['timestamp']).total_seconds() < 900  # 15 minutes
        ]
        
        # If no alerts, risk is low
        if not self.alert_history:
            self.risk_level = 0
            return self.risk_level
            
        # Calculate time-weighted alert frequency
        total_weight = 0
        weighted_sum = 0
        
        # Recent alerts have higher weight
        for alert in self.alert_history:
            # Time difference in minutes
            time_diff = (now - alert['timestamp']).total_seconds() / 60.0
            
            # Exponential decay weight: recent alerts matter more
            # Steeper decay - alerts older than 3 minutes have much less impact
            weight = max(0.05, 2.5 * np.exp(-1.0 * time_diff))
            
            # Higher weight for drowsiness and microsleep than yawning
            if alert['type'] == 'drowsiness':
                weight *= 1.5
            elif alert['type'] == 'microsleep':
                weight *= 2.0
            elif alert['type'] == 'yawning':
                weight *= 0.7  # Reduce impact of yawning
            
            weighted_sum += weight
            total_weight += 1
        
        # Calculate weighted average
        if total_weight > 0:
            alert_score = weighted_sum / total_weight
            
            # Natural decay - if most recent alert is older than 3 minutes, reduce score
            most_recent_time = max([alert['timestamp'] for alert in self.alert_history])
            minutes_since_last_alert = (now - most_recent_time).total_seconds() / 60.0
            
            if minutes_since_last_alert > 3:
                # Apply decay factor - the longer since the last alert, the more we reduce
                decay_factor = min(0.8, 0.2 * (minutes_since_last_alert - 3))
                alert_score = max(0, alert_score - decay_factor)
            
            # Convert score to risk level
            if alert_score < 0.5:
                risk = 0  # Low risk
            elif alert_score < 1.0:
                risk = 1  # Medium risk
            elif alert_score < 2.0:
                risk = 2  # High risk
            else:
                risk = 3  # Critical risk
                
            # Check for rapid increase in alerts (more than 3 in last 5 minutes)
            recent_alerts = sum(1 for alert in self.alert_history 
                             if (now - alert['timestamp']).total_seconds() < 300)
            if recent_alerts >= 3:
                risk = max(2, risk)  # At least high risk
            
            # Check for multiple microsleeps
            recent_microsleeps = sum(1 for alert in self.alert_history 
                                 if alert['type'] == 'microsleep' and 
                                 (now - alert['timestamp']).total_seconds() < 600)
            if recent_microsleeps >= 2:
                risk = 3  # Critical risk
                
            # Ensure risk level decreases over time without new alerts
            # Drastically reduced from minutes to just 10 seconds (0.1667 minutes)
            if minutes_since_last_alert > (10/60) and self.risk_level >= 2:
                # Step down by one level
                risk = max(1, self.risk_level - 1)
                
            self.risk_level = risk
        else:
            # Fallback - should not happen as we check for empty alert_history above
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
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            
            # Increase contrast
            alpha = 1.5  # Contrast control
            beta = 30    # Brightness control
            enhanced = cv2.convertScaleAbs(enhanced, alpha=alpha, beta=beta)
            
            # Reduce noise with bilateral filter (preserves edges)
            enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
            
            return enhanced, True
        else:
            # For normal light, mild enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            return enhanced, False
    
    def process_frame(self, frame):
        """Process a video frame for drowsiness detection"""
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
                    return viz_frame, False, None
            else:
                # Max attempts reached, show error
                viz_frame = frame.copy()
                message = "Authentication failed. Please contact administrator."
                cv2.putText(viz_frame, message, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                return viz_frame, False, None
        
        # Continue with existing processing
        # Make a copy of the frame for visualization
        viz_frame = frame.copy()
        timestamp = time.time()
        
        # Apply image enhancement for better detection in various lighting
        gray, is_low_light = self.enhance_image_for_detection(frame)
        
        # Display low-light mode indicator
        if is_low_light:
            cv2.putText(viz_frame, "LOW LIGHT MODE", (frame.shape[1] - 200, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
        
        # Try multiple detection approaches for better reliability
        faces = []
        
        # First try dlib face detector if enabled
        if self.use_dlib:
            try:
                # Adjust parameters based on lighting conditions
                if is_low_light:
                    # In low light, use lower threshold (more sensitive)
                    faces = self.detector(gray, 1)  # Second param: upsampling for better detection
                else:
                    faces = self.detector(gray, 0)
            except Exception as e:
                print(f"dlib face detection error: {e}")
                self.use_dlib = False  # Fall back to OpenCV on failure
        
        # If no faces found with dlib or dlib is disabled, try OpenCV
        if len(faces) == 0:
            try:
                # Adjust parameters based on lighting
                min_neighbors = 3 if is_low_light else 5
                scale_factor = 1.05 if is_low_light else 1.1
                
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
                        
                    if len(faces) > 0:
                        print("Using OpenCV face detection (fallback)")
            except Exception as e:
                print(f"OpenCV face detection error: {e}")
        
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
                
                # Check for yawning - with improved threshold logic
                if mor > self.MOR_THRESHOLD:
                    self.mor_counter += 1
                    if self.mor_counter >= self.CONSECUTIVE_FRAMES // 2:  # Less frames needed for yawn detection
                        current_yawning = True
                        # Only set alert type to yawning if no drowsiness detected (drowsiness is higher priority)
                        if not current_drowsy:
                            alert_type = 'yawning'
                        cv2.putText(viz_frame, "YAWNING DETECTED!", (10, 150), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    self.mor_counter = 0

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
        
        # Return both the processed frame and detection status
        return viz_frame, self.drowsy or self.yawning, alert_type

    def verify_faces_with_eyes(self, gray, face_rects):
        """Verify faces by checking if eyes are detected, reduces false positives"""
        verified_faces = []
        
        for face_rect in face_rects:
            x = face_rect.left()
            y = face_rect.top()
            w = face_rect.width()
            h = face_rect.height()
            
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Skip tiny regions that might cause errors
            if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
                continue
                
            # Detect eyes in face region
            eyes = self.eye_detector.detectMultiScale(face_roi)
            
            # If at least one eye is detected, consider the face valid
            if len(eyes) > 0:
                verified_faces.append(face_rect)
                
        return verified_faces


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
