"""
Driver Face Recognition Module
Handles driver identification and verification using facial recognition
"""

import os
import cv2  # Ensure cv2 is imported at the very top
import numpy as np
import pickle
import base64
import requests
import json
from pathlib import Path
from datetime import datetime
from io import BytesIO
from PIL import Image

class DriverRecognizer:
    """Handles driver face recognition using OpenCV's face recognition capabilities"""
    
    def __init__(self, model_path=None, server_url=None, api_key=None, driver_id=None):
        global cv2  # Declare cv2 as global to ensure it is recognized
        """
        Initialize the driver recognition system
        
        Args:
            model_path: Path to pre-trained face recognition model
            server_url: URL of the server for fetching reference images
            api_key: API key for server authentication
            driver_id: ID of the driver to authenticate
        """
        self.model_path = model_path
        self.server_url = server_url
        self.api_key = api_key
        self.driver_id = driver_id
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to create the face recognizer with error handling for missing face module
        try:
            # Check if cv2.face is available
            if hasattr(cv2, 'face'):
                self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                self.opencv_face_available = True
            else:
                # Try to import the face module directly
                try:
                    import cv2.face
                    self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
                    self.opencv_face_available = True
                except ImportError:
                    print("WARNING: OpenCV face module not available. Face recognition features will be limited.")
                    print("To fix this, install OpenCV with contrib modules: pip install opencv-contrib-python")
                    self.face_recognizer = None
                    self.opencv_face_available = False
        except Exception as e:
            print(f"ERROR: Could not initialize face recognizer: {e}")
            print("Face recognition features will be limited.")
            self.face_recognizer = None
            self.opencv_face_available = False
            
        self.recognition_threshold = 70.0  # Lower is more strict (better match)
        
        # Driver data mapping ID to features
        self.driver_data = {}
        self.reference_image = None
        self.reference_face = None
        
        # Load model if provided or try to fetch from server
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif server_url and driver_id:
            print(f"Fetching driver model from server: {server_url}")
            self.fetch_driver_model()
        else:
            print("No recognition model provided or model doesn't exist.")
            # Try to find model in common locations
            default_paths = [
                "models/driver_recognition_model.pkl",
                "driver_recognition_model.pkl",
                "../models/driver_recognition_model.pkl"
            ]
            for path in default_paths:
                if os.path.exists(path):
                    print(f"Found model at {path}")
                    self.load_model(path)
                    break
    
    def fetch_driver_model(self):
        """Fetch driver recognition model from the server"""
        if not self.server_url or not self.driver_id:
            print("Server URL and driver ID are required to fetch model")
            return False
            
        try:
            # Create headers with API key if available
            headers = {}
            if self.api_key:
                headers['Authorization'] = f'Bearer {self.api_key}'
                
            # Fetch driver model from server API endpoint
            endpoint = f"{self.server_url.rstrip('/')}/api/driver_model/{self.driver_id}"
            print(f"Requesting model from: {endpoint}")
            
            response = requests.get(endpoint, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Error fetching model: HTTP {response.status_code}")
                print(response.text)
                return False
            
            data = response.json()
            
            if not data.get('success'):
                print(f"Error: {data.get('error', 'Unknown error')}")
                return False
            
            model_data = data['model_data']
            
            # Store driver info
            self.driver_data[self.driver_id] = {
                'id': model_data['driver_id'],
                'username': model_data['username']
            }
            
            # Process reference image if available
            if model_data.get('reference_image'):
                print("Reference image received, setting up for recognition")
                
                # Store the base64 image for simple matching
                self.reference_image = model_data['reference_image']
                
                # Convert base64 to image for facial feature extraction
                try:
                    # Convert base64 to image
                    image_bytes = base64.b64decode(model_data['reference_image'])
                    
                    # Convert to numpy array for OpenCV
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if img is None:
                        raise ValueError("Failed to decode image")
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Store the entire image for template matching fallback
                    self.reference_face = gray
                    
                    # Detect face
                    faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
                    
                    if len(faces) > 0:
                        # Get largest face
                        face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = face
                        
                        # Extract face region
                        face_roi = gray[y:y+h, x:x+w]
                        
                        # Save the extracted face for template matching
                        self.reference_face = cv2.resize(face_roi, (100, 100))
                        
                        # If OpenCV face module is available, train recognizer with this face
                        if self.opencv_face_available and self.face_recognizer:
                            # Standard size for face recognition
                            face_roi = cv2.resize(face_roi, (100, 100))
                            
                            # Train recognizer with this face
                            self.face_recognizer.train([face_roi], np.array([self.driver_id]))
                            print("Face recognizer trained with reference image")
                        else:
                            print("OpenCV face module not available, will use template matching instead")
                    else:
                        print("No face detected in reference image, will use full image for comparison")
                        
                except Exception as e:
                    print(f"Error processing reference image: {e}")
            else:
                print("No reference image available in model data")
                
            print("Driver model successfully loaded from server")
            return True
            
        except Exception as e:
            print(f"Error fetching driver model: {e}")
            return False
    
    def load_model(self, model_path):
        """Load a pre-trained face recognition model"""
        try:
            # Check if we have OpenCV face support
            if not self.opencv_face_available:
                print("WARNING: Cannot load face recognition model without OpenCV face module.")
                return False
                
            # Check if it's a pickle file containing both model and driver data
            if model_path.endswith('.pkl'):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.driver_data = data.get('driver_data', {})
                    model_file = data.get('model_file')
                    self.reference_image = data.get('reference_image')
                    
                    # If the pickle includes a temporary model file path
                    if model_file:
                        self.face_recognizer.read(model_file)
                    else:
                        print("No model file found in pickle data.")
            
            # Direct model file
            else:
                self.face_recognizer.read(model_path)
                
            print(f"Loaded driver recognition model with {len(self.driver_data)} known drivers")
            return True
            
        except Exception as e:
            print(f"Error loading recognition model: {e}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess image for face recognition"""
        if image is None or image.size == 0:
            return None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Equalize histogram for better recognition in different lighting
        gray = cv2.equalizeHist(gray)
        
        # Resize to standard size
        standard_size = (100, 100)
        return cv2.resize(gray, standard_size)
    
    def identify_driver(self, face_image):
        """
        Identify a driver from their face
        
        Args:
            face_image: Image containing the driver's face
            
        Returns:
            tuple: (driver_id, confidence) or (None, 0) if not recognized
        """
        if face_image is None:
            return None, 0
            
        # If OpenCV face recognition not available, fall back to template matching
        if not self.opencv_face_available or not self.face_recognizer:
            return self._identify_by_template_matching(face_image)
            
        if len(self.driver_data) == 0:
            print("No driver data available for recognition")
            return None, 0
            
        # Preprocess the image
        processed_face = self.preprocess_image(face_image)
        if processed_face is None:
            return None, 0
            
        try:
            # Perform recognition
            label, confidence = self.face_recognizer.predict(processed_face)
            
            # Convert confidence to 0-1 range (higher is better)
            # LBPH gives lower values for better matches
            normalized_confidence = max(0, min(100 - confidence, 100)) / 100.0
            
            # Only return ID if confidence is high enough
            if normalized_confidence > 0.6:  # 60% confident
                return label, normalized_confidence
                
        except Exception as e:
            print(f"Error during face recognition: {e}")
            
        # If OpenCV recognition failed, try template matching as fallback
        return self._identify_by_template_matching(face_image)
    
    def _identify_by_template_matching(self, face_image):
        """Fallback recognition using template matching"""
        if self.reference_face is None:
            return None, 0
            
        try:
            # Preprocess image to match reference face
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
                
            # Resize to same dimensions as reference face
            resized = cv2.resize(gray, (self.reference_face.shape[1], self.reference_face.shape[0]))
            
            # Use template matching to compare faces
            result = cv2.matchTemplate(resized, self.reference_face, cv2.TM_CCOEFF_NORMED)
            _, confidence = cv2.minMaxLoc(result)
            
            # Consider it a match if confidence is above threshold
            if confidence > 0.6:
                return self.driver_id, confidence
                
        except Exception as e:
            print(f"Error in template matching: {e}")
            
        return None, 0
    
    def verify_driver(self, face_image, driver_id=None):
        """
        Verify if the face matches the specified driver ID
        
        Args:
            face_image: Image containing the driver's face
            driver_id: ID of the driver to verify against (uses self.driver_id if None)
            
        Returns:
            tuple: (is_verified, confidence)
        """
        if driver_id is None:
            driver_id = self.driver_id
            
        if driver_id is None:
            print("No driver ID specified for verification")
            return False, 0
            
        # First identify the driver
        detected_id, confidence = self.identify_driver(face_image)
        
        # Check if the detected ID matches the requested driver_id
        if detected_id is not None and detected_id == driver_id:
            return True, confidence
            
        # For simple images without faces, use direct image comparison
        if self.reference_image and detected_id is None:
            try:
                # Encode the current frame
                _, buffer = cv2.imencode('.jpg', face_image)
                current_image_b64 = base64.b64encode(buffer).decode('utf-8')
                
                # Compare image hashes (simple comparison)
                import imagehash
                from PIL import Image
                from io import BytesIO
                
                # Get current image hash
                current_bytes = BytesIO(base64.b64decode(current_image_b64))
                current_hash = imagehash.average_hash(Image.open(current_bytes))
                
                # Get reference image hash
                ref_bytes = BytesIO(base64.b64decode(self.reference_image))
                ref_hash = imagehash.average_hash(Image.open(ref_bytes))
                
                # Compare hashes
                hash_diff = current_hash - ref_hash
                
                # Convert difference to similarity score (lower diff = higher similarity)
                similarity = max(0, 1 - (hash_diff / 64))  # 64 is max hash difference
                
                if similarity > 0.7:  # Threshold for similarity
                    return True, similarity
            except Exception as e:
                print(f"Error in image hash comparison: {e}")
            
        return False, confidence

    def save_model(self, output_path):
        """Save the face recognition model for later use"""
        if not self.opencv_face_available:
            print("Cannot save model without OpenCV face module")
            return False
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            # Temporary file for face recognizer model
            temp_model_file = output_path + ".xml"
            
            # Save the recognizer model
            self.face_recognizer.write(temp_model_file)
            
            # Prepare data for pickle
            data = {
                'driver_data': self.driver_data,
                'model_file': temp_model_file,
                'reference_image': self.reference_image
            }
            
            # Save everything to pickle file
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
                
            print(f"Saved driver recognition model to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving recognition model: {e}")
            return False
