import cv2
import argparse
import time
from drowsiness_detector import DrowsinessDetector
from data_sender import DataSender
import requests
import json
import base64
from io import BytesIO
import numpy as np
from PIL import Image
import sys
import os

def main():
    parser = argparse.ArgumentParser(description="Driver Monitoring Application")
    parser.add_argument("--server", type=str, help="Server URL", required=True)
    parser.add_argument("--driver-id", type=int, help="Driver ID (optional, will auto-detect if not provided)", required=False, default=None)
    parser.add_argument("--api-key", type=str, help="API Key (optional)", required=False)
    args = parser.parse_args()

    detector = DrowsinessDetector()

    data_sender = None
    if args.server:
        # Initialize DataSender with server URL, API key is optional
        data_sender = DataSender(server_url=args.server, api_key=args.api_key if hasattr(args, 'api_key') else None)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    # Add throttling variables
    last_alert_time = 0
    alert_cooldown = 10  # seconds between alerts
    last_yawn_alert_time = 0
    yawn_cooldown = 20  # seconds between yawn alerts
    
    # Batch alerts to reduce API calls
    queued_alerts = []
    last_send_time = time.time()
    send_interval = 15  # seconds between batch sends
    
    # Add risk update variables
    last_risk_update_time = 0
    risk_update_interval = 3  # Reduced interval to 3 seconds for more frequent updates
    
    # Last sent risk level to avoid sending duplicates
    last_sent_risk_level = -1

    # Driver authentication variables
    active_driver_id = args.driver_id
    active_driver_username = None
    reference_images = {}
    authentication_attempts = 0
    max_authentication_attempts = 10
    authentication_interval = 5  # seconds between authentication attempts
    last_authentication_time = 0
    
    # Add suspension variables
    is_suspended = False
    last_suspension_check = 0
    suspension_check_interval = 30  # Check every 30 seconds
    
    if active_driver_id is None:
        print("No driver ID provided. Will attempt to automatically authenticate driver.")
        # Fetch available driver reference images
        reference_images = fetch_driver_references(args.server, args.api_key)
        if not reference_images:
            print("Error: Could not fetch any driver reference images from the server.")
            print("Please check your server connection or provide a specific driver ID.")
            return
        print(f"Fetched {len(reference_images)} driver references. Starting automatic authentication...")
    else:
        # If driver ID is provided, try to get the username for display purposes
        driver_info = get_driver_info(args.server, args.driver_id, args.api_key)
        if driver_info:
            active_driver_username = driver_info.get('username')
            print(f"Driver monitoring started for: {active_driver_username} (ID: {active_driver_id})")
        else:
            print(f"Driver monitoring started for ID: {active_driver_id}")
    
    print(f"Connected to server: {args.server}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_time = time.time()
        
        # If no driver ID provided, attempt to authenticate the driver
        if active_driver_id is None and authentication_attempts < max_authentication_attempts:
            if current_time - last_authentication_time > authentication_interval:
                print(f"Attempting driver authentication (attempt {authentication_attempts + 1}/{max_authentication_attempts})")
                driver_info = authenticate_driver(frame, reference_images, detector)
                last_authentication_time = current_time
                authentication_attempts += 1
                
                if driver_info:
                    active_driver_id = driver_info['id']
                    active_driver_username = driver_info['username']
                    print(f"Driver authenticated! {active_driver_username} (ID: {active_driver_id})")
                    
                    # Add an overlay to show successful authentication
                    cv2.putText(frame, f"Driver Authenticated: {active_driver_username}", (30, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.imshow("Driver Monitor", frame)
                    cv2.waitKey(1000)  # Show the success message for 1 second

        # If we've exhausted authentication attempts without success
        if active_driver_id is None and authentication_attempts >= max_authentication_attempts:
            # Show error message on frame
            cv2.putText(frame, "Authentication failed. Please restart.", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Driver Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Check for suspension status periodically
        if active_driver_id and current_time - last_suspension_check > suspension_check_interval:
            suspension_status = check_suspension_status(args.server, active_driver_id, args.api_key)
            if suspension_status['suspended'] != is_suspended:
                is_suspended = suspension_status['suspended']
                if is_suspended:
                    print(f"DRIVER SUSPENDED: {suspension_status['message']}")
                    # Display suspension message on screen
                    suspension_frame = create_suspension_message(suspension_status['message'])
                    cv2.imshow('Driver Monitoring', suspension_frame)
                    # Play audio message
                    detector.play_suspension_message(suspension_status['message'])
                else:
                    print(f"DRIVER UNSUSPENDED: {suspension_status['message']}")
                    # Display unsuspension message
                    unsuspension_frame = create_unsuspension_message(suspension_status['message'])
                    cv2.imshow('Driver Monitoring', unsuspension_frame)
                    # Play audio message
                    detector.play_suspension_message(suspension_status['message'])
            last_suspension_check = current_time

        # Skip processing if suspended
        if is_suspended:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Display suspension message
            suspension_frame = create_suspension_message("YOU ARE SUSPENDED")
            cv2.imshow('Driver Monitoring', suspension_frame)
            
            # Check for key press to exit
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
            continue

        # Only process frames if we have an authenticated driver
        if active_driver_id:
            processed_frame, alert_detected, alert_type = detector.process_frame(frame)
            
            # Add driver username to the display
            if active_driver_username:
                cv2.putText(processed_frame, f"Driver: {active_driver_username}", (10, processed_frame.shape[0] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
            cv2.imshow("Driver Monitor", processed_frame)

            # Get current risk level
            risk_level = detector.risk_level
            risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
            
            # Check if we should record this alert based on cooldown
            if alert_detected and data_sender and active_driver_id:
                record_alert = False
                
                if alert_type == "drowsiness" and (current_time - last_alert_time) > alert_cooldown:
                    record_alert = True
                    last_alert_time = current_time
                elif alert_type == "yawning" and (current_time - last_yawn_alert_time) > yawn_cooldown:
                    record_alert = True
                    last_yawn_alert_time = current_time
                elif alert_type == "microsleep":  # Always record microsleep events
                    record_alert = True
                    last_alert_time = current_time
                    
                if record_alert:
                    # Always include risk level in the detection data
                    detection_data = {
                        "type": alert_type,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "risk_level": risk_level,
                        "risk_label": risk_labels[risk_level]
                    }
                    # Add to queue instead of sending immediately
                    queued_alerts.append(detection_data)
                    print(f"Alert queued: {alert_type} with risk level: {risk_labels[risk_level]}")
                    
                    # Send risk level update directly - now API key is optional
                    send_direct_risk_update(args.server, active_driver_id, risk_level, risk_labels[risk_level], args.api_key)
            
            # Send periodic risk level updates even when no alerts are detected
            if data_sender and active_driver_id and (current_time - last_risk_update_time) > risk_update_interval:
                # Only send if risk level has changed or it's been a while since the last update
                if risk_level != last_sent_risk_level or (current_time - last_risk_update_time) > 30:
                    # Send risk level update directly instead of queuing it as an alert
                    send_direct_risk_update(args.server, active_driver_id, risk_level, risk_labels[risk_level], args.api_key)
                    # Store data about this driver's status without adding to alert queue
                    last_risk_update_time = current_time
                    last_sent_risk_level = risk_level
                    print(f"Status update sent directly with risk level: {risk_labels[risk_level]} for driver: {active_driver_username}")
                
            # Send batched alerts periodically
            if queued_alerts and (current_time - last_send_time) > send_interval:
                if data_sender:
                    print(f"Sending batch of {len(queued_alerts)} alerts/updates")
                    data_sender.send_batch_detection_data(queued_alerts, active_driver_id)
                    queued_alerts = []
                    last_send_time = current_time
        else:
            # If we don't have a driver ID yet, show the basic frame
            cv2.putText(frame, "Attempting to authenticate driver...", (30, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
            cv2.imshow("Driver Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # NEW: Restart the script when 'r' key is pressed
        if key == ord('r'):
            print("Restarting driver monitor...")
            python = sys.executable
            os.execl(python, python, *sys.argv)

    # Send any remaining alerts before exiting
    if queued_alerts and data_sender and active_driver_id:
        data_sender.send_batch_detection_data(queued_alerts, active_driver_id)
        
    cap.release()
    cv2.destroyAllWindows()

def fetch_driver_references(server_url, api_key=None):
    """Fetch all driver reference images from the server"""
    if not server_url:
        return {}
    
    try:
        # Construct the URL for fetching driver references
        references_url = f"{server_url.rstrip('/')}/api/driver_references"
        
        # Prepare headers - only add Authorization if API key is provided
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        # Send the request
        response = requests.get(
            references_url,
            headers=headers,
            timeout=10  # Longer timeout for fetching multiple images
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'success' in data and data['success'] and 'drivers' in data:
                reference_images = {}
                # Process each driver reference image
                for driver in data['drivers']:
                    if 'id' in driver and 'reference_image' in driver and driver['reference_image']:
                        try:
                            # Convert base64 image to numpy array for processing
                            img_data = driver['reference_image']
                            # Remove base64 prefix if present
                            if ',' in img_data:
                                img_data = img_data.split(',', 1)[1]
                            # Decode base64 to image
                            image = Image.open(BytesIO(base64.b64decode(img_data)))
                            # Convert PIL Image to numpy array for OpenCV
                            ref_image = np.array(image)
                            if len(ref_image.shape) == 3 and ref_image.shape[2] == 4:  # If RGBA
                                ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGBA2BGR)
                            elif len(ref_image.shape) == 3 and ref_image.shape[2] == 3:  # If RGB
                                ref_image = cv2.cvtColor(ref_image, cv2.COLOR_RGB2BGR)
                            # Store reference image with driver ID
                            reference_images[driver['id']] = {
                                'image': ref_image,
                                'username': driver.get('username', f"Driver {driver['id']}")
                            }
                        except Exception as e:
                            print(f"Error processing reference image for driver {driver['id']}: {e}")
                
                print(f"Successfully fetched {len(reference_images)} driver reference images")
                return reference_images
            else:
                print("Invalid response format from server")
                return {}
        else:
            print(f"Error fetching driver references: {response.status_code}")
            print(response.text)
            return {}
            
    except Exception as e:
        print(f"Error in fetch_driver_references: {e}")
        return {}

def get_driver_info(server_url, driver_id, api_key=None):
    """Get driver information by ID"""
    if not server_url or not driver_id:
        return None
    
    try:
        # Construct the URL for fetching driver info
        driver_info_url = f"{server_url.rstrip('/')}/api/driver/{driver_id}/info"
        
        # Prepare headers - only add Authorization if API key is provided
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        # Send the request
        response = requests.get(
            driver_info_url,
            headers=headers,
            timeout=5
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'success' in data and data['success'] and 'driver' in data:
                return data['driver']
            else:
                print("Invalid response format from server when fetching driver info")
                return None
        else:
            print(f"Error fetching driver info: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error in get_driver_info: {e}")
        return None

def authenticate_driver(frame, reference_images, detector):
    """Authenticate the driver against reference images"""
    if not reference_images:
        return None
    
    try:
        # Use the detector to find faces in the current frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        is_low_light = False
        faces = detector.detect_faces_with_timeout(gray, is_low_light)
        
        if not faces:
            print("No face detected for authentication")
            return None
        
        # Use the largest face for authentication
        largest_face = max(faces, key=lambda face: face.width() * face.height())
        face_roi = gray[largest_face.top():largest_face.bottom(), largest_face.left():largest_face.right()]
        
        # Calculate histogram for the current face
        curr_hist = detector._calculate_face_histogram(face_roi)
        
        if curr_hist is None:
            print("Could not calculate histogram for current face")
            return None
        
        # Compare with all reference images
        best_match = None
        best_correlation = -1
        best_username = None
        
        for driver_id, ref_data in reference_images.items():
            # Skip entries without valid images
            if 'image' not in ref_data:
                continue
                
            try:
                # Convert reference image to grayscale
                ref_gray = cv2.cvtColor(ref_data['image'], cv2.COLOR_BGR2GRAY)
                
                # Detect faces in reference image
                ref_faces = detector.detect_faces_with_timeout(ref_gray, is_low_light)
                
                if ref_faces:
                    # Use the largest face in reference image
                    ref_largest = max(ref_faces, key=lambda face: face.width() * face.height())
                    ref_face_roi = ref_gray[ref_largest.top():ref_largest.bottom(), 
                                            ref_largest.left():ref_largest.right()]
                    
                    # Calculate histogram for reference face
                    ref_hist = detector._calculate_face_histogram(ref_face_roi)
                    
                    if ref_hist is not None:
                        # Compare histograms using correlation
                        correlation = cv2.compareHist(
                            curr_hist,
                            ref_hist,
                            cv2.HISTCMP_CORREL
                        )
                        
                        print(f"Driver {driver_id} ({ref_data.get('username', 'Unknown')}) correlation: {correlation:.2f}")
                        
                        # Keep track of best match
                        if correlation > best_correlation:
                            best_correlation = correlation
                            best_match = driver_id
                            best_username = ref_data.get('username', f"Driver {driver_id}")
                
            except Exception as e:
                print(f"Error comparing with driver {driver_id}: {e}")
        
        # Determine if we have a good match
        if best_correlation > 0.45:  # Same threshold used in drowsiness_detector.py
            print(f"Authentication successful with correlation {best_correlation:.2f}")
            return {
                'id': best_match,
                'username': best_username,
                'correlation': best_correlation
            }
        else:
            print(f"Best correlation {best_correlation:.2f} below threshold (0.45)")
            return None
            
    except Exception as e:
        print(f"Error in authenticate_driver: {e}")
        return None

def send_direct_risk_update(server_url, driver_id, risk_level, risk_label, api_key=None):
    """Send a direct risk level update to the server's risk_level endpoint"""
    if not server_url or not driver_id:
        return False

    try:
        # Construct the risk update URL
        risk_update_url = f"{server_url.rstrip('/')}/api/update_risk_level/{driver_id}"

        # Prepare headers - only add Authorization if API key is provided
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'

        # Prepare payload
        payload = {
            'risk_level': risk_level,
            'risk_label': risk_label
        }

        # Retry logic
        max_retries = 3
        timeout_duration = 10  # Increased timeout duration
        for attempt in range(max_retries):
            try:
                # Send the request
                response = requests.post(
                    risk_update_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout_duration
                )

                if response.status_code == 200:
                    print(f"Successfully sent direct risk level update: {risk_label}")
                    return True
                else:
                    print(f"Error sending risk level update (attempt {attempt + 1}/{max_retries}): {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Request exception on attempt {attempt + 1}/{max_retries}: {e}")

            # Wait before retrying
            time.sleep(2)

        print("Failed to send risk level update after multiple attempts.")
        return False

    except Exception as e:
        print(f"Error sending direct risk level update: {e}")
        return False

def check_suspension_status(server_url, driver_id, api_key=None):
    """Check if the driver is suspended"""
    try:
        headers = {}
        if api_key:
            headers['X-API-Key'] = api_key
            
        response = requests.get(f"{server_url}/api/check_suspension/{driver_id}", headers=headers)
        if response.status_code == 200:
            data = response.json()
            return {
                'suspended': data.get('suspended', False),
                'message': data.get('message', '')
            }
        return {'suspended': False, 'message': ''}
    except Exception as e:
        print(f"Error checking suspension status: {str(e)}")
        return {'suspended': False, 'message': ''}

def create_suspension_message(message):
    """Create an image with the suspension message"""
    # Create a black background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add red background for warning
    cv2.rectangle(frame, (0, 0), (640, 480), (0, 0, 150), -1)
    
    # Add suspension text
    cv2.putText(frame, "DRIVER SUSPENDED", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add the message - handle multi-line messages
    y_pos = 200
    for line in message.split('\n'):
        # Wrap long lines
        words = line.split(' ')
        line_parts = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            # Check if we need to wrap
            if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] > 600:
                line_parts.append(current_line)
                current_line = word + " "
            else:
                current_line = test_line
                
        if current_line:
            line_parts.append(current_line)
            
        for part in line_parts:
            cv2.putText(frame, part, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
            
    # Add instruction to contact supervisor
    cv2.putText(frame, "Contact your supervisor for assistance", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return frame

def create_unsuspension_message(message):
    """Create an image with the unsuspension message"""
    # Create a black background
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Add green background for success
    cv2.rectangle(frame, (0, 0), (640, 480), (0, 150, 0), -1)
    
    # Add unsuspension text
    cv2.putText(frame, "DRIVER UNSUSPENDED", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add the message - handle multi-line messages
    y_pos = 200
    for line in message.split('\n'):
        # Wrap long lines as in the suspension message function
        words = line.split(' ')
        line_parts = []
        current_line = ""
        for word in words:
            test_line = current_line + word + " "
            if cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] > 600:
                line_parts.append(current_line)
                current_line = word + " "
            else:
                current_line = test_line
                
        if current_line:
            line_parts.append(current_line)
            
        for part in line_parts:
            cv2.putText(frame, part, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos += 30
    
    # Add instruction to resume driving
    cv2.putText(frame, "Monitoring will resume shortly", (150, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return frame

if __name__ == "__main__":
    main()
