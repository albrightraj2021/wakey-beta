import cv2
import argparse
import time
from drowsiness_detector import DrowsinessDetector
from data_sender import DataSender
import requests
import json

def main():
    parser = argparse.ArgumentParser(description="Driver Monitoring Application")
    parser.add_argument("--server", type=str, help="Server URL", required=False)
    parser.add_argument("--driver-id", type=int, help="Driver ID", required=False)
    parser.add_argument("--api-key", type=str, help="API Key", required=False)
    args = parser.parse_args()

    detector = DrowsinessDetector()

    data_sender = None
    if args.server and args.driver_id:
        data_sender = DataSender(server_url=args.server, api_key=args.api_key)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, alert_detected, alert_type = detector.process_frame(frame)
        cv2.imshow("Driver Monitor", processed_frame)

        current_time = time.time()
        
        # Get current risk level
        risk_level = detector.risk_level
        risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        
        # Check if we should record this alert based on cooldown
        if alert_detected and data_sender and args.driver_id:
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
                
                # Send risk level update directly
                send_direct_risk_update(args.server, args.driver_id, risk_level, risk_labels[risk_level], args.api_key)
        
        # Send periodic risk level updates even when no alerts are detected
        if data_sender and args.driver_id and (current_time - last_risk_update_time) > risk_update_interval:
            # Always send updates to ensure risk level is current
            status_update = {
                "type": "status_update",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "risk_level": risk_level,
                "risk_label": risk_labels[risk_level]
            }
            queued_alerts.append(status_update)
            last_risk_update_time = current_time
            last_sent_risk_level = risk_level
            print(f"Status update queued with risk level: {risk_labels[risk_level]}")
            
            # Send risk level update directly and force update
            send_direct_risk_update(args.server, args.driver_id, risk_level, risk_labels[risk_level], args.api_key)
            
        # Send batched alerts periodically
        if queued_alerts and (current_time - last_send_time) > send_interval:
            if data_sender:
                print(f"Sending batch of {len(queued_alerts)} alerts/updates")
                data_sender.send_batch_detection_data(queued_alerts, args.driver_id)
                queued_alerts = []
                last_send_time = current_time

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Send any remaining alerts before exiting
    if queued_alerts and data_sender and args.driver_id:
        data_sender.send_batch_detection_data(queued_alerts, args.driver_id)
        
    cap.release()
    cv2.destroyAllWindows()

def send_direct_risk_update(server_url, driver_id, risk_level, risk_label, api_key=None):
    """Send a direct risk level update to the server's risk_level endpoint"""
    if not server_url or not driver_id:
        return False
    
    try:
        # Construct the risk update URL
        risk_update_url = f"{server_url.rstrip('/')}/api/update_risk_level/{driver_id}"
        
        # Prepare headers
        headers = {'Content-Type': 'application/json'}
        if api_key:
            headers['Authorization'] = f'Bearer {api_key}'
        
        # Prepare payload
        payload = {
            'risk_level': risk_level,
            'risk_label': risk_label
        }
        
        # Print debug info
        print(f"Sending risk update to {risk_update_url}")
        print(f"Payload: {payload}")
        
        # Send the request
        response = requests.post(
            risk_update_url,
            headers=headers,
            json=payload,
            timeout=3  # Short timeout for quick updates
        )
        
        # Print response for debugging
        print(f"Risk update response: {response.status_code}")
        print(f"Response body: {response.text}")
        
        if response.status_code == 200:
            print(f"Successfully sent direct risk level update: {risk_label}")
            
            # Verify the update was successful by checking the current risk level
            verify_url = f"{server_url.rstrip('/')}/api/risk_level/{driver_id}"
            verify_response = requests.get(verify_url, headers=headers)
            if verify_response.status_code == 200:
                print(f"Verification response: {verify_response.json()}")
            
            return True
        else:
            print(f"Error sending risk level update: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"Error sending direct risk level update: {e}")
        return False

if __name__ == "__main__":
    main()
