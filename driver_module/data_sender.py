"""
Data Sender Module
Handles sending detection data to the central server
"""
import requests
import json
import threading
import time
from datetime import datetime
from queue import Queue

class DataSender:
    """Class for sending detection data to the server"""
    
    def __init__(self, server_url=None, api_key=None):
        """
        Initialize the DataSender with server URL and optional API key
        
        Args:
            server_url: Base URL of the central server
            api_key: Optional API key for authentication
        """
        self.server_url = server_url
        self.api_key = api_key
        self.endpoint = f"{server_url.rstrip('/')}/api/driver_alerts"
        self.headers = {
            'Content-Type': 'application/json'
        }
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
            
        # Queue for storing detection events
        self.event_queue = Queue()
        self.max_batch_size = 5
        
        # Start the background sender thread
        self.sender_thread = threading.Thread(target=self._background_sender, daemon=True)
        self.sender_thread.start()
    
    def send_detection_data(self, detection_data, driver_id):
        """Send a single detection event to the server"""
        if not self.server_url:
            print("No server URL configured")
            return False
        
        try:
            payload = {
                'driver_id': driver_id,
                'detection_events': [detection_data]
            }
            
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"Successfully sent {detection_data['type']} event to server")
                return True
            else:
                print(f"Error sending data: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"Error sending data: {e}")
            return False
    
    def send_batch_detection_data(self, detection_events, driver_id):
        """Send multiple detection events in a single request"""
        if not self.server_url:
            print("No server URL configured")
            return False
        
        if not detection_events:
            return True  # Nothing to send
            
        try:
            payload = {
                'driver_id': driver_id,
                'detection_events': detection_events
            }
            
            response = requests.post(
                self.endpoint,
                headers=self.headers,
                json=payload,
                timeout=10  # Longer timeout for batch requests
            )
            
            if response.status_code == 200:
                print(f"Successfully sent batch of {len(detection_events)} events to server")
                return True
            else:
                print(f"Error sending batch data: {response.status_code}")
                print(response.text)
                return False
                
        except Exception as e:
            print(f"Error sending batch data: {e}")
            return False
    
    def _background_sender(self):
        """Background thread that sends batched events to the server"""
        batch = []
        last_send = time.time()
        
        while True:
            try:
                # Wait for an item but with a timeout to check batch conditions
                try:
                    item = self.event_queue.get(timeout=1.0)
                    batch.append(item['data'])
                except:
                    # Timeout - check if we should send what we have
                    pass
                
                # Send batch if it's full or if enough time has passed
                current_time = time.time()
                if (len(batch) >= self.max_batch_size or 
                   (len(batch) > 0 and current_time - last_send > 10)):
                    
                    if len(batch) > 0:
                        driver_id = item['driver_id']  # Use the last item's driver ID
                        self.send_batch_detection_data(batch, driver_id)
                        batch = []
                        last_send = current_time
            
            except Exception as e:
                print(f"Error in background sender: {e}")
                # Sleep a bit to avoid tight loops on errors
                time.sleep(1)
