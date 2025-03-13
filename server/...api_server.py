"""
API Server for Driver Drowsiness Detection system
Handles API requests from remote driver modules
"""

from flask import Flask, request, jsonify, g
import time
import base64
import os
import json
from datetime import datetime
from werkzeug.security import check_password_hash

# Fix the import path for database connection
# Change this line:
# from server.app import get_db_connection

# To a path that doesn't assume the directory structure:
import sys
import os
# Add the server directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import get_db_connection

api = Flask(__name__)
api.secret_key = 'drowsiness_detection_api_key'

# API key validation
def validate_api_key():
    """Validate the API key in the request header"""
    auth_header = request.headers.get('Authorization')
    if not auth_header or not auth_header.startswith('Bearer '):
        return False
        
    # Extract token
    token = auth_header.split(' ')[1]
    
    # In a real application, this would validate against a database of API keys
    # For demonstration, we'll use a simple check
    valid_tokens = ['test_api_key', 'driver_module_key']
    
    return token in valid_tokens

# Driver validation
def validate_driver(driver_id, api_key=None):
    """Validate that the driver exists and is associated with the API key"""
    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Check if driver exists
    cursor.execute("SELECT * FROM users WHERE id = %s AND user_type = 'driver'", (driver_id,))
    driver = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return driver is not None

@api.route('/api/driver_alerts', methods=['POST'])
def receive_driver_alerts():
    """API endpoint to receive alert data from driver modules"""
    # Validate API key (optional)
    # if not validate_api_key():
    #     return jsonify({'success': False, 'error': 'Invalid API key'}), 401
        
    try:
        # Parse the data
        data = request.json
        driver_id = data.get('driver_id')
        detection_events = data.get('detection_events', [])
        
        if not driver_id:
            return jsonify({'success': False, 'error': 'Driver ID not provided'}), 400
        
        if not validate_driver(driver_id):
            return jsonify({'success': False, 'error': 'Invalid driver ID'}), 400
        
        if not detection_events:
            return jsonify({'success': True, 'message': 'No detection events to process'}), 200
        
        # Process each detection event
        processed_count = 0
        conn = get_db_connection()
        cursor = conn.cursor()
        
        for event in detection_events:
            try:
                # Extract data
                event_type = event.get('type', 'drowsiness')
                timestamp_str = event.get('timestamp')
                
                # Parse timestamp if provided, otherwise use current time
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    except:
                        timestamp = datetime.now()
                else:
                    timestamp = datetime.now()
                
                # Insert alert
                cursor.execute(
                    "INSERT INTO alerts (user_id, alert_type, timestamp) VALUES (%s, %s, %s)",
                    (driver_id, event_type, timestamp)
                )
                processed_count += 1
                
                # Store image if provided
                if 'image' in event:
                    # In a production system, we'd store this in the database or filesystem
                    pass
                    
            except Exception as e:
                print(f"Error processing event: {e}")
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({
            'success': True,
            'message': f'Processed {processed_count} detection events',
            'events_processed': processed_count
        }), 200
        
    except Exception as e:
        print(f"Error in driver_alerts endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@api.route('/api/driver_model/<int:driver_id>', methods=['GET'])
def get_driver_model(driver_id):
    """API endpoint to provide driver recognition model data"""
    # Validate API key (optional)
    # if not validate_api_key():
    #     return jsonify({'success': False, 'error': 'Invalid API key'}), 401
        
    try:
        if not validate_driver(driver_id):
            return jsonify({'success': False, 'error': 'Invalid driver ID'}), 400
        
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Get driver info and reference image
        cursor.execute("""
            SELECT id, username, email, reference_image
            FROM users
            WHERE id = %s AND user_type = 'driver'
        """, (driver_id,))
        
        driver = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not driver or not driver['reference_image']:
            return jsonify({'success': False, 'error': 'Driver has no reference image'}), 404
        
        # Create a simple model from the reference image
        # In a real system, this would be a proper face recognition model
        model_data = {
            'driver_id': driver['id'],
            'username': driver['username'],
            'reference_image': driver['reference_image'],  # Base64 image data
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify({
            'success': True,
            'model_data': model_data
        }), 200
        
    except Exception as e:
        print(f"Error in get_driver_model endpoint: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    # If run directly, start on port 5001 to avoid conflict with main app
    api.run(host='0.0.0.0', port=5001)