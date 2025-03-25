from flask import Flask, render_template, request, redirect, url_for, session, Response, g, jsonify
import mysql.connector
import cv2
import numpy as np
import imagehash
from PIL import Image
import pyttsx3
import os
import io
import base64
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import time
import threading

import os
import sys
# Add the path so we can import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__, 
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'),
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'))
app.secret_key = 'drowsiness_detection_secret_key'

# Database connection
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="root",
        database="distracted_driver"
    )

# Routes for user authentication
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user_type = request.form['user_type']
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return render_template('register.html', error="Username already exists")
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        # Insert new user
        cursor.execute(
            "INSERT INTO users (username, password, email, user_type) VALUES (%s, %s, %s, %s)",
            (username, hashed_password, email, user_type)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if user and check_password_hash(user['password'], password):
            # Store user_id (which is the DRIVER_ID for drivers) in session
            session['user_id'] = user['id']  # This is the DRIVER_ID when user is a driver
            session['username'] = user['username']
            session['user_type'] = user['user_type']
            
            # Check if this is a driver who needs to provide a reference photo
            if user['user_type'] == 'driver':
                # Check if reference image exists and if face recognition is required
                require_face = user.get('require_face_recognition', True)  # Default to True if column doesn't exist
                has_image = user.get('reference_image') is not None
                
                if require_face and not has_image:
                    # Redirect to capture reference photo
                    return redirect(url_for('capture_reference'))
            
            if user['user_type'] == 'owner':
                return redirect(url_for('owner_dashboard'))
            else:
                return redirect(url_for('driver_dashboard'))
        else:
            return render_template('login.html', error="Invalid username or password")
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# Dashboard routes
@app.route('/owner/dashboard')
def owner_dashboard():
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get all drivers associated with this owner
    cursor.execute("""
        SELECT u.id, u.username, u.email 
        FROM users u
        JOIN driver_owner do ON u.id = do.driver_id
        WHERE do.owner_id = %s AND u.user_type = 'driver'
    """, (session['user_id'],))
    
    drivers = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('owner_dashboard.html', drivers=drivers)

@app.route('/driver/dashboard')
def driver_dashboard():
    if 'user_id' not in session or session['user_type'] != 'driver':
        return redirect(url_for('login'))
    
    return render_template('driver_dashboard.html')

# Driver registration and reference photo capture
@app.route('/register_driver', methods=['GET', 'POST'])
def register_driver():
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        driver_username = request.form['username']
        driver_password = request.form['password']
        driver_email = request.form['email']
        
        # Get optional fields
        first_name = request.form.get('firstName', '')
        last_name = request.form.get('lastName', '')
        phone_number = request.form.get('phoneNumber', '')
        
        # Get face recognition settings
        require_face_recognition = 'requireFaceRecognition' in request.form
        face_recognition_option = request.form.get('faceRecognitionOption', 'onLogin')
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Check if username already exists
        cursor.execute("SELECT * FROM users WHERE username = %s", (driver_username,))
        if cursor.fetchone():
            cursor.close()
            conn.close()
            return render_template('register_driver.html', error="Username already exists")
        
        # Process reference photo if uploaded
        reference_image = None
        if require_face_recognition and face_recognition_option == 'upload':
            file = request.files.get('referencePhoto')
            if file and file.filename:
                try:
                    # Read file data
                    img_data = file.read()
                    
                    # Convert to base64 for database storage
                    reference_image = base64.b64encode(img_data).decode('utf-8')
                    
                except Exception as e:
                    print(f"Error processing reference photo: {e}")
                    return render_template('register_driver.html', error=f"Error processing photo: {str(e)}")
            elif face_recognition_option == 'upload':
                # If upload option was selected but no file was provided
                return render_template('register_driver.html', error="Please select a reference photo file or choose 'Capture on first login'")
        
        # Insert new driver
        hashed_password = generate_password_hash(driver_password)
        
        try:
            query = (
                "INSERT INTO users "
                "(username, password, email, user_type, first_name, last_name, " 
                "phone_number, require_face_recognition, reference_image) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            )
            values = (
                driver_username, 
                hashed_password, 
                driver_email, 
                'driver',
                first_name,
                last_name,
                phone_number,
                require_face_recognition,
                reference_image
            )
            cursor.execute(query, values)
        except mysql.connector.Error as err:
            # If error is due to missing columns, add them
            if "Unknown column" in str(err):
                try:
                    # Add missing columns
                    cursor.execute("ALTER TABLE users ADD COLUMN first_name VARCHAR(50)")
                    cursor.execute("ALTER TABLE users ADD COLUMN last_name VARCHAR(50)")
                    cursor.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)")
                    cursor.execute("ALTER TABLE users ADD COLUMN require_face_recognition BOOLEAN DEFAULT TRUE")
                    if "reference_image" not in str(err):
                        cursor.execute("ALTER TABLE users ADD COLUMN reference_image LONGTEXT")
                    
                    # Try insert again
                    cursor.execute(query, values)
                except Exception as e:
                    # If still fails, use the original insert without the new columns
                    cursor.execute(
                        "INSERT INTO users (username, password, email, user_type) VALUES (%s, %s, %s, %s)",
                        (driver_username, hashed_password, driver_email, 'driver')
                    )
        
        conn.commit()
        
        # Get the new driver ID
        driver_id = cursor.lastrowid
        
        # Associate driver with owner
        cursor.execute(
            "INSERT INTO driver_owner (driver_id, owner_id) VALUES (%s, %s)",
            (driver_id, session['user_id'])
        )
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return redirect(url_for('owner_dashboard'))
    
    return render_template('register_driver.html')

# Fix the import path for the drowsiness detection module
# Change this line:
# from driver_module.advanced_detection import detect_drowsiness_in_feed

# To:
import sys
import os
# Add the project root to the path so we can import from any module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Define the detection function directly to avoid circular imports
def detect_drowsiness_in_feed(user_id=None, db_connection_func=None):
    """Generate video frames with drowsiness detection"""
    # Import here to avoid circular imports
    from driver_module.drowsiness_detector import DrowsinessDetector
    import cv2
    import time
    from datetime import datetime
    
    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        yield None
        return
    
    try:
        # Get reference image for user if available
        reference_image = None
        if user_id and db_connection_func:
            try:
                conn = db_connection_func()
                cursor = conn.cursor()
                cursor.execute("SELECT reference_image FROM users WHERE id = %s", (user_id,))
                result = cursor.fetchone()
                if result and result[0]:
                    reference_image = result[0]
                cursor.close()
                conn.close()
            except Exception as e:
                print(f"Error retrieving reference image: {e}")
        
        # Initialize our improved drowsiness detector with reference image
        detector = DrowsinessDetector(reference_image)
        print("Advanced drowsiness detector initialized")
        
        # Record when last alert was triggered to avoid alert spam
        last_alert_time = time.time()
        alert_cooldown = 10  # seconds between alerts
        
        # Track yawning separately to avoid mixing with drowsiness
        last_yawn_alert_time = time.time()
        yawn_alert_cooldown = 20  # longer cooldown for yawning alerts
        yawn_confidence = 0  # confidence level for yawning detection
        
        # Track risk level history
        risk_history = []
        last_risk_update = time.time()
        
        while True:
            # Capture frame
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame from camera")
                camera.release()
                camera = cv2.VideoCapture(0)
                time.sleep(1)
                continue
            
            # Process frame with advanced detection
            try:
                processed_frame, alert_triggered, alert_type = detector.process_frame(frame)
                
                # Get risk level
                risk_level = detector.risk_level
                
                # Make risk level more prominent in the UI
                risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
                risk_colors = [(0, 255, 0), (0, 255, 255), (0, 165, 255), (0, 0, 255)]
                
                # Add risk indicator
                cv2.rectangle(processed_frame, 
                              (processed_frame.shape[1] - 200, 10), 
                              (processed_frame.shape[1] - 10, 50), 
                              risk_colors[risk_level], -1)
                cv2.putText(processed_frame, f"RISK: {risk_labels[risk_level]}", 
                           (processed_frame.shape[1] - 190, 35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Store risk level in database periodically
                current_time = time.time()
                if db_connection_func and user_id and (current_time - last_risk_update) > 60:
                    try:
                        conn = db_connection_func()
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE users SET risk_level = %s WHERE id = %s",
                            (risk_level, user_id)
                        )
                        conn.commit()
                        cursor.close()
                        conn.close()
                        last_risk_update = current_time
                    except Exception as e:
                        print(f"Error updating risk level: {e}")
                
                # Check if we should trigger an alert
                should_alert = False
                if alert_triggered:
                    if alert_type == "yawning":
                        yawn_confidence += 1
                        if yawn_confidence >= 3 and (current_time - last_yawn_alert_time) > yawn_alert_cooldown:
                            should_alert = True
                            last_yawn_alert_time = current_time
                            yawn_confidence = 0
                    elif alert_type == "drowsiness" and (current_time - last_alert_time) > alert_cooldown:
                        should_alert = True
                        last_alert_time = current_time
                else:
                    yawn_confidence = max(0, yawn_confidence - 0.5)
                        
                if should_alert:
                    # Record alert in database
                    if db_connection_func and user_id:
                        try:
                            conn = db_connection_func()
                            cursor = conn.cursor()
                            cursor.execute(
                                "INSERT INTO alerts (user_id, alert_type, timestamp) VALUES (%s, %s, %s)",
                                (user_id, alert_type, datetime.now())
                            )
                            conn.commit()
                            cursor.close()
                            conn.close()
                            print(f"{alert_type.capitalize()} alert recorded for user {user_id}")
                        except Exception as e:
                            print(f"Error recording alert: {e}")
                    
                    # Alert notification (voice or visual)
                    # Non-blocking alert that doesn't require pyttsx3 (which can cause threading issues)
                    if os.name == 'nt':  # Windows
                        os.system(f'start /min cmd /c "echo Alert! {alert_type} detected! && timeout /t 1"')
                    else:  # Linux/Mac
                        os.system(f'echo "Alert! {alert_type} detected!" | espeak 2>/dev/null &')
                    
            except Exception as e:
                print(f"Error in drowsiness detection: {e}")
                cv2.putText(frame, f"Detection error: {str(e)[:50]}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                processed_frame = frame
            
            # Add user_id to frame
            if user_id:
                cv2.putText(processed_frame, f"User ID: {user_id}", 
                           (10, processed_frame.shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Convert frame to JPEG for streaming
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
    except Exception as e:
        print(f"Fatal error in detection: {e}")
    finally:
        camera.release()

# Video stream and drowsiness detection
def generate_frames(user_id=None):
    """Generate video frames with drowsiness detection"""
    # This function is called outside request context, so we need to pass user_id
    print(f"Starting video feed for user ID: {user_id}")
    
    # Use our advanced detection module that handles the camera, detection, and yields frames
    return detect_drowsiness_in_feed(user_id=user_id, db_connection_func=get_db_connection)

@app.route('/video_feed')
def video_feed():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Extract user_id from session before entering the generator function
    user_id = session.get('user_id')
    
    # Pass the user_id directly to generate_frames
    return Response(generate_frames(user_id=user_id), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_reference', methods=['GET', 'POST'])
def capture_reference():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Get the captured image from the form
        image_data = request.form['image_data']
        
        # Ensure we have image data
        if not image_data or 'data:image' not in image_data:
            return render_template('capture_reference.html', error="No image data provided. Please capture a photo.")
        
        try:
            # Remove the prefix from base64 string (e.g., 'data:image/jpeg;base64,')
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Save to database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "UPDATE users SET reference_image = %s WHERE id = %s",
                (image_data, session['user_id'])
            )
            conn.commit()
            cursor.close()
            conn.close()
            
            # Redirect based on user type
            if session.get('user_type') == 'driver':
                return redirect(url_for('driver_dashboard'))
            else:
                return redirect(url_for('owner_dashboard'))
                
        except Exception as e:
            print(f"Error processing reference image: {e}")
            return render_template('capture_reference.html', error=f"Error saving image: {str(e)}")
    
    return render_template('capture_reference.html')

@app.route('/view_alerts')
def view_alerts():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Get filter parameters
    date_filter = request.args.get('date', '')
    driver_filter = request.args.get('driver', '')
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Get all drivers for owner's dropdown filter
    drivers = []
    if session['user_type'] == 'owner':
        cursor.execute("""
            SELECT u.id, u.username
            FROM users u
            JOIN driver_owner do ON u.id = do.driver_id
            WHERE do.owner_id = %s AND u.user_type = 'driver'
        """, (session['user_id'],))
        drivers = cursor.fetchall()
    
    # Base query
    if session['user_type'] == 'owner':
        query = """
            SELECT a.*, u.username 
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            JOIN driver_owner do ON u.id = do.driver_id
            WHERE do.owner_id = %s
        """
        params = [session['user_id']]

        # Add filters
        if date_filter:
            query += " AND DATE(a.timestamp) = %s"
            params.append(date_filter)
        if driver_filter:
            query += " AND u.id = %s"
            params.append(int(driver_filter))  # Convert to int to prevent injection
        
        query += " ORDER BY a.timestamp DESC"
    else:
        # Fix: Use alias for alerts table in the driver's query too
        query = """
            SELECT a.*, u.username 
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id = %s
        """
        params = [session['user_id']]
        
        # Add filters
        if date_filter:
            query += " AND DATE(a.timestamp) = %s"
            params.append(date_filter)
        
        query += " ORDER BY a.timestamp DESC"
    
    cursor.execute(query, tuple(params))
    alerts = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('view_alerts.html', alerts=alerts, drivers=drivers, request=request)

# API routes for AJAX calls
@app.route('/api/recent_alerts')
def api_recent_alerts():
    if 'user_id' not in session:
        return {"error": "Unauthorized"}, 401
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    if session['user_type'] == 'owner':
        # Get alerts for all drivers associated with this owner
        cursor.execute("""
            SELECT a.*, u.username 
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            JOIN driver_owner do ON u.id = do.driver_id
            WHERE do.owner_id = %s
            ORDER BY a.timestamp DESC
            LIMIT 5
        """, (session['user_id'],))
    else:
        # Get alerts for current driver
        cursor.execute("""
            SELECT * FROM alerts 
            WHERE user_id = %s
            ORDER BY timestamp DESC 
            LIMIT 5
        """, (session['user_id'],))
    
    alerts = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Convert datetime objects to strings for JSON serialization 
    for alert in alerts:
        alert['timestamp'] = alert['timestamp'].isoformat()
    
    return {"alerts": alerts}

@app.route('/api/alert_count')
def api_alert_count():
    if 'user_id' not in session:
        return {"error": "Unauthorized"}, 401
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    if session['user_type'] == 'owner':
        # Count alerts for all drivers associated with this owner
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM alerts a
            JOIN driver_owner do ON a.user_id = do.driver_id
            WHERE do.owner_id = %s
        """, (session['user_id'],))
    else:
        # Count alerts for current driver
        cursor.execute("""
            SELECT COUNT(*) as count 
            FROM alerts 
            WHERE user_id = %s
        """, (session['user_id'],))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return {"count": result['count']}

@app.route('/api/alert/<int:alert_id>')
def api_alert_details(alert_id):
    if 'user_id' not in session:
        return {"error": "Unauthorized", "success": False}, 401
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    if session['user_type'] == 'owner':
        # Get alert details for owner, ensuring it belongs to one of their drivers
        cursor.execute("""
            SELECT a.*, u.username 
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            JOIN driver_owner do ON u.id = do.driver_id
            WHERE a.id = %s AND do.owner_id = %s
        """, (alert_id, session['user_id']))
    else:
        # Get alert details for driver, ensuring it belongs to them
        cursor.execute("""
            SELECT a.*, u.username 
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            WHERE a.id = %s AND a.user_id = %s
        """, (alert_id, session['user_id']))
    
    alert = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if alert:
        # Convert datetime objects to strings for JSON serialization
        alert['timestamp'] = alert['timestamp'].isoformat()
        return {"success": True, "alert": alert}
    else:
        return {"success": False, "error": "Alert not found"}

@app.route('/api/export_alerts')
def api_export_alerts():
    if 'user_id' not in session:
        return "Unauthorized", 401
    
    import csv
    from io import StringIO
    from flask import make_response
    
    # Get filter parameters
    date_filter = request.args.get('date', '')
    driver_filter = request.args.get('driver', '')
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Base query
    if session['user_type'] == 'owner':
        query = """
            SELECT a.id, a.alert_type, a.timestamp, u.username, u.email
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            JOIN driver_owner do ON u.id = do.driver_id
            WHERE do.owner_id = %s
        """
        params = [session['user_id']]
    else:
        query = """
            SELECT a.id, a.alert_type, a.timestamp, u.username, u.email
            FROM alerts a
            JOIN users u ON a.user_id = u.id
            WHERE a.user_id = %s
        """
        params = [session['user_id']]
    
    # Add filters
    if date_filter:
        query += " AND DATE(a.timestamp) = %s"
        params.append(date_filter)
    if driver_filter and session['user_type'] == 'owner':
        query += " AND u.id = %s"
        params.append(driver_filter)
    
    query += " ORDER BY a.timestamp DESC"
    
    cursor.execute(query, tuple(params))
    alerts = cursor.fetchall()
    cursor.close()
    conn.close()
    
    # Create CSV file
    output = StringIO()
    csv_writer = csv.writer(output)
    
    # Write header
    csv_writer.writerow(['Alert ID', 'Type', 'Timestamp', 'Driver', 'Email'])
    
    # Write data
    for alert in alerts:
        csv_writer.writerow([
            alert['id'],
            alert['alert_type'],
            alert['timestamp'],
            alert['username'],
            alert['email']
        ])
    
    # Create response
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=alerts_export.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/driver_alerts/<int:driver_id>')
def driver_alerts(driver_id):
    """View all alerts for a specific driver"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
    
    # Verify relationship between owner and driver
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner 
            WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or not result['is_associated']:
        cursor.close()
        conn.close()
        return redirect(url_for('owner_dashboard'))
    
    # Get driver info
    cursor.execute("SELECT username FROM users WHERE id = %s", (driver_id,))
    driver = cursor.fetchone()
    
    # Get all alerts for this driver
    cursor.execute("""
        SELECT * FROM alerts
        WHERE user_id = %s
        ORDER BY timestamp DESC    
    """, (driver_id,))
    alerts = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return render_template('driver_alerts.html', driver=driver, alerts=alerts, driver_id=driver_id)

@app.route('/reset_password/<int:driver_id>', methods=['GET', 'POST'])
def reset_password(driver_id):
    """Reset password for a driver"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
    
    # Verify relationship between owner and driver
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner 
            WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or not result['is_associated']:
        cursor.close()
        conn.close()
        return redirect(url_for('owner_dashboard'))
    
    # Get driver info
    cursor.execute("SELECT username FROM users WHERE id = %s", (driver_id,))
    driver = cursor.fetchone()
    
    if request.method == 'POST':
        new_password = request.form['new_password']
        
        # Update password
        hashed_password = generate_password_hash(new_password)
        cursor.execute(
            "UPDATE users SET password = %s WHERE id = %s",
            (hashed_password, driver_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        # Redirect back to driver view with success message
        return redirect(url_for('view_driver', driver_id=driver_id))
    
    cursor.close()
    conn.close()
    
    return render_template('reset_password.html', driver=driver, driver_id=driver_id)

@app.route('/delete_driver/<int:driver_id>', methods=['POST'])
def delete_driver(driver_id):
    """Delete a driver from the system"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
    
    # Verify relationship between owner and driver
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner 
            WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or result[0] == 0:  # Fixed condition: should check if relationship DOESN'T exist
        cursor.close()
        conn.close()
        return redirect(url_for('owner_dashboard'))
    
    # Delete the driver-owner relationship first
    cursor.execute("DELETE FROM driver_owner WHERE driver_id = %s", (driver_id,))
    conn.commit()
    
    # Then delete the user (driver)
    cursor.execute("DELETE FROM users WHERE id = %s", (driver_id,))
    conn.commit()
    cursor.close()
    conn.close()
    
    return redirect(url_for('owner_dashboard'))

@app.route('/view_driver/<int:driver_id>')
def view_driver(driver_id):
    """View detailed information about a specific driver"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Verify that this driver is associated with the logged-in owner
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner 
            WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or not result['is_associated']:
        cursor.close()
        conn.close()
        return redirect(url_for('owner_dashboard'))
    
    # Get driver details
    cursor.execute("""
        SELECT u.id, u.username, u.email, u.registration_date
        FROM users u
        WHERE u.id = %s AND u.user_type = 'driver'
    """, (driver_id,))
    driver = cursor.fetchone()
    
    # Get recent alerts for this driver
    cursor.execute("""
        SELECT id, alert_type, timestamp
        FROM alerts
        WHERE user_id = %s
        ORDER BY timestamp DESC
        LIMIT 10
    """, (driver_id,))
    alerts = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return render_template('view_driver.html', driver=driver, alerts=alerts)

@app.route('/api/risk_level/<int:driver_id>')
def api_risk_level(driver_id):
    """Get the current risk level for a driver"""
    if 'user_id' not in session:
        return {"error": "Unauthorized", "success": False}, 401
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Check permissions
    if session['user_type'] == 'owner':
        # Owner can check any of their drivers
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM driver_owner 
                WHERE driver_id = %s AND owner_id = %s
            ) as is_associated
        """, (driver_id, session['user_id']))
        
        result = cursor.fetchone()
        if not result or not result['is_associated']:
            cursor.close()
            conn.close()
            return {"error": "Not authorized to view this driver", "success": False}, 403
    elif session['user_type'] == 'driver':
        # Drivers can only check themselves
        if driver_id != session['user_id']:
            return {"error": "Not authorized to view this driver", "success": False}, 403
    
    # Get risk level from database
    cursor.execute("SELECT risk_level FROM users WHERE id = %s", (driver_id,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if not result:
        return {"error": "Driver not found", "success": False}, 404
    
    # Map numeric levels to descriptions
    risk_levels = ["Low", "Medium", "High", "Critical"]
    risk_level = result['risk_level'] if result['risk_level'] is not None else 0
    
    return {
        "success": True,
        "risk_level": risk_level,
        "risk_label": risk_levels[min(risk_level, 3)]
    }

# Add new API endpoint for driver module authentication
@app.route('/api/driver/authenticate', methods=['POST'])
def api_driver_authenticate():
    """API endpoint for driver module to authenticate a driver"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if not username or not password:
            return jsonify({'success': False, 'error': 'Username and password required'})
        
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s AND user_type = 'driver'", (username,))
        driver = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        if driver and check_password_hash(driver['password'], password):
            return jsonify({
                'success': True,
                'driver_id': driver['id'],
                'username': driver['username']
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid credentials'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Add a new route to manage driver modules
@app.route('/manage_driver_modules')
def manage_driver_modules():
    """Admin page to manage driver modules"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))
        
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get all drivers associated with this owner
        cursor.execute("""
            SELECT u.id, u.username, u.email, 
                   COALESCE(u.risk_level, 0) as risk_level,
                   (SELECT MAX(timestamp) FROM alerts WHERE user_id = u.id) as last_alert
            FROM users u
            JOIN driver_owner do ON u.id = do.driver_id
            WHERE do.owner_id = %s AND u.user_type = 'driver'
        """, (session['user_id'],))
            
        drivers = cursor.fetchall()
    except mysql.connector.Error as err:
        print(f"Database error: {err}")
        drivers = []
        
    cursor.close()
    conn.close()
    
    return render_template('manage_driver_modules.html', drivers=drivers)

# Add endpoint to generate API key for driver modules
@app.route('/api/generate_driver_key/<int:driver_id>', methods=['POST'])
def generate_driver_key(driver_id):
    """Generate a new API key for a driver module"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    # Verify relationship between owner and driver
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner 
            WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or not result['is_associated']:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'Not authorized for this driver'}), 403
    
    # Generate a new API key - in reality, use a more secure method
    import secrets
    api_key = secrets.token_urlsafe(32)
    
    # Store the API key in the database
    # For this example, we'll add an api_key column to the users table if needed
    try:
        cursor.execute("SHOW COLUMNS FROM users LIKE 'api_key'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE users ADD COLUMN api_key VARCHAR(64)")
        
        cursor.execute(
            "UPDATE users SET api_key = %s WHERE id = %s",
            (api_key, driver_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'api_key': api_key})
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

# Add endpoint to check driver module status
@app.route('/api/driver_module_status/<int:driver_id>')
def driver_module_status(driver_id):
    """Check the status of a driver module"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    
    # Check permissions
    if session['user_type'] == 'owner':
        # Owner can check any of their drivers
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM driver_owner 
                WHERE driver_id = %s AND owner_id = %s
            ) as is_associated
        """, (driver_id, session['user_id']))
        
        result = cursor.fetchone()
        if not result or not result['is_associated']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Not authorized for this driver'}), 403
    elif session['user_id'] != driver_id:
        # Drivers can only check themselves
        return jsonify({'success': False, 'error': 'Not authorized for this driver'}), 403
    
    # For this example, we'll check when the latest alert was recorded
    # In a real system, the driver module would periodically ping the server with its status
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT MAX(timestamp) as last_alert, 
               COUNT(*) as total_alerts,
               risk_level
        FROM alerts a
        JOIN users u ON a.user_id = u.id
        WHERE a.user_id = %s
        GROUP BY u.risk_level
    """, (driver_id,))
    
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    
    if result and result['last_alert']:
        last_alert_time = result['last_alert']
        # Calculate how long ago the last alert was
        time_diff = datetime.now() - last_alert_time
        minutes_ago = time_diff.total_seconds() / 60
        
        # Consider the module active if there was an alert in the last 15 minutes
        is_active = minutes_ago < 15
        
        return jsonify({
            'success': True,
            'is_active': is_active,
            'last_alert': last_alert_time.isoformat(),
            'minutes_since_last_alert': round(minutes_ago, 1),
            'total_alerts': result['total_alerts'],
            'risk_level': result['risk_level']
        })
    else:
        return jsonify({
            'success': True,
            'is_active': False,
            'message': 'No alerts recorded'
        })

@app.route('/download_driver_module')
def download_driver_module():
    """Endpoint for downloading the driver module package"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    try:
        # Import the driver module creation function
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from download_driver import create_driver_module_zip
        import tempfile
        
        # Create a temporary file for the ZIP
        fd, temp_path = tempfile.mkstemp(suffix='.zip')
        os.close(fd)
        
        # Get server URL
        server_url = request.url_root.rstrip('/')
        
        # If user is a driver, include their ID in the config
        driver_id = None
        api_key = None
        if session.get('user_type') == 'driver':
            driver_id = session.get('user_id')
            
            # Generate an API key if needed
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            cursor.execute("SELECT api_key FROM users WHERE id = %s", (driver_id,))
            result = cursor.fetchone()
            
            if result and result.get('api_key'):
                api_key = result['api_key']
            else:
                # Generate a new API key
                import secrets
                api_key = secrets.token_urlsafe(32)
                
                # Check if api_key column exists
                cursor.execute("SHOW COLUMNS FROM users LIKE 'api_key'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE users ADD COLUMN api_key VARCHAR(64)")
                
                # Save the API key
                cursor.execute(
                    "UPDATE users SET api_key = %s WHERE id = %s",
                    (api_key, driver_id)
                )
                conn.commit()
            
            cursor.close()
            conn.close()
        
        # Create the ZIP file
        success = create_driver_module_zip(
            temp_path, 
            server_url=server_url,
            driver_id=driver_id,
            api_key=api_key
        )
        
        if success:
            # Create response with the file
            from flask import send_file
            return send_file(
                temp_path,
                as_attachment=True,
                download_name='driver_module.zip',
                mimetype='application/zip'
            )
        else:
            return render_template('error.html', error="Failed to create driver module package.")
            
    except Exception as e:
        print(f"Error creating driver module package: {e}")
        return render_template('error.html', error=f"Error: {str(e)}")

# Add new API endpoint for driver module to get reference image
@app.route('/api/driver_model/<int:driver_id>', methods=['GET'])
def get_driver_model(driver_id):
    """API endpoint to provide driver recognition model data"""
    # Validate API key if provided
    auth_header = request.headers.get('Authorization')
    if (auth_header and auth_header.startswith('Bearer ')):
        api_key = auth_header.split(' ')[1]
        # Verify API key matches the driver's key
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT api_key FROM users WHERE id = %s AND user_type = 'driver'", (driver_id,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not result or result.get('api_key') != api_key:
            return jsonify({'success': False, 'error': 'Invalid API key'}), 401
    
    try:
        # Get driver info and reference image
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        cursor.execute("""
            SELECT id, username, email, reference_image
            FROM users
            WHERE id = %s AND user_type = 'driver'
        """, (driver_id,))
        
        driver = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not driver or not driver.get('reference_image'):
            return jsonify({'success': False, 'error': 'Driver has no reference image'}), 404
        
        # Create model data from the reference image
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

# Add endpoint to receive alerts from driver modules
@app.route('/api/driver_alerts', methods=['POST'])
def receive_driver_alerts():
    """API endpoint to receive alert data from driver modules"""
    try:
        # Parse the data
        data = request.json
        driver_id = data.get('driver_id')
        detection_events = data.get('detection_events', [])
        
        if not driver_id:
            return jsonify({'success': False, 'error': 'Driver ID not provided'}), 400
        
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
                
                # Store image if provided (in a real system, we'd store to filesystem/database)
                if 'image' in event:
                    # Here we'd save the image - omitted for brevity
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

# Add new endpoint for risk level updates
@app.route('/api/update_risk_level/<int:driver_id>', methods=['POST'])
def update_risk_level(driver_id):
    """Endpoint for updating driver risk level data"""
    if not request.is_json:
        return jsonify({'success': False, 'error': 'Invalid request format'}), 400
    
    data = request.get_json()
    
    # Check for required fields
    if 'risk_level' not in data:
        return jsonify({'success': False, 'error': 'Missing risk_level field'}), 400
    
    try:
        # Store the risk level in the database
        risk_level = int(data['risk_level'])
        risk_label = data.get('risk_label', '')
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # First, verify the columns exist or create them
        # This is more robust than just checking if they exist
        try:
            # Use ALTER TABLE IGNORE to ignore errors if columns already exist
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS risk_level INT DEFAULT 0")
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS risk_label VARCHAR(20) DEFAULT 'LOW'")
            cursor.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS last_risk_update TIMESTAMP")
            conn.commit()
        except Exception as e:
            # If MySQL version doesn't support IF NOT EXISTS, catch the error and check manually
            try:
                # Check if columns exist one by one
                cursor.execute("SHOW COLUMNS FROM users LIKE 'risk_level'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE users ADD COLUMN risk_level INT DEFAULT 0")
                
                cursor.execute("SHOW COLUMNS FROM users LIKE 'risk_label'") 
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE users ADD COLUMN risk_label VARCHAR(20) DEFAULT 'LOW'")
                
                cursor.execute("SHOW COLUMNS FROM users LIKE 'last_risk_update'")
                if not cursor.fetchone():
                    cursor.execute("ALTER TABLE users ADD COLUMN last_risk_update TIMESTAMP")
                
                conn.commit()
            except Exception as inner_e:
                print(f"Error adding columns: {inner_e}")
                # Continue anyway, we'll try to update just the risk_level below
        
        try:
            # Update all columns if available
            cursor.execute('''
                UPDATE users 
                SET risk_level = %s, risk_label = %s, last_risk_update = %s 
                WHERE id = %s
            ''', (risk_level, risk_label, datetime.now().isoformat(), driver_id))
        except Exception as update_error:
            # If updating with risk_label fails, try without it
            if "risk_label" in str(update_error):
                print("Falling back to only updating risk_level")
                cursor.execute('''
                    UPDATE users 
                    SET risk_level = %s 
                    WHERE id = %s
                ''', (risk_level, driver_id))
            else:
                # If some other error, re-raise it
                raise update_error
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Risk level updated successfully'})
    
    except Exception as e:
        print(f"Error updating risk level: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# Fix the get_risk_level function as well
@app.route('/api/risk_level/<int:driver_id>', methods=['GET'])
def get_risk_level(driver_id):
    """Get current risk level for a driver"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # FIX: Use the correct table and use cursor.execute
        cursor.execute(
            'SELECT risk_level, risk_label FROM users WHERE id = %s', 
            (driver_id,)
        )
        
        driver = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if not driver:
            return jsonify({'success': False, 'error': 'Driver not found'}), 404
        
        # Use default values if fields are None
        risk_level = driver.get('risk_level', 0) if driver.get('risk_level') is not None else 0
        risk_label = driver.get('risk_label', 'LOW') if driver.get('risk_label') else 'LOW'
        
        return jsonify({
            'success': True,
            'risk_level': risk_level,
            'risk_label': risk_label
        })
        
    except Exception as e:
        print(f"Error getting risk level: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")