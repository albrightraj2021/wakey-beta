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
from datetime import datetime, timedelta
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

def ensure_columns_exist():
    """Ensure the required columns exist in the users table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check and add the 'suspended' column if it doesn't exist
        cursor.execute("SHOW COLUMNS FROM users LIKE 'suspended'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE users ADD COLUMN suspended BOOLEAN DEFAULT FALSE")
        
        # Check and add the 'suspension_message' column if it doesn't exist
        cursor.execute("SHOW COLUMNS FROM users LIKE 'suspension_message'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE users ADD COLUMN suspension_message TEXT")
        
        conn.commit()
    except Exception as e:
        print(f"Error ensuring columns exist: {e}")
    finally:
        cursor.close()
        conn.close()

# Call the function during app initialization
ensure_columns_exist()

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

    # Get all vehicles owned by this owner
    cursor.execute("""
        SELECT id, vehicle_name, license_plate
        FROM vehicles
        WHERE owner_id = %s
    """, (session['user_id'],))
    vehicles = cursor.fetchall()

    cursor.close()
    conn.close()
    
    return render_template('owner_dashboard.html', drivers=drivers, vehicles=vehicles)

@app.route('/driver/dashboard')
def driver_dashboard():
    if 'user_id' not in session or session['user_type'] != 'driver':
        return redirect(url_for('login'))
    
    # Fetch vehicle information for this driver
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # First check for active vehicle sessions
    cursor.execute("""
        SELECT vs.id, vs.vehicle_name, vs.license_plate, vs.session_start 
        FROM vehicle_sessions vs 
        WHERE vs.driver_id = %s AND vs.session_end IS NULL
        ORDER BY vs.session_start DESC 
        LIMIT 1
    """, (session['user_id'],))
    
    active_vehicle = cursor.fetchone()
    
    # If no active session, get the most recent vehicle used by this driver
    if not active_vehicle:
        cursor.execute("""
            SELECT vs.id, vs.vehicle_name, vs.license_plate, vs.session_start, vs.session_end
            FROM vehicle_sessions vs
            WHERE vs.driver_id = %s
            ORDER BY vs.session_start DESC
            LIMIT 1
        """, (session['user_id'],))
        recent_vehicle = cursor.fetchone()
        vehicle = recent_vehicle
    else:
        vehicle = active_vehicle
    
    cursor.close()
    conn.close()
    
    return render_template('driver_dashboard.html', vehicle=vehicle)

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
        
        # Process reference photo - handle both file upload and webcam capture
        reference_image = None
        if require_face_recognition:
            # Check for webcam-captured image data
            webcam_image_data = request.form.get('referenceImageData')
            if webcam_image_data:
                # Use the webcam image data directly (already base64 encoded)
                reference_image = webcam_image_data
            else:
                # Handle file upload
                file = request.files.get('referencePhoto')
                if file and file.filename:
                    try:
                        img_data = file.read()
                        reference_image = base64.b64encode(img_data).decode('utf-8')
                    except Exception as e:
                        return render_template('register_driver.html', error=f"Error processing photo: {str(e)}")
                elif face_recognition_option == 'upload':
                    return render_template('register_driver.html', error="Please capture a photo or select a reference photo file")
        
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
            if "Unknown column" in str(err):
                try:
                    cursor.execute("ALTER TABLE users ADD COLUMN first_name VARCHAR(50)")
                    cursor.execute("ALTER TABLE users ADD COLUMN last_name VARCHAR(50)")
                    cursor.execute("ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)")
                    cursor.execute("ALTER TABLE users ADD COLUMN require_face_recognition BOOLEAN DEFAULT TRUE")
                    if "reference_image" not in str(err):
                        cursor.execute("ALTER TABLE users ADD COLUMN reference_image LONGTEXT")
                    cursor.execute(query, values)
                except Exception as e:
                    cursor.execute(
                        "INSERT INTO users (username, password, email, user_type) VALUES (%s, %s, %s, %s)",
                        (driver_username, hashed_password, driver_email, 'driver')
                    )
        conn.commit()
        driver_id = cursor.lastrowid
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
    camera = cv2.VideoCapture(2)
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
                if db_connection_func and user_id and (current_time - last_risk_update) > 10:
                    try:
                        conn = db_connection_func()
                        cursor = conn.cursor()
                        cursor.execute(
                            "UPDATE users SET risk_level = %s, last_risk_update = %s WHERE id = %s",
                            (risk_level, datetime.now().isoformat(), user_id)
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
            import numpy as np
            import base64
            from io import BytesIO
            from PIL import Image
            
            # Extract the base64 part
            if ',' in image_data:
                image_data_b64 = image_data.split(',')[1]
            else:
                image_data_b64 = image_data
                
            # Decode and convert to OpenCV format
            img_bytes = base64.b64decode(image_data_b64)
            img = Image.open(BytesIO(img_bytes))
            img_np = np.array(img)
            
            # Use OpenCV for face detection
            gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
            
            if len(faces) == 0:
                return render_template('capture_reference.html', 
                                      error="No face detected in the image. Please try again with better lighting and positioning.")
            
            if len(faces) > 1:
                return render_template('capture_reference.html', 
                                      error="Multiple faces detected. Please ensure only your face is in the frame.")
            
            # Save to database - original image_data includes the MIME type prefix
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
            AND a.alert_type != 'status_update'
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
            AND a.alert_type != 'status_update'
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

@app.route('/update_reference/<int:driver_id>', methods=['POST'])
def update_reference(driver_id):
    """Update a driver's reference image from the owner dashboard"""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    # Verify owner-driver relationship
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or not result['is_associated']:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'Not authorized for this driver'}), 403

    data = request.get_json()
    image_data = data.get('image_data', '')
    if not image_data or 'data:image' not in image_data:
        return jsonify({'success': False, 'error': 'Invalid image data provided'}), 400

    try:
        import numpy as np
        import base64
        from io import BytesIO
        from PIL import Image

        # Extract base64 portion if present
        if ',' in image_data:
            image_data_b64 = image_data.split(',')[1]
        else:
            image_data_b64 = image_data

        img_bytes = base64.b64decode(image_data_b64)
        img = Image.open(BytesIO(img_bytes))
        img_np = np.array(img)

        # Use OpenCV for face detection
        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) == 0:
            return jsonify({'success': False, 'error': 'No face detected. Please try again.'}), 400
        if len(faces) > 1:
            return jsonify({'success': False, 'error': 'Multiple faces detected. Please capture only one face.'}), 400

        cursor.execute("UPDATE users SET reference_image = %s WHERE id = %s", (image_data, driver_id))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'message': 'Reference image updated successfully'})
    except Exception as e:
        print(f"Error processing reference image: {e}")
        return jsonify({'success': False, 'error': f"Error saving image: {str(e)}"}), 500

@app.route('/view_driver/<int:driver_id>')
def view_driver(driver_id):
    if 'user_id' not in session or session['user_type'] != 'owner':
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    # Check owner-driver relationship
    cursor.execute("""
        SELECT u.*, do.owner_id
        FROM users u
        JOIN driver_owner do ON u.id = do.driver_id
        WHERE u.id = %s AND do.owner_id = %s
    """, (driver_id, session['user_id']))
    driver = cursor.fetchone()
    if not driver:
        cursor.close()
        conn.close()
        return "Driver not found or not associated with this owner.", 404

    # Get recent alerts for this driver
    cursor.execute("""
        SELECT a.*
        FROM alerts a
        WHERE a.user_id = %s
        AND a.alert_type != 'status_update'
        ORDER BY a.timestamp DESC
        LIMIT 10
    """, (driver_id,))
    alerts = cursor.fetchall()

    # Get all vehicle assignments for this driver
    cursor.execute("""
        SELECT id, vehicle_name, license_plate, session_start, session_end
        FROM vehicle_sessions
        WHERE driver_id = %s
        ORDER BY session_start DESC
    """, (driver_id,))
    vehicle_assignments = cursor.fetchall()
    # Format datetimes for frontend
    for v in vehicle_assignments:
        if v.get('session_start'):
            v['session_start'] = v['session_start'].isoformat() if hasattr(v['session_start'], 'isoformat') else str(v['session_start'])
        if v.get('session_end'):
            v['session_end'] = v['session_end'].isoformat() if v['session_end'] and hasattr(v['session_end'], 'isoformat') else (str(v['session_end']) if v['session_end'] else None)

    cursor.close()
    conn.close()
    return render_template('view_driver.html', driver=driver, alerts=alerts, vehicle_assignments=vehicle_assignments)

# --- Vehicle assignment endpoints for scheduling ---
@app.route('/api/assign_vehicle', methods=['POST'])
def assign_vehicle():
    """Owner assigns a vehicle to a driver for a scheduled period."""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    data = request.json
    driver_id = data.get('driver_id')
    vehicle_name = data.get('vehicle_name')
    license_plate = data.get('license_plate')
    session_start = data.get('session_start')
    session_end = data.get('session_end')

    if not (driver_id and vehicle_name and license_plate and session_start and session_end):
        return jsonify({'success': False, 'error': 'Missing required fields'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT EXISTS(
            SELECT 1 FROM driver_owner WHERE driver_id = %s AND owner_id = %s
        ) as is_associated
    """, (driver_id, session['user_id']))
    result = cursor.fetchone()
    if not result or not result['is_associated']:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'Not authorized for this driver'}), 403

    # --- Prevent double-booking of the same vehicle ---
    # Check if the vehicle is already assigned to another driver during the requested period
    cursor.execute("""
        SELECT vs.id, vs.driver_id, vs.session_start, vs.session_end
        FROM vehicle_sessions vs
        WHERE vs.vehicle_name = %s AND vs.license_plate = %s
        AND (
            (vs.session_start <= %s AND (vs.session_end IS NULL OR vs.session_end >= %s)) OR
            (vs.session_start >= %s AND vs.session_start <= %s)
        )
    """, (
        vehicle_name, license_plate,
        session_end, session_start,  # Overlap: existing session includes requested start
        session_start, session_end   # Overlap: existing session starts within requested window
    ))
    conflict = cursor.fetchone()
    if conflict:
        cursor.close()
        conn.close()
        return jsonify({
            'success': False,
            'error': 'This vehicle is already assigned to another driver during the selected time period.'
        }), 409

    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                driver_id INT NOT NULL,
                vehicle_name VARCHAR(100),
                license_plate VARCHAR(20),
                session_start DATETIME NOT NULL,
                session_end DATETIME,
                FOREIGN KEY (driver_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Error ensuring vehicle_sessions table: {e}")

    try:
        cursor.execute("""
            INSERT INTO vehicle_sessions
            (driver_id, vehicle_name, license_plate, session_start, session_end)
            VALUES (%s, %s, %s, %s, %s)
        """, (driver_id, vehicle_name, license_plate, session_start, session_end))
        conn.commit()
        session_id = cursor.lastrowid
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'session_id': session_id}), 200
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/end_vehicle_assignment/<int:session_id>', methods=['POST'])
def end_vehicle_assignment(session_id):
    """Owner ends a vehicle assignment early by setting session_end to now."""
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT vs.id
        FROM vehicle_sessions vs
        JOIN driver_owner do ON vs.driver_id = do.driver_id
        WHERE vs.id = %s AND do.owner_id = %s
    """, (session_id, session['user_id']))
    result = cursor.fetchone()
    if not result:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'Not authorized for this assignment'}), 403

    cursor.execute("""
        UPDATE vehicle_sessions SET session_end = %s WHERE id = %s
    """, (datetime.now(), session_id))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({'success': True, 'message': 'Assignment ended'})

@app.route('/api/driver_vehicle_assignments/<int:driver_id>')
def get_driver_vehicle_assignments(driver_id):
    """Get all vehicle assignments (past, current, future) for a driver."""
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401

    if session['user_type'] == 'owner':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM driver_owner WHERE driver_id = %s AND owner_id = %s
            ) as is_associated
        """, (driver_id, session['user_id']))
        result = cursor.fetchone()
        if not result or not result['is_associated']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
    elif session['user_type'] == 'driver':
        if driver_id != session['user_id']:
            return jsonify({'success': False, 'error': 'Not authorized'}), 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT id, vehicle_name, license_plate, session_start, session_end
        FROM vehicle_sessions
        WHERE driver_id = %s
        ORDER BY session_start DESC
    """, (driver_id,))
    assignments = cursor.fetchall()
    cursor.close()
    conn.close()
    for a in assignments:
        if a.get('session_start'):
            a['session_start'] = a['session_start'].isoformat()
        if a.get('session_end'):
            a['session_end'] = a['session_end'].isoformat() if a['session_end'] else None
    return jsonify({'success': True, 'assignments': assignments})

# --- API endpoints for dashboard AJAX requests ---

@app.route('/api/alert_count')
def api_alert_count():
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM alerts a
        JOIN driver_owner do ON a.user_id = do.driver_id
        WHERE do.owner_id = %s
        AND a.alert_type != 'status_update'
    """, (session['user_id'],))
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return jsonify({'success': True, 'count': count})

@app.route('/api/recent_alerts')
def api_recent_alerts():
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT a.id, a.user_id, a.alert_type, a.timestamp, u.username
        FROM alerts a
        JOIN users u ON a.user_id = u.id
        JOIN driver_owner do ON u.id = do.driver_id
        WHERE do.owner_id = %s
        AND a.alert_type != 'status_update'
        ORDER BY a.timestamp DESC
        LIMIT 10
    """, (session['user_id'],))
    alerts = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify({'success': True, 'alerts': alerts})

@app.route('/api/driver_vehicle/<int:driver_id>')
def api_driver_vehicle(driver_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    # Owner can view their drivers, driver can view self
    if session['user_type'] == 'owner':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM driver_owner WHERE driver_id = %s AND owner_id = %s
            ) as is_associated
        """, (driver_id, session['user_id']))
        result = cursor.fetchone()
        if not result or not result['is_associated']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
    elif session['user_type'] == 'driver':
        if driver_id != session['user_id']:
            return jsonify({'success': False, 'error': 'Not authorized'}), 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT id, vehicle_name, license_plate, session_start, session_end
            FROM vehicle_sessions
            WHERE driver_id = %s
            ORDER BY session_end IS NULL DESC, session_start DESC
            LIMIT 1
        """, (driver_id,))
        vehicle = cursor.fetchone()
    except Exception as e:
        vehicle = None
    cursor.close()
    conn.close()
    # Always return success, even if vehicle is None
    return jsonify({'success': True, 'vehicle': vehicle})

@app.route('/api/risk_level/<int:driver_id>')
def api_risk_level(driver_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    # Owner can view their drivers, driver can view self
    if session['user_type'] == 'owner':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM driver_owner WHERE driver_id = %s AND owner_id = %s
            ) as is_associated
        """, (driver_id, session['user_id']))
        result = cursor.fetchone()
        if not result or not result['is_associated']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
    elif session['user_type'] == 'driver':
        if driver_id != session['user_id']:
            return jsonify({'success': False, 'error': 'Not authorized'}), 403

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT risk_level FROM users WHERE id = %s", (driver_id,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    risk_level = row['risk_level'] if row and row['risk_level'] is not None else 0
    risk_labels = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    return jsonify({'success': True, 'risk_level': risk_level, 'risk_label': risk_labels[risk_level]})

@app.route('/api/driver_module_status/<int:driver_id>')
def api_driver_module_status(driver_id):
    # Dummy implementation: always inactive
    # Replace with real status check if available
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    # Owner can view their drivers, driver can view self
    if session['user_type'] == 'owner':
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("""
            SELECT EXISTS(
                SELECT 1 FROM driver_owner WHERE driver_id = %s AND owner_id = %s
            ) as is_associated
        """, (driver_id, session['user_id']))
        result = cursor.fetchone()
        if not result or not result['is_associated']:
            cursor.close()
            conn.close()
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
        cursor.close()
        conn.close()
    elif session['user_type'] == 'driver':
        if driver_id != session['user_id']:
            return jsonify({'success': False, 'error': 'Not authorized'}), 403
    # For demo, always return inactive
    return jsonify({'success': True, 'is_active': False})

@app.route('/api/add_vehicle', methods=['POST'])
def api_add_vehicle():
    if 'user_id' not in session or session['user_type'] != 'owner':
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    data = request.get_json()
    vehicle_name = data.get('vehicle_name', '').strip()
    license_plate = data.get('license_plate', '').strip()
    if not vehicle_name or not license_plate:
        return jsonify({'success': False, 'error': 'Vehicle name and number plate are required.'}), 400

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    # Ensure vehicles table exists
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INT AUTO_INCREMENT PRIMARY KEY,
                owner_id INT NOT NULL,
                vehicle_name VARCHAR(100) NOT NULL,
                license_plate VARCHAR(50) NOT NULL,
                UNIQUE(owner_id, license_plate),
                FOREIGN KEY (owner_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
    except Exception as e:
        print(f"Error ensuring vehicles table: {e}")

    # Check for duplicate
    cursor.execute("SELECT id FROM vehicles WHERE owner_id = %s AND license_plate = %s", (session['user_id'], license_plate))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'A vehicle with this number plate already exists.'}), 409

    try:
        cursor.execute(
            "INSERT INTO vehicles (owner_id, vehicle_name, license_plate) VALUES (%s, %s, %s)",
            (session['user_id'], vehicle_name, license_plate)
        )
        conn.commit()
        vehicle_id = cursor.lastrowid
        cursor.execute("SELECT id, vehicle_name, license_plate FROM vehicles WHERE id = %s", (vehicle_id,))
        vehicle = cursor.fetchone()
        cursor.close()
        conn.close()
        return jsonify({'success': True, 'vehicle': vehicle})
    except Exception as e:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/alert_details/<int:alert_id>')
def api_alert_details(alert_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    # Get alert and driver info
    cursor.execute("""
        SELECT a.*, u.username, u.email
        FROM alerts a
        JOIN users u ON a.user_id = u.id
        WHERE a.id = %s
    """, (alert_id,))
    alert = cursor.fetchone()
    if not alert:
        cursor.close()
        conn.close()
        return jsonify({'success': False, 'error': 'Alert not found'}), 404

    # Get vehicle info for the driver at the alert time
    cursor.execute("""
        SELECT vehicle_name, license_plate, session_start, session_end
        FROM vehicle_sessions
        WHERE driver_id = %s
        AND session_start <= %s
        AND (session_end IS NULL OR session_end >= %s)
        ORDER BY session_start DESC
        LIMIT 1
    """, (alert['user_id'], alert['timestamp'], alert['timestamp']))
    vehicle = cursor.fetchone()
    cursor.close()
    conn.close()
    # Format datetimes for frontend
    if vehicle:
        if vehicle.get('session_start'):
            vehicle['session_start'] = vehicle['session_start'].isoformat()
        if vehicle.get('session_end'):
            vehicle['session_end'] = vehicle['session_end'].isoformat() if vehicle['session_end'] else None
    return jsonify({'success': True, 'alert': alert, 'vehicle': vehicle})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")