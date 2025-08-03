from flask import Flask, render_template, request, jsonify, session, url_for, redirect, flash
from joblib import load
import mysql.connector
import numpy as np
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta, datetime
import os
from werkzeug.utils import secure_filename
import pickle
from sklearn.preprocessing import LabelEncoder

from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__, template_folder='templates')

app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'static', 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ======================
# Manually define class labels for encoders
# ======================
protocol_types = ["icmp", "tcp", "udp"]
service_types = [
    "aol", "auth", "bgp", "courier", "csnet_ns", "ctf", "daytime", "discard", "domain", "domain_u",
    "echo", "eco_i", "ecr_i", "efs", "exec", "finger", "ftp", "ftp_data", "gopher", "harvest",
    "hostnames", "http", "http_2784", "http_443", "http_8001", "imap4", "IRC", "iso_tsap", "klogin",
    "kshell", "ldap", "link", "login", "mtp", "name", "netbios_dgm", "netbios_ns", "netbios_ssn",
    "netstat", "nnsp", "nntp", "ntp_u", "other", "pm_dump", "pop_2", "pop_3", "printer", "private",
    "red_i", "remote_job", "rje", "shell", "smtp", "sql_net", "ssh", "sunrpc", "supdup", "systat",
    "telnet", "tftp_u", "tim_i", "time", "urh_i", "urp_i", "uucp", "uucp_path", "vmnet", "whois", "X11",
    "Z39_50"
]
flag_types = [
    "OTH", "REJ", "RSTO", "RSTOS0", "RSTR", "S0", "S1", "S2", "S3", "SF", "SH"
]

# ======================
# Fit encoders in memory
# ======================
model = load('models/xgboost_best_model.pkl')
scaler = load('models/scaler.pkl')
service_encoder = load('models/service_encoder.pkl')
label_encoder = load('models/label_encoder.pkl')
flag_encoder = load('models/flag_encoder.pkl')
protocol_type_encoder = load('models/protocol_type_encoder.pkl')

feature_columns = [
'protocol_type','service','flag','src_bytes','dst_bytes','logged_in','count',
'srv_count','serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate',
'dst_host_srv_count','dst_host_serror_rate'
]
feature_mapping = {name: idx for idx, name in enumerate(feature_columns)}


def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='intrusion_detection_system'
    )


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    # Check if user is logged in
    if not session.get('user_id'):
        return render_template('predict.html')

    # Pass encoder options for dropdowns
    # protocol_type_options = protocol_type_encoder.classes_ # Removed encoder options
    # service_options = service_encoder.classes_ # Removed encoder options
    # flag_options = flag_encoder.classes_ # Removed encoder options

    conn = None
    cursor = None
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()

        if request.method == 'POST':
            # Get the input values from the form
            form_data = request.form.to_dict()

            print("\n=== Processing New Prediction ===")
            print(f"Input data: {form_data}")

            # Initialize a zero array for 15 features (matching the reduced set)
            features = np.zeros(15)

            # Use LabelEncoders and scaler for categorical and numerical features
            try:
                protocol = form_data['protocol_type'].strip().lower()
                features[feature_mapping['protocol_type']] = protocol_type_encoder.transform([protocol])[0]
            except Exception as e:
                return render_template("predict.html", error=f"Invalid protocol_type: {form_data.get('protocol_type')}", feature_mapping=feature_mapping)

            try:
                service = form_data['service'].strip().lower()
                features[feature_mapping['service']] = service_encoder.transform([service])[0]
            except Exception as e:
                return render_template("predict.html", error=f"Invalid service: {form_data.get('service')}", feature_mapping=feature_mapping)

            try:
                flag = form_data['flag'].strip().upper()
                features[feature_mapping['flag']] = flag_encoder.transform([flag])[0]
            except Exception as e:
                return render_template("predict.html", error=f"Invalid flag: {form_data.get('flag')}", feature_mapping=feature_mapping)

            # Robustly handle logged_in field
            if 'logged_in' in form_data:
                logged_in_val = form_data['logged_in']
                if str(logged_in_val).lower() in ['no', '0', 'false']:
                    features[feature_mapping['logged_in']] = 0
                else:
                    features[feature_mapping['logged_in']] = 1

            # Process numerical features
            for field, index in feature_mapping.items():
                if field not in ['protocol_type', 'service', 'flag', 'logged_in']:
                    value = form_data.get(field, '0')
                    try:
                        features[index] = float(value)
                    except ValueError:
                        return render_template("predict.html",
                                               error=f"Invalid value for {field}. Please enter a valid number.",
                                               feature_mapping=feature_mapping)

            # Scale features
            features_scaled = scaler.transform([features])[0]
            print("Features (scaled):", features_scaled)

            # Debug: Print features and model output
            print("Features sent to model:", features_scaled)
            try:
                prediction = model.predict([features_scaled])[0]
                probabilities = model.predict_proba([features_scaled])[0]
                print("Model raw prediction:", prediction)
                print("Model probabilities:", probabilities)
            except Exception as e:
                print("Model prediction error:", e)
                return render_template("predict.html", error="Model prediction error: {}".format(e), feature_mapping=feature_mapping)

            confidence = float(max(probabilities))
            prediction_str = label_encoder.inverse_transform([prediction])[0] if hasattr(label_encoder, 'inverse_transform') else ('normal' if prediction == 0 else 'anomaly')

            print(f"\nPrediction: {prediction_str}")
            print(f"Confidence: {confidence}")

            # Save to database
            cursor.execute(
                "INSERT INTO detections (prediction, confidence, user_id) VALUES (%s, %s, %s)",
                (prediction_str, confidence, session['user_id'])
            )
            conn.commit()

            # Store prediction results in session for the result page
            session['last_prediction'] = {
                'prediction': prediction_str,
                'confidence': f"{confidence:.2%}"
            }

            return redirect(url_for('result'))

        return render_template(
            "predict.html",
            # protocol_type_options=protocol_type_options, # Removed encoder options
            # service_options=service_options, # Removed encoder options
            # flag_options=flag_options, # Removed encoder options
            feature_mapping=feature_mapping
        )
    except Exception as e:
        print(f"Database error in predict: {str(e)}")
        return render_template("predict.html", error="An error occurred while processing your request.",
                               feature_mapping=feature_mapping)
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()


@app.route('/result')
def result():
    if not session.get('user_id'):
        return redirect(url_for('login'))
    prediction = session.get('last_prediction', {})
    return render_template('result.html', prediction=prediction)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['role'] = user['role']
            session['profile_image'] = user.get('profile_image')
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.', 'error')
            return render_template('login.html', username=username)

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/statistics')
def statistics():
    conn = get_db_connection()
    is_admin = session.get('role') == 'admin'
    user_id = session.get('user_id')
    if is_admin:
        # Admin: show all statistics
        cursor1 = conn.cursor(dictionary=True)
        cursor1.execute("SELECT prediction, COUNT(*) as count FROM detections GROUP BY prediction")
        rows = cursor1.fetchall()
        stats = {'normal': 0, 'anomaly': 0}
        total = 0
        for row in rows:
            if row['prediction'] in stats:
                stats[row['prediction']] = row['count']
                total += row['count']
        cursor1.close()

        cursor2 = conn.cursor(dictionary=True)
        cursor2.execute("SELECT COUNT(*) as user_count FROM users")
        user_count = cursor2.fetchone()['user_count']
        cursor2.close()

        cursor3 = conn.cursor(dictionary=True)
        cursor3.execute("""
            SELECT u.username, COUNT(d.id) as count
            FROM users u
            LEFT JOIN detections d ON u.id = d.user_id
            GROUP BY u.id, u.username
        """)
        user_rows = cursor3.fetchall()
        user_stats = []
        for row in user_rows:
            percentage = (row['count'] / total * 100) if total > 0 else 0
            user_stats.append({
                'username': row['username'],
                'count': row['count'],
                'percentage': f"{percentage:.2f}%"
            })
        cursor3.close()
    else:
        # Regular user: show only their statistics
        cursor1 = conn.cursor(dictionary=True)
        cursor1.execute("SELECT prediction, COUNT(*) as count FROM detections WHERE user_id=%s GROUP BY prediction", (user_id,))
        rows = cursor1.fetchall()
        stats = {'normal': 0, 'anomaly': 0}
        total = 0
        for row in rows:
            if row['prediction'] in stats:
                stats[row['prediction']] = row['count']
                total += row['count']
        cursor1.close()
        user_count = 1
        cursor2 = conn.cursor(dictionary=True)
        cursor2.execute("SELECT username FROM users WHERE id=%s", (user_id,))
        username_row = cursor2.fetchone()
        # Always fetch all results before closing
        while cursor2.fetchone() is not None:
            pass
        cursor2.close()
        username = username_row['username'] if username_row else 'You'
        user_stats = [{
            'username': username,
            'count': total,
            'percentage': '100%' if total > 0 else '0%'
        }]
    conn.close()
    return render_template(
        'statistics.html',
        stats=stats,
        total=total,
        is_admin=is_admin,
        user_stats=user_stats,
        user_count=user_count
    )


@app.route('/admin/users', methods=['GET', 'POST'])
def admin_users():
    # Check if user is logged in and is admin
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # Verify user is admin and get current user data
    cursor.execute('SELECT id, username, email, role, profile_image FROM users WHERE id = %s', (session['user_id'],))
    current_user = cursor.fetchone()
    if not current_user or current_user['role'] != 'admin':
        flash('Access denied. Admin privileges required.', 'error')
        return redirect(url_for('index'))
    
    # Ensure session data is preserved and current
    session['username'] = current_user['username']
    session['role'] = current_user['role']
    if current_user.get('profile_image'):
        session['profile_image'] = current_user['profile_image']
    
    # Mark session as modified to ensure it's saved
    session.modified = True
    # Handle add, delete, and role update
    if request.method == 'POST':
        method = request.form.get('_method', '').upper()
        if method == 'DELETE':
            user_id = request.form.get('user_id')
            if user_id:
                # Check if admin is trying to delete themselves
                if int(user_id) == session.get('user_id'):
                    return jsonify({'success': False, 'message': 'You cannot delete your own account'}), 400
                
                try:
                    # First delete all detections associated with this user
                    cursor.execute('DELETE FROM detections WHERE user_id=%s', (user_id,))
                    # Then delete the user
                    cursor.execute('DELETE FROM users WHERE id=%s', (user_id,))
                    conn.commit()
                    flash('User deleted successfully.', 'success')
                    return jsonify({'success': True, 'message': 'User deleted successfully'})
                except Exception as e:
                    conn.rollback()
                    print(f"Error deleting user: {e}")
                    flash('Error deleting user. Please try again.', 'error')
                    return jsonify({'success': False, 'message': 'Error deleting user'}), 500
        elif method == 'PATCH':
            user_id = request.form.get('user_id')
            new_role = request.form.get('role')
            if user_id and new_role:
                try:
                    # Check if admin is trying to change their own role
                    if int(user_id) == session.get('user_id'):
                        return jsonify({'success': False, 'message': 'You cannot change your own role'}), 400
                    
                    cursor.execute('UPDATE users SET role=%s WHERE id=%s', (new_role, user_id))
                    conn.commit()
                    flash('User role updated successfully.', 'success')
                    return jsonify({'success': True, 'message': 'User role updated successfully'})
                except Exception as e:
                    conn.rollback()
                    print(f"Error updating user role: {e}")
                    flash('Error updating user role. Please try again.', 'error')
                    return jsonify({'success': False, 'message': 'Error updating user role'}), 500
        else:
            # Add user
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')
            full_name = request.form.get('full_name')
            role = request.form.get('role', 'user')
            if not (username and email and password and full_name):
                flash('All fields are required to add a user.', 'error')
            else:
                cursor.execute('SELECT id FROM users WHERE username=%s OR email=%s', (username, email))
                if cursor.fetchone():
                    flash('Username or email already exists.', 'error')
                else:
                    hashed_pw = generate_password_hash(password)
                    cursor.execute('INSERT INTO users (username, email, password, full_name, role) VALUES (%s, %s, %s, %s, %s)',
                                   (username, email, hashed_pw, full_name, role))
                    conn.commit()
                    flash('User added successfully.', 'success')
    # Query all users and their detection counts
    cursor.execute('''
        SELECT u.id, u.username, u.email, u.role, u.full_name, u.profile_image, u.created_at, COUNT(d.id) as detections
        FROM users u
        LEFT JOIN detections d ON u.id = d.user_id
        GROUP BY u.id, u.username, u.email, u.role, u.full_name, u.profile_image, u.created_at
        ORDER BY u.created_at DESC
    ''')
    users = []
    for row in cursor.fetchall():
        users.append({
            'id': row['id'],
            'username': row['username'],
            'email': row['email'],
            'role': row['role'],
            'full_name': row['full_name'],
            'profile_image': row['profile_image'],
            'detections': row['detections'],
            'joined': row['created_at'].strftime('%Y-%m-%d') if row['created_at'] else '',
        })
    cursor.close()
    conn.close()
    
    # Ensure session is preserved after all operations
    if 'user_id' in session:
        # Refresh session data from database
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT username, email, role, profile_image FROM users WHERE id = %s', (session['user_id'],))
        user_data = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user_data:
            session['username'] = user_data['username']
            session['role'] = user_data['role']
            if user_data.get('profile_image'):
                session['profile_image'] = user_data['profile_image']
            
            # Mark session as modified to ensure it's saved
            session.modified = True
    
    return render_template('admin_users.html', users=users)


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    message = None
    if request.method == 'POST':
        # Handle profile update (full_name, email, password, profile image, etc.)
        full_name = request.form.get('full_name')
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        profile_image = None
        if 'profile_image' in request.files:
            file = request.files['profile_image']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                import time
                unique_filename = f"{session['user_id']}_{int(time.time())}_{filename}"
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], unique_filename))
                profile_image = unique_filename
                cursor.execute('UPDATE users SET profile_image=%s WHERE id=%s', (profile_image, session['user_id']))
                # Update session with new profile image
                session['profile_image'] = profile_image
        # Update full_name and email
        cursor.execute('UPDATE users SET full_name=%s, email=%s WHERE id=%s', (full_name, email, session['user_id']))
        # Update password if provided
        if new_password:
            hashed_pw = generate_password_hash(new_password)
            cursor.execute('UPDATE users SET password=%s WHERE id=%s', (hashed_pw, session['user_id']))
        conn.commit()
        message = 'Profile updated successfully.'
    cursor.execute('SELECT * FROM users WHERE id = %s', (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    return render_template('profile.html', user=user, message=message)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        full_name = request.form.get('full_name')
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if not (full_name and username and email and password):
            flash('All fields are required.', 'error')
            return render_template('signup.html')
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT id FROM users WHERE username=%s OR email=%s', (username, email))
        if cursor.fetchone():
            flash('Username or email already exists.', 'error')
            cursor.close()
            conn.close()
            return render_template('signup.html')
        hashed_pw = generate_password_hash(password)
        cursor.execute('INSERT INTO users (username, email, password, full_name, role) VALUES (%s, %s, %s, %s, %s)',
                       (username, email, hashed_pw, full_name, 'user'))
        conn.commit()
        cursor.close()
        conn.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/debug_session')
def debug_session():
    if 'user_id' not in session:
        return "Not logged in"
    
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (session['user_id'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return {
        'session_profile_image': session.get('profile_image'),
        'db_profile_image': user.get('profile_image'),
        'user_id': session['user_id'],
        'username': session['username']
    }


@app.context_processor
def inject_profile_image():
    if 'user_id' in session:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT username, email, role, profile_image FROM users WHERE id = %s", (session['user_id'],))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            # Preserve all session data
            session['username'] = user.get('username')
            session['role'] = user.get('role')
            session['profile_image'] = user.get('profile_image')
            
            # Mark session as modified to ensure it's saved
            session.modified = True
    return dict()

if __name__ == '__main__':
    app.run(debug=True)
