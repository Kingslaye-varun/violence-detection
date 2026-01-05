"""
Enhanced Streamlit Dashboard for Violence Detection System
Features: Violence + Weapon + Gender Detection with Detailed Analysis
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
import os
from collections import deque
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import traceback

# Page config
st.set_page_config(
    page_title="Enhanced Violence Detection Dashboard",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS WITH ERROR HANDLING
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models with comprehensive error handling"""
    models = {
        'cnn': None,
        'lstm': None,
        'gender': None,
        'yolo': None,
        'pose': None,
        'face_cascade': None,
        'errors': []
    }
    
    try:
        # Find model helper
        def find_model(filename):
            if os.path.exists(filename):
                return filename
            elif os.path.exists(os.path.join('models', filename)):
                return os.path.join('models', filename)
            else:
                raise FileNotFoundError(f"Model not found: {filename}")
        
        # Load Violence Detection Models
        try:
            cnn_path = find_model('mobilenet_feature_extractor.tflite')
            models['cnn'] = tf.lite.Interpreter(model_path=cnn_path)
            models['cnn'].allocate_tensors()
            st.success("✅ Violence CNN model loaded")
        except Exception as e:
            models['errors'].append(f"CNN model error: {str(e)}")
            st.error(f"❌ CNN model failed: {str(e)}")
        
        try:
            lstm_path = find_model('violence_detection_lstm.tflite')
            models['lstm'] = tf.lite.Interpreter(model_path=lstm_path)
            models['lstm'].allocate_tensors()
            st.success("✅ Violence LSTM model loaded")
        except Exception as e:
            models['errors'].append(f"LSTM model error: {str(e)}")
            st.error(f"❌ LSTM model failed: {str(e)}")
        
        # Load Gender Model
        try:
            gender_path = find_model('gender_classification.tflite')
            models['gender'] = tf.lite.Interpreter(model_path=gender_path)
            models['gender'].allocate_tensors()
            st.success("✅ Gender classification model loaded")
        except Exception as e:
            models['errors'].append(f"Gender model error: {str(e)}")
            st.warning(f"⚠️ Gender model not available: {str(e)}")
        
        # Load YOLO Weapon Detection
        try:
            from ultralytics import YOLO
            if os.path.exists('yolov8n.pt'):
                models['yolo'] = YOLO('yolov8n.pt')
            else:
                with st.spinner("📥 Downloading YOLOv8 model (first time only)..."):
                    models['yolo'] = YOLO('yolov8n.pt')
            st.success("✅ YOLOv8 weapon detection loaded")
        except ImportError:
            models['errors'].append("YOLOv8 not installed (pip install ultralytics)")
            st.warning("⚠️ Weapon detection unavailable - install ultralytics")
        except Exception as e:
            models['errors'].append(f"YOLO error: {str(e)}")
            st.warning(f"⚠️ Weapon detection error: {str(e)}")
        
        # Load MediaPipe
        try:
            models['pose'] = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
            st.success("✅ MediaPipe pose tracking loaded")
        except Exception as e:
            models['errors'].append(f"MediaPipe error: {str(e)}")
            st.error(f"❌ MediaPipe failed: {str(e)}")
        
        # Load Face Cascade
        try:
            models['face_cascade'] = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            st.success("✅ Face detection loaded")
        except Exception as e:
            models['errors'].append(f"Face cascade error: {str(e)}")
            st.warning(f"⚠️ Face detection error: {str(e)}")
        
    except Exception as e:
        st.error(f"❌ Critical error loading models: {str(e)}")
        models['errors'].append(f"Critical error: {str(e)}")
    
    return models


# ============================================================================
# DETECTION FUNCTIONS WITH ERROR HANDLING
# ============================================================================

# Dangerous objects configuration
DANGEROUS_CLASSES = {
    'knife': 'Knife',
    'scissors': 'Scissors',
    'fork': 'Sharp Object',
    'bottle': 'Bottle (Weapon)',
    'baseball bat': 'Bat',
    'sports ball': 'Thrown Object'
}

def detect_weapons(frame, yolo_model):
    """Detect dangerous objects using YOLO with error handling"""
    if yolo_model is None:
        return [], 0, None
    
    try:
        results = yolo_model(frame, verbose=False, conf=0.4)
        detected_weapons = []
        weapon_count = 0
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                try:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[cls].lower()
                    
                    if class_name in DANGEROUS_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        weapon_name = DANGEROUS_CLASSES[class_name]
                        
                        detected_weapons.append({
                            'name': weapon_name,
                            'conf': conf,
                            'box': (x1, y1, x2, y2),
                            'class': class_name
                        })
                        weapon_count += 1
                        
                        # Draw red box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"{weapon_name} {conf:.2f}"
                        
                        # Background for text
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w, y1), (0, 0, 255), -1)
                        cv2.putText(frame, label, (x1, y1-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except Exception as e:
                    continue
        
        return detected_weapons, weapon_count, None
    
    except Exception as e:
        return [], 0, f"Weapon detection error: {str(e)}"

def detect_aggressive_pose(landmarks):
    """Detect aggressive postures with error handling"""
    if not landmarks:
        return 0.0, [], None
    
    try:
        mp_pose = mp.solutions.pose
        score = 0.0
        reasons = []
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        
        # Raised arms detection
        if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
            score += 0.3
            reasons.append("Raised arms")
        
        # Extended arms detection
        if abs(left_wrist.x - left_shoulder.x) > 0.3 or abs(right_wrist.x - right_shoulder.x) > 0.3:
            score += 0.3
            reasons.append("Extended arms")
        
        # Punching motion (elbow extension)
        left_arm_extended = abs(left_wrist.x - left_elbow.x) > 0.2
        right_arm_extended = abs(right_wrist.x - right_elbow.x) > 0.2
        
        if left_arm_extended or right_arm_extended:
            score += 0.2
            reasons.append("Arm extension")
        
        return min(score, 1.0), reasons, None
    
    except Exception as e:
        return 0.0, [], f"Pose detection error: {str(e)}"


def detect_gender(frame, face_cascade, gender_model):
    """Detect faces and classify gender with error handling"""
    if face_cascade is None or gender_model is None:
        return 0, 0, 0, None
    
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        male_count = 0
        female_count = 0
        face_details = []
        
        for (x, y, w, h) in faces:
            try:
                face = frame[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (128, 128))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                face_normalized = np.expand_dims(face_rgb / 255.0, axis=0).astype(np.float32)
                
                gender_input = gender_model.get_input_details()
                gender_output = gender_model.get_output_details()
                
                gender_model.set_tensor(gender_input[0]['index'], face_normalized)
                gender_model.invoke()
                gender_pred = gender_model.get_tensor(gender_output[0]['index'])[0, 0]
                
                if gender_pred > 0.5:
                    female_count += 1
                    label = "F"
                    color = (255, 0, 255)
                    gender_text = "Female"
                else:
                    male_count += 1
                    label = "M"
                    color = (255, 0, 0)
                    gender_text = "Male"
                
                face_details.append({
                    'gender': gender_text,
                    'confidence': float(gender_pred if gender_pred > 0.5 else 1 - gender_pred),
                    'box': (x, y, w, h)
                })
                
                # Draw box and label
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x+5, y+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Confidence text
                conf_text = f"{gender_pred:.2f}" if gender_pred > 0.5 else f"{1-gender_pred:.2f}"
                cv2.putText(frame, conf_text, (x+5, y+h-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            except Exception as e:
                continue
        
        return male_count, female_count, len(faces), None
    
    except Exception as e:
        return 0, 0, 0, f"Gender detection error: {str(e)}"


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Custom CSS
    st.markdown("""
        <style>
        .big-font {
            font-size:20px !important;
            font-weight: bold;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin: 10px 0;
        }
        .alert-box {
            background-color: #ffebee;
            padding: 10px;
            border-left: 5px solid #f44336;
            margin: 5px 0;
        }
        .success-box {
            background-color: #e8f5e9;
            padding: 10px;
            border-left: 5px solid #4caf50;
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title
    st.title("🚨 Enhanced Violence Detection Dashboard")
    st.markdown("**Real-time AI-powered Violence, Weapon & Gender Detection**")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    st.sidebar.subheader("Detection Settings")
    violence_threshold = st.sidebar.slider(
        "Violence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Higher = fewer false positives"
    )
    
    weapon_confidence = st.sidebar.slider(
        "Weapon Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Higher = fewer false weapon detections"
    )
    
    frame_skip = st.sidebar.slider(
        "Frame Skip",
        min_value=1,
        max_value=5,
        value=2,
        help="Process every Nth frame (higher = faster)"
    )
    
    st.sidebar.subheader("Display Options")
    show_skeleton = st.sidebar.checkbox("Show Skeleton", value=True)
    show_gender = st.sidebar.checkbox("Show Gender Detection", value=True)
    show_weapons = st.sidebar.checkbox("Show Weapon Detection", value=True)
    show_fps = st.sidebar.checkbox("Show FPS", value=True)
    
    # Camera source selection
    st.sidebar.subheader("Camera Source")
    camera_type = st.sidebar.radio(
        "Select Source",
        ["Webcam", "IP Camera (RTSP)"],
        help="Choose between local webcam or IP camera"
    )
    
    if camera_type == "Webcam":
        camera_index = st.sidebar.number_input(
            "Camera Index",
            min_value=0,
            max_value=5,
            value=0,
            help="Change if webcam not detected"
        )
        camera_source = int(camera_index)
    else:
        st.sidebar.info("💡 Enter your IP camera RTSP URL")
        
        # RTSP URL builder
        use_builder = st.sidebar.checkbox("Use URL Builder", value=True)
        
        if use_builder:
            camera_ip = st.sidebar.text_input("Camera IP", "192.168.1.100")
            camera_user = st.sidebar.text_input("Username", "admin")
            camera_pass = st.sidebar.text_input("Password", "password", type="password")
            camera_port = st.sidebar.number_input("Port", value=554, min_value=1, max_value=65535)
            
            # Common RTSP formats
            rtsp_format = st.sidebar.selectbox(
                "Camera Brand/Format",
                [
                    "Hikvision",
                    "Dahua",
                    "Generic",
                    "TP-Link",
                    "Axis",
                    "Custom"
                ]
            )
            
            if rtsp_format == "Hikvision":
                camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/Streaming/Channels/101"
            elif rtsp_format == "Dahua":
                camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/cam/realmonitor?channel=1&subtype=0"
            elif rtsp_format == "TP-Link":
                camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/stream1"
            elif rtsp_format == "Axis":
                camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/axis-media/media.amp"
            elif rtsp_format == "Generic":
                camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/stream1"
            else:
                camera_source = st.sidebar.text_input("Custom RTSP URL", "rtsp://user:pass@ip:port/stream")
            
            st.sidebar.code(camera_source, language="text")
        else:
            camera_source = st.sidebar.text_input(
                "RTSP URL",
                "rtsp://admin:password@192.168.1.100:554/stream1",
                help="Full RTSP URL"
            )
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        models = load_models()
    
    # Check critical models
    if models['cnn'] is None or models['lstm'] is None:
        st.error("❌ Critical models failed to load. Cannot proceed.")
        st.info("Please ensure model files are in the 'models' folder")
        return
    
    # Show model status
    with st.expander("📊 Model Status"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Violence Detection", "✅" if models['lstm'] else "❌")
            st.metric("Pose Tracking", "✅" if models['pose'] else "❌")
        with col2:
            st.metric("Gender Classification", "✅" if models['gender'] else "❌")
            st.metric("Face Detection", "✅" if models['face_cascade'] else "❌")
        with col3:
            st.metric("Weapon Detection", "✅" if models['yolo'] else "❌")
        
        if models['errors']:
            st.warning("⚠️ Some features unavailable:")
            for error in models['errors']:
                st.text(f"  • {error}")
    
    # Layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Feed")
        video_placeholder = st.empty()
    
    with col2:
        st.subheader("📊 Real-time Statistics")
        stats_placeholder = st.empty()
        
        st.subheader("🚨 Recent Alerts")
        alerts_placeholder = st.empty()
    
    # Charts row
    st.subheader("📈 Analysis Charts")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        violence_chart_placeholder = st.empty()
    
    with chart_col2:
        detection_chart_placeholder = st.empty()
    
    # Detailed analysis
    with st.expander("📋 Detailed Analysis", expanded=False):
        analysis_placeholder = st.empty()
    
    # Control buttons
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        start_button = st.button("▶️ Start Monitoring")
    
    with col_btn2:
        stop_button = st.button("⏹️ Stop")
    
    with col_btn3:
        reset_button = st.button("🔄 Reset Stats")
    
    with col_btn4:
        export_button = st.button("📥 Export Report")
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'violence_count': 0,
            'weapon_count': 0,
            'male_total': 0,
            'female_total': 0,
            'session_start': None,
            'alerts': [],
            'violence_scores': [],
            'weapon_detections': [],
            'gender_history': [],
            'pose_alerts': [],
            'errors': []
        }
    
    if start_button:
        st.session_state.running = True
        st.session_state.stats['session_start'] = time.time()
        st.rerun()
    
    if stop_button:
        st.session_state.running = False
        st.rerun()
    
    if reset_button:
        st.session_state.stats = {
            'violence_count': 0,
            'weapon_count': 0,
            'male_total': 0,
            'female_total': 0,
            'session_start': time.time(),
            'alerts': [],
            'violence_scores': [],
            'weapon_detections': [],
            'gender_history': [],
            'pose_alerts': [],
            'errors': []
        }
        st.rerun()
    
    if export_button:
        export_report()
    
    # Main monitoring loop
    if st.session_state.running:
        try:
            # Connect to camera
            if isinstance(camera_source, int):
                cap = cv2.VideoCapture(camera_source)
            else:
                # RTSP connection
                cap = cv2.VideoCapture(camera_source)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce latency for IP cameras
            
            if not cap.isOpened():
                st.error(f"❌ Could not connect to camera")
                if isinstance(camera_source, str):
                    st.info("💡 Check RTSP URL, credentials, and network connection")
                    st.code(camera_source)
                else:
                    st.info(f"💡 Try changing the Camera Index (current: {camera_source})")
                st.session_state.running = False
                return
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Buffers
            SEQ_LENGTH = 30
            sequence_buffer = np.zeros((1, SEQ_LENGTH, 1280), dtype=np.float32)
            prediction_history = deque(maxlen=5)
            
            frame_count = 0
            fps_time = time.time()
            fps = 0
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to read frame from camera")
                    break
                
                display_frame = frame.copy()
                h, w = display_frame.shape[:2]
                
                # FPS calculation
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time + 0.001)
                    fps_time = time.time()
                
                # Weapon detection (every frame for safety)
                weapons = []
                weapon_count = 0
                weapon_error = None
                
                if show_weapons and models['yolo']:
                    weapons, weapon_count, weapon_error = detect_weapons(display_frame, models['yolo'])
                    if weapon_error:
                        st.session_state.stats['errors'].append(weapon_error)
                    if weapon_count > 0:
                        st.session_state.stats['weapon_count'] += 1
                        st.session_state.stats['weapon_detections'].append({
                            'time': datetime.now().strftime('%H:%M:%S'),
                            'weapons': [w['name'] for w in weapons],
                            'count': weapon_count
                        })
                
                # MediaPipe Pose
                pose_score = 0.0
                pose_reasons = []
                pose_error = None
                
                if models['pose']:
                    try:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_frame.flags.writeable = False
                        pose_results = models['pose'].process(rgb_frame)
                        rgb_frame.flags.writeable = True
                        
                        if pose_results.pose_landmarks:
                            if show_skeleton:
                                mp.solutions.drawing_utils.draw_landmarks(
                                    display_frame,
                                    pose_results.pose_landmarks,
                                    mp.solutions.pose.POSE_CONNECTIONS,
                                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(0, 255, 0), thickness=3, circle_radius=3
                                    ),
                                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                                        color=(0, 255, 0), thickness=2
                                    )
                                )
                            
                            landmarks = pose_results.pose_landmarks.landmark
                            pose_score, pose_reasons, pose_error = detect_aggressive_pose(landmarks)
                            
                            if pose_error:
                                st.session_state.stats['errors'].append(pose_error)
                    except Exception as e:
                        pose_error = f"Pose processing error: {str(e)}"
                        st.session_state.stats['errors'].append(pose_error)
                
                # Process frame for violence detection
                if frame_count % frame_skip == 0:
                    try:
                        # Violence detection
                        input_frame = cv2.resize(frame, (128, 128))
                        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                        input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
                        input_frame = (input_frame / 127.5) - 1.0
                        
                        cnn_input = models['cnn'].get_input_details()
                        cnn_output = models['cnn'].get_output_details()
                        
                        models['cnn'].set_tensor(cnn_input[0]['index'], input_frame)
                        models['cnn'].invoke()
                        feature_vector = models['cnn'].get_tensor(cnn_output[0]['index'])
                        
                        sequence_buffer = np.roll(sequence_buffer, shift=-1, axis=1)
                        sequence_buffer[0, -1, :] = feature_vector[0, :]
                        
                        # Gender detection
                        male_count = 0
                        female_count = 0
                        total_faces = 0
                        gender_error = None
                        
                        if show_gender and models['gender'] and models['face_cascade']:
                            male_count, female_count, total_faces, gender_error = detect_gender(
                                display_frame,
                                models['face_cascade'],
                                models['gender']
                            )
                            if gender_error:
                                st.session_state.stats['errors'].append(gender_error)
                            
                            st.session_state.stats['male_total'] += male_count
                            st.session_state.stats['female_total'] += female_count
                            st.session_state.stats['gender_history'].append({
                                'time': datetime.now().strftime('%H:%M:%S'),
                                'male': male_count,
                                'female': female_count
                            })
                        
                        # LSTM violence classification
                        if frame_count >= SEQ_LENGTH * frame_skip:
                            try:
                                lstm_input = models['lstm'].get_input_details()
                                lstm_output = models['lstm'].get_output_details()
                                
                                models['lstm'].set_tensor(lstm_input[0]['index'], sequence_buffer)
                                models['lstm'].invoke()
                                lstm_pred = models['lstm'].get_tensor(lstm_output[0]['index'])[0, 0]
                                
                                # Combined score with weapon boost
                                weapon_boost = 0.3 if weapon_count > 0 else 0.0
                                combined_score = (lstm_pred * 0.7) + (pose_score * 0.3) + weapon_boost
                                combined_score = np.clip(combined_score, 0, 1)
                                
                                prediction_history.append(combined_score)
                                smoothed = np.mean(prediction_history)
                                
                                # Store for charts
                                st.session_state.stats['violence_scores'].append({
                                    'time': frame_count,
                                    'score': float(smoothed),
                                    'lstm': float(lstm_pred),
                                    'pose': float(pose_score),
                                    'weapon_boost': float(weapon_boost)
                                })
                                
                                # Keep only last 100 scores
                                if len(st.session_state.stats['violence_scores']) > 100:
                                    st.session_state.stats['violence_scores'].pop(0)
                                
                                # Check violence
                                recent_high = sum(1 for p in prediction_history if p > violence_threshold)
                                is_violence = smoothed > violence_threshold and recent_high >= 3
                                is_weapon = weapon_count > 0
                                
                                # Determine threat level
                                if is_violence or is_weapon:
                                    st.session_state.stats['violence_count'] += 1
                                    
                                    # Create alert message
                                    if is_weapon:
                                        threat_level = "🚨 HIGH THREAT"
                                        v_status = f"{threat_level} - WEAPON DETECTED"
                                        v_color = (0, 0, 255)
                                        weapon_list = ', '.join([w['name'] for w in weapons])
                                        alert_msg = f"{datetime.now().strftime('%H:%M:%S')} - {threat_level}: {weapon_list}"
                                    else:
                                        threat_level = "⚠️ VIOLENCE"
                                        v_status = f"{threat_level} DETECTED"
                                        v_color = (0, 0, 255)
                                        alert_msg = f"{datetime.now().strftime('%H:%M:%S')} - Violence: {smoothed:.2f}"
                                    
                                    if pose_reasons:
                                        alert_msg += f" | Pose: {', '.join(pose_reasons)}"
                                        st.session_state.stats['pose_alerts'].append({
                                            'time': datetime.now().strftime('%H:%M:%S'),
                                            'reasons': pose_reasons
                                        })
                                    
                                    st.session_state.stats['alerts'].insert(0, alert_msg)
                                    if len(st.session_state.stats['alerts']) > 20:
                                        st.session_state.stats['alerts'].pop()
                                    
                                    # Draw alert border
                                    cv2.rectangle(display_frame, (0, 0), (w, h), v_color, 10)
                                else:
                                    v_status = "✓ NORMAL"
                                    v_color = (0, 255, 0)
                                
                                # Display info panel
                                panel_h = 160
                                cv2.rectangle(display_frame, (0, 0), (w, panel_h), (0, 0, 0), -1)
                                cv2.rectangle(display_frame, (0, 0), (w, panel_h), v_color, 3)
                                
                                cv2.putText(display_frame, v_status, (10, 35),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, v_color, 2)
                                
                                cv2.putText(display_frame, f"Violence: {smoothed:.2f} (AI:{lstm_pred:.2f} Pose:{pose_score:.2f})",
                                           (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                                
                                if weapon_count > 0:
                                    weapon_text = f"Weapons: {weapon_count} - {', '.join([w['name'] for w in weapons])}"
                                    cv2.putText(display_frame, weapon_text, (10, 90),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                
                                if pose_reasons:
                                    cv2.putText(display_frame, f"Pose: {', '.join(pose_reasons)}", (10, 115),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                
                                if total_faces > 0:
                                    cv2.putText(display_frame, f"People: M:{male_count} F:{female_count}", (10, 140),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            except Exception as e:
                                error_msg = f"LSTM processing error: {str(e)}"
                                st.session_state.stats['errors'].append(error_msg)
                        
                        else:
                            # Initialization progress
                            progress = (frame_count / (SEQ_LENGTH * frame_skip)) * 100
                            cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
                            cv2.putText(display_frame, f"Initializing AI... {progress:.0f}%", (10, 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    
                    except Exception as e:
                        error_msg = f"Frame processing error: {str(e)}"
                        st.session_state.stats['errors'].append(error_msg)
                
                # FPS display
                if show_fps:
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display video
                video_placeholder.image(display_frame, channels="BGR")
                
                frame_count += 1
                
                # Update UI every 10 frames
                if frame_count % 10 == 0:
                    update_dashboard(
                        stats_placeholder,
                        alerts_placeholder,
                        violence_chart_placeholder,
                        detection_chart_placeholder,
                        analysis_placeholder,
                        violence_threshold
                    )
                
                time.sleep(0.01)
            
            cap.release()
        
        except Exception as e:
            st.error(f"❌ Critical error: {str(e)}")
            st.code(traceback.format_exc())
            st.session_state.running = False
    
    else:
        st.info("👆 Click 'Start Monitoring' to begin detection")
        
        # Show sample stats if available
        if st.session_state.stats['session_start']:
            update_dashboard(
                stats_placeholder,
                alerts_placeholder,
                violence_chart_placeholder,
                detection_chart_placeholder,
                analysis_placeholder,
                violence_threshold
            )


# ============================================================================
# DASHBOARD UPDATE FUNCTIONS
# ============================================================================

def update_dashboard(stats_placeholder, alerts_placeholder, violence_chart_placeholder, 
                     detection_chart_placeholder, analysis_placeholder, violence_threshold):
    """Update all dashboard components"""
    
    # Calculate session duration
    if st.session_state.stats['session_start']:
        session_duration = int(time.time() - st.session_state.stats['session_start'])
    else:
        session_duration = 0
    
    # Update statistics
    with stats_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("⏱️ Duration", f"{session_duration}s")
            st.metric("🚨 Violence Alerts", st.session_state.stats['violence_count'])
        
        with col2:
            st.metric("🔪 Weapon Detections", st.session_state.stats['weapon_count'])
            total_people = st.session_state.stats['male_total'] + st.session_state.stats['female_total']
            st.metric("👥 People Detected", total_people)
        
        with col3:
            st.metric("👨 Male", st.session_state.stats['male_total'])
            st.metric("👩 Female", st.session_state.stats['female_total'])
        
        with col4:
            if total_people > 0:
                male_ratio = (st.session_state.stats['male_total'] / total_people) * 100
                st.metric("📊 Male Ratio", f"{male_ratio:.1f}%")
            else:
                st.metric("📊 Male Ratio", "N/A")
            
            error_count = len(st.session_state.stats['errors'])
            st.metric("⚠️ Errors", error_count)
    
    # Update alerts
    if st.session_state.stats['alerts']:
        with alerts_placeholder.container():
            for alert in st.session_state.stats['alerts'][:10]:
                if "HIGH THREAT" in alert or "WEAPON" in alert:
                    st.markdown(f'<div class="alert-box">🚨 {alert}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="alert-box">⚠️ {alert}</div>', unsafe_allow_html=True)
    
    # Update violence chart
    if len(st.session_state.stats['violence_scores']) > 0:
        with violence_chart_placeholder.container():
            scores = st.session_state.stats['violence_scores']
            
            fig = make_subplots(
                rows=1, cols=1,
                subplot_titles=["Violence Score Over Time"]
            )
            
            fig.add_trace(go.Scatter(
                y=[s['score'] for s in scores],
                mode='lines',
                name='Combined Score',
                line=dict(color='red', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                y=[s['lstm'] for s in scores],
                mode='lines',
                name='AI Score',
                line=dict(color='orange', width=1, dash='dot')
            ))
            
            fig.add_trace(go.Scatter(
                y=[s['pose'] for s in scores],
                mode='lines',
                name='Pose Score',
                line=dict(color='blue', width=1, dash='dot')
            ))
            
            fig.add_hline(y=violence_threshold, line_dash="dash", line_color="yellow",
                         annotation_text="Threshold")
            
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                xaxis_title="Frame",
                yaxis_title="Score",
                yaxis_range=[0, 1],
                showlegend=True,
                legend=dict(x=0, y=1)
            )
            
            st.plotly_chart(fig, key=f"violence_chart_{time.time()}")
    
    # Update detection summary chart
    with detection_chart_placeholder.container():
        detection_data = {
            'Category': ['Violence', 'Weapons', 'Male', 'Female'],
            'Count': [
                st.session_state.stats['violence_count'],
                st.session_state.stats['weapon_count'],
                st.session_state.stats['male_total'],
                st.session_state.stats['female_total']
            ],
            'Color': ['red', 'darkred', 'blue', 'pink']
        }
        
        fig = go.Figure(data=[
            go.Bar(
                x=detection_data['Category'],
                y=detection_data['Count'],
                marker_color=detection_data['Color'],
                text=detection_data['Count'],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title="Detection Summary",
            height=300,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_title="Category",
            yaxis_title="Count"
        )
        
        st.plotly_chart(fig, key=f"detection_chart_{time.time()}")
    
    # Update detailed analysis
    with analysis_placeholder.container():
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "🔪 Weapons", "👥 Gender", "⚠️ Errors"])
        
        with tab1:
            st.write("**Session Summary**")
            summary_df = pd.DataFrame({
                'Metric': ['Duration', 'Violence Alerts', 'Weapon Detections', 'Male Detections', 'Female Detections', 'Total People'],
                'Value': [
                    f"{session_duration}s",
                    st.session_state.stats['violence_count'],
                    st.session_state.stats['weapon_count'],
                    st.session_state.stats['male_total'],
                    st.session_state.stats['female_total'],
                    st.session_state.stats['male_total'] + st.session_state.stats['female_total']
                ]
            })
            st.dataframe(summary_df, hide_index=True)
        
        with tab2:
            if st.session_state.stats['weapon_detections']:
                st.write("**Weapon Detection History**")
                weapon_df = pd.DataFrame(st.session_state.stats['weapon_detections'])
                weapon_df['weapons'] = weapon_df['weapons'].apply(lambda x: ', '.join(x))
                st.dataframe(weapon_df, hide_index=True)
            else:
                st.info("No weapons detected")
        
        with tab3:
            if st.session_state.stats['gender_history']:
                st.write("**Gender Detection History (Last 20)**")
                gender_df = pd.DataFrame(st.session_state.stats['gender_history'][-20:])
                st.dataframe(gender_df, hide_index=True)
            else:
                st.info("No gender data available")
        
        with tab4:
            if st.session_state.stats['errors']:
                st.write("**Error Log (Last 10)**")
                for error in st.session_state.stats['errors'][-10:]:
                    st.text(f"• {error}")
            else:
                st.success("No errors detected")

def export_report():
    """Export session report as JSON"""
    report = {
        'session_duration': int(time.time() - st.session_state.stats['session_start']) if st.session_state.stats['session_start'] else 0,
        'violence_count': st.session_state.stats['violence_count'],
        'weapon_count': st.session_state.stats['weapon_count'],
        'male_total': st.session_state.stats['male_total'],
        'female_total': st.session_state.stats['female_total'],
        'alerts': st.session_state.stats['alerts'],
        'weapon_detections': st.session_state.stats['weapon_detections'],
        'errors': st.session_state.stats['errors']
    }
    
    import json
    report_json = json.dumps(report, indent=2)
    
    st.download_button(
        label="📥 Download Report (JSON)",
        data=report_json,
        file_name=f"violence_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )

if __name__ == "__main__":
    main()
