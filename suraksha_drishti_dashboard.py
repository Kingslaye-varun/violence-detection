"""
Suraksha Drishti - AI-Powered Security Surveillance System
Government of India Initiative
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import time
import os
from collections import deque
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Suraksha Drishti - Security Surveillance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# LOAD MODELS SILENTLY
# ============================================================================

@st.cache_resource
def load_models():
    """Load all AI models silently"""
    models = {'cnn': None, 'lstm': None, 'gender': None, 'yolo': None, 'face_cascade': None}
    
    def find_model(filename):
        if os.path.exists(filename):
            return filename
        elif os.path.exists(os.path.join('models', filename)):
            return os.path.join('models', filename)
        return None
    
    try:
        # CNN Model
        cnn_path = find_model('mobilenet_feature_extractor.tflite')
        if cnn_path:
            models['cnn'] = tf.lite.Interpreter(model_path=cnn_path)
            models['cnn'].allocate_tensors()
        
        # LSTM Model
        lstm_path = find_model('violence_detection_lstm.h5')
        if lstm_path:
            models['lstm'] = tf.keras.models.load_model(lstm_path, compile=False)
        
        # Gender Model
        gender_path = find_model('gender_classification.tflite')
        if gender_path:
            models['gender'] = tf.lite.Interpreter(model_path=gender_path)
            models['gender'].allocate_tensors()
        
        # YOLO Weapon Detection
        try:
            from ultralytics import YOLO
            models['yolo'] = YOLO('yolov8n.pt', verbose=False)
        except:
            pass
        
        # Face Detection
        models['face_cascade'] = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    except:
        pass
    
    return models

# Dangerous objects
DANGEROUS_CLASSES = {
    'knife': 'Knife', 'scissors': 'Scissors', 'fork': 'Sharp Object',
    'bottle': 'Bottle', 'baseball bat': 'Bat', 'sports ball': 'Projectile'
}

def detect_weapons(frame, yolo_model):
    if yolo_model is None:
        return [], 0
    try:
        results = yolo_model(frame, verbose=False, conf=0.4)
        detected_weapons = []
        weapon_count = 0
        
        for result in results:
            for box in result.boxes:
                try:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = result.names[cls].lower()
                    
                    if class_name in DANGEROUS_CLASSES:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        weapon_name = DANGEROUS_CLASSES[class_name]
                        detected_weapons.append({'name': weapon_name, 'conf': conf, 'box': (x1, y1, x2, y2)})
                        weapon_count += 1
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                        label = f"{weapon_name}"
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(frame, (x1, y1-text_h-10), (x1+text_w, y1), (0, 0, 255), -1)
                        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                except:
                    continue
        return detected_weapons, weapon_count
    except:
        return [], 0

def detect_gender(frame, face_cascade, gender_model):
    if face_cascade is None or gender_model is None:
        return 0, 0, 0
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
        
        male_count = 0
        female_count = 0
        
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
                else:
                    male_count += 1
                    label = "M"
                    color = (255, 0, 0)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x+5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            except:
                continue
        
        return male_count, female_count, len(faces)
    except:
        return 0, 0, 0

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Custom CSS - Government Style
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
            color: white;
        }
        .main-title {
            font-size: 42px;
            font-weight: bold;
            margin: 0;
            color: white;
        }
        .main-subtitle {
            font-size: 18px;
            margin: 5px 0 0 0;
            color: #e0e0e0;
        }
        .status-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #2a5298;
            margin: 10px 0;
        }
        .alert-high {
            background-color: #ffebee;
            border-left: 4px solid #d32f2f;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .alert-medium {
            background-color: #fff3e0;
            border-left: 4px solid #f57c00;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .alert-safe {
            background-color: #e8f5e9;
            border-left: 4px solid #388e3c;
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }
        .metric-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            color: #2a5298;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div class="main-header">
            <div class="main-title">🛡️ सुरक्षा दृष्टि</div>
            <div class="main-title">Suraksha Drishti</div>
            <div class="main-subtitle">AI-Powered Security Surveillance System</div>
            <div class="main-subtitle">Government of India Initiative</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Load models silently
    with st.spinner("Initializing system..."):
        models = load_models()
    
    # Check if critical models loaded
    system_ready = models['cnn'] is not None and models['lstm'] is not None
    
    if not system_ready:
        st.error("⚠️ System initialization failed. Please contact technical support.")
        return
    
    # Configuration Section
    st.markdown("### 📹 Camera Configuration")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        camera_type = st.radio(
            "Select Camera Source",
            ["Webcam", "IP Camera (RTSP)"],
            horizontal=True
        )
    
    with col2:
        if camera_type == "Webcam":
            st.info("💡 Using local webcam for monitoring")
            camera_source = 0
        else:
            st.info("💡 Enter RTSP stream details below")
            
            with st.expander("🔧 RTSP Configuration", expanded=True):
                camera_ip = st.text_input("Camera IP Address", "192.168.1.100")
                camera_user = st.text_input("Username", "admin")
                camera_pass = st.text_input("Password", "password", type="password")
                camera_port = st.number_input("Port", value=554, min_value=1, max_value=65535)
                
                camera_brand = st.selectbox(
                    "Camera Brand",
                    ["Hikvision", "Dahua", "Generic", "TP-Link", "Axis"]
                )
                
                if camera_brand == "Hikvision":
                    camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/Streaming/Channels/101"
                elif camera_brand == "Dahua":
                    camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/cam/realmonitor?channel=1&subtype=0"
                elif camera_brand == "TP-Link":
                    camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/stream1"
                elif camera_brand == "Axis":
                    camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/axis-media/media.amp"
                else:
                    camera_source = f"rtsp://{camera_user}:{camera_pass}@{camera_ip}:{camera_port}/stream1"
                
                st.code(camera_source, language="text")
    
    st.markdown("---")
    
    # Detection Settings
    with st.expander("⚙️ Detection Settings"):
        col1, col2 = st.columns(2)
        with col1:
            violence_threshold = st.slider("Threat Sensitivity", 0.0, 1.0, 0.7, 0.05)
        with col2:
            frame_skip = st.slider("Processing Speed", 1, 5, 2, help="Higher = Faster but less accurate")
    
    # Layout
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("### 📺 Live Monitoring")
        video_placeholder = st.empty()
    
    with col_right:
        st.markdown("### 📊 Real-Time Analysis")
        status_placeholder = st.empty()
        metrics_placeholder = st.empty()
        alerts_placeholder = st.empty()
    
    # Charts
    st.markdown("### 📈 Threat Analysis")
    chart_placeholder = st.empty()
    
    # Control Buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        start_button = st.button("▶️ Start Monitoring", use_container_width=True, type="primary")
    
    with col2:
        stop_button = st.button("⏹️ Stop", use_container_width=True)
    
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
            'violence_scores': []
        }
    
    if start_button:
        st.session_state.running = True
        st.session_state.stats['session_start'] = time.time()
        st.rerun()
    
    if stop_button:
        st.session_state.running = False
        st.rerun()
    
    # Main monitoring loop
    if st.session_state.running:
        try:
            cap = cv2.VideoCapture(camera_source)
            
            if not cap.isOpened():
                st.error("❌ Unable to connect to camera. Please check configuration.")
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
                    st.error("❌ Camera feed interrupted")
                    break
                
                display_frame = frame.copy()
                
                # FPS
                if frame_count % 10 == 0:
                    fps = 10 / (time.time() - fps_time + 0.001)
                    fps_time = time.time()
                
                # Weapon detection
                weapons, weapon_count = detect_weapons(display_frame, models['yolo'])
                
                # Process frame
                if frame_count % frame_skip == 0:
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
                    male_count, female_count, total_faces = detect_gender(display_frame, models['face_cascade'], models['gender'])
                    st.session_state.stats['male_total'] += male_count
                    st.session_state.stats['female_total'] += female_count
                    
                    # LSTM prediction
                    if frame_count >= SEQ_LENGTH * frame_skip:
                        lstm_pred = models['lstm'].predict(sequence_buffer, verbose=0)[0, 0]
                        weapon_boost = 0.3 if weapon_count > 0 else 0.0
                        combined_score = (lstm_pred * 0.7) + weapon_boost
                        combined_score = np.clip(combined_score, 0, 1)
                        
                        prediction_history.append(combined_score)
                        smoothed = np.mean(prediction_history)
                        
                        st.session_state.stats['violence_scores'].append({
                            'time': frame_count,
                            'score': float(smoothed)
                        })
                        
                        if len(st.session_state.stats['violence_scores']) > 100:
                            st.session_state.stats['violence_scores'].pop(0)
                        
                        # Threat assessment
                        recent_high = sum(1 for p in prediction_history if p > violence_threshold)
                        is_violence = smoothed > violence_threshold and recent_high >= 3
                        is_weapon = weapon_count > 0
                        
                        if is_violence or is_weapon:
                            st.session_state.stats['violence_count'] += 1
                            
                            if is_weapon:
                                threat_level = "HIGH THREAT"
                                status_text = "WEAPON DETECTED"
                                color = (0, 0, 255)
                            elif smoothed > 0.85:
                                threat_level = "CRITICAL"
                                status_text = "VIOLENCE DETECTED"
                                color = (0, 0, 255)
                            else:
                                threat_level = "ELEVATED"
                                status_text = "SUSPICIOUS ACTIVITY"
                                color = (0, 165, 255)
                            
                            alert_msg = f"{datetime.now().strftime('%H:%M:%S')} - {threat_level}: {status_text}"
                            if alert_msg not in [a['msg'] for a in st.session_state.stats['alerts'][-5:]]:
                                st.session_state.stats['alerts'].append({
                                    'time': datetime.now().strftime('%H:%M:%S'),
                                    'level': threat_level,
                                    'msg': alert_msg
                                })
                            
                            cv2.putText(display_frame, f"ALERT: {status_text}", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
                        else:
                            cv2.putText(display_frame, "STATUS: NORMAL", (10, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        # Display info
                        cv2.putText(display_frame, f"Threat Level: {smoothed*100:.1f}%", (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(display_frame, f"People: {total_faces} (M:{male_count} F:{female_count})", (10, 85),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display video
                video_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                
                # Update status
                with status_placeholder.container():
                    if st.session_state.stats['violence_scores']:
                        current_threat = st.session_state.stats['violence_scores'][-1]['score']
                        if current_threat > 0.7:
                            st.markdown(f'<div class="alert-high">🚨 <b>HIGH THREAT</b> - Threat Level: {current_threat*100:.1f}%</div>', unsafe_allow_html=True)
                        elif current_threat > 0.5:
                            st.markdown(f'<div class="alert-medium">⚠️ <b>ELEVATED</b> - Threat Level: {current_threat*100:.1f}%</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="alert-safe">✅ <b>NORMAL</b> - Threat Level: {current_threat*100:.1f}%</div>', unsafe_allow_html=True)
                
                # Update metrics
                with metrics_placeholder.container():
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f'<div class="metric-box"><div class="metric-value">{st.session_state.stats["violence_count"]}</div><div class="metric-label">Incidents</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="metric-box"><div class="metric-value">{st.session_state.stats["weapon_count"]}</div><div class="metric-label">Weapons</div></div>', unsafe_allow_html=True)
                    with col3:
                        total_people = st.session_state.stats['male_total'] + st.session_state.stats['female_total']
                        st.markdown(f'<div class="metric-box"><div class="metric-value">{total_people}</div><div class="metric-label">People Detected</div></div>', unsafe_allow_html=True)
                
                # Update alerts
                with alerts_placeholder.container():
                    st.markdown("#### 🚨 Recent Alerts")
                    if st.session_state.stats['alerts']:
                        for alert in st.session_state.stats['alerts'][-5:]:
                            if "HIGH" in alert['level'] or "CRITICAL" in alert['level']:
                                st.markdown(f'<div class="alert-high">{alert["msg"]}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="alert-medium">{alert["msg"]}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No alerts")
                
                # Update chart
                if st.session_state.stats['violence_scores']:
                    with chart_placeholder.container():
                        df = pd.DataFrame(st.session_state.stats['violence_scores'])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            y=df['score'],
                            mode='lines',
                            name='Threat Level',
                            line=dict(color='#d32f2f', width=2),
                            fill='tozeroy'
                        ))
                        fig.add_hline(y=violence_threshold, line_dash="dash", line_color="orange", annotation_text="Threshold")
                        fig.update_layout(
                            height=250,
                            margin=dict(l=0, r=0, t=20, b=0),
                            xaxis_title="Time",
                            yaxis_title="Threat Level",
                            yaxis_range=[0, 1],
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                frame_count += 1
                time.sleep(0.01)
            
            cap.release()
        
        except Exception as e:
            st.error(f"❌ System error: {str(e)}")
            st.session_state.running = False
    else:
        video_placeholder.info("👆 Click 'Start Monitoring' to begin surveillance")
        status_placeholder.empty()
        metrics_placeholder.empty()
        alerts_placeholder.empty()
        chart_placeholder.empty()

if __name__ == "__main__":
    main()
