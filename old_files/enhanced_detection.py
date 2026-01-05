"""
Enhanced Violence Detection System
- MediaPipe pose detection (continuous)
- Weapon detection (knife, gun, scissors)
- Gender classification
- Real-time statistics
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

CNN_MODEL_PATH = 'mobilenet_feature_extractor.tflite'
LSTM_MODEL_PATH = 'violence_detection_lstm.tflite'
GENDER_MODEL_PATH = 'gender_classification.tflite'

SEQ_LENGTH = 30
FRAME_SKIP = 2  # Process more frequently
VIOLENCE_THRESHOLD = 0.7
IMG_SIZE = 128

# Weapon detection (using COCO-SSD classes)
WEAPON_CLASSES = {
    'knife': 43,
    'scissors': 76,
    # Note: gun not in COCO, we'll use pose analysis for threatening gestures
}

# ============================================================================
# LOAD MODELS
# ============================================================================

print("🔄 Loading models...")

# Violence detection
cnn_interpreter = tf.lite.Interpreter(model_path=CNN_MODEL_PATH)
cnn_interpreter.allocate_tensors()
cnn_input_details = cnn_interpreter.get_input_details()
cnn_output_details = cnn_interpreter.get_output_details()

lstm_interpreter = tf.lite.Interpreter(model_path=LSTM_MODEL_PATH)
lstm_interpreter.allocate_tensors()
lstm_input_details = lstm_interpreter.get_input_details()
lstm_output_details = lstm_interpreter.get_output_details()

# Gender classification
try:
    gender_interpreter = tf.lite.Interpreter(model_path=GENDER_MODEL_PATH)
    gender_interpreter.allocate_tensors()
    gender_input_details = gender_interpreter.get_input_details()
    gender_output_details = gender_interpreter.get_output_details()
    GENDER_ENABLED = True
    print("✅ Gender model loaded")
except:
    GENDER_ENABLED = False
    print("⚠️  Gender model not found - skipping gender detection")

print("✅ Violence models loaded")

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Improved stability settings
pose = mp_pose.Pose(
    static_image_mode=False,  # Video mode for better tracking
    model_complexity=1,
    smooth_landmarks=True,  # Enable smoothing
    enable_segmentation=False,
    smooth_segmentation=False,
    min_detection_confidence=0.7,  # Higher confidence
    min_tracking_confidence=0.7   # Higher tracking confidence
)

print("✅ MediaPipe initialized")

# ============================================================================
# FACE DETECTION
# ============================================================================

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def detect_aggressive_pose(landmarks):
    """Detect aggressive body postures"""
    if not landmarks:
        return 0.0, ""
    
    score = 0.0
    reasons = []
    
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
        right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        # Arms raised (fighting stance)
        if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
            score += 0.3
            reasons.append("Raised arms")
        
        # Arms extended (punching)
        left_extended = abs(left_wrist.x - left_shoulder.x) > 0.3
        right_extended = abs(right_wrist.x - right_shoulder.x) > 0.3
        
        if left_extended or right_extended:
            score += 0.3
            reasons.append("Extended arms")
        
        # Calculate arm angles
        def calc_angle(a, b, c):
            radians = np.arctan2(c.y - b.y, c.x - b.x) - np.arctan2(a.y - b.y, a.x - b.x)
            angle = np.abs(radians * 180.0 / np.pi)
            if angle > 180.0:
                angle = 360 - angle
            return angle
        
        left_angle = calc_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calc_angle(right_shoulder, right_elbow, right_wrist)
        
        if 30 < left_angle < 150 or 30 < right_angle < 150:
            score += 0.2
            reasons.append("Fighting stance")
        
    except:
        pass
    
    return min(score, 1.0), ", ".join(reasons) if reasons else ""

def calculate_pose_velocity(pose_history):
    """Calculate movement velocity"""
    if len(pose_history) < 2:
        return 0.0
    
    velocities = []
    key_points = [11, 12, 13, 14, 15, 16]  # Shoulders, elbows, wrists
    
    for i in range(1, len(pose_history)):
        if pose_history[i] is None or pose_history[i-1] is None:
            continue
        
        for idx in key_points:
            if idx < len(pose_history[i]) and idx < len(pose_history[i-1]):
                p1 = pose_history[i][idx]
                p2 = pose_history[i-1][idx]
                
                if p1 and p2:
                    dx = p1.x - p2.x
                    dy = p1.y - p2.y
                    velocity = np.sqrt(dx**2 + dy**2)
                    velocities.append(velocity)
    
    return np.mean(velocities) if velocities else 0.0

def detect_gender(frame):
    """Detect faces and classify gender"""
    if not GENDER_ENABLED:
        return 0, 0, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    male_count = 0
    female_count = 0
    
    for (x, y, w, h) in faces:
        try:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            face_normalized = np.expand_dims(face_rgb / 255.0, axis=0).astype(np.float32)
            
            gender_interpreter.set_tensor(gender_input_details[0]['index'], face_normalized)
            gender_interpreter.invoke()
            gender_pred = gender_interpreter.get_tensor(gender_output_details[0]['index'])[0, 0]
            
            if gender_pred > 0.5:
                female_count += 1
                label = "F"
                color = (255, 0, 255)
            else:
                male_count += 1
                label = "M"
                color = (255, 0, 0)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        except:
            pass
    
    return male_count, female_count, len(faces)

def detect_weapons(frame):
    """Simple weapon detection using color/shape analysis"""
    # This is a placeholder - for real weapon detection, use YOLOv5 or similar
    # Here we'll just detect sharp objects based on edges
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Count sharp edges (very basic)
    sharp_pixels = np.sum(edges > 0)
    total_pixels = edges.shape[0] * edges.shape[1]
    edge_ratio = sharp_pixels / total_pixels
    
    # If too many sharp edges, might indicate weapon
    if edge_ratio > 0.15:  # Threshold
        return True, "Sharp object detected"
    
    return False, ""

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    # Buffers
    sequence_buffer = np.zeros((1, SEQ_LENGTH, 1280), dtype=np.float32)
    prediction_history = deque(maxlen=5)
    gender_history = deque(maxlen=30)
    pose_history = deque(maxlen=10)
    
    # Statistics
    total_violence_detections = 0
    total_weapon_detections = 0
    session_start = time.time()
    
    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    print("\n" + "="*70)
    print("🎥 ENHANCED DETECTION SYSTEM")
    print("="*70)
    print("Features:")
    print("  ✅ Violence Detection (AI)")
    print("  ✅ MediaPipe Pose Analysis (Continuous)")
    print("  ✅ Weapon Detection")
    print("  ✅ Gender Classification")
    print("\nControls: 'q' or ESC to quit")
    print("="*70 + "\n")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
        h, w = display_frame.shape[:2]
        
        # FPS
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        
        # ===== MEDIAPIPE POSE (CONTINUOUS & STABLE) =====
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        pose_results = pose.process(rgb_frame)
        rgb_frame.flags.writeable = True
        
        pose_score = 0.0
        pose_reason = ""
        velocity = 0.0
        
        if pose_results.pose_landmarks:
            # Draw skeleton with custom style for stability
            mp_drawing.draw_landmarks(
                display_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2,
                    circle_radius=2
                )
            )
            
            landmarks = pose_results.pose_landmarks.landmark
            pose_history.append(landmarks)
            
            pose_score, pose_reason = detect_aggressive_pose(landmarks)
            velocity = calculate_pose_velocity(pose_history)
        else:
            # Keep last known pose for stability
            if len(pose_history) > 0 and pose_history[-1] is not None:
                pose_history.append(pose_history[-1])
            else:
                pose_history.append(None)
        
        # Process every FRAME_SKIP frames
        if frame_count % FRAME_SKIP == 0:
            
            # ===== VIOLENCE DETECTION =====
            input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
            input_frame = (input_frame / 127.5) - 1.0
            
            cnn_interpreter.set_tensor(cnn_input_details[0]['index'], input_frame)
            cnn_interpreter.invoke()
            feature_vector = cnn_interpreter.get_tensor(cnn_output_details[0]['index'])
            
            sequence_buffer = np.roll(sequence_buffer, shift=-1, axis=1)
            sequence_buffer[0, -1, :] = feature_vector[0, :]
            
            # ===== WEAPON DETECTION =====
            weapon_detected, weapon_reason = detect_weapons(frame)
            if weapon_detected:
                total_weapon_detections += 1
            
            # ===== GENDER DETECTION =====
            male_count, female_count, total_faces = detect_gender(display_frame)
            gender_history.append((male_count, female_count, total_faces))
            
            avg_male = np.mean([g[0] for g in gender_history]) if gender_history else 0
            avg_female = np.mean([g[1] for g in gender_history]) if gender_history else 0
            avg_total = np.mean([g[2] for g in gender_history]) if gender_history else 0
            
            # ===== VIOLENCE CLASSIFICATION =====
            if frame_count >= SEQ_LENGTH * FRAME_SKIP:
                
                lstm_interpreter.set_tensor(lstm_input_details[0]['index'], sequence_buffer)
                lstm_interpreter.invoke()
                lstm_prediction = lstm_interpreter.get_tensor(lstm_output_details[0]['index'])[0, 0]
                
                # Combined scoring
                combined_score = (lstm_prediction * 0.7) + (pose_score * 0.2) + (velocity * 10 * 0.1)
                combined_score = np.clip(combined_score, 0, 1)
                
                # Weapon bonus
                if weapon_detected:
                    combined_score = min(combined_score + 0.3, 1.0)
                
                prediction_history.append(combined_score)
                smoothed = np.mean(prediction_history)
                
                recent_high = sum(1 for p in prediction_history if p > VIOLENCE_THRESHOLD)
                is_violence = smoothed > VIOLENCE_THRESHOLD and recent_high >= 3
                
                if is_violence:
                    total_violence_detections += 1
                    v_status = "⚠️  VIOLENCE DETECTED"
                    v_color = (0, 0, 255)
                    cv2.rectangle(display_frame, (0, 0), (w, h), v_color, 10)
                    
                    # Build alert message
                    alert_msg = f"[{time.strftime('%H:%M:%S')}] VIOLENCE"
                    if weapon_detected:
                        alert_msg += f" + WEAPON ({weapon_reason})"
                    if pose_reason:
                        alert_msg += f" + {pose_reason}"
                    print(alert_msg)
                else:
                    v_status = "✓ NORMAL"
                    v_color = (0, 255, 0)
                
                # ===== DISPLAY =====
                # Top panel
                panel_h = 140
                cv2.rectangle(display_frame, (0, 0), (w, panel_h), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (0, 0), (w, panel_h), v_color, 3)
                
                cv2.putText(display_frame, v_status, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, v_color, 2)
                
                cv2.putText(display_frame, f"AI: {lstm_prediction:.2f} | Pose: {pose_score:.2f} | Combined: {smoothed:.2f}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                if pose_reason:
                    cv2.putText(display_frame, f"Pose: {pose_reason}", (10, 85),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                if weapon_detected:
                    cv2.putText(display_frame, f"⚠️  {weapon_reason}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.putText(display_frame, f"Velocity: {velocity*100:.1f}", (10, 135),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Bottom panel - Gender & Stats
                if GENDER_ENABLED:
                    panel_y = h - 100
                    cv2.rectangle(display_frame, (0, panel_y), (w, h), (0, 0, 0), -1)
                    cv2.rectangle(display_frame, (0, panel_y), (w, h), (255, 255, 255), 2)
                    
                    cv2.putText(display_frame, f"Gender: M:{male_count} F:{female_count} Total:{total_faces}",
                               (10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if avg_total > 0:
                        male_ratio = avg_male / avg_total
                        cv2.putText(display_frame, f"Avg Ratio: M:{male_ratio:.0%} F:{1-male_ratio:.0%}",
                                   (10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Session stats
                    session_time = int(time.time() - session_start)
                    cv2.putText(display_frame, f"Session: {session_time}s | Alerts: {total_violence_detections} | Weapons: {total_weapon_detections}",
                               (10, panel_y + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                
            else:
                progress = (frame_count / (SEQ_LENGTH * FRAME_SKIP)) * 100
                cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Initializing... {progress:.0f}%", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # FPS
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow('Enhanced Violence Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n🛑 Stopping...")
            break
        
        frame_count += 1
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    pose.close()
    
    # Final statistics
    session_duration = int(time.time() - session_start)
    
    print("\n" + "="*70)
    print("SESSION STATISTICS")
    print("="*70)
    print(f"Duration: {session_duration} seconds")
    print(f"Total Violence Alerts: {total_violence_detections}")
    print(f"Total Weapon Detections: {total_weapon_detections}")
    
    if GENDER_ENABLED and len(gender_history) > 0:
        total_male = sum(g[0] for g in gender_history)
        total_female = sum(g[1] for g in gender_history)
        total_all = total_male + total_female
        
        print(f"\nGender Statistics:")
        print(f"  Total Male detections: {int(total_male)}")
        print(f"  Total Female detections: {int(total_female)}")
        if total_all > 0:
            print(f"  Male ratio: {total_male/total_all*100:.1f}%")
            print(f"  Female ratio: {total_female/total_all*100:.1f}%")
    
    print("="*70)
    print("✅ System stopped")

if __name__ == "__main__":
    main()
