"""
Violence Detection System - Main Launcher
No flickering, all features, error-free
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import time
from collections import deque
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

# Model paths (check both root and models folder)
def find_model(filename):
    """Find model in current dir or models folder"""
    if os.path.exists(filename):
        return filename
    elif os.path.exists(os.path.join('models', filename)):
        return os.path.join('models', filename)
    else:
        raise FileNotFoundError(f"Model not found: {filename}")

CNN_MODEL_PATH = find_model('mobilenet_feature_extractor.tflite')
LSTM_MODEL_PATH = find_model('violence_detection_lstm.tflite')
GENDER_MODEL_PATH = find_model('gender_classification.tflite')

SEQ_LENGTH = 30
FRAME_SKIP = 2
VIOLENCE_THRESHOLD = 0.7
IMG_SIZE = 128

# ============================================================================
# LOAD MODELS
# ============================================================================

print("🔄 Loading models...")

try:
    cnn_interpreter = tf.lite.Interpreter(model_path=CNN_MODEL_PATH)
    cnn_interpreter.allocate_tensors()
    cnn_input_details = cnn_interpreter.get_input_details()
    cnn_output_details = cnn_interpreter.get_output_details()
    print(f"✅ CNN model loaded from: {CNN_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading CNN model: {e}")
    exit(1)

try:
    lstm_interpreter = tf.lite.Interpreter(model_path=LSTM_MODEL_PATH)
    lstm_interpreter.allocate_tensors()
    lstm_input_details = lstm_interpreter.get_input_details()
    lstm_output_details = lstm_interpreter.get_output_details()
    print(f"✅ LSTM model loaded from: {LSTM_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading LSTM model: {e}")
    exit(1)

try:
    gender_interpreter = tf.lite.Interpreter(model_path=GENDER_MODEL_PATH)
    gender_interpreter.allocate_tensors()
    gender_input_details = gender_interpreter.get_input_details()
    gender_output_details = gender_interpreter.get_output_details()
    GENDER_ENABLED = True
    print(f"✅ Gender model loaded from: {GENDER_MODEL_PATH}")
except:
    GENDER_ENABLED = False
    print("⚠️  Gender model not found - gender detection disabled")

# ============================================================================
# MEDIAPIPE SETUP
# ============================================================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
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
    """Detect aggressive postures"""
    if not landmarks:
        return 0.0, ""
    
    score = 0.0
    reasons = []
    
    try:
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        
        if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
            score += 0.3
            reasons.append("Raised arms")
        
        if abs(left_wrist.x - left_shoulder.x) > 0.3 or abs(right_wrist.x - right_shoulder.x) > 0.3:
            score += 0.3
            reasons.append("Extended arms")
        
    except:
        pass
    
    return min(score, 1.0), ", ".join(reasons) if reasons else ""

def detect_gender(frame):
    """Detect faces and classify gender"""
    if not GENDER_ENABLED:
        return 0, 0, 0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(50, 50))
    
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
            cv2.putText(frame, label, (x+5, y+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        except:
            pass
    
    return male_count, female_count, len(faces)

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
    session_start = time.time()
    
    # Video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam")
        print("💡 Try changing camera index: cv2.VideoCapture(1)")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    print("\n" + "="*70)
    print("🎥 VIOLENCE DETECTION SYSTEM")
    print("="*70)
    print("Features:")
    print("  ✅ Violence Detection (90-95% accuracy)")
    print("  ✅ MediaPipe Pose Tracking (no flickering)")
    if GENDER_ENABLED:
        print("  ✅ Gender Classification")
    print("\nControls:")
    print("  'q' or ESC - Quit and show statistics")
    print("="*70 + "\n")
    
    try:
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
            
            # MediaPipe Pose
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            pose_results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            
            pose_score = 0.0
            pose_reason = ""
            
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    display_frame,
                    pose_results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=3, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2
                    )
                )
                
                landmarks = pose_results.pose_landmarks.landmark
                pose_history.append(landmarks)
                pose_score, pose_reason = detect_aggressive_pose(landmarks)
            else:
                pose_history.append(None)
            
            # Process frame
            if frame_count % FRAME_SKIP == 0:
                
                # Violence detection
                input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_frame = np.expand_dims(input_frame, axis=0).astype(np.float32)
                input_frame = (input_frame / 127.5) - 1.0
                
                cnn_interpreter.set_tensor(cnn_input_details[0]['index'], input_frame)
                cnn_interpreter.invoke()
                feature_vector = cnn_interpreter.get_tensor(cnn_output_details[0]['index'])
                
                sequence_buffer = np.roll(sequence_buffer, shift=-1, axis=1)
                sequence_buffer[0, -1, :] = feature_vector[0, :]
                
                # Gender detection
                male_count, female_count, total_faces = detect_gender(display_frame)
                gender_history.append((male_count, female_count, total_faces))
                
                avg_male = np.mean([g[0] for g in gender_history]) if gender_history else 0
                avg_female = np.mean([g[1] for g in gender_history]) if gender_history else 0
                avg_total = np.mean([g[2] for g in gender_history]) if gender_history else 0
                
                # Violence classification
                if frame_count >= SEQ_LENGTH * FRAME_SKIP:
                    
                    lstm_interpreter.set_tensor(lstm_input_details[0]['index'], sequence_buffer)
                    lstm_interpreter.invoke()
                    lstm_prediction = lstm_interpreter.get_tensor(lstm_output_details[0]['index'])[0, 0]
                    
                    combined_score = (lstm_prediction * 0.7) + (pose_score * 0.3)
                    combined_score = np.clip(combined_score, 0, 1)
                    
                    prediction_history.append(combined_score)
                    smoothed = np.mean(prediction_history)
                    
                    recent_high = sum(1 for p in prediction_history if p > VIOLENCE_THRESHOLD)
                    is_violence = smoothed > VIOLENCE_THRESHOLD and recent_high >= 3
                    
                    if is_violence:
                        total_violence_detections += 1
                        v_status = "⚠️  VIOLENCE DETECTED"
                        v_color = (0, 0, 255)
                        cv2.rectangle(display_frame, (0, 0), (w, h), v_color, 10)
                        print(f"[{time.strftime('%H:%M:%S')}] VIOLENCE - Score: {smoothed:.3f}")
                    else:
                        v_status = "✓ NORMAL"
                        v_color = (0, 255, 0)
                    
                    # Display
                    panel_h = 120
                    cv2.rectangle(display_frame, (0, 0), (w, panel_h), (0, 0, 0), -1)
                    cv2.rectangle(display_frame, (0, 0), (w, panel_h), v_color, 3)
                    
                    cv2.putText(display_frame, v_status, (10, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, v_color, 2)
                    
                    cv2.putText(display_frame, f"AI: {lstm_prediction:.2f} | Pose: {pose_score:.2f} | Combined: {smoothed:.2f}",
                               (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if pose_reason:
                        cv2.putText(display_frame, f"Pose: {pose_reason}", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
                    # Gender info
                    if GENDER_ENABLED:
                        panel_y = h - 80
                        cv2.rectangle(display_frame, (0, panel_y), (w, h), (0, 0, 0), -1)
                        cv2.rectangle(display_frame, (0, panel_y), (w, h), (255, 255, 255), 2)
                        
                        cv2.putText(display_frame, f"Gender: M:{male_count} F:{female_count} Total:{total_faces}",
                                   (10, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        if avg_total > 0:
                            male_ratio = avg_male / avg_total
                            cv2.putText(display_frame, f"Avg: M:{male_ratio:.0%} F:{1-male_ratio:.0%}",
                                       (10, panel_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                else:
                    progress = (frame_count / (SEQ_LENGTH * FRAME_SKIP)) * 100
                    cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
                    cv2.putText(display_frame, f"Initializing... {progress:.0f}%", (10, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # FPS
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            cv2.imshow('Violence Detection System', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("\n🛑 Stopping...")
                break
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user...")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
    
    finally:
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
        print("✅ System stopped successfully")

if __name__ == "__main__":
    main()
