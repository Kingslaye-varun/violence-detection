"""
Combined Violence Detection + Gender Ratio Detection
Real-time webcam application
"""

import cv2
import numpy as np
import tensorflow as tf
import time
from collections import deque

# ============================================================================
# CONFIGURATION
# ============================================================================

# Violence detection models
CNN_MODEL_PATH = 'mobilenet_feature_extractor.tflite'
LSTM_MODEL_PATH = 'violence_detection_lstm.tflite'

# Gender classification model
GENDER_MODEL_PATH = 'gender_classification.tflite'

# Parameters
SEQ_LENGTH = 30
FRAME_SKIP = 3
VIOLENCE_THRESHOLD = 0.7
IMG_SIZE = 128

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
gender_interpreter = tf.lite.Interpreter(model_path=GENDER_MODEL_PATH)
gender_interpreter.allocate_tensors()
gender_input_details = gender_interpreter.get_input_details()
gender_output_details = gender_interpreter.get_output_details()

print("✅ All models loaded")

# ============================================================================
# FACE DETECTION FOR GENDER
# ============================================================================

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_gender(frame):
    """Detect faces and classify gender"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    male_count = 0
    female_count = 0

    for (x, y, w, h) in faces:
        # Extract face
        face = frame[y:y+h, x:x+w]

        # Preprocess for gender model
        face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        face_normalized = np.expand_dims(
            face_rgb / 255.0, axis=0).astype(np.float32)

        # Predict gender
        gender_interpreter.set_tensor(
            gender_input_details[0]['index'], face_normalized)
        gender_interpreter.invoke()
        gender_pred = gender_interpreter.get_tensor(
            gender_output_details[0]['index'])[0, 0]

        # 0 = Male, 1 = Female
        if gender_pred > 0.5:
            female_count += 1
            label = "Female"
            color = (255, 0, 255)  # Pink
        else:
            male_count += 1
            label = "Male"
            color = (255, 0, 0)  # Blue

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"{label} ({gender_pred:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return male_count, female_count, len(faces)

# ============================================================================
# MAIN LOOP
# ============================================================================


def main():
    # Buffers
    sequence_buffer = np.zeros((1, SEQ_LENGTH, 1280), dtype=np.float32)
    prediction_history = deque(maxlen=5)
    gender_history = deque(maxlen=10)  # Track gender ratio over time

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
    print("🎥 COMBINED DETECTION SYSTEM ACTIVE")
    print("="*70)
    print("Features:")
    print("  - Violence Detection")
    print("  - Gender Classification")
    print("  - Male/Female Ratio")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("="*70 + "\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # FPS calculation
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()

        # Process frame
        if frame_count % FRAME_SKIP == 0:

            # ===== VIOLENCE DETECTION =====
            input_frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
            input_frame = np.expand_dims(
                input_frame, axis=0).astype(np.float32)
            input_frame = (input_frame / 127.5) - 1.0

            cnn_interpreter.set_tensor(
                cnn_input_details[0]['index'], input_frame)
            cnn_interpreter.invoke()
            feature_vector = cnn_interpreter.get_tensor(
                cnn_output_details[0]['index'])

            sequence_buffer = np.roll(sequence_buffer, shift=-1, axis=1)
            sequence_buffer[0, -1, :] = feature_vector[0, :]

            # ===== GENDER DETECTION =====
            male_count, female_count, total_faces = detect_gender(
                display_frame)
            gender_history.append((male_count, female_count, total_faces))

            # Calculate average gender ratio
            if len(gender_history) > 0:
                avg_male = np.mean([g[0] for g in gender_history])
                avg_female = np.mean([g[1] for g in gender_history])
                avg_total = np.mean([g[2] for g in gender_history])
            else:
                avg_male = avg_female = avg_total = 0

            # ===== VIOLENCE CLASSIFICATION =====
            if frame_count >= SEQ_LENGTH * FRAME_SKIP:

                lstm_interpreter.set_tensor(
                    lstm_input_details[0]['index'], sequence_buffer)
                lstm_interpreter.invoke()
                prediction = lstm_interpreter.get_tensor(
                    lstm_output_details[0]['index'])[0, 0]

                prediction_history.append(prediction)
                smoothed_prediction = np.mean(prediction_history)

                # Require consistent high scores
                recent_high = sum(
                    1 for p in prediction_history if p > VIOLENCE_THRESHOLD)
                is_violence = smoothed_prediction > VIOLENCE_THRESHOLD and recent_high >= 3

                # Display violence status
                if is_violence:
                    v_status = "⚠️  VIOLENCE DETECTED"
                    v_color = (0, 0, 255)  # Red
                    cv2.rectangle(display_frame, (0, 0), (w, h), v_color, 8)
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {v_status} - Score: {smoothed_prediction:.3f}")
                else:
                    v_status = "✓ NORMAL"
                    v_color = (0, 255, 0)  # Green

                # ===== DISPLAY INFO =====
                # Top panel - Violence status
                cv2.rectangle(display_frame, (0, 0), (w, 100), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (0, 0), (w, 100), v_color, 3)

                cv2.putText(display_frame, v_status, (10, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, v_color, 2)
                cv2.putText(display_frame, f"Confidence: {smoothed_prediction:.1%}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Bottom panel - Gender info
                panel_y = h - 120
                cv2.rectangle(display_frame, (0, panel_y),
                              (w, h), (0, 0, 0), -1)
                cv2.rectangle(display_frame, (0, panel_y),
                              (w, h), (255, 255, 255), 2)

                cv2.putText(display_frame, "GENDER DETECTION", (10, panel_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.putText(display_frame, f"Male: {male_count} (Avg: {avg_male:.1f})",
                            (10, panel_y + 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                cv2.putText(display_frame, f"Female: {female_count} (Avg: {avg_female:.1f})",
                            (10, panel_y + 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                cv2.putText(display_frame, f"Total Faces: {total_faces}",
                            (10, panel_y + 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Gender ratio bar
                if avg_total > 0:
                    male_ratio = avg_male / avg_total
                    bar_width = int((w - 40) * male_ratio)

                    # Male bar (blue)
                    cv2.rectangle(display_frame, (w//2 + 10, panel_y + 50),
                                  (w//2 + 10 + bar_width, panel_y + 70), (255, 0, 0), -1)
                    # Female bar (pink)
                    cv2.rectangle(display_frame, (w//2 + 10 + bar_width, panel_y + 50),
                                  (w - 20, panel_y + 70), (255, 0, 255), -1)

                    cv2.putText(display_frame, f"M/F Ratio: {male_ratio:.0%}/{1-male_ratio:.0%}",
                                (w//2 + 10, panel_y + 95),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            else:
                # Initializing
                progress = (frame_count / (SEQ_LENGTH * FRAME_SKIP)) * 100
                cv2.rectangle(display_frame, (0, 0), (w, 60), (0, 0, 0), -1)
                cv2.putText(display_frame, f"Initializing... {progress:.0f}%", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # FPS counter
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (w - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # Display
        cv2.imshow('Violence + Gender Detection', display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            print("\n🛑 Stopping...")
            break

        frame_count += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    # Final statistics
    if len(gender_history) > 0:
        total_male = sum(g[0] for g in gender_history)
        total_female = sum(g[1] for g in gender_history)
        total_all = total_male + total_female

        print("\n" + "="*70)
        print("SESSION STATISTICS")
        print("="*70)
        print(f"Total Male detections: {total_male}")
        print(f"Total Female detections: {total_female}")
        if total_all > 0:
            print(f"Male ratio: {total_male/total_all*100:.1f}%")
            print(f"Female ratio: {total_female/total_all*100:.1f}%")
        print("="*70)

    print("✅ System stopped")


if __name__ == "__main__":
    main()
