"""
Quick CCTV Connection Test Script
Test your IP camera before deploying
"""

import cv2
import time

print("="*70)
print("CCTV CONNECTION TEST")
print("="*70)

# ============================================================================
# CONFIGURE YOUR CAMERA HERE
# ============================================================================

CAMERA_IP = "192.168.1.100"  # Change to your camera IP
USERNAME = "admin"            # Change to your username
PASSWORD = "password"         # Change to your password
PORT = 554                    # Usually 554 for RTSP

# ============================================================================
# RTSP URL FORMATS (Try each one)
# ============================================================================

rtsp_urls = [
    # Hikvision
    f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{PORT}/Streaming/Channels/101",
    
    # Dahua
    f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{PORT}/cam/realmonitor?channel=1&subtype=0",
    
    # Generic
    f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{PORT}/stream1",
    f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{PORT}/stream2",
    
    # TP-Link
    f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{PORT}/live/ch00_0",
    
    # Axis
    f"rtsp://{USERNAME}:{PASSWORD}@{CAMERA_IP}:{PORT}/axis-media/media.amp",
    
    # Without credentials (if camera allows)
    f"rtsp://{CAMERA_IP}:{PORT}/stream1",
]

print(f"\nTesting camera at: {CAMERA_IP}")
print(f"Username: {USERNAME}")
print(f"Port: {PORT}\n")

# ============================================================================
# TEST EACH URL
# ============================================================================

successful_url = None

for i, url in enumerate(rtsp_urls, 1):
    print(f"\n[{i}/{len(rtsp_urls)}] Testing: {url}")
    print("-" * 70)
    
    try:
        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.isOpened():
            print("❌ Failed to open connection")
            continue
        
        print("⏳ Waiting for frames...")
        
        # Try to read frames for 5 seconds
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            
            if ret and frame is not None:
                frame_count += 1
                
                if frame_count == 1:
                    print(f"✅ SUCCESS! Received first frame")
                    print(f"   Frame size: {frame.shape[1]}x{frame.shape[0]}")
                    print(f"   Channels: {frame.shape[2]}")
                    successful_url = url
                    break
            
            time.sleep(0.1)
        
        cap.release()
        
        if frame_count > 0:
            print(f"✅ Total frames received: {frame_count}")
            break
        else:
            print("❌ No frames received within 5 seconds")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")

# ============================================================================
# RESULTS
# ============================================================================

print("\n" + "="*70)
print("TEST RESULTS")
print("="*70)

if successful_url:
    print("\n✅ CONNECTION SUCCESSFUL!")
    print(f"\nWorking RTSP URL:")
    print(f"  {successful_url}")
    print(f"\nUse this URL in your dashboard:")
    print(f"  camera_source = \"{successful_url}\"")
    print(f"\nOr in sidebar:")
    print(f"  IP: {CAMERA_IP}")
    print(f"  Username: {USERNAME}")
    print(f"  Password: {PASSWORD}")
    print(f"  Port: {PORT}")
else:
    print("\n❌ CONNECTION FAILED")
    print("\nTroubleshooting:")
    print("  1. Check camera IP is correct")
    print("  2. Verify username and password")
    print("  3. Ensure camera is on same network")
    print("  4. Check RTSP is enabled in camera settings")
    print("  5. Try accessing camera web interface first")
    print("  6. Check firewall settings")
    print("  7. Ping camera IP to verify connectivity:")
    print(f"     ping {CAMERA_IP}")

print("\n" + "="*70)

# ============================================================================
# LIVE TEST (if successful)
# ============================================================================

if successful_url:
    print("\nWould you like to see live video? (Press 'q' to quit)")
    
    try:
        cap = cv2.VideoCapture(successful_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print("Opening video window...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            
            if ret:
                # Add text overlay
                cv2.putText(frame, "CCTV Test - Press 'q' to quit", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('CCTV Test', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError during live test: {str(e)}")

print("\nTest complete!")
