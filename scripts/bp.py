import cv2
import mediapipe as mp
import numpy as np

def extract_kinematic_tensor(video_path, target_time_steps=180):
    # 1. Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False, # Optimizes for video tracking
        model_complexity=1,      # Standard complexity for mobile-level accuracy
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    sequence_data = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # MediaPipe requires RGB format, while OpenCV reads in BGR
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_landmarks = []
            # 2. Extract 33 joints, grabbing only x and y (discarding z)
            for landmark in results.pose_landmarks.landmark:
                frame_landmarks.extend([landmark.x, landmark.y])
            
            sequence_data.append(frame_landmarks)
        else:
            # If tracking fails for a frame, append zeros to maintain temporal alignment
            sequence_data.append(np.zeros(66).tolist())

    cap.release()
    pose.close()

    # Convert the dynamic list into a rigid NumPy array (actual_frames, 66)
    sequence_tensor = np.array(sequence_data)

    # 3. Standardize the sequence length to target_time_steps
    actual_frames = sequence_tensor.shape[0]
    
    if actual_frames < target_time_steps:
        # Pad the end of the sequence with zeros
        padding = np.zeros((target_time_steps - actual_frames, 66))
        sequence_tensor = np.vstack((sequence_tensor, padding))
    elif actual_frames > target_time_steps:
        # Truncate the sequence to fit the fixed time dimension
        sequence_tensor = sequence_tensor[:target_time_steps, :]

    return sequence_tensor

# --- Assembling the Batch ---
# When preparing data for the GRU model:
video_batch = ['CoachA_Lob_001.mp4']

# Process all videos and stack them into a single tensor
batch_data = [extract_kinematic_tensor(vid) for vid in video_batch]
final_batch_tensor = np.array(batch_data)

print(f"Final Tensor Shape: {final_batch_tensor.shape}") 
# Output will be: Final Tensor Shape: (3, 180, 66)
# (batch_size, time_steps, features)