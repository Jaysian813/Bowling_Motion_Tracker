import cv2
import mediapipe as mp
import numpy as np
import math


# Mediapipe Global Dictionary
KP = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    
    # Torso
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_hip': 23, 'right_hip': 24,
    
    # Arms & Hands
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    
    # Legs & Feet
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

class BowlingSwingAnalyzer:
    def __init__(self, fps=30):
        self.fps = fps
        self.dt = 1.0 / fps
        
        print("--- Environment Setup ---")
        print("AI Engine: MediaPipe Pose (BlazePose 33-Keypoint)")
        print(f"Analysis FPS: {self.fps}")
        print("--------------------------")
        
        # 1. Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2, # Level 2 is the most accurate for sports
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.pose_history = []

   # ==========================================
    # BIOMECHANICS & FUNDAMENTALS LOGIC
    # ==========================================

    def get_velocity_and_acceleration(self, positions):
        if len(positions) < 3:
            return np.array([0]), np.array([0])
        velocities = np.gradient(positions, self.dt, axis=0)
        accelerations = np.gradient(velocities, self.dt, axis=0)
        return velocities, accelerations

    def check_head_stability(self):
        if len(self.pose_history) < 3: return True, 0, 0
        
        nose_x = [frame[KP['nose']][0] for frame in self.pose_history]
        nose_y = [frame[KP['nose']][1] for frame in self.pose_history]
        
        _, accel_x = self.get_velocity_and_acceleration(nose_x)
        _, accel_y = self.get_velocity_and_acceleration(nose_y)
        
        max_accel_x, max_accel_y = np.max(np.abs(accel_x)), np.max(np.abs(accel_y))
        
        # Thresholds (pixels/sec^2)
        is_stable = (max_accel_x < 500) and (max_accel_y < 500) 
        return is_stable, max_accel_x, max_accel_y
    
    def check_lead_foot_plant(self):
        if len(self.pose_history) < 5: return False, 0
        
        # Using left_foot_index (toe) or left_ankle for the slide
        foot_x = [frame[self.KP['left_foot_index']][0] for frame in self.pose_history]
        velocities, _ = self.get_velocity_and_acceleration(foot_x)
        
        final_velocities = velocities[-5:] 
        avg_final_speed = np.mean(np.abs(final_velocities))
        
        is_planted = avg_final_speed < 15 
        return is_planted, avg_final_speed
    
    def check_center_of_mass(self, current_frame):
        wrist_x = current_frame[KP['right_wrist']][0]
        ankle_x = current_frame[KP['left_ankle']][0]
        return abs(wrist_x - ankle_x)

    def calculate_spine_angle(self, current_frame):
        shoulder = current_frame[KP['right_shoulder']]
        hip = current_frame[KP['right_hip']]
        
        delta_y = hip[1] - shoulder[1]
        delta_x = hip[0] - shoulder[0]
        return math.degrees(math.atan2(delta_y, delta_x))

    # ==========================================
    # VIDEO PROCESSING LOOP
    # ==========================================

    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video at {video_path}")
            return

        # 1. Ask the video file for its exact frame rate
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 2. Update the physics variables to match reality
        if actual_fps > 0:
            self.fps = actual_fps
            self.dt = 1.0 / self.fps
            print(f"Success: Video loaded. Auto-detected FPS: {self.fps}")

        # Get video dimensions to convert normalized coordinates to pixels
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # MediaPipe requires RGB images
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(image_rgb)

            if results.pose_landmarks:
                # 1. Draw the skeleton on the frame
                self.mp_draw.draw_landmarks(
                    frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

                # 2. Extract coordinates and convert to actual pixels
                current_frame_pixels = {}
                for idx, lm in enumerate(results.pose_landmarks.landmark):
                    current_frame_pixels[idx] = (int(lm.x * width), int(lm.y * height))
                
                self.pose_history.append(current_frame_pixels)

                # 3. Run Biomechanics Checks
                stable_head, _, _ = self.check_head_stability()
                com_distance = self.check_center_of_mass(current_frame_pixels)
                spine_angle = self.calculate_spine_angle(current_frame_pixels)

                # 4. Display the Data on Screen
                cv2.putText(frame, f"Head Stable: {stable_head}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if stable_head else (0, 0, 255), 2)
                cv2.putText(frame, f"COM Dist: {int(com_distance)} px", (20, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                cv2.putText(frame, f"Spine Angle: {int(spine_angle)} deg", (20, 130), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Show the video playing live
            cv2.imshow('Bowling Motion Tracker', frame)

            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Run the script
if __name__ == "__main__":
    tracker = BowlingSwingAnalyzer(fps=30)
    
    # Replace with the name of a test video in your folder
    # Or use your Google Drive file picker function!
    tracker.process_video("test_clip.mp4")