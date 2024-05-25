import cv2
import numpy as np
import mediapipe as mp
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from collections import deque, Counter
from mediapipe.python.solutions import pose as mp_pose


class FullBodyPoseEmbedder(object):
  """Converts 3D pose landmarks into 3D embedding."""

  def __init__(self, torso_size_multiplier=2.5):
    # Multiplier to apply to the torso to get minimal body size.
    self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
    self._landmark_names = [
        'nose',
        'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer',
        'left_ear', 'right_ear',
        'mouth_left', 'mouth_right',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_pinky_1', 'right_pinky_1',
        'left_index_1', 'right_index_1',
        'left_thumb_2', 'right_thumb_2',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_ankle', 'right_ankle',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index',
    ]

  def __call__(self, landmarks):
    """Normalizes pose landmarks and converts to embedding

    Args:
      landmarks - NumPy array with 3D landmarks of shape (N, 3).

    Result:
      Numpy array with pose embedding of shape (M, 3) where `M` is the number of
      pairwise distances defined in `_get_pose_distance_embedding`.
    """
    assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

    # Get pose landmarks.
    landmarks = np.copy(landmarks)

    # Normalize landmarks.
    landmarks = self._normalize_pose_landmarks(landmarks)

    # Get embedding.
    # Get embedding.
    distance_embedding = self._get_pose_distance_embedding(landmarks)
    angle_embedding = self._get_pose_angle_embedding(landmarks)
    distance3D_embedding = self._get_pose_3Ddistance_embedding(landmarks)
    return landmarks, distance_embedding, distance3D_embedding,angle_embedding

  def _normalize_pose_landmarks(self, landmarks):
    """Normalizes landmarks translation and scale."""
    landmarks = np.copy(landmarks)

    # Normalize translation.
    pose_center = self._get_pose_center(landmarks)
    landmarks -= pose_center

    # Normalize scale.
    pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
    landmarks /= pose_size
    # Multiplication by 100 is not required, but makes it eaasier to debug.
    landmarks *= 100

    return landmarks

  def _get_pose_center(self, landmarks):
    """Calculates pose center as point between hips."""
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    center = (left_hip + right_hip) * 0.5
    return center

  def _get_pose_size(self, landmarks, torso_size_multiplier):
    """Calculates pose size.

    It is the maximum of two values:
      * Torso size multiplied by `torso_size_multiplier`
      * Maximum distance from pose center to any pose landmark
    """
    # This approach uses only 2D landmarks to compute pose size.
    landmarks = landmarks[:, :2]

    # Hips center.
    left_hip = landmarks[self._landmark_names.index('left_hip')]
    right_hip = landmarks[self._landmark_names.index('right_hip')]
    hips = (left_hip + right_hip) * 0.5

    # Shoulders center.
    left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
    right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
    shoulders = (left_shoulder + right_shoulder) * 0.5

    # Torso size as the minimum body size.
    torso_size = np.linalg.norm(shoulders - hips)

    # Max dist to pose center.
    pose_center = self._get_pose_center(landmarks)
    max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

    return max(torso_size * torso_size_multiplier, max_dist)

  def _get_pose_distance_embedding(self, landmarks):
      """Converts pose landmarks into 3D embedding.

      We use several pairwise 3D distances to form pose embedding. All distances
      include X and Y components with sign. We differnt types of pairs to cover
      different pose classes. Feel free to remove some or add new.

      Args:
        landmarks - NumPy array with 3D landmarks of shape (N, 3).

      Result:
        Numpy array with pose embedding of shape (M, 3) where `M` is the number of
        pairwise distances.
      """
      embedding = np.array([
          # One joint.

          # self._get_distance(
          #    self._get_average_by_names(landmarks, 'left_hip', 'right_hip')[0],
          #    self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder'))[0],
          #
          # self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow')[0],
          # self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow')[0],
          #
          # self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist')[0],
          # self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist')[0],
          #
          # self._get_distance_by_names(landmarks, 'left_hip', 'left_knee')[0],
          # self._get_distance_by_names(landmarks, 'right_hip', 'right_knee')[0],
          #
          # self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle')[0],
          # self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle')[0],

          # Two joints.

          self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist')[0],
          self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist')[0],

          self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle')[0],
          self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle')[0],

          # Four joints.

          self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist')[0],
          self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist')[0],

          # Five joints.

          self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle')[0],
          self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle')[0],

          self._get_distance_by_names(landmarks, 'left_hip', 'right_wrist')[0],
          self._get_distance_by_names(landmarks, 'right_hip', 'left_wrist')[0],

          # Cross body.

          self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow')[0],
          self._get_distance_by_names(landmarks, 'left_knee', 'right_knee')[0],

          self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist')[0],
          self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle')[0],

          # Body bent direction.

          self._get_distance(
              self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
              landmarks[self._landmark_names.index('left_hip')])[0],
          self._get_distance(
              self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
              landmarks[self._landmark_names.index('right_hip')])[0],

      ])

      return embedding
  def _get_pose_3Ddistance_embedding(self, landmarks):
      """Converts pose landmarks into 3D embedding.

      We use several pairwise 3D distances to form pose embedding. All distances
      include X and Y components with sign. We differnt types of pairs to cover
      different pose classes. Feel free to remove some or add new.

      Args:
        landmarks - NumPy array with 3D landmarks of shape (N, 3).

      Result:
        Numpy array with pose embedding of shape (M, 3) where `M` is the number of
        pairwise distances.
      """
      embedding_3d = np.array([
          #One joint.

          # self._get_distance(
          #    self._get_average_by_names(landmarks, 'left_hip', 'right_hip')[1],
          #    self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder'))[1],
          #
          # self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow')[1],
          # self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow')[1],
          #
          # self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist')[1],
          # self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist')[1],
          #
          # self._get_distance_by_names(landmarks, 'left_hip', 'left_knee')[1],
          # self._get_distance_by_names(landmarks, 'right_hip', 'right_knee')[1],
          #
          # self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle')[1],
          # self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle')[1],

          # Two joints.

          self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist')[1],
          self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist')[1],

          self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle')[1],
          self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle')[1],

          # Four joints.

          self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist')[1],
          self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist')[1],

          # Five joints.

          self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle')[1],
          self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle')[1],

          self._get_distance_by_names(landmarks, 'left_hip', 'right_wrist')[1],
          self._get_distance_by_names(landmarks, 'right_hip', 'left_wrist')[1],

          # Cross body.

          self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow')[1],
          self._get_distance_by_names(landmarks, 'left_knee', 'right_knee')[1],

          self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist')[1],
          self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle')[1],

          # Body bent direction.

          self._get_distance(
              self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
              landmarks[self._landmark_names.index('left_hip')])[1],
          self._get_distance(
              self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
              landmarks[self._landmark_names.index('right_hip')])[1],

      ])

      return embedding_3d

  def _get_pose_angle_embedding(self, landmarks):
      angle_embedding = np.array([
          self._get_angle_by_names(landmarks, 'right_elbow', 'right_shoulder', 'right_hip'),
          self._get_angle_by_names(landmarks, 'left_elbow', 'left_shoulder', 'left_hip'),

          self._get_angle_by_names(landmarks, 'right_knee', 'mid_hip', 'left_knee'),

          self._get_angle_by_names(landmarks, 'right_hip', 'right_knee', 'right_ankle'),
          self._get_angle_by_names(landmarks, 'left_hip', 'left_knee', 'left_ankle'),

          self._get_angle_by_names(landmarks, 'right_wrist', 'right_elbow', 'right_shoulder'),
          self._get_angle_by_names(landmarks, 'left_wrist', 'left_elbow', 'left_shoulder')
      ])
      return angle_embedding
  def _get_average_by_names(self, landmarks, name_from, name_to):
    lmk_from = landmarks[self._landmark_names.index(name_from)]
    lmk_to = landmarks[self._landmark_names.index(name_to)]
    return (lmk_from + lmk_to) * 0.5

  def _get_distance_by_names(self, landmarks, name_from, name_to):
      lmk_from = landmarks[self._landmark_names.index(name_from)]
      lmk_to = landmarks[self._landmark_names.index(name_to)]
      # name_list.append("x_"+name_from+"_"+name_to)
      # name_list.append("y_"+name_from+"_"+name_to)
      # name_list.append("z_"+name_from+"_"+name_to)

      squared_dist = np.sum((lmk_from - lmk_to) ** 2, axis=0)
      dist_3D = np.sqrt(squared_dist)

      return self._get_distance(lmk_from, lmk_to)[0], self._get_distance(lmk_from, lmk_to)[1]

  def _get_distance(self, lmk_from, lmk_to):
    squared_dist = np.sum((lmk_from-lmk_to)**2, axis=0)
    dist_3D = np.sqrt(squared_dist)
    return lmk_to - lmk_from, dist_3D

  def _get_angle_by_names(self, landmarks, lmk1_name, lmk2_name, lmk3_name):
      lmk1 = landmarks[self._landmark_names.index(lmk1_name)]
      if lmk2_name == 'mid_hip':
          lmk2 = self._get_average_by_names(landmarks, 'left_hip', 'right_hip')
      else:
          lmk2 = landmarks[self._landmark_names.index(lmk2_name)]
      lmk3 = landmarks[self._landmark_names.index(lmk3_name)]
      # name_list.append(lmk1_name+"_"+lmk2_name+"_"+lmk3_name)
      return self._get_angle(lmk1, lmk2, lmk3)

  def _get_angle(self, a, b, c):
      ba = a - b
      bc = c - b

      cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
      angle = np.arccos(cosine_angle)

      return np.degrees(angle)

# Load models
randomForestClassifier = joblib.load("./random_forest.joblib")
label_encoder = joblib.load("./label_encoder.joblib")

# Set up MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Video capture setup
cap = cv2.VideoCapture('Jumping Jack Nasıl Yapılır_.mp4')  # Change this to the appropriate video file for testing
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (frame_width, frame_height))

pose_embedder = FullBodyPoseEmbedder()

# Initialize counters and statuses for each exercise
counters = {
    'pushups': 0,
    'jumping_jacks': 0,
    'pullups': 0,
    'situps': 0,
    'squats': 0
}
statuses = {
    'pushups': '',
    'jumping_jacks': '',
    'pullups': '',
    'situps': '',
    'squats': ''
}

# Initialize deque for label smoothing
label_window = deque(maxlen=10)

# Function to update counters and statuses
def update_counter(label_str, last_status, count, up_label, down_label):
    if label_str == up_label:
        last_status = "up"
    if label_str == down_label and last_status == "up":
        last_status = "down"
        count += 1
    return last_status, count

# Read and process the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = pose.process(frame_rgb)
    frame_rgb.flags.writeable = True

    # Draw pose landmarks
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # Extract pose landmarks as numpy array
        pose_landmarks = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark], dtype=np.float32)

        # Process landmarks through the embedder
        if pose_landmarks.shape == (33, 3):  # Ensure correct landmarks shape
            landmarks, distance_embedding, distance3D_embedding, angle_embedding = pose_embedder(pose_landmarks)
            features = np.concatenate((distance3D_embedding, angle_embedding), axis=0)
            features = np.reshape(features, (1, features.size))

            # Predict label with probability
            label_probs = randomForestClassifier.predict_proba(features)
            label_numeric = label_probs.argmax(axis=1)
            confidence_score = label_probs.max(axis=1)[0]

            # Initialize label_str to handle cases where confidence is low
            label_str = ""

            # Only update if confidence score is above threshold
            confidence_threshold = 0.5  # Lowered the threshold
            if confidence_score >= confidence_threshold:
                label_str = label_encoder.inverse_transform(label_numeric)[0]  # Convert numeric label to string

                # Add label to deque for smoothing only if it's not empty
                if label_str:
                    label_window.append(label_str)
                    most_common_label = Counter(label_window).most_common(1)[0][0]

                    # Display label on the video
                    cv2.putText(frame, most_common_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                    # Update counters and statuses for each exercise
                    statuses['pushups'], counters['pushups'] = update_counter(most_common_label, statuses['pushups'], counters['pushups'], "pushups_up", "pushups_down")
                    statuses['jumping_jacks'], counters['jumping_jacks'] = update_counter(most_common_label, statuses['jumping_jacks'], counters['jumping_jacks'], "jumping_jacks_up", "jumping_jacks_down")
                    statuses['pullups'], counters['pullups'] = update_counter(most_common_label, statuses['pullups'], counters['pullups'], "pullups_up", "pullups_down")
                    statuses['situps'], counters['situps'] = update_counter(most_common_label, statuses['situps'], counters['situps'], "situp_up", "situp_down")
                    statuses['squats'], counters['squats'] = update_counter(most_common_label, statuses['squats'], counters['squats'], "squats_up", "squats_down")

                    # Display the counter for the current exercise only
                    if most_common_label in ["pushups_up", "pushups_down"]:
                        cv2.putText(frame, f"Push-ups: {counters['pushups']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif most_common_label in ["jumping_jacks_up", "jumping_jacks_down"]:
                        cv2.putText(frame, f"Jumping Jacks: {counters['jumping_jacks']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif most_common_label in ["pullups_up", "pullups_down"]:
                        cv2.putText(frame, f"Pull-ups: {counters['pullups']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif most_common_label in ["situp_up", "situp_down"]:
                        cv2.putText(frame, f"Sit-ups: {counters['situps']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    elif most_common_label in ["squats_up", "squats_down"]:
                        cv2.putText(frame, f"Squats: {counters['squats']}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            #else:
            #    # For debugging: show confidence score
            #    cv2.putText(frame, f"Confidence: {confidence_score:.2f}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()