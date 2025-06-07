import cv2
import face_recognition
import torch
from facenet_pytorch import MTCNN
from face_embedding_db import FaceEmbeddingDB
import numpy as np
from PIL import Image

def is_high_quality(face_image, min_resolution=80, blur_threshold=150):
    """Check for sharpness and size of face image"""
    h, w = face_image.shape[:2]
    if h < min_resolution or w < min_resolution:
        return False

    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var > blur_threshold

def is_front_face(face_landmarks):
    """Check for near-frontal face using eye and nose landmarks"""
    left_eye = face_landmarks.get("left_eye")
    right_eye = face_landmarks.get("right_eye")
    nose_tip = face_landmarks.get("nose_tip")

    if not left_eye or not right_eye or not nose_tip:
        return False

    # Use mid-point of both eyes
    left = np.mean(left_eye, axis=0)
    right = np.mean(right_eye, axis=0)
    nose = np.mean(nose_tip, axis=0)

    # Horizontal alignment: eyes should be roughly on same y-level
    eye_y_diff = abs(left[1] - right[1])
    eye_x_diff = abs(left[0] - right[0])
    if eye_y_diff / (eye_x_diff + 1e-5) > 0.1:
        return False

    # Nose should be centered between eyes
    mid_eye_x = (left[0] + right[0]) / 2
    nose_offset_ratio = abs(nose[0] - mid_eye_x) / (eye_x_diff + 1e-5)
    return nose_offset_ratio < 0.2  # tighter threshold for more frontal

class FaceRecognizer:
    def __init__(self):
        # self.face_db = FaceEmbeddingDB()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=device)  # Detect all faces
        self.face_db = FaceEmbeddingDB()  # your existing DB logic
    

    def detect_faces_from_response(self, image_content):
        """Detect faces in an image and return locations"""
        image_array = np.frombuffer(image_content, dtype=np.uint8)

        # image = cv2.imread(image_path)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # # Find all face locations
        # pil_image = Image.fromarray(rgb_image)

        # # Detect boxes and cropped face tensors
        # boxes, probs, landmarks = self.mtcnn.detect(pil_image, landmarks=True)
        
        # # Convert to format: (top, right, bottom, left)
        # face_locations = []
        # face_landmarks_list = []

        # if boxes is not None and landmarks is not None:
        #     for box, lm in zip(boxes, landmarks):
        #         x1, y1, x2, y2 = box
        #         top, right, bottom, left = int(y1), int(x2), int(y2), int(x1)
        #         face_locations.append((top, right, bottom, left))

        #         # Convert MTCNN 5-point to 68-point-style keys
        #         face_landmarks_list.append({
        #             'left_eye': [tuple(lm[0])],
        #             'right_eye': [tuple(lm[1])],
        #             'nose_tip': [tuple(lm[2])],
        #             'top_lip': [tuple(lm[3])],  # left mouth corner
        #             'bottom_lip': [tuple(lm[4])]  # right mouth corner
        #         })

        # return face_locations, face_landmarks_list, rgb_image

        face_locations = face_recognition.face_locations(rgb_image)
        face_landmarks_list = face_recognition.face_landmarks(rgb_image, face_locations)

        return face_locations, face_landmarks_list, rgb_image

    def identify_faces(self, image_content):
        """Detect and identify faces in an image"""
        face_locations, face_landmarks_list, rgb_image = self.detect_faces_from_response(image_content)

        # filtered_locations = []
        # for i, landmarks in enumerate(face_landmarks_list):
        #     # Heuristic: if both eyes or mouth corners are not detected, likely a side profile

        #     if is_front_face(landmarks) and is_high_quality(rgb_image):
        #         filtered_locations.append(face_locations[i])

        results = []
        for face_location in face_locations:
            # Extract the face
            top, right, bottom, left = face_location
            face_image = rgb_image[top:bottom, left:right]

            # Get persistent face ID
            face_id = self.face_db.get_face_id(rgb_image, location=face_location) # face id is old face id if previously seen, else a new face id is assigned

            if face_id:
                results.append({"face_id": face_id, "location": face_location})

        return results
