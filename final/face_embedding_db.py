import os
import pickle
import face_recognition
from numpy import dot
from numpy.linalg import norm


class FaceEmbeddingDB:
    def __init__(self, db_path="final/face_embeddings_db.pkl"):
        self.db_path = db_path
        self.embeddings = {}  # face_id -> embedding
        self.load_db()


        # # Load ArcFace model once
        # self.model = insightface.app.FaceAnalysis(name='buffalo_l')
        # self.model.prepare(ctx_id=0, det_size=(640, 640))  # ctx_id=-1 for CPU

    def _cosine_similarity(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.embeddings = pickle.load(f)
                print(f"Loaded {len(self.embeddings)} face embeddings from database")
            except Exception as e:
                print(f"Error loading face database: {e}")
                self.embeddings = {}

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        print(f"Saved {len(self.embeddings)} face embeddings to database")

    # def get_face_id(self, face_image, location=None, threshold=0.5):
    #     """Get face ID using ArcFace, create new one if needed"""
    #     faces = self.model.get(face_image)
    #     if not faces:
    #         return None

    #     face_embedding = faces[0].embedding

    #     # Compare against known embeddings
    #     best_score = -1
    #     best_id = None
    #     for face_id, stored_embedding in self.embeddings.items():
    #         score = self._cosine_similarity(stored_embedding, face_embedding)
    #         if score > best_score:
    #             best_score = score
    #             best_id = face_id

    #     if best_score >= threshold:
    #         return best_id

    #     # No match, create new ID
    #     new_id = self._generate_new_id()
    #     self.embeddings[new_id] = face_embedding
    #     self.save_db()
    #     return new_id

    def get_face_id(self, face_image, location=None, threshold=0.4):
        """Get face ID for a face image, creating a new one if needed"""
        # Get embedding for this face
        face_encoding = face_recognition.face_encodings(
            face_image, known_face_locations=[location]
        )

        if not face_encoding:
            return None  # No face detected

        face_encoding = face_encoding[0]

        # Check if this face matches any known face
        minimum_distance = +100.0
        minimum_distance_face_id = None
        for face_id, stored_encoding in self.embeddings.items():
            # Compare face encodings
            distance = face_recognition.face_distance([stored_encoding], face_encoding)[
                0
            ]

            # Watch for all matches!!!

            if(distance < minimum_distance):
                minimum_distance = distance
                minimum_distance_face_id = face_id
            # If match found, return existing ID
            # if distance < threshold:
                # is this match valid?????

                # return face_id

        if(minimum_distance < threshold):
            return minimum_distance_face_id

        # No match found, create new ID
        new_id = self._generate_new_id()
        self.embeddings[new_id] = face_encoding
        self.save_db()
        return new_id

    def _generate_new_id(self):
        """Generate a new unique face ID"""
        if not self.embeddings:
            return 1
        return max(self.embeddings.keys()) + 1
