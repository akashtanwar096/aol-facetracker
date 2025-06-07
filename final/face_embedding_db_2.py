import os
import pickle
import face_recognition
from numpy import dot
from numpy.linalg import norm
import os
import pickle
import numpy as np
from numpy.linalg import norm
from numpy import dot
from insightface.app import FaceAnalysis


class FaceEmbeddingDB:
    def __init__(self, db_path="final/face_embeddings_db.pkl", device="cuda"):
        self.db_path = db_path
        self.embeddings = {}  # face_id -> embedding
        self.load_db()

        # Load ArcFace model
        self.model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

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

    def match_embedding(self, face_embedding, threshold=0.6):
        best_score = -1
        best_id = None

        face_embedding = face_embedding / np.linalg.norm(face_embedding)

        for face_id, stored_embedding in self.embeddings.items():
            stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)
            score = np.dot(stored_embedding, face_embedding)
            if score > best_score:
                best_score = score
                best_id = face_id

        # print(f"best_score:{best_score}   threshold:{threshold}  best_face_id:{best_id}")
        if best_score >= threshold:
            return best_id

        new_id = self._generate_new_id()
        # print(f"generating new face id: {new_id}")
        self.embeddings[new_id] = face_embedding
        self.save_db()
        return new_id

    def get_face_id(self, face_image, threshold=0.6):
        """Get face ID using ArcFace, create new one if needed"""

        faces = self.model.get(face_image)
        if not faces:
            return None

        face_embedding = faces[0].embedding

        # Compare against known embeddings
        best_score = -1
        best_id = None
        for face_id, stored_embedding in self.embeddings.items():
            score = self._cosine_similarity(stored_embedding, face_embedding)
            if score > best_score:
                best_score = score
                best_id = face_id

        if best_score >= threshold:
            return best_id
        
        print(f"best_score:{best_score}   threshold:{threshold}")
        # No match, create new ID
        new_id = self._generate_new_id()
        self.embeddings[new_id] = face_embedding
        self.save_db()
        return new_id

    def _generate_new_id(self):
        if not self.embeddings:
            return 1
        return max(self.embeddings.keys()) + 1



        