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
    def __init__(self, db_path="final/face_embeddings_db.pkl", face_map_path="final/faces_map_db.pkl", device="cuda"):
        self.db_path = db_path
        # self.emb_count = 0
        self.embeddings = {}  # face_id -> embedding , 0 marks the number of faces seen till now
        self.faces_map = {} # specific face_ids that are merged with another face_id, these face_ids have no existence of their own, rather use the value of the dict for their id as the original face id
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
                with open(self.face_map_path, "rb") as f:
                    self.faces_map = pickle.load(f)
                print(f"Loaded {len(self.embeddings)} face embeddings from database")
            except Exception as e:
                print(f"Error loading face database: {e}")
                self.embeddings = {}

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        with open(self.face_map_path, "wb") as f:
            pickle.dump(self.faces_map, f)
        print(f"Saved {len(self.embeddings)} face embeddings to database. and {len(self.faces_map)} faces_map to database.")

    def match_embedding(self, face_embedding, aug_embeddings, threshold=0.6): 
        # return the best matched id, along with all the other ids that need to be changed to this best id
        best_score = -1
        best_id = None

        face_embedding = face_embedding / np.linalg.norm(face_embedding)

        all_matches = []
        for face_id, stored_embeddings in self.embeddings.items():
            for stored_embedding in stored_embeddings:
                stored_embedding = stored_embedding / np.linalg.norm(stored_embedding)

                # We need augmented embeddings for the same face
                # So everytime we save a new face we should also store its augmented face_embeddings
                # In case of multiple matches we need to combine an old face id into another old face id
                # We can store this face id as an augmentation for future matching
                # The face_id needs to be matched in the back end
                # So we need an additional related_face_ids table in DB

                for aug_embedding_ in aug_embeddings+[face_embedding]:
                    aug_embedding_ = aug_embedding_ / np.linalg.norm(aug_embedding_)
                    score = np.dot(stored_embedding, aug_embedding_)
                    
                    if score>threshold:
                        all_matches.append( (face_id, score) )

                    if score > best_score:
                        best_score = score
                        best_id = face_id

        
        # print(f"best_score:{best_score}   threshold:{threshold}  best_face_id:{best_id}")
        if best_score >= threshold:
            # although we are returning best_id, but the new face which is of best_id can add more information to best_id!
            # hence we need to store everything for best_id!
            # so....

            self.embeddings[best_id].append(face_embedding)
            self.save_db()

            # It is also possible that there are multiple matches for a face
            # Means 2 face ids getting reported, so we need to store only 1
            # At the same time we need to remove one of the face_ids, but need to maintain the last known face_id

            return best_id

        new_id = self._generate_new_id()
        # print(f"generating new face id: {new_id}")
        self.embeddings[new_id] = [face_embedding]
        self.save_db()

        return new_id

    

    def _generate_new_id(self):
        if not self.embeddings:
            return 1
        
        # self.emb_count += 1
        
        # return self.emb_count

        return max(self.embeddings.keys()) + 1



        