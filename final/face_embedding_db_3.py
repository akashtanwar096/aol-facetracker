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
    def __init__(self, db_path="final/face_embeddings_db.pkl", cluster_map_path="final/cluster_db.pkl", device="cuda"):
        self.db_path = db_path
        self.cluster_map_path = cluster_map_path
        
        
        self.embeddings = {}  # face_id -> embedding , 0 marks the number of faces seen till now
        self.clusters = {} # cluster id and list of face ids in the cluster
        self.reverse_cluster_map = {}

        self.load_db()

        # Load ArcFace model
        self.model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))

    def _cosine_similarity(self, a, b):
        return dot(a, b) / (norm(a) * norm(b))

    def _populate_reverse_cluster_map(self):
        for k in self.clusters.keys():
            for v in self.clusters[k]:
                self.reverse_cluster_map[v] = k

    def load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "rb") as f:
                    self.embeddings = pickle.load(f)
                with open(self.cluster_map_path, "rb") as f:
                    self.clusters = pickle.load(f)
                    self._populate_reverse_cluster_map()
                print(f"Loaded {len(self.embeddings)} face embeddings from database and {len(self.clusters)} clusters")
            except Exception as e:
                print(f"🔴 ERROR loading face database: {e}")
                self.embeddings = {}
                self.clusters = {}
                self.reverse_cluster_map = {}

    def save_db(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.embeddings, f)
        with open(self.cluster_map_path, "wb") as f:
            pickle.dump(self.clusters, f)
        print(f"Saved {len(self.embeddings)} face embeddings to database. and {len(self.clusters)} clusters to database.")

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
                    
                    if score>=threshold:
                        all_matches.append( (face_id, score) )

                    if score > best_score:
                        best_score = score
                        best_id = face_id

        
        # print(f"best_score:{best_score}   threshold:{threshold}  best_face_id:{best_id}")

        if best_score >= threshold:
            # Although we are returning best_id, but the new face which is of best_id can add more information to best_id!
            # hence we need to store everything for best_id!
            # so....

            # We need to merge the clusters of all_matches
            # Consider cluster id of each face id and re-assign the cluster if needed
            for fid,sc in all_matches:
                # assign all fids the best cluster id
                if(self.reverse_cluster_map[fid] != self.reverse_cluster_map[best_id]):
                    print(f"CLUSTERS MERGED: of face_id: {fid}  from clust_id:{self.reverse_cluster_map[fid]}  to  new clust_id:{self.reverse_cluster_map[best_id]}")
                    self._assign_cluster(fid, self.reverse_cluster_map[best_id])
                    # when some faceid went to best_id later cluster we did not update it in faces.db
                    # so this will be updated too


            self.embeddings[best_id].append(face_embedding)

            # for all the other matches of face_id -> a mapping needs to be made to the best face_id
            # print(f"a  embedding length:{len(self.embeddings.keys())}")
            self.save_db()

            # It is also possible that there are multiple matches for a face
            # Means 2 face ids getting reported, so we need to store only 1
            # At the same time we need to remove one of the face_ids, but need to maintain the last known face_id

            return self.reverse_cluster_map[best_id], best_id, all_matches

        new_cluster_id, new_face_id = self._generate_new_id()

        # print(f"generating new face id: {new_id}")
        self.embeddings[new_face_id] = [face_embedding]
        if(new_cluster_id not in self.clusters.keys()):
            self.clusters[new_cluster_id] = [new_face_id]

        # print(f"b new embedding length:{len(self.embeddings.keys())}")
        self.save_db()

        return new_cluster_id, new_face_id, []

    def _generate_new_cluster_id(self):
        if not self.clusters:
            return 1
        else:
            return max(self.clusters.keys()) + 1

    def _assign_cluster(self, face_id, clust_id=None):

        if(clust_id is None):
            clust_id = self._generate_new_cluster_id()
            # self.clusters[clust_id] = [face_id] # initialize

        print(f"assigning cluster : {clust_id}  to face_id :{face_id} ")
        if(face_id not in self.reverse_cluster_map.keys()):
            self.clusters[clust_id] = [face_id]
            self.reverse_cluster_map[face_id] = clust_id
        else:
            # means face_id is already assigned a cluster before
            # first remove from the previous cluster
            
            if( self.reverse_cluster_map[face_id] == clust_id ):
                # clust id is already the same
                return clust_id

            if face_id in self.clusters[self.reverse_cluster_map[face_id]]:
                self.clusters[self.reverse_cluster_map[face_id]].remove(face_id)

            # now add the face_id to the clust_id
            if(clust_id in self.clusters):
                self.clusters[clust_id].append(face_id)
            else:
                self.clusters[clust_id] = [face_id]

            self.reverse_cluster_map[face_id] = clust_id
        return clust_id


    def merge_clusters(self, clust_id_1, clust_id_2):
        # all the face_ids in clust_id_2 must move to clust_id_1
        # and clust_id_2 must become empty
        if( ( clust_id_1 not in self.clusters.keys() ) or ( clust_id_2 not in self.clusters.keys() ) ):
            print(f"ERROR: either {clust_id_1} is not in self.clusters or {clust_id_2}")
            return
        
        fids_2 = list(self.clusters[clust_id_2])
        print(f"{fids_2}  - list of faces in : cluster id {clust_id_2}")
        for fid2 in fids_2:
            print(f"fid:{fid2} is assigned the cluster:{clust_id_1}")
            self._assign_cluster(fid2, clust_id_1)
        print(f"clust_id_2 is now empty. total_faces in clusterid2: {len(fids_2)}")
        self.clusters[clust_id_2] = [] # empty the cluster 2, all in cluster 1
        self.save_db()
        return fids_2,clust_id_1 # return all the faceids whose cluster was changed to cluster_id_1


    def unmerge_faces(self, face_id_1, face_id_2):
        # face_id_1 and face_id_2 should have the same cluster id
        # Now find any other empty cluster id and assign face_id_2 that

        if(self.reverse_cluster_map[face_id_1] != self.reverse_cluster_map[face_id_2]):
            # they already have different clusters, nothing to do
            print(f"ERROR: clust id of f1:{face_id_1} - {self.reverse_cluster_map[face_id_1]}!= clust id of f2:{face_id_2} - {self.reverse_cluster_map[face_id_2]}")
            return
        
        existing_cluster = self.reverse_cluster_map[face_id_1]

        chosen_cluster = None
        for k in self.clusters.keys():
            if(len(self.clusters[k])==0):
                print(f"choosing the cluster id : {k}  because it has 0 faceids.")
                chosen_cluster = k
                break
        
        if(chosen_cluster is None):
            chosen_cluster = self._generate_new_cluster_id()
            print(f"No empty cluster found so assigning new : {chosen_cluster}")
        
        assign_cluster = self._assign_cluster(face_id_2, chosen_cluster)
        
        self.save_db()

        return assign_cluster


    def _generate_new_id(self):
        if not self.embeddings:
            clust_id = self._assign_cluster(1)
            return clust_id,1
        
        # a new face demands a new cluster id and a new face id
        new_face_id = max(self.embeddings.keys()) + 1

        clust_id = self._assign_cluster(new_face_id)

        return clust_id,new_face_id

