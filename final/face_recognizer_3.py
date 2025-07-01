import cv2
import face_recognition
import torch
from facenet_pytorch import MTCNN
from face_embedding_db_2 import FaceEmbeddingDB
import numpy as np
from PIL import Image
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from final.face_embedding_db_3 import FaceEmbeddingDB  # Adjust path accordingly

import logging
logging.getLogger("insightface").setLevel(logging.WARNING)

def is_valid_rgbimage(img):
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return False
    return True

def is_valid_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None or img.shape[0] == 0 or img.shape[1] == 0:
        return False
    return True



def expand_or_contract_box(box, scale, image_shape):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    center_x, center_y = x1 + w // 2, y1 + h // 2
    new_w, new_h = int(w * scale), int(h * scale)
    x1_new = max(0, center_x - new_w // 2)
    y1_new = max(0, center_y - new_h // 2)
    x2_new = min(image_shape[1], center_x + new_w // 2)
    y2_new = min(image_shape[0], center_y + new_h // 2)
    return [x1_new, y1_new, x2_new, y2_new]

def mirror_image(image):
    return cv2.flip(image, 1)

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)



class FaceRecognizer:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_db = FaceEmbeddingDB(device=device)

        self.model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))

    def _reload_fdb(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_db = FaceEmbeddingDB(device=device)

    def detect_faces_from_response(self, rgb_image):
        """Detect faces using ArcFace and return bounding boxes and RGB image"""
        
        faces = self.model.get(rgb_image)

        #import pdb; pdb.set_trace()

        face_locations = []
        fin_faces = []
        for face in faces:
            box = face.bbox.astype(int)  # [x1, y1, x2, y2]
            
            det_score = getattr(face, 'det_score', 1.0)  # May not be available in all builds
            top, right, bottom, left = box[1], box[2], box[3], box[0]

            face_width = right - left
            face_height = bottom - top

            # Filter conditions
            if face_width < 30 or face_height < 30:
                # print(f"Skipped face: too small  face_width:{face_width}  face_height:{face_height}")
                continue
            
            if det_score < 0.83:
                # print(f"Skipped face: low quality (det_score={det_score:.2f})")
                continue
            
            face_locations.append((top, right, bottom, left))
            fin_faces.append(face)

        return face_locations, fin_faces

    def identify_faces(self, image_content):
        if(not is_valid_image(image_content)):
            print("A ERROR! EMPTY IMAGE.")
            return [],[]

        image_array = np.frombuffer(image_content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if(not is_valid_rgbimage(rgb_image)):
            print("B ERROR! EMPTY IMAGE.")
            return [],[]

        face_locations, faces = self.detect_faces_from_response(rgb_image)
        self._reload_fdb()

        #faces = self.model.get(rgb_image)
        results = []

        for face in faces:
            box = face.bbox.astype(int)
            top, right, bottom, left = box[1], box[2], box[3], box[0]
            embedding = face.embedding
            
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Now augment and collect embeddings
            aug_embeddings = []

            # Expand & Contracted
            expanded_box = expand_or_contract_box(box, 1.2, rgb_image.shape)
            contracted_box = expand_or_contract_box(box, 0.8, rgb_image.shape)

            for bx in [expanded_box, contracted_box]:
                x1_e, y1_e, x2_e, y2_e = bx
                if(y2_e>y1_e and x2_e>x1_e):
                    crop = rgb_image[y1_e:y2_e, x1_e:x2_e]
                    # face_aug = self.model.get()
                    _,face_aug = self.detect_faces_from_response(crop)
                    if face_aug: aug_embeddings.append(face_aug[0].embedding)

            # Mirror
            if(y2>y1 and x2>x1 and y2>0 and x2>0 and x1>0 and y1>0):
                face_crop = rgb_image[y1:y2, x1:x2]
                if (face_crop is None) or (face_crop.size == 0):
                    print(f"[WARN] Empty face_crop at box: {(x1, y1, x2, y2)}")
                    # import pdb; pdb.set_trace()

                mirrored = mirror_image(face_crop)
                if (mirrored is None) or (mirrored.size == 0):
                    print(f"[WARN] Empty mirrored at box: {(x1, y1, x2, y2)}")
                    # import pdb; pdb.set_trace()

                # face_aug = self.model.get(mirrored)
                _,face_aug = self.detect_faces_from_response(crop)
                if face_aug: aug_embeddings.append(face_aug[0].embedding)

                # Rotations
                for angle in [20, -20]:
                    rotated = rotate_image(face_crop, angle)
                    # face_aug = self.model.get(rotated)
                    _,face_aug = self.detect_faces_from_response(crop)
                    if face_aug: aug_embeddings.append(face_aug[0].embedding)

                if not ( (mirrored is None) or (mirrored.size == 0) ):
                    # Mirrored rotations
                    for angle in [20, -20]:
                        rot_mirror = rotate_image(mirrored, angle)
                        # face_aug = self.model.get(rot_mirror)
                        _,face_aug = self.detect_faces_from_response(crop)
                        if face_aug: aug_embeddings.append(face_aug[0].embedding)

            clust_id, face_id = self.face_db.match_embedding(embedding, aug_embeddings)

            
            
            if face_id:
                results.append({"clust_id":clust_id, "face_id": face_id, "location": (top, right, bottom, left)})

        return results

