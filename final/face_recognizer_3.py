import cv2
import face_recognition
import torch
from facenet_pytorch import MTCNN
from face_embedding_db_2 import FaceEmbeddingDB
import numpy as np
from PIL import Image
# import insightface
# import onnx
# import onnxruntime
import cv2
import numpy as np
import torch
from insightface.app import FaceAnalysis
from final.face_embedding_db_2 import FaceEmbeddingDB  # Adjust path accordingly

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


class FaceRecognizer:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_db = FaceEmbeddingDB(device=device)

        self.model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
        self.model.prepare(ctx_id=0 if device == 'cuda' else -1, det_size=(640, 640))

    def _reload_fdb(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.face_db = FaceEmbeddingDB(device=device)

    def detect_faces_from_response(self, image_content):
        """Detect faces using ArcFace and return bounding boxes and RGB image"""
        if(not is_valid_image(image_content)):
            print("A ERROR! EMPTY IMAGE.")
            return [],[]

        image_array = np.frombuffer(image_content, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if(not is_valid_rgbimage(rgb_image)):
            print("B ERROR! EMPTY IMAGE.")
            return [],[]
        
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
        face_locations, faces = self.detect_faces_from_response(image_content)
        self._reload_fdb()

        #faces = self.model.get(rgb_image)
        results = []

        for face in faces:
            box = face.bbox.astype(int)
            top, right, bottom, left = box[1], box[2], box[3], box[0]
            embedding = face.embedding
            face_id = self.face_db.match_embedding(embedding)
            if face_id:
                results.append({"face_id": face_id, "location": (top, right, bottom, left)})

        return results
    
    # def identify_faces(self, image_content):
    #     """Detect and identify faces in an image"""
    #     face_locations, rgb_image = self.detect_faces_from_response(image_content)
        
    #     self._reload_fdb()

    #     results = []
    #     for face_location in face_locations:
    #         top, right, bottom, left = face_location
    #         face_image = rgb_image[top:bottom, left:right]
    #         if(not is_valid_rgbimage(face_image)):
    #             print(f"C ERROR! EMPTY FACE IMAGE.  {top}:{bottom}:{left}:{right}")
    #             continue
    #         face_id = self.face_db.get_face_id(face_image)
    #         if face_id:
    #             results.append({"face_id": face_id, "location": face_location})

    #     return results

        