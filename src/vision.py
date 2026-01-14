# vision.py — Brenda Vision (YOLO fuse bug fixed)
import os
import cv2
import numpy as np
import base64
import time
import threading
from typing import Dict, Any, Optional
from ultralytics import YOLO
import tensorflow as tf

# ----------------------------------------------------------------------
# 1. MODELS
# ----------------------------------------------------------------------
FACE_PROTO = 'models/deploy.prototxt'
FACE_MODEL = 'models/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL) if os.path.exists(FACE_MODEL) else None

# ----------------------------------------------------------------------
# 2. SAFE YOLO INITIALISATION
# ----------------------------------------------------------------------
print("Loading YOLO pose model …")
dummy = np.zeros((640, 640, 3), dtype=np.uint8)          # tiny black image

def _load_yolo(path: str):
    """Load a model and run a single dummy inference so the backend is built."""
    model = YOLO(path)
    # One forward pass forces AutoBackend → fuse → any hidden errors
    try:
        _ = model(dummy, imgsz=640, conf=0.4, verbose=False)
        print(f"   {path} loaded and warmed up")
        return model
    except Exception as e:
        print(f"   {path} failed ({e})")
        return None

# Try x first → fall back to nano
yolo = _load_yolo('yolov8x-pose.pt')
if yolo is None:
    yolo = _load_yolo('yolov8n-pose.pt')
    if yolo is None:
        raise RuntimeError("Could not load any YOLO pose model")

# ----------------------------------------------------------------------
# 3. EMOTION & RECOGNITION MODELS (unchanged)
# ----------------------------------------------------------------------
emotion_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(48,48,1)),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(2),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
])
emotion_labels = ['angry','disgust','fear','happy','sad','surprise','neutral']

rec_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3), include_top=False, pooling='avg', weights='imagenet')
rec_model = tf.keras.Model(
    rec_model.input,
    tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(rec_model.output))

def get_embedding(face):
    try:
        x = cv2.resize(face, (224,224)).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)
        return rec_model.predict(x, verbose=0)[0]
    except:
        return np.zeros(1280, dtype=np.float32)

# ----------------------------------------------------------------------
# 4. DB, TRACKER, GESTURE, POSTURE (unchanged)
# ----------------------------------------------------------------------
class EmbeddingDB:
    def __init__(self): self.db = {}; self.lock = threading.Lock()
    def add(self, name, emb):
        with self.lock:
            if name not in self.db: self.db[name] = []
            self.db[name].append(emb)
    def match(self, emb, thresh=0.6):
        if not self.db: return "Unknown", {}
        emb = emb / (np.linalg.norm(emb)+1e-8)
        best, score = "Unknown", 0
        for n, embs in self.db.items():
            sim = max(np.dot(emb, e/(np.linalg.norm(e)+1e-8)) for e in embs)
            if sim > score: score, best = sim, n
        return (best if score > thresh else "Unknown"), {"similarity": round(score, 3)}
emb_db = EmbeddingDB()

class Tracker:
    def __init__(self): self.objects = {}; self.next_id = 1
    def update(self, boxes):
        results = {}
        for box in boxes:
            cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
            match = min(self.objects.items(),
                        key=lambda x: abs(x[1][0]-cx) + abs(x[1][1]-cy),
                        default=None)
            if match and abs(match[1][0]-cx) < 120 and abs(match[1][1]-cy) < 120:
                tid, _ = match
                self.objects[tid] = (cx, cy)
            else:
                tid = self.next_id; self.next_id += 1; self.objects[tid] = (cx, cy)
            results[tid] = box
        self.objects = {k: v for k, v in self.objects.items() if k in results}
        return results
tracker = Tracker()

def _detect_gesture(kps: Optional[np.ndarray]) -> str:
    if kps is None or len(kps) < 17:
        return "None"
    def kp(i):
        p = kps[i]
        return p if len(p) >= 3 and p[2] > 0.3 else None
    nose, l_wr, r_wr = kp(0), kp(10), kp(9)
    if not all(x is not None for x in [nose, l_wr, r_wr]):
        return "None"
    nx, ny = nose[0], nose[1]
    lx, ly = l_wr[0], l_wr[1]
    rx, ry = r_wr[0], r_wr[1]
    if ly < ny and ry < ny: return "HandsUp"
    if abs(lx - rx) < 50 and ly < ny: return "Prayer"
    if rx > nx + 80: return "PointingRight"
    if rx < nx - 80: return "PointingLeft"
    return "None"

def get_posture(kps: Optional[np.ndarray]) -> str:
    if kps is None or len(kps) < 17: return "Unknown"
    hip_l, hip_r = kps[11], kps[12]
    if len(hip_l) >= 3 and len(hip_r) >= 3 and hip_l[2] > 0.3 and hip_r[2] > 0.3:
        return "Standing"
    return "Sitting"

# ----------------------------------------------------------------------
# 5. MAIN CLASS – **NO `fuse=` ARGUMENT ANYWHERE**
# ----------------------------------------------------------------------
class BrendaVision:
    def __init__(self):
        self.yolo = yolo
        self.emb_db = emb_db
        self.tracker = tracker

    def analyze_image_b64(self, b64_str: str) -> Dict[str, Any]:
        try:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64_str), np.uint8),
                              cv2.IMREAD_COLOR)
            if img is None:
                return {"status": "failed", "error": "Invalid image"}
            h, w = img.shape[:2]
            t0 = time.time()

            # *** IMPORTANT: NO `fuse=` argument ***
            results = self.yolo(img, imgsz=640, conf=0.4, verbose=False)[0]
            kps_list = results.keypoints.xy.cpu().numpy() if results.keypoints else []
            pose_boxes = [list(map(int, b)) for b in results.boxes.xyxy.cpu().numpy()] \
                         if results.boxes else []

            # Face detection (Caffe DNN)
            if net is None:
                return {"status": "failed", "error": "Face model not loaded"}
            blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104,117,123])
            net.setInput(blob); dets = net.forward()
            face_boxes = []
            for i in range(dets.shape[2]):
                conf = dets[0,0,i,2]
                if conf > 0.7:
                    x1 = int(dets[0,0,i,3]*w); y1 = int(dets[0,0,i,4]*h)
                    x2 = int(dets[0,0,i,5]*w); y2 = int(dets[0,0,i,6]*h)
                    face_boxes.append([x1, y1, x2, y2])

            track_map = self.tracker.update(pose_boxes)
            output = []

            for tid, bbox in track_map.items():
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                idx = next((i for i, t in enumerate(track_map.items()) if t[1] == bbox), -1)
                kps = kps_list[idx] if 0 <= idx < len(kps_list) else None

                # ----- match face -----
                best_face = None
                min_dist = float('inf')
                for fx1, fy1, fx2, fy2 in face_boxes:
                    fcx, fcy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    dist = abs(fcx - cx) + abs(fcy - cy)
                    if dist < min_dist and dist < 150:
                        min_dist = dist
                        best_face = [fx1, fy1, fx2, fy2]

                face_crop = None
                name, metrics = "Unknown", {}
                emo, conf = "neutral", 0.0

                if best_face:
                    fx1, fy1, fx2, fy2 = best_face
                    if fy2 > fy1 and fx2 > fx1 and fx2 - fx1 > 10 and fy2 - fy1 > 10:
                        face_crop = img[fy1:fy2, fx1:fx2]
                        # emotion
                        try:
                            gray = cv2.cvtColor(cv2.resize(face_crop, (48,48)),
                                                cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                            gray = gray.reshape(1,48,48,1)
                            pred = emotion_model.predict(gray, verbose=0)[0]
                            idx = int(np.argmax(pred))
                            emo, conf = emotion_labels[idx], float(pred[idx])
                        except: pass
                        # recognition
                        try:
                            emb = get_embedding(face_crop)
                            name, metrics = self.emb_db.match(emb)
                        except: pass

                gesture = _detect_gesture(kps)
                posture = get_posture(kps)

                response = ""
                if "Unknown" not in name:
                    response = f"{name} is {emo} and {gesture}".strip()
                elif gesture != "None":
                    response = f"Person is {emo} and {gesture}".strip()

                output.append({
                    "track_id": tid,
                    "name": name,
                    "emotion": emo,
                    "emotion_confidence": round(conf, 3),
                    "gesture": gesture,
                    "posture": posture,
                    "location": [x1, y1, x2, y2],
                    "face_location": best_face or [],
                    "pose_keypoints": kps.tolist() if kps is not None else [],
                    "identity_score": metrics.get("similarity", 0),
                    "response": response
                })

            latency = int((time.time() - t0) * 1000)
            result = {
                "status": "success",
                "faces": output,
                "count": len(output),
                "_latency_ms": latency
            }
            if output and output[0].get("response"):
                result["speak"] = output[0]["response"]
            return result

        except Exception as e:
            import traceback
            print("Vision Error:", traceback.format_exc())
            return {"status": "failed", "error": str(e)}

    def enroll(self, name: str, b64_str: str):
        try:
            img = cv2.imdecode(np.frombuffer(base64.b64decode(b64_str), np.uint8),
                              cv2.IMREAD_COLOR)
            if img is None: return {"status": "failed", "error": "Invalid image"}
            h, w = img.shape[:2]
            if net is None:
                return {"status": "failed", "error": "Face model not loaded"}
            blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104,117,123])
            net.setInput(blob); dets = net.forward()
            i = np.argmax(dets[0,0,:,2])
            if dets[0,0,i,2] < 0.7:
                return {"status": "failed", "error": "Low face confidence"}
            x1 = int(dets[0,0,i,3]*w); y1 = int(dets[0,0,i,4]*h)
            x2 = int(dets[0,0,i,5]*w); y2 = int(dets[0,0,i,6]*h)
            if x2 <= x1 or y2 <= y1:
                return {"status": "failed", "error": "Invalid face bbox"}
            face = img[y1:y2, x1:x2]
            emb = get_embedding(face)
            self.emb_db.add(name, emb)
            os.makedirs("face_db", exist_ok=True)
            cv2.imwrite(f"face_db/{name}_{len(self.emb_db.db.get(name,[]))}.jpg", face)
            return {"status": "enrolled", "name": name}
        except Exception as e:
            return {"status": "failed", "error": f"Enroll failed: {e}"}