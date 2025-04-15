import cv2
import numpy as np
import time
import os
import pickle
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage
try:
    from mtcnn.mtcnn import MTCNN
    from keras_facenet import FaceNet
    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"[LỖI] Không thể nhập MTCNN hoặc FaceNet: {e}")
    MODELS_AVAILABLE = False
    class MTCNN: pass
    class FaceNet: pass

from PIL import Image
try:
    from sklearn.metrics.pairwise import euclidean_distances
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[LỖI] Không thể nhập sklearn. Vui lòng cài đặt scikit-learn.")
    SKLEARN_AVAILABLE = False

# Cấu hình
RECOGNITION_THRESHOLD = 1.05  # Ngưỡng nhận diện khuôn mặt
REQUIRED_FACE_SIZE = (160, 160)  # Kích thước ảnh khuôn mặt
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

# Tín hiệu giao tiếp với giao diện
class RecognitionSignals(QObject):
    frame_ready = pyqtSignal(QImage)  # Gửi khung hình
    recognition_result = pyqtSignal(np.ndarray, str, str)  # Kết quả nhận diện
    no_recognition = pyqtSignal()  # Không nhận diện được
    error = pyqtSignal(str)  # Lỗi
    embeddings_loaded = pyqtSignal(int)  # Số lượng embedding đã tải

# Luồng xử lý nhận diện khuôn mặt
class RecognitionWorker(QThread):
    def __init__(self, detector: MTCNN, embedder: FaceNet, embedding_filepath: str, parent=None):
        super().__init__(parent)

        # Kiểm tra thư viện cần thiết
        if not MODELS_AVAILABLE or not SKLEARN_AVAILABLE:
            self.signals = RecognitionSignals()
            self.running = False
            print("[LỖI] Thiếu thư viện MTCNN/FaceNet hoặc scikit-learn.")
            self.detector = None
            self.embedder = None
            self.embedding_file = None
            self.known_people = []
            self.known_embeddings = np.array([])
            self._prevent_run = True
            return
        else:
            self._prevent_run = False

        self.detector = detector
        self.embedder = embedder
        self.embedding_file = embedding_filepath
        self.signals = RecognitionSignals()
        self.running = False
        self.known_people = []
        self.known_embeddings = np.array([])
        self._load_embeddings()  # Tải dữ liệu embedding khi khởi tạo

    def _load_embeddings(self):
        """Tải dữ liệu embedding từ file."""
        if not self.embedding_file or not isinstance(self.embedding_file, str):
            print("[LỖI] Đường dẫn file embedding không hợp lệ.")
            self.known_people = []
            self.known_embeddings = np.array([])
            self.signals.embeddings_loaded.emit(0)
            return

        try:
            if os.path.exists(self.embedding_file) and os.path.getsize(self.embedding_file) > 0:
                with open(self.embedding_file, 'rb') as file:
                    data = pickle.load(file)

                if isinstance(data, list):
                    valid_data = [
                        item for item in data
                        if isinstance(item, dict) and 'id' in item and 'name' in item and 'embedding' in item
                        and isinstance(item['embedding'], np.ndarray)
                    ]
                    self.known_people = valid_data
                    self.known_embeddings = np.array([person['embedding'] for person in valid_data]) if valid_data else np.array([])
                else:
                    print("[LỖI] File embedding không đúng định dạng.")
                    self.known_people = []
                    self.known_embeddings = np.array([])

            else:
                print("[CẢNH BÁO] Không tìm thấy file embedding.")
                self.known_people = []
                self.known_embeddings = np.array([])

            self.signals.embeddings_loaded.emit(len(self.known_people))

        except Exception as e:
            print(f"[LỖI] Không thể tải file embedding: {e}")
            self.known_people = []
            self.known_embeddings = np.array([])
            self.signals.embeddings_loaded.emit(-1)

    def reload_embeddings(self):
        """Tải lại dữ liệu embedding."""
        self._load_embeddings()

    def run(self):
        """Xử lý camera và nhận diện khuôn mặt."""
        if self._prevent_run:
            print("[LỖI] Worker không chạy do lỗi khởi tạo.")
            return

        self.running = True
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                cap = cv2.VideoCapture(1)
                if not cap.isOpened():
                    raise IOError("Không thể mở camera.")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        except Exception as e:
            print(f"[LỖI] Không thể mở camera: {e}")
            self.signals.error.emit(f"Lỗi camera: {e}")
            self.running = False
            if cap:
                cap.release()
            return

        last_recognition_time = time.time()
        last_sent_id = None

        while self.running:
            try:
                ret, frame_bgr = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                processed_frame = frame_bgr.copy()
                found_person = False
                best_match = None
                min_distance = float('inf')

                # Nhận diện nếu có dữ liệu embedding
                if self.known_embeddings.size > 0 and self.detector and self.embedder:
                    faces = self.detector.detect_faces(frame_rgb)
                    for face in faces:
                        x1, y1, width, height = face['box']
                        x1, y1 = max(0, x1), max(0, y1)
                        x2 = min(frame_rgb.shape[1], x1 + width)
                        y2 = min(frame_rgb.shape[0], y1 + height)

                        if x2 <= x1 or y2 <= y1:
                            continue

                        face_pixels = frame_rgb[y1:y2, x1:x2]
                        if face_pixels.size == 0:
                            continue

                        # Tạo embedding từ khuôn mặt
                        face_image = Image.fromarray(face_pixels).resize(REQUIRED_FACE_SIZE)
                        face_array = np.asarray(face_image)
                        samples = np.expand_dims(face_array, axis=0)
                        live_embedding = self.embedder.embeddings(samples)[0]

                        # Tìm người khớp nhất
                        distances = euclidean_distances([live_embedding], self.known_embeddings)[0]
                        min_distance_idx = np.argmin(distances)
                        distance = distances[min_distance_idx]

                        color = (0, 0, 255)
                        text = "Unknown"

                        if distance < RECOGNITION_THRESHOLD:
                            color = (0, 255, 0)
                            person = self.known_people[min_distance_idx]
                            text = person['name']
                            if distance < min_distance:
                                min_distance = distance
                                best_match = (frame_bgr[y1:y2, x1:x2].copy(), person['name'], person['id'])
                                found_person = True

                        # Vẽ khung và tên lên ảnh
                        cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                        text_y = y1 - 10 if y1 > 20 else y1 + 15
                        cv2.putText(processed_frame, text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # Gửi tín hiệu nhận diện
                now = time.time()
                if best_match:
                    face_crop, name, id_ = best_match
                    if id_ != last_sent_id or (now - last_recognition_time) > 1.0:
                        self.signals.recognition_result.emit(face_crop, name, id_)
                        last_recognition_time = now
                        last_sent_id = id_
                elif not found_person and (last_sent_id is not None or (now - last_recognition_time) > 1.0):
                    self.signals.no_recognition.emit()
                    last_recognition_time = now
                    last_sent_id = None

                # Gửi khung hình cho giao diện
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                h, w, ch = processed_rgb.shape
                qt_image = QImage(processed_rgb.data, w, h, ch * w, QImage.Format_RGB888)
                self.signals.frame_ready.emit(qt_image.copy())

                time.sleep(0.01)

            except Exception as e:
                print(f"[LỖI] Lỗi trong vòng lặp: {e}")
                self.signals.error.emit(f"Lỗi xử lý: {e}")
                time.sleep(0.5)

        # Giải phóng camera
        if cap and cap.isOpened():
            cap.release()

    def stop(self):
        """Dừng luồng xử lý."""
        self.running = False