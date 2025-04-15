# -*- coding: utf-8 -*-
import cv2
import numpy as np
import time
import os
import pickle
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from PyQt5.QtGui import QImage
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances

# --- Hằng số cấu hình ---
EMBEDDING_FOLDER = "EmbeddingPicture"
EMBEDDING_FILENAME = "Embeddings_Facenet.p"
EMBEDDING_FILEPATH = os.path.join(EMBEDDING_FOLDER, EMBEDDING_FILENAME)
RECOGNITION_THRESHOLD = 1.05  # Ngưỡng khoảng cách euclidean để xem là cùng một người
REQUIRED_FACE_SIZE = (160, 160) # Kích thước ảnh input chuẩn cho FaceNet

# --- Tín hiệu giao tiếp giữa Worker và GUI ---
class RecognitionSignals(QObject):
    frame_ready = pyqtSignal(QImage)            # Gửi frame camera đã xử lý (QImage)
    recognition_result = pyqtSignal(np.ndarray, str, str) # Gửi (ảnh mặt BGR, tên, id) khi nhận diện được
    no_recognition = pyqtSignal()               # Báo không nhận diện được ai
    error = pyqtSignal(str)                     # Gửi thông báo lỗi (vd: không mở được cam)
    embeddings_loaded = pyqtSignal(int)         # Báo số lượng embeddings đã tải xong

# --- Luồng xử lý nền cho Camera và Nhận diện ---
# Chạy riêng để không làm đơ giao diện chính
class RecognitionWorker(QThread):
    def __init__(self, detector: MTCNN, embedder: FaceNet, parent=None):
        super().__init__(parent)
        self.detector = detector # Model phát hiện khuôn mặt
        self.embedder = embedder # Model tạo embedding
        self.signals = RecognitionSignals() # Tạo đối tượng chứa tín hiệu
        self.running = False # Biến cờ để kiểm soát vòng lặp chạy/dừng
        self.known_people_data = [] # Danh sách chứa {'id':..., 'name':..., 'embedding':...}
        self.known_embeddings_np = np.array([]) # Mảng numpy chứa các embedding đã biết
        self._load_embeddings() # Tải dữ liệu lần đầu khi khởi tạo worker

    def _load_embeddings(self):
        """Tải hoặc tải lại dữ liệu embeddings từ file pickle."""
        print(f"Worker: Đang tải embeddings từ {EMBEDDING_FILEPATH}...")
        try:
            if os.path.exists(EMBEDDING_FILEPATH):
                with open(EMBEDDING_FILEPATH, 'rb') as file:
                    loaded_data = pickle.load(file)
                # Kiểm tra định dạng cơ bản của dữ liệu tải được
                if isinstance(loaded_data, list) and all(isinstance(item, dict) and 'id' in item and 'name' in item and 'embedding' in item for item in loaded_data):
                    self.known_people_data = loaded_data
                    if self.known_people_data: # Nếu danh sách không rỗng
                        self.known_embeddings_np = np.array([person['embedding'] for person in self.known_people_data])
                        print(f"Worker: Đã tải {len(self.known_people_data)} embeddings.")
                    else:
                        self.known_embeddings_np = np.array([])
                        print("Worker: File embeddings rỗng.")
                else: # Sai định dạng
                    self.known_people_data = []
                    self.known_embeddings_np = np.array([])
                    print("Worker: Định dạng file embeddings không hợp lệ.")
            else: # File không tồn tại
                self.known_people_data = []
                self.known_embeddings_np = np.array([])
                print(f"Worker: Không tìm thấy file embeddings: {EMBEDDING_FILEPATH}.")

            # Gửi tín hiệu báo số lượng embeddings đã tải (kể cả khi là 0)
            self.signals.embeddings_loaded.emit(len(self.known_people_data))

        except Exception as e:
            self.signals.error.emit(f"Lỗi khi tải embeddings: {e}")
            self.known_people_data = []
            self.known_embeddings_np = np.array([])
            self.signals.embeddings_loaded.emit(0)

    def reload_embeddings(self):
        """Phương thức công khai để yêu cầu tải lại embeddings từ bên ngoài."""
        self._load_embeddings()

    def run(self):
        """Vòng lặp chính của luồng: đọc camera, phát hiện, nhận diện, gửi tín hiệu."""
        self.running = True
        print("Worker: Bắt đầu luồng nhận diện...")
        cap = cv2.VideoCapture(0) # Mở webcam mặc định
        if not cap.isOpened():
            self.signals.error.emit("Không thể mở webcam.")
            self.running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while self.running:
            ret, frame_bgr = cap.read()
            if not ret:
                time.sleep(0.05) # Đợi một chút nếu không đọc được frame
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) # Chuyển sang RGB cho model
            processed_frame_bgr = frame_bgr.copy() # Tạo bản sao để vẽ lên mà không ảnh hưởng frame gốc
            recognition_found_this_frame = False # Cờ kiểm tra xem có nhận diện được ai trong frame này không

            # Chỉ thực hiện nhận diện nếu có dữ liệu embeddings đã tải
            if len(self.known_people_data) > 0:
                try:
                    faces = self.detector.detect_faces(frame_rgb) # Phát hiện khuôn mặt
                    best_match_info = None # Lưu thông tin (ảnh, tên, id) của người khớp nhất trong frame
                    min_overall_distance = float('inf') # Lưu khoảng cách nhỏ nhất tìm được trong frame

                    for face in faces:
                        x1, y1, width, height = face['box']
                        x1, y1 = abs(x1), abs(y1) # Đảm bảo tọa độ không âm
                        x2, y2 = x1 + width, y1 + height
                        face_pixels_rgb = frame_rgb[y1:y2, x1:x2] # Cắt vùng khuôn mặt

                        if face_pixels_rgb.size == 0: continue # Bỏ qua nếu vùng cắt rỗng

                        # Chuẩn bị và tạo embedding cho khuôn mặt từ camera
                        face_image = Image.fromarray(face_pixels_rgb).resize(REQUIRED_FACE_SIZE)
                        face_array = np.asarray(face_image)
                        samples = np.expand_dims(face_array, axis=0)
                        live_embedding = self.embedder.embeddings(samples)[0]

                        # So sánh embedding vừa tạo với các embedding đã biết
                        distances = euclidean_distances([live_embedding], self.known_embeddings_np)[0]
                        min_distance_idx = np.argmin(distances) # Tìm index của embedding gần nhất
                        min_distance = distances[min_distance_idx] # Lấy khoảng cách nhỏ nhất

                        # Kiểm tra với ngưỡng và tìm người khớp nhất (khoảng cách nhỏ nhất)
                        if min_distance < RECOGNITION_THRESHOLD and min_distance < min_overall_distance:
                            min_overall_distance = min_distance # Cập nhật khoảng cách nhỏ nhất
                            person_info = self.known_people_data[min_distance_idx]
                            rec_id = person_info['id']
                            rec_name = person_info['name']
                            display_text = f"{rec_name} ({rec_id})"
                            color = (0, 255, 0) # Màu xanh lá cây cho người nhận diện được
                            # Lưu thông tin người khớp nhất để gửi tín hiệu sau
                            best_match_info = (frame_bgr[y1:y2, x1:x2].copy(), rec_name, rec_id) # Lưu ảnh crop BGR gốc
                            # Vẽ hình chữ nhật và tên lên frame sẽ hiển thị
                            cv2.rectangle(processed_frame_bgr, (x1, y1), (x2, y2), color, 2)
                            text_y = y1 - 10 if y1 > 20 else y1 + 15 # Vị trí hiển thị text
                            cv2.putText(processed_frame_bgr, display_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            cv2.putText(processed_frame_bgr, f"d:{min_distance:.2f}", (x2 - 60, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                        # Không cần vẽ 'Unknown' cho mọi khuôn mặt không khớp, chỉ vẽ nếu *không có ai* khớp tốt trong frame
                        # (Việc xử lý không vẽ Unknown sẽ được làm sau vòng lặp faces nếu best_match_info vẫn là None)

                except Exception as e:
                    print(f"Worker: Lỗi trong vòng lặp nhận diện: {e}")
                    cv2.putText(processed_frame_bgr, "Loi Nhan Dien", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                # Sau khi duyệt hết các khuôn mặt trong frame:
                if best_match_info:
                    # Gửi tín hiệu chứa thông tin người khớp nhất
                    face_crop_bgr, name, id_ = best_match_info
                    self.signals.recognition_result.emit(face_crop_bgr, name, id_)
                    recognition_found_this_frame = True
                # else: Không cần vẽ Unknown ở đây, vì có thể frame sau sẽ nhận ra

            # --- Gửi frame (đã vẽ vời nếu có) cho GUI hiển thị ---
            processed_frame_rgb = cv2.cvtColor(processed_frame_bgr, cv2.COLOR_BGR2RGB) # Chuyển sang RGB cho QImage
            h, w, ch = processed_frame_rgb.shape
            qt_image = QImage(processed_frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            self.signals.frame_ready.emit(qt_image.copy()) # Gửi bản sao để tránh vấn đề về bộ nhớ

            # Gửi tín hiệu báo không nhận diện được ai TRONG FRAME NÀY
            if not recognition_found_this_frame:
                self.signals.no_recognition.emit()

            time.sleep(0.01) # Nghỉ cực ngắn để giảm tải CPU và cho phép xử lý sự kiện khác

        # --- Dọn dẹp khi vòng lặp kết thúc ---
        print("Worker: Giải phóng camera và dừng luồng.")
        cap.release()

    def stop(self):
        """Phương thức để yêu cầu dừng luồng từ bên ngoài."""
        self.running = False # Đặt cờ dừng
        print("Worker: Yêu cầu dừng...")
        self.wait() # Đợi luồng kết thúc hoàn toàn
        print("Worker: Luồng đã dừng.")