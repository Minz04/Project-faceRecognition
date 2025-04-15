import cv2
import numpy as np
import os
import pickle
import time
import traceback  
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image

# --- Hằng số ---
EMBEDDING_FOLDER = "EmbeddingPicture"
EMBEDDING_FILENAME = "Embeddings_Facenet.p"
EMBEDDING_FILEPATH = os.path.join(EMBEDDING_FOLDER, EMBEDDING_FILENAME)
RECOGNITION_THRESHOLD = 1.05  
REQUIRED_FACE_SIZE = (160, 160)

print("Khởi tạo mô hình...")
try:
    detector = MTCNN()
    embedder = FaceNet()
    print("Mô hình đã sẵn sàng.")
except Exception as e:
    print(f"[LỖI] Không thể khởi tạo MTCNN hoặc FaceNet: {e}")
    exit()

# --- Tải dữ liệu embeddings ---
known_people_data = []  # Danh sách chứa thông tin người đã biết
known_embeddings_np = np.array([])
print(f"Đang tải dữ liệu embeddings từ {EMBEDDING_FILEPATH}...")
try:
    if os.path.exists(EMBEDDING_FILEPATH):
        with open(EMBEDDING_FILEPATH, 'rb') as file:
            loaded_data = pickle.load(file)
            # Kiểm tra định dạng dữ liệu
            if isinstance(loaded_data, list):
                if all(isinstance(item, dict) and 'id' in item and 'name' in item and 'embedding' in item for item in loaded_data):
                    known_people_data = loaded_data
                    if known_people_data:  # Nếu danh sách không rỗng
                        known_embeddings_np = np.array([person['embedding'] for person in known_people_data])
                        print(f"Đã tải thành công {len(known_people_data)} embeddings.")
                    else:
                        print("  - Cảnh báo: File embeddings rỗng (chứa danh sách trống).")
                elif not loaded_data:
                    print("  - Cảnh báo: File embeddings rỗng.")
                else:
                    print("  - Lỗi: File embeddings không đúng định dạng (phải là danh sách các từ điển chứa 'id', 'name', 'embedding').")
                    known_people_data = []  
            else:
                print("  - Lỗi: File embeddings không chứa danh sách.")
                known_people_data = []  
    else:
        print(f"  - Lỗi: Không tìm thấy file embeddings tại {EMBEDDING_FILEPATH}.")

except Exception as e:
    print(f"[LỖI] Không thể tải hoặc phân tích file embeddings: {e}")
    known_people_data = []  # Đặt lại nếu lỗi

if not known_people_data:
    print("Không có dữ liệu nhận diện hợp lệ. Không thể tiếp tục nhận diện.")
    exit() 

# --- Mở webcam ---
print("Đang mở webcam...")
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("[LỖI] Không thể mở webcam.")
    exit()
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam đã mở thành công: {frame_width}x{frame_height}")

if known_people_data:
    print("\nBắt đầu nhận diện...")
else:
    print("\nChỉ hiển thị camera (Không có dữ liệu nhận diện)...")

# --- Vòng lặp chính để nhận diện ---
while True:
    ret, frame_bgr = cam.read()
    if not ret:
        print("[LỖI] Không thể đọc khung hình từ webcam.")
        time.sleep(0.1)  
        continue

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    processed_frame = frame_bgr.copy()  

    if known_people_data:
        try:
            faces = detector.detect_faces(frame_rgb)

            for face in faces:
                # Trích xuất khuôn mặt
                x1, y1, width, height = face['box']
                x1, y1 = abs(x1), abs(y1)  
                x2, y2 = x1 + width, y1 + height

                face_pixels = frame_rgb[y1:y2, x1:x2]

                if face_pixels.size == 0:
                    continue  # Bỏ qua khuôn mặt này nếu không có dữ liệu

                try:
                    # Resize và chuẩn bị để tạo embedding
                    face_image = Image.fromarray(face_pixels).resize(REQUIRED_FACE_SIZE)
                    face_array = np.asarray(face_image)
                    samples = np.expand_dims(face_array, axis=0)  

                    # Tạo embedding cho khuôn mặt
                    live_embedding = embedder.embeddings(samples)[0]

                    # So sánh với embeddings đã biết
                    distances = euclidean_distances([live_embedding], known_embeddings_np)[0]
                    min_distance_index = np.argmin(distances)
                    min_distance = distances[min_distance_index]

                    # Nhận diện người
                    if min_distance < RECOGNITION_THRESHOLD:
                        person_info = known_people_data[min_distance_index]
                        rec_id = person_info['id']
                        rec_name = person_info['name']
                        display_text = f"{rec_name} ({rec_id})"
                        color = (0, 255, 0)  # Màu xanh lá
                    else:
                        display_text = "Unknow"
                        color = (0, 255, 255)  # Màu vàng

                    # Vẽ khung và hiển thị thông tin
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                    text_y = y1 - 10 if y1 > 20 else y1 + 15  
                    cv2.putText(processed_frame, display_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(processed_frame, f"d:{min_distance:.2f}", (x2 - 60, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                except Exception as face_proc_e:
                    print(f"[Cảnh báo] Lỗi khi xử lý khuôn mặt: {face_proc_e}")
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 0, 255), 1)

        except Exception as loop_e:
            print(f"[LỖI] Lỗi trong vòng lặp nhận diện: {loop_e}")
            traceback.print_exc()  
            cv2.putText(processed_frame, "Lỗi nhận diện", (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # --- Hiển thị khung hình kết quả ---
    cv2.imshow("Nhan dien khuon mat", processed_frame)


    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        print("\nĐã nhấn ESC, thoát...")
        break

print("Đang giải phóng webcam và đóng cửa sổ...")
cam.release()
cv2.destroyAllWindows()
print("Ứng dụng đã đóng.")