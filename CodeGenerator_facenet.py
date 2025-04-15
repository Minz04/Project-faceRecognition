import cv2
import numpy as np
import pickle
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image

IMAGES_FOLDER = "dataset"
OUTPUT_FOLDER = "EmbeddingPicture"
OUTPUT_FILENAME = "Embeddings_Facenet.p"
OUTPUT_FILEPATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
REQUIRED_FACE_SIZE = (160, 160)
VALID_IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')

print("Khởi tạo mô hình...")
try:
    DETECTOR = MTCNN()
    EMBEDDER = FaceNet()
    print("Mô hình đã sẵn sàng.")
except Exception as e:
    print(f"[LỖI] Không thể khởi tạo MTCNN hoặc FaceNet: {e}")
    DETECTOR = None
    EMBEDDER = None

def generate_and_save_embeddings():
    if not DETECTOR or not EMBEDDER:
        print("[LỖI] Mô hình chưa được khởi tạo.")
        return False

    embeddingsData = []

    if not os.path.exists(IMAGES_FOLDER):
        print(f"[LỖI] Không tìm thấy thư mục: {IMAGES_FOLDER}")
        return False
    if not os.path.isdir(IMAGES_FOLDER):
        print(f"[LỖI] Đường dẫn không phải thư mục: {IMAGES_FOLDER}")
        return False

    for person_folder_name in os.listdir(IMAGES_FOLDER):
        person_folder_path = os.path.join(IMAGES_FOLDER, person_folder_name)
        if not os.path.isdir(person_folder_path) or person_folder_name.startswith('.'):
            continue

        user_id = ""
        user_name = ""

        if '_' in person_folder_name:
            try:
                user_id, user_name = person_folder_name.split('_', 1)
                user_id = user_id.strip()
                user_name = user_name.strip()
                if not user_id or not user_name:
                    print(f"[Cảnh báo] Tên thư mục không hợp lệ: {person_folder_name}")
                    continue
            except ValueError:
                user_id = person_folder_name.strip()
                user_name = person_folder_name.strip()
        else:
            user_id = person_folder_name.strip()
            user_name = person_folder_name.strip()

        image_count = 0
        for filename in os.listdir(person_folder_path):
            if not filename.lower().endswith(VALID_IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(person_folder_path, filename)
            image_count += 1
            print(f"  [{image_count}] Xử lý ảnh: {filename}...")

            try:
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    print(f"  [LỖI] Không thể đọc ảnh: {filename}")
                    continue

                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                results = DETECTOR.detect_faces(img_rgb)

                if not results:
                    print(f"  [!] Không phát hiện khuôn mặt: {filename}")
                    continue

                # Lấy khuôn mặt lớn nhất nếu có nhiều khuôn mặt
                if len(results) > 1:
                    best_face_idx = np.argmax([res['box'][2] * res['box'][3] for res in results])
                    face_data = results[best_face_idx]
                else:
                    face_data = results[0]

                x1, y1, width, height = face_data['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face_pixels = img_rgb[y1:y2, x1:x2]

                if face_pixels.size == 0:
                    print(f"  [LỖI] Không thể cắt ảnh: {filename}")
                    continue

                # Resize và tạo embedding
                face_image = Image.fromarray(face_pixels).resize(REQUIRED_FACE_SIZE)
                face_array = np.asarray(face_image)
                samples = np.expand_dims(face_array, axis=0)
                embedding = EMBEDDER.embeddings(samples)[0]

                embeddingsData.append({
                    'id': user_id,
                    'name': user_name,
                    'embedding': embedding
                })

            except Exception as e:
                print(f"  [LỖI] Khi xử lý ảnh {filename}: {e}")
                continue

        if image_count == 0:
            print(f"[!] Thư mục '{person_folder_name}' không có ảnh hợp lệ.")

    print(f"\nTổng số embeddings đã tạo: {len(embeddingsData)}")

    try:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        with open(OUTPUT_FILEPATH, 'wb') as file:
            pickle.dump(embeddingsData, file)

        if embeddingsData:
            print(f"Đã lưu embeddings vào: {OUTPUT_FILEPATH}")
        else:
            print(f"Đã lưu file rỗng (không có ảnh hợp lệ): {OUTPUT_FILEPATH}")
        return True
    except Exception as e:
        print(f"[LỖI] Không thể lưu embeddings: {e}")
        return False

if __name__ == "__main__":
    print("Đang chạy CodeGenerator...")
    if generate_and_save_embeddings():
        print("Tạo embeddings thành công.")
    else:
        print("Tạo embeddings thất bại.")
