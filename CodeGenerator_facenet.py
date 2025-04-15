import cv2
import numpy as np
import pickle
import os
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from PIL import Image
import traceback

# --- Hằng số ---
IMAGES_FOLDER = os.path.join('Resources', 'Images')
OUTPUT_FOLDER = "EmbeddingPicture"
OUTPUT_FILENAME = "Embeddings_Facenet.p"
OUTPUT_FILEPATH = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
REQUIRED_FACE_SIZE = (160, 160)

# --- Mô hình toàn cục ---
print("Đang khởi tạo mô hình...")
try:
    DETECTOR = MTCNN()
    EMBEDDER = FaceNet()
    print("CodeGenerator: Mô hình đã sẵn sàng.")
except Exception as e:
    print(f"[LỖI] Không thể khởi tạo MTCNN hoặc FaceNet: {e}")
    DETECTOR = None
    EMBEDDER = None
    

def generate_and_save_embeddings():
    # Kiểm tra xem mô hình đã được khởi tạo chưa
    if not DETECTOR or not EMBEDDER:
        print("[LỖI] CodeGenerator: Mô hình chưa được khởi tạo. Không thể tạo embeddings.")
        return False

    user_data_list = []
    print(f"\nĐang đọc ảnh từ: {IMAGES_FOLDER}")
    if not os.path.exists(IMAGES_FOLDER):
        print(f"[LỖI] Không tìm thấy thư mục ảnh: {IMAGES_FOLDER}")
        return False

    # Lặp qua tất cả các file trong thư mục ảnh
    for filename in os.listdir(IMAGES_FOLDER):
        full_path = os.path.join(IMAGES_FOLDER, filename)
        if os.path.isfile(full_path) and not filename.startswith('.'):
            base_name = os.path.splitext(filename)[0]
            user_id = ""
            user_name = ""

            # Kiểm tra xem tên file có chứa dấu gạch dưới hay không
            if '_' in base_name:
                try:
                    user_id, user_name = base_name.split('_', 1)
                    user_id = user_id.strip()
                    user_name = user_name.strip()
                except ValueError:
                    print(f"  [Cảnh báo] Tên file '{filename}' có dấu gạch dưới nhưng không đúng định dạng ID_Ten. Sử dụng tên gốc làm Tên.")
                    user_id = base_name.strip()
                    user_name = base_name.strip()
            else:
                # Không có dấu gạch dưới, giả định toàn bộ tên gốc là Tên (và cũng dùng làm ID)
                user_id = base_name.strip()
                user_name = base_name.strip()
                print(f"  [Thông tin]  Tên file '{filename}' không có dấu gạch dưới. Sử dụng '{user_name}' làm cả ID và Tên.")

            if not user_id or not user_name:
                 print(f"  [Cảnh báo]  Không thể trích xuất ID/Tên từ '{filename}'. Bỏ qua.")
                 continue

            user_data_list.append({'id': user_id, 'name': user_name, 'path': full_path})
            print(f"  - Tìm thấy: ID='{user_id}', Tên='{user_name}', Đường dẫn='{filename}'")

    # Kiểm tra xem có ảnh nào hợp lệ không
    print(f" Đã tìm thấy tổng cộng {len(user_data_list)} ảnh có thể sử dụng để xử lý.")
    if not user_data_list:
        print(" Không tìm thấy ảnh hợp lệ để xử lý.")
        try:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            with open(OUTPUT_FILEPATH, 'wb') as file:
                pickle.dump([], file)
            print(f" Đã lưu danh sách embeddings rỗng vào {OUTPUT_FILEPATH}")
            return True
        except Exception as e:
            print(f"[LỖI]  Không thể lưu file embeddings rỗng: {e}")
            return False

    # Danh sách các dict: [{'id': id, 'name': name, 'embedding': vector}, ...]
    embeddingsData = [] 
    print("\n Bắt đầu tạo embeddings...")

    for i, user_info in enumerate(user_data_list):
        img_path = user_info['path']
        current_id = user_info['id']
        current_name = user_info['name']
        print(f" [{i+1}/{len(user_data_list)}] Đang xử lý: {os.path.basename(img_path)} (ID: {current_id}, Tên: {current_name})...")

        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f" [LỖI] Không thể đọc ảnh: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            results = DETECTOR.detect_faces(img_rgb)

            if not results:
                print(f" [Cảnh báo] Không phát hiện khuôn mặt trong: {img_path}")
                continue
            if len(results) > 1:
                print(f" [Cảnh báo] Phát hiện nhiều khuôn mặt trong: {img_path}. Sử dụng khuôn mặt lớn nhất.")
                best_face_idx = np.argmax([res['box'][2] * res['box'][3] for res in results])
                face_data = results[best_face_idx]
            else:
                 face_data = results[0]

            x1, y1, width, height = face_data['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height
            face_pixels = img_rgb[y1:y2, x1:x2]

            if face_pixels.size == 0:
                 print(f" [LỖI] Không thể trích xuất pixel khuôn mặt từ: {img_path}")
                 continue

            face_image = Image.fromarray(face_pixels).resize(REQUIRED_FACE_SIZE)
            face_array = np.asarray(face_image)
            samples = np.expand_dims(face_array, axis=0)
            embedding = EMBEDDER.embeddings(samples)[0]

            embeddingsData.append({
                'id': current_id,
                'name': current_name,
                'embedding': embedding
            })
            print(f" Đã tạo embedding thành công.")

        except Exception as e:
            print(f"[LỖI] Lỗi khi xử lý {img_path}: {e}")
            traceback.print_exc()

    print(f"\n Hoàn tất! Đã tạo {len(embeddingsData)} embeddings.")

    # Lưu kết quả
    if embeddingsData:
        try:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            with open(OUTPUT_FILEPATH, 'wb') as file:
                pickle.dump(embeddingsData, file)
            print(f" Đã lưu embeddings thành công vào: {OUTPUT_FILEPATH}")
            return True
        except Exception as e:
            print(f"\n[LỖI] Không thể lưu file embeddings: {e}")
            return False
    else:
        print(" Không có embeddings nào được tạo để lưu.")
        try:
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            with open(OUTPUT_FILEPATH, 'wb') as file:
                pickle.dump([], file)
            print(f" Đã lưu danh sách embeddings rỗng vào {OUTPUT_FILEPATH}")
            return True
        except Exception as e:
            print(f"[LỖI] Không thể lưu file embeddings rỗng: {e}")
            return False

if __name__ == "__main__":
    print("Đang chạy CodeGenerator...")
    if generate_and_save_embeddings():
        print("\nQuá trình tạo embeddings đã thành công.")
    else:
        print("\nQuá trình tạo embeddings đã thất bại.")