import sys
import os

# Tạo đường dẫn đến thư mục chứa file này
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot
from ui_form_FaceRecognition import Ui_MainWindow
from handleFormUI.worker import RecognitionWorker
from handleFormUI.add_user import AddUserDialog

# --- Định nghĩa đường dẫn thư mục ảnh gốc ---
IMAGES_FOLDER = os.path.join(project_root, 'Resources', 'Images')
print(f"Thư mục ảnh gốc được xác định là: {IMAGES_FOLDER}")
if not os.path.isdir(IMAGES_FOLDER):
    print(f"[CẢNH BÁO] Không tìm thấy thư mục ảnh gốc tại: {IMAGES_FOLDER}")

print("Đang khởi tạo model MTCNN và FaceNet...")
MODELS_LOADED = False
DETECTOR = None
EMBEDDER = None
try:
    from mtcnn.mtcnn import MTCNN
    from keras_facenet import FaceNet
    DETECTOR = MTCNN()
    EMBEDDER = FaceNet()
    print("Khởi tạo models thành công.")
    MODELS_LOADED = True
except Exception as e:
    print(f"[LỖI] Không thể khởi tạo models: {e}")

EMBEDDING_FILEPATH = os.path.join('EmbeddingPicture', 'Embeddings_Facenet.p')
if MODELS_LOADED and not os.path.exists(EMBEDDING_FILEPATH):
    print(f"File embeddings không tồn tại tại {EMBEDDING_FILEPATH}. Đang thử tạo lần đầu...")
    try:
        from CodeGenerator_facenet import generate_and_save_embeddings
        if generate_and_save_embeddings():
            print("Đã tạo file embeddings ban đầu thành công.")
        else:
            print("[CẢNH BÁO] Tạo file embeddings ban đầu thất bại.")
    except Exception as gen_e:
        print(f"[CẢNH BÁO] Lỗi trong quá trình tạo embeddings ban đầu: {gen_e}")

# 
class FaceRecognitionApp(QMainWindow, Ui_MainWindow):
    def __init__(self, detector, embedder, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Ứng dụng Nhận Diện Khuôn Mặt")

        self.detector = detector
        self.embedder = embedder
        self.recognition_worker = None
        self.add_user_dialog = None

        if not MODELS_LOADED:
            self.labelCamera.setText("Lỗi: Không tải được model nhận diện.")
            self.btnAddPerson.setEnabled(False)
            QMessageBox.critical(self, "Lỗi Model", "Không thể khởi tạo model nhận diện. Ứng dụng không thể hoạt động.")
            return

        print("Đang khởi động worker nhận diện...")
        self.recognition_worker = RecognitionWorker(self.detector, self.embedder)

        self.recognition_worker.signals.frame_ready.connect(self.update_camera_feed)
        self.recognition_worker.signals.recognition_result.connect(self.update_recognition_info) # Kết nối đến hàm đã sửa đổi
        self.recognition_worker.signals.no_recognition.connect(self.clear_recognition_info)
        self.recognition_worker.signals.error.connect(self.show_worker_error)
        self.recognition_worker.signals.embeddings_loaded.connect(self.update_status_bar)

        self.recognition_worker.start()

        self.btnAddPerson.clicked.connect(self.open_add_user_dialog)
        self.labelPicturePerson.setAlignment(Qt.AlignCenter)
        self.txt_name_person.setReadOnly(True)
        self.txt_id_person.setReadOnly(True)
        self.clear_recognition_info()

    @pyqtSlot(QImage)
    def update_camera_feed(self, qt_image):
        pixmap = QPixmap.fromImage(qt_image)
        self.labelCamera.setPixmap(pixmap.scaled(self.labelCamera.size(),
                                                  Qt.KeepAspectRatio, Qt.SmoothTransformation))

    # --- SỬA ĐỔI HÀM NÀY ---
    @pyqtSlot(np.ndarray, str, str) # Vẫn nhận face_crop_bgr nhưng không dùng để hiển thị nữa
    def update_recognition_info(self, face_crop_bgr, name, id_):
        """Hiển thị tên, id và ảnh gốc của người được nhận diện."""
        self.txt_name_person.setText(name)
        self.txt_id_person.setText(id_)

        # --- Tìm và hiển thị ảnh gốc từ thư mục Images ---
        found_image_path = None
        # Kiểm tra xem thư mục ảnh có tồn tại không
        if os.path.isdir(IMAGES_FOLDER):
            try:
                # Duyệt qua các file trong thư mục ảnh
                for filename in os.listdir(IMAGES_FOLDER):
                    # Kiểm tra xem tên file có bắt đầu bằng "ID_" không
                    if filename.startswith(id_ + "_"):
                        found_image_path = os.path.join(IMAGES_FOLDER, filename)
                        break 
            except Exception as e:
                print(f"Lỗi khi tìm kiếm ảnh cho ID {id_}: {e}")

        # Hiển thị ảnh nếu tìm thấy
        if found_image_path and os.path.exists(found_image_path):
            pixmap = QPixmap(found_image_path) # Tải ảnh gốc
            if not pixmap.isNull(): # Kiểm tra xem ảnh có tải được không
                self.labelPicturePerson.setPixmap(pixmap.scaled(self.labelPicturePerson.size(),
                                                                 Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                print(f"Không thể tải ảnh: {found_image_path}")
                self.labelPicturePerson.setText("Ảnh lỗi")
        else:
            # Không tìm thấy ảnh hoặc đường dẫn không hợp lệ
            print(f"Không tìm thấy ảnh gốc cho ID: {id_} trong thư mục {IMAGES_FOLDER}")
            self.labelPicturePerson.setText("Không có ảnh")
        # --- Kết thúc phần sửa đổi ---

    @pyqtSlot()
    def clear_recognition_info(self):
        """Xóa thông tin người dùng khỏi giao diện."""
        if self.txt_id_person.toPlainText() != "":
            self.txt_name_person.clear()
            self.txt_id_person.clear()
            # Đặt lại label ảnh về trạng thái mặc định
            self.labelPicturePerson.setText("(Chưa nhận diện)")

    @pyqtSlot(str)
    def show_worker_error(self, error_message):
        QMessageBox.warning(self, "Lỗi Worker", error_message)

    @pyqtSlot(int)
    def update_status_bar(self, count):
         if count > 0:
             self.statusBar().showMessage(f"Sẵn sàng nhận diện ({count} người đã biết)")
         else:
             self.statusBar().showMessage("Chưa có dữ liệu nhận diện. Vui lòng thêm người dùng.")

    def open_add_user_dialog(self):
        if not self.recognition_worker:
            QMessageBox.critical(self,"Lỗi", "Worker nhận diện chưa sẵn sàng.")
            return

        worker_was_running = False
        if self.recognition_worker.isRunning():
            print("Đang dừng worker chính để mở dialog Thêm Người Dùng...")
            self.recognition_worker.stop()
            worker_was_running = True
            print("Worker chính đã dừng.")
        else:
             print("Worker chính không chạy, không cần dừng.")

        if self.add_user_dialog is None:
            self.add_user_dialog = AddUserDialog(self)
            self.add_user_dialog.user_added.connect(self.handle_user_added)

        print("Đang mở dialog Thêm Người Dùng...")
        result = self.add_user_dialog.exec_()
        print(f"Dialog Thêm Người Dùng đã đóng với kết quả: {'Đồng ý' if result == QDialog.Accepted else 'Hủy/Đóng'}")

        if worker_was_running and not self.recognition_worker.isRunning():
             print("Đang khởi động lại worker chính...")
             self.recognition_worker.start()
             print("Worker chính đã khởi động lại.")
        elif not worker_was_running:
             print("Worker chính không chạy trước đó, không cần khởi động lại.")

    @pyqtSlot()
    def handle_user_added(self):
        print("AppMain: Nhận tín hiệu user_added. Yêu cầu worker tải lại embeddings...")
        if self.recognition_worker:
            self.recognition_worker.reload_embeddings()

    def closeEvent(self, event):
        print("Đang đóng ứng dụng chính...")
        if self.recognition_worker and self.recognition_worker.isRunning():
            print("Đang dừng worker nhận diện...")
            self.recognition_worker.stop()
        if self.add_user_dialog and self.add_user_dialog.isVisible():
             self.add_user_dialog.close()
        print("Dọn dẹp hoàn tất. Thoát ứng dụng.")
        event.accept()

# --- Điểm bắt đầu thực thi ứng dụng ---
if __name__ == "__main__":
    app = QApplication(sys.argv)

    if not MODELS_LOADED:
        sys.exit(1)

    main_window = FaceRecognitionApp(DETECTOR, EMBEDDER)
    main_window.show()

    sys.exit(app.exec_())