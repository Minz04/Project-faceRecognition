import sys
import os

# Lấy thư mục gốc của project
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Thêm thư mục gốc vào sys.path nếu chưa có
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QLabel, QDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSlot
from ui_form_FaceRecognition import Ui_MainWindow

# Nhập các module worker và add_user
try:
    from handleFormUI.worker import RecognitionWorker
    from handleFormUI.add_user import AddUserDialog
except ImportError:
    try:
        from worker import RecognitionWorker
        from add_user import AddUserDialog
    except ImportError as e:
        print(f"[LỖI] Không thể nhập RecognitionWorker hoặc AddUserDialog: {e}")
        sys.exit(1)

# Xác định đường dẫn thư mục dataset
dataset_folder = os.path.join(project_root, 'dataset')

# Tạo thư mục dataset nếu chưa tồn tại
if not os.path.isdir(dataset_folder):
    try:
        os.makedirs(dataset_folder, exist_ok=True)
    except Exception as e:
        print(f"[LỖI] Không thể tạo thư mục dataset: {e}")

# Khởi tạo model MTCNN và FaceNet
models_loaded = False
detector = None
embedder = None
try:
    from mtcnn.mtcnn import MTCNN
    from keras_facenet import FaceNet
    detector = MTCNN()
    embedder = FaceNet()
    models_loaded = True
except Exception as e:
    print(f"[LỖI] Không thể khởi tạo model: {e}")

# Xác định đường dẫn file embedding
embedding_folder = os.path.join(project_root, 'EmbeddingPicture')
embedding_file = os.path.join(embedding_folder, 'Embeddings_Facenet.p')

# Tạo file embedding nếu chưa tồn tại và model đã tải
if models_loaded and not os.path.exists(embedding_file):
    try:
        from CodeGenerator_facenet import generate_and_save_embeddings
        if generate_and_save_embeddings():
            print("Tạo file embedding ban đầu thành công.")
        else:
            print("[CẢNH BÁO] Không thể tạo file embedding ban đầu.")
    except Exception as e:
        print(f"[LỖI] Lỗi khi tạo file embedding: {e}")

class FaceRecognitionApp(QMainWindow, Ui_MainWindow):
    def __init__(self, detector, embedder, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Ứng dụng Nhận Diện Khuôn Mặt (FaceNet + MTCNN)")

        self.detector = detector
        self.embedder = embedder
        self.recognition_worker = None
        self.add_user_dialog = None

        # Xử lý khi model không tải được
        if not models_loaded:
            self.labelCamera.setText("LỖI:\nKhông thể tải model MTCNN hoặc FaceNet.\nVui lòng kiểm tra cài đặt.")
            self.labelCamera.setAlignment(Qt.AlignCenter)
            self.labelCamera.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            self.btnAddPerson.setEnabled(False)
            self.statusBar().showMessage("Lỗi khởi tạo model!")
            QMessageBox.critical(self, "Lỗi Model", "Không thể khởi tạo model MTCNN/FaceNet. Kiểm tra console để biết chi tiết.")
        else:
            # Khởi động worker nhận diện nếu model tải thành công
            self.recognition_worker = RecognitionWorker(self.detector, self.embedder, embedding_file, parent=self)
            self.recognition_worker.signals.frame_ready.connect(self.update_camera_feed)
            self.recognition_worker.signals.recognition_result.connect(self.update_recognition_info)
            self.recognition_worker.signals.no_recognition.connect(self.clear_recognition_info)
            self.recognition_worker.signals.error.connect(self.show_worker_error)
            self.recognition_worker.signals.embeddings_loaded.connect(self.update_status_bar)
            self.recognition_worker.start()
            self.statusBar().showMessage("Đang khởi động worker và tải embedding...")

        # Thiết lập giao diện ban đầu
        self.btnAddPerson.clicked.connect(self.open_add_user_dialog)
        self.labelPicturePerson.setAlignment(Qt.AlignCenter)
        self.labelPicturePerson.setText("(Chưa nhận diện)")
        self.txt_name_person.setReadOnly(True)
        self.txt_id_person.setReadOnly(True)
        self.clear_recognition_info()

    @pyqtSlot(QImage)
    def update_camera_feed(self, qt_image):
        # Cập nhật khung hình camera
        if hasattr(self, 'labelCamera') and models_loaded:
            try:
                pixmap = QPixmap.fromImage(qt_image)
                self.labelCamera.setPixmap(pixmap.scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            except Exception as e:
                print(f"Lỗi khi cập nhật khung hình camera: {e}")

    @pyqtSlot(np.ndarray, str, str)
    def update_recognition_info(self, face_crop_bgr, name, id_):
        """Cập nhật giao diện với thông tin tên, ID và ảnh của người được nhận diện."""
        self.txt_name_person.setText(name)
        self.txt_id_person.setText(id_)

        # Tìm thư mục và ảnh của người dùng
        photo_path = None
        person_folder = None

        if os.path.isdir(dataset_folder):
            folder_prefix = f"{id_}_"
            try:
                for item in os.listdir(dataset_folder):
                    item_path = os.path.join(dataset_folder, item)
                    if os.path.isdir(item_path) and item.startswith(folder_prefix):
                        person_folder = item_path
                        for file in os.listdir(person_folder):
                            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                photo_path = os.path.join(person_folder, file)
                                break
                        break
            except Exception as e:
                print(f"[LỖI] Lỗi khi tìm ảnh cho ID {id_}: {e}")

        # Hiển thị ảnh nếu tìm thấy
        if photo_path and os.path.exists(photo_path):
            pixmap = QPixmap(photo_path)
            if not pixmap.isNull():
                self.labelPicturePerson.setPixmap(pixmap.scaled(self.labelPicturePerson.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.labelPicturePerson.setText("Ảnh không hợp lệ")
        elif person_folder:
            self.labelPicturePerson.setText("Không có ảnh trong thư mục")
        else:
            self.labelPicturePerson.setText("Không tìm thấy ảnh")

    @pyqtSlot()
    def clear_recognition_info(self):
        """Xóa thông tin người dùng khỏi giao diện khi không nhận diện được."""
        if self.txt_id_person.toPlainText():
            self.txt_name_person.clear()
            self.txt_id_person.clear()
            self.labelPicturePerson.setText("(Chưa nhận diện)")
            self.labelPicturePerson.setPixmap(QPixmap())

    @pyqtSlot(str)
    def show_worker_error(self, error_message):
        # Hiển thị lỗi từ worker
        self.statusBar().showMessage(f"Lỗi Worker: {error_message}", 10000)
        QMessageBox.warning(self, "Lỗi Worker", error_message)

    @pyqtSlot(int)
    def update_status_bar(self, count):
        # Cập nhật thanh trạng thái
        if not models_loaded:
            self.statusBar().showMessage("LỖI KHỞI TẠO MODEL!")
            return
        if count > 0:
            self.statusBar().showMessage(f"Sẵn sàng nhận diện ({count} người đã biết)")
        elif count == 0:
            self.statusBar().showMessage("Chưa có dữ liệu nhận diện. Vui lòng thêm người dùng.")
        else:
            self.statusBar().showMessage("Lỗi tải dữ liệu nhận diện.")

    def open_add_user_dialog(self):
        """Mở cửa sổ thêm người dùng, tạm dừng worker nếu đang chạy."""
        if not models_loaded:
            QMessageBox.critical(self, "Lỗi", "Model chưa tải. Không thể thêm người dùng.")
            return
        if not self.recognition_worker:
            QMessageBox.critical(self, "Lỗi", "Worker nhận diện chưa khởi tạo.")
            return

        worker_was_running = False
        if self.recognition_worker.isRunning():
            self.recognition_worker.stop()
            worker_was_running = True

        self.add_user_dialog = AddUserDialog(self)
        self.add_user_dialog.user_added.connect(self.handle_user_added)
        result = self.add_user_dialog.exec_()

        if worker_was_running and not self.recognition_worker.isRunning():
            self.recognition_worker.start()
            self.statusBar().showMessage("Đang khởi động lại worker nhận diện...")

        self.add_user_dialog.deleteLater()
        self.add_user_dialog = None

    @pyqtSlot()
    def handle_user_added(self):
        """Xử lý khi thêm người dùng thành công và tải lại embedding."""
        if self.recognition_worker:
            self.recognition_worker.reload_embeddings()
            self.statusBar().showMessage("Đã thêm người dùng. Đang cập nhật dữ liệu nhận diện...")

    def closeEvent(self, event):
        """Dọn dẹp trước khi đóng ứng dụng."""
        if self.recognition_worker and self.recognition_worker.isRunning():
            self.recognition_worker.stop()
        if self.add_user_dialog and self.add_user_dialog.isVisible():
            self.add_user_dialog.reject()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = FaceRecognitionApp(detector, embedder)
    main_window.show()
    sys.exit(app.exec_())