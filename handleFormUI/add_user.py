import cv2
import sys
import os
import random
import numpy as np
import traceback
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from ui_form_ChupAnh import Ui_Form

try:
    from CodeGenerator_facenet import generate_and_save_embeddings
except ImportError:
    print("[LỖI] Không thể import 'generate_and_save_embeddings'")
    generate_and_save_embeddings = None

IMAGES_FOLDER = "dataset"

class AddUserDialog(QDialog, Ui_Form):
    user_added = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Thêm Người Dùng Mới")

        self.capture = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_preview)
        self.captured_image = None

        self.btnChupAnh.clicked.connect(self.capture_image_action)
        self.btnDongY.clicked.connect(self.confirm_action)
        self.btnHuy.clicked.connect(self.cancel_action)

        self.init_camera()
        self.reset_ui_to_capture_mode()

    def init_camera(self):
        try:
            self.capture = cv2.VideoCapture(0)
            if not self.capture or not self.capture.isOpened():
                print("Không mở được camera 0, thử camera 1...")
                self.capture = cv2.VideoCapture(1)
                if not self.capture or not self.capture.isOpened():
                    raise ValueError("Không thể mở camera.")

            self.timer.start(30)
            print("Camera đã khởi tạo.")
            self.btnChupAnh.setEnabled(True)

        except Exception as e:
            QMessageBox.warning(self, "Lỗi Camera", f"Không thể mở webcam.\nChi tiết: {e}")
            self.btnChupAnh.setEnabled(False)
            self.capture = None

    def update_preview(self):
        if self.capture and self.capture.isOpened() and self.timer.isActive():
            ret, frame_bgr = self.capture.read()
            if ret:
                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.labelCamera.setPixmap(pixmap.scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
                except Exception as e:
                    print(f"Lỗi hiển thị frame: {e}")

    def reset_ui_to_capture_mode(self):
        self.txtTenNguoiMoi.hide()
        self.btnDongY.hide()
        self.btnHuy.hide()
        self.labelCamera.clear()
        self.labelCamera.setText("(Hướng camera vào mặt và nhấn Chụp Ảnh)")
        self.captured_image = None
        self.txtTenNguoiMoi.clear()

        self.btnChupAnh.show()
        self.btnChupAnh.setEnabled(self.capture is not None and self.capture.isOpened())

        if self.capture and self.capture.isOpened() and not self.timer.isActive():
            self.timer.start(30)

    def capture_image_action(self):
        if self.capture and self.capture.isOpened():
            ret, frame_bgr = self.capture.read()
            if ret and frame_bgr is not None:
                self.timer.stop()
                self.captured_image = frame_bgr.copy()

                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    h, w, ch = frame_rgb.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qt_image)
                    self.labelCamera.setPixmap(pixmap.scaled(self.labelCamera.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

                    self.btnChupAnh.hide()
                    self.txtTenNguoiMoi.show()
                    self.btnDongY.show()
                    self.btnHuy.show()
                    self.txtTenNguoiMoi.setFocus()
                except Exception as e:
                    QMessageBox.warning(self, "Lỗi", f"Lỗi hiển thị ảnh: {e}")
                    traceback.print_exc()
                    self.reset_ui_to_capture_mode()
            else:
                QMessageBox.warning(self, "Lỗi", "Không thể chụp ảnh. Thử lại.")
                self.reset_ui_to_capture_mode()
        else:
            QMessageBox.warning(self, "Lỗi", "Camera chưa sẵn sàng.")
            self.reset_ui_to_capture_mode()

    def confirm_action(self):
        user_name_raw = self.txtTenNguoiMoi.text()
        user_name = ''.join(c for c in user_name_raw if c.isalnum() or c in [' ', '_', '-']).strip()
        user_name = '_'.join(user_name.split())

        if not user_name:
            QMessageBox.warning(self, "Thiếu Thông Tin", "Vui lòng nhập tên hợp lệ.")
            return

        if self.captured_image is None:
            QMessageBox.critical(self, "Lỗi", "Chưa có ảnh được chụp.")
            self.reset_ui_to_capture_mode()
            return

        if not generate_and_save_embeddings:
            QMessageBox.critical(self, "Lỗi Hệ Thống", "Hàm generate_and_save_embeddings không khả dụng.")
            return

        # Tạo thư mục tên duy nhất
        new_id_str = ""
        folder_name = ""
        for _ in range(100):
            temp_id = random.randint(100, 999)
            formatted_id = f"{temp_id:03d}"
            potential_folder_name = f"{formatted_id}_{user_name}"
            potential_folder_path = os.path.join(IMAGES_FOLDER, potential_folder_name)

            if not os.path.exists(potential_folder_path):
                new_id_str = formatted_id
                folder_name = potential_folder_name
                break

        if not new_id_str:
            QMessageBox.critical(self, "Lỗi", "Không thể tạo thư mục duy nhất.")
            self.reset_ui_to_capture_mode()
            return

        folder_path = os.path.join(IMAGES_FOLDER, folder_name)
        image_filename = f"{folder_name}.png"
        save_path = os.path.join(folder_path, image_filename)

        try:
            os.makedirs(folder_path, exist_ok=True)
            success = cv2.imwrite(save_path, self.captured_image)
            if not success:
                raise IOError(f"Lưu ảnh thất bại tại '{save_path}'")
            print(f"Đã lưu ảnh: {save_path}")

            print("Đang cập nhật embeddings...")
            generate_and_save_embeddings()
            QMessageBox.information(self, "Thành công", "Người dùng đã được thêm.")
            self.user_added.emit()
            self.close()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi", f"Có lỗi khi lưu ảnh hoặc tạo embedding: {e}")
            traceback.print_exc()
            self.reset_ui_to_capture_mode()

    def cancel_action(self):
        self.reset_ui_to_capture_mode()
