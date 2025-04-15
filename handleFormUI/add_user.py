# -*- coding: utf-8 -*-
import cv2
import os
import random
import numpy as np
import traceback # Để in chi tiết lỗi nếu có
from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, pyqtSignal

# Import class giao diện đã được pyuic tạo ra
from ui_form_ChupAnh import Ui_Form

# Quan trọng: Import hàm tạo embedding từ file CodeGenerator
# Đảm bảo file CodeGenerator_facenet.py có thể chạy độc lập hoặc hàm của nó có thể được gọi
try:
    from CodeGenerator_facenet import generate_and_save_embeddings
except ImportError:
    print("[LỖI] Không thể import 'generate_and_save_embeddings' từ CodeGenerator_facenet.py")
    # Nếu không import được, chức năng thêm người sẽ không thể cập nhật embeddings
    generate_and_save_embeddings = None

IMAGES_FOLDER = os.path.join('Resources', 'Images') # Đường dẫn thư mục lưu ảnh người dùng

class AddUserDialog(QDialog, Ui_Form):
    # Tín hiệu sẽ được phát ra khi người dùng mới được thêm thành công
    # để cửa sổ chính biết và yêu cầu worker tải lại embeddings
    user_added = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Thêm Người Dùng Mới")

        self.capture = None       # Đối tượng VideoCapture để truy cập camera
        self.timer = QTimer(self) # Timer để cập nhật preview từ camera
        self.timer.timeout.connect(self.update_preview)
        self.captured_image = None # Biến lưu ảnh đã chụp (dạng numpy array BGR)

        # Kết nối các nút với hàm xử lý tương ứng
        self.btnChupAnh.clicked.connect(self.capture_image_action)
        self.btnDongY.clicked.connect(self.confirm_action)
        self.btnHuy.clicked.connect(self.cancel_action)

        self.init_camera() # Khởi tạo camera
        self.reset_ui_to_capture_mode() # Đặt giao diện về trạng thái ban đầu (chờ chụp)

    def init_camera(self):
        """Khởi tạo kết nối với webcam."""
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            QMessageBox.warning(self, "Lỗi Camera", "Không thể mở webcam.")
            self.btnChupAnh.setEnabled(False) # Vô hiệu hóa nút chụp nếu không có cam
        else:
            # Bắt đầu timer để hiển thị preview (chỉ khi camera mở thành công)
            self.timer.start(30) # Cập nhật khoảng 30ms một lần

    def update_preview(self):
        """Liên tục đọc frame từ camera và hiển thị lên labelCamera."""
        # Chỉ chạy khi camera tồn tại, đang mở và timer đang hoạt động (chế độ preview)
        if self.capture and self.capture.isOpened() and self.timer.isActive():
            ret, frame_bgr = self.capture.read()
            if ret:
                # Chuyển BGR sang RGB để QPixmap hiển thị đúng màu
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qt_image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                # Hiển thị ảnh, co dãn để vừa với label nhưng giữ tỷ lệ
                self.labelCamera.setPixmap(pixmap.scaled(self.labelCamera.size(),
                                                          Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def reset_ui_to_capture_mode(self):
        """Đặt lại giao diện về trạng thái chờ người dùng bấm nút chụp ảnh."""
        self.txtTenNguoiMoi.hide() # Ẩn ô nhập tên
        self.btnDongY.hide()       # Ẩn nút Đồng ý
        self.btnHuy.hide()         # Ẩn nút Hủy
        self.labelCamera.clear()   # Xóa ảnh đang hiển thị (nếu có)
        self.captured_image = None # Xóa ảnh đã chụp trước đó (nếu có)
        self.txtTenNguoiMoi.clear()# Xóa tên đã nhập trước đó (nếu có)

        self.btnChupAnh.show()     # Hiện nút Chụp ảnh
        # Kích hoạt nút chụp chỉ khi camera sẵn sàng
        self.btnChupAnh.setEnabled(self.capture is not None and self.capture.isOpened())

        # Nếu timer chưa chạy (ví dụ sau khi nhấn Hủy), khởi động lại để hiển thị preview
        if not self.timer.isActive() and self.capture and self.capture.isOpened():
            self.timer.start(30)

    def capture_image_action(self):
        """Thực hiện chụp ảnh từ camera và chuyển sang giao diện nhập tên."""
        if self.capture and self.capture.isOpened():
            ret, frame_bgr = self.capture.read() # Đọc frame hiện tại từ camera
            if ret:
                self.timer.stop() # Dừng cập nhật preview, giữ nguyên ảnh vừa chụp
                self.captured_image = frame_bgr # Lưu ảnh đã chụp (BGR)

                # Hiển thị ảnh vừa chụp lên labelCamera
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                h, w, ch = frame_rgb.shape
                qt_image = QImage(frame_rgb.data, w, h, w * ch, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.labelCamera.setPixmap(pixmap.scaled(self.labelCamera.size(),
                                                          Qt.KeepAspectRatio, Qt.SmoothTransformation))

                # Chuyển đổi giao diện: ẩn nút chụp, hiện các control nhập liệu
                self.btnChupAnh.hide()
                self.txtTenNguoiMoi.show()
                self.btnDongY.show()
                self.btnHuy.show()
                self.txtTenNguoiMoi.setFocus() # Đặt con trỏ vào ô nhập tên
            else:
                QMessageBox.warning(self, "Lỗi Chụp Ảnh", "Không thể đọc frame từ camera.")
                self.reset_ui_to_capture_mode() # Quay lại trạng thái chờ chụp
        else:
             QMessageBox.warning(self, "Lỗi Camera", "Camera chưa sẵn sàng.")

    def confirm_action(self):
        """Xử lý khi nhấn nút Đồng ý: Lưu ảnh, tạo ID, chạy lại embedding, đóng dialog."""
        user_name = self.txtTenNguoiMoi.text().strip() # Lấy tên và xóa khoảng trắng thừa
        if not user_name:
            QMessageBox.warning(self, "Thiếu Thông Tin", "Vui lòng nhập tên người dùng.")
            return
        if self.captured_image is None:
             QMessageBox.critical(self, "Lỗi Logic", "Chưa có ảnh nào được chụp để lưu.")
             self.reset_ui_to_capture_mode()
             return
        # Kiểm tra xem hàm tạo embedding có sẵn sàng không
        if not generate_and_save_embeddings:
            QMessageBox.critical(self,"Lỗi Hệ Thống", "Chức năng tạo embedding không hoạt động (kiểm tra import).")
            return

        # --- Tạo ID ngẫu nhiên và đảm bảo không trùng ---
        new_id_str = ""
        filename = ""
        max_tries = 100 # Giới hạn số lần thử tạo ID để tránh vòng lặp vô hạn
        for _ in range(max_tries):
            temp_id = random.randint(1, 9999) # Random ID từ 1 đến 9999
            formatted_id = str(temp_id)      # Chuyển thành chuỗi
            # Tạo tên file dự kiến theo định dạng "ID_Tên.png"
            potential_filename = f"{formatted_id}_{user_name}.png" # Lưu dưới dạng PNG
            id_exists = False # Cờ kiểm tra ID đã tồn tại chưa
            if os.path.exists(IMAGES_FOLDER):
                 # Duyệt qua các file trong thư mục Images để kiểm tra ID
                 for existing_file in os.listdir(IMAGES_FOLDER):
                     # Kiểm tra xem tên file có bắt đầu bằng "ID_" không
                     if existing_file.startswith(formatted_id + "_"):
                         id_exists = True
                         break # Đã tìm thấy ID trùng, không cần kiểm tra nữa
            if not id_exists: # Nếu ID này chưa tồn tại
                new_id_str = formatted_id # Lưu ID hợp lệ
                filename = potential_filename # Lưu tên file hợp lệ
                break # Thoát vòng lặp vì đã tìm được ID duy nhất

        if not new_id_str: # Nếu sau max_tries vẫn không tìm được ID duy nhất
             QMessageBox.critical(self, "Lỗi Tạo ID", f"Không thể tạo ID duy nhất sau {max_tries} lần thử. Thư mục Images có thể đã đầy hoặc có lỗi.")
             self.reset_ui_to_capture_mode() # Cho phép người dùng thử lại
             return

        # --- Lưu ảnh và gọi hàm tạo lại embeddings ---
        save_path = os.path.join(IMAGES_FOLDER, filename)
        try:
            os.makedirs(IMAGES_FOLDER, exist_ok=True) # Tạo thư mục nếu chưa có
            # Lưu ảnh đã chụp (dạng BGR) vào file
            success = cv2.imwrite(save_path, self.captured_image)
            if not success:
                 raise IOError("Lưu ảnh thất bại (cv2.imwrite trả về false).")
            print(f"Ảnh đã được lưu thành công: {save_path}")

            # *** Rất quan trọng: Gọi lại hàm để cập nhật file embeddings ***
            print("Đang tạo lại file embeddings...")
            # Hàm này sẽ đọc lại toàn bộ thư mục Images (bao gồm cả ảnh mới)
            # và ghi đè lại file Embeddings_Facenet.p
            embedding_success = generate_and_save_embeddings()

            if embedding_success:
                QMessageBox.information(self, "Thành Công", f"Đã lưu người dùng '{user_name}' (ID: {new_id_str}) và cập nhật embeddings.")
                self.user_added.emit() # Phát tín hiệu báo đã thêm thành công
                self.accept() # Đóng dialog với trạng thái thành công (OK)
            else:
                 # Lưu ảnh thành công nhưng embedding lỗi
                 QMessageBox.warning(self, "Lưu Ảnh OK, Embedding Lỗi", f"Đã lưu ảnh '{filename}' nhưng cập nhật file embeddings thất bại. Nhận diện có thể không chính xác cho người mới. Hãy thử chạy lại CodeGenerator thủ công.")
                 # Vẫn phát tín hiệu và đóng dialog, nhưng cảnh báo người dùng
                 self.user_added.emit() # Vẫn phát tín hiệu để cửa sổ chính biết
                 self.accept()

        except Exception as e:
            QMessageBox.critical(self, "Lỗi Lưu File hoặc Embedding", f"Quá trình lưu bị lỗi!\nChi tiết: {e}")
            traceback.print_exc() # In lỗi chi tiết ra console để debug
            # Không đóng dialog, cho phép người dùng thử lại sau khi xem lỗi
            self.reset_ui_to_capture_mode()

    def cancel_action(self):
        """Xử lý khi nhấn nút Hủy: Quay lại trạng thái chờ chụp ảnh."""
        self.reset_ui_to_capture_mode()

    def closeEvent(self, event):
        """Được gọi tự động khi cửa sổ dialog đóng (bằng nút X hoặc các lệnh accept/reject)."""
        print("Đang đóng cửa sổ Thêm Người Dùng...")
        self.timer.stop() # Dừng timer cập nhật preview
        if self.capture and self.capture.isOpened():
            self.capture.release() # Giải phóng camera
            print("Camera của cửa sổ Thêm Người Dùng đã được giải phóng.")
        self.capture = None # Đặt lại biến camera
        super().closeEvent(event) # Gọi hàm closeEvent của lớp cha