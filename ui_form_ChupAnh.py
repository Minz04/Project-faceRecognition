# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui/form_ChupAnh.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(940, 476)
        self.labelCamera = QtWidgets.QLabel(Form)
        self.labelCamera.setGeometry(QtCore.QRect(30, 30, 431, 411))
        self.labelCamera.setText("")
        self.labelCamera.setScaledContents(True)
        self.labelCamera.setObjectName("labelCamera")
        self.txtTenNguoiMoi = QtWidgets.QLineEdit(Form)
        self.txtTenNguoiMoi.setGeometry(QtCore.QRect(590, 100, 271, 31))
        self.txtTenNguoiMoi.setObjectName("txtTenNguoiMoi")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(590, 70, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.btnChupAnh = QtWidgets.QPushButton(Form)
        self.btnChupAnh.setGeometry(QtCore.QRect(690, 190, 75, 23))
        self.btnChupAnh.setObjectName("btnChupAnh")
        self.btnDongY = QtWidgets.QPushButton(Form)
        self.btnDongY.setGeometry(QtCore.QRect(590, 270, 81, 41))
        self.btnDongY.setObjectName("btnDongY")
        self.btnHuy = QtWidgets.QPushButton(Form)
        self.btnHuy.setGeometry(QtCore.QRect(780, 270, 81, 41))
        self.btnHuy.setObjectName("btnHuy")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label.setText(_translate("Form", "Nhập tên của bạn:"))
        self.btnChupAnh.setText(_translate("Form", "Chụp ảnh"))
        self.btnDongY.setText(_translate("Form", "Đồng ý"))
        self.btnHuy.setText(_translate("Form", "Hủy"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
