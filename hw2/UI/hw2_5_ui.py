# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'UI/hw2_5_ui.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(706, 546)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout.setObjectName("verticalLayout")
        self.btn_load = QtWidgets.QPushButton(self.groupBox)
        self.btn_load.setObjectName("btn_load")
        self.verticalLayout.addWidget(self.btn_load)
        self.btn_1 = QtWidgets.QPushButton(self.groupBox)
        self.btn_1.setObjectName("btn_1")
        self.verticalLayout.addWidget(self.btn_1)
        self.btn_2 = QtWidgets.QPushButton(self.groupBox)
        self.btn_2.setObjectName("btn_2")
        self.verticalLayout.addWidget(self.btn_2)
        self.btn_3 = QtWidgets.QPushButton(self.groupBox)
        self.btn_3.setObjectName("btn_3")
        self.verticalLayout.addWidget(self.btn_3)
        self.btn_4 = QtWidgets.QPushButton(self.groupBox)
        self.btn_4.setObjectName("btn_4")
        self.verticalLayout.addWidget(self.btn_4)
        self.btn_5 = QtWidgets.QPushButton(self.groupBox)
        self.btn_5.setObjectName("btn_5")
        self.verticalLayout.addWidget(self.btn_5)
        self.horizontalLayout.addWidget(self.groupBox)
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_image = QtWidgets.QLabel(self.widget)
        self.label_image.setObjectName("label_image")
        self.verticalLayout_2.addWidget(self.label_image)
        self.label_prediction = QtWidgets.QLabel(self.widget)
        self.label_prediction.setAlignment(QtCore.Qt.AlignCenter)
        self.label_prediction.setObjectName("label_prediction")
        self.verticalLayout_2.addWidget(self.label_prediction)
        self.verticalLayout_2.setStretch(0, 7)
        self.verticalLayout_2.setStretch(1, 1)
        self.horizontalLayout.addWidget(self.widget)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 706, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "2022 CvDl HW2"))
        self.groupBox.setTitle(_translate("MainWindow", "5. ResNet50"))
        self.btn_load.setText(_translate("MainWindow", "Load Image"))
        self.btn_1.setText(_translate("MainWindow", "1. Show Images"))
        self.btn_2.setText(_translate("MainWindow", "2. Show Distribution"))
        self.btn_3.setText(_translate("MainWindow", "3. Show Model Structure"))
        self.btn_4.setText(_translate("MainWindow", "4. Show Comparison"))
        self.btn_5.setText(_translate("MainWindow", "5. Inference"))
        self.label_image.setText(_translate("MainWindow", "TextLabel"))
        self.label_prediction.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
