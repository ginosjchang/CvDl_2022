# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'dl_ui.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(680, 550)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btn_load = QtWidgets.QPushButton(self.groupBox)
        self.btn_load.setObjectName("btn_load")
        self.verticalLayout_2.addWidget(self.btn_load)
        self.btn_5_1 = QtWidgets.QPushButton(self.groupBox)
        self.btn_5_1.setObjectName("btn_5_1")
        self.verticalLayout_2.addWidget(self.btn_5_1)
        self.btn_5_2 = QtWidgets.QPushButton(self.groupBox)
        self.btn_5_2.setObjectName("btn_5_2")
        self.verticalLayout_2.addWidget(self.btn_5_2)
        self.btn_5_3 = QtWidgets.QPushButton(self.groupBox)
        self.btn_5_3.setObjectName("btn_5_3")
        self.verticalLayout_2.addWidget(self.btn_5_3)
        self.btn_5_4 = QtWidgets.QPushButton(self.groupBox)
        self.btn_5_4.setObjectName("btn_5_4")
        self.verticalLayout_2.addWidget(self.btn_5_4)
        self.btn_5_5 = QtWidgets.QPushButton(self.groupBox)
        self.btn_5_5.setObjectName("btn_5_5")
        self.verticalLayout_2.addWidget(self.btn_5_5)
        self.horizontalLayout.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setObjectName("label_2")
        self.verticalLayout.addWidget(self.label_2)
        self.label = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 8)
        self.horizontalLayout.addWidget(self.groupBox_2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 680, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "VGG19"))
        self.groupBox.setTitle(_translate("MainWindow", "5. VGG19"))
        self.btn_load.setText(_translate("MainWindow", "Load Image"))
        self.btn_5_1.setText(_translate("MainWindow", "1. Show Train Images"))
        self.btn_5_2.setText(_translate("MainWindow", "2. Show Model Structure"))
        self.btn_5_3.setText(_translate("MainWindow", "3. Show Data Augmentation"))
        self.btn_5_4.setText(_translate("MainWindow", "4. Show Accuracy and Loss"))
        self.btn_5_5.setText(_translate("MainWindow", "5. Inference"))
        self.label_2.setText(_translate("MainWindow", "TextLabel"))
        self.label.setText(_translate("MainWindow", "TextLabel"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
