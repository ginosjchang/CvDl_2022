from lib2to3.pytree import Base
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import ui.dl_ui as ui

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import func

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        (self.train_image, self.train_label), test_data = tf.keras.datasets.cifar10.load_data()

        self.labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.model = tf.keras.models.load_model('models/model.h5')

        self.btn_load.clicked.connect(self.btn_load_clicked)
        self.btn_5_1.clicked.connect(self.btn_5_1_clicked)
        self.btn_5_2.clicked.connect(self.btn_5_2_clicked)
        self.btn_5_3.clicked.connect(self.btn_5_3_clicked)
        self.btn_5_4.clicked.connect(self.btn_5_4_clicked)
        self.btn_5_5.clicked.connect(self.btn_5_5_clicked)

        self.label.setText("")
        self.label_2.setText("")
    
    def btn_load_clicked(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './',"Image files (*.jpg *.png *.bmp)")  #Select file
            self.image = cv2.resize(cv2.imread(fname[0]), (32,32)) #Load file
            self.label.setPixmap(QPixmap(fname[0]))
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    
    def btn_5_1_clicked(self):
        plt.figure()
        for i in range(9):
            n = 331 + i
            plt.subplot(n)
            plt.imshow(self.train_image[i])
            plt.title(self.labels[int(self.train_label[i])])   
        plt.show()

    def btn_5_2_clicked(self):
        self.model.summary()
    
    def btn_5_3_clicked(self):
        image, label = func.augmentation(np.expand_dims(self.image, axis=0), np.array([[1]]))

        plt.figure()
        for i in range(3):
            n = 131 + i
            plt.subplot(n)
            plt.imshow(cv2.cvtColor(image[i],cv2.COLOR_BGR2RGB))
        plt.show()

    def btn_5_4_clicked(self):
        acc = cv2.imread('accuracy.png')
        loss = cv2.imread('loss.png')

        cv2.imshow("Accuracy", acc)
        cv2.imshow("Loss", loss)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def btn_5_5_clicked(self):
        image = np.expand_dims(self.image, axis = 0)
        result = self.model.predict(image)
        self.label_2.setText(self.labels[np.argmax(result)])

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())