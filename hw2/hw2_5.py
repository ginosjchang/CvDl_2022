from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import tensorflow as tf
import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt

import UI.hw2_5_ui as ui

def createLabels(data):
    for item in data:
        height = item.get_height()
        plt.text(
            item.get_x()+item.get_width()/2., 
            height*1.05, 
            '%d' % int(height),
            ha = "center",
            va = "bottom",
        )

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.label_image.setText("")
        self.label_prediction.setText("")

        self.model = tf.keras.models.load_model('models/binary/binary20221202-233205.h5')

        self.inference_data = tf.keras.utils.image_dataset_from_directory("Dataset_CvDl_Hw2_Q5/inference_dataset", label_mode='binary')

        self.btn_load.clicked.connect(self.btn_load_clicked)
        self.btn_1.clicked.connect(self.btn_1_clicked)
        self.btn_2.clicked.connect(self.btn_2_clicked)
        self.btn_3.clicked.connect(self.btn_3_clicked)
        self.btn_4.clicked.connect(self.btn_4_clicked)
        self.btn_5.clicked.connect(self.btn_5_clicked)
    
    def btn_load_clicked(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './',"Image files (*.jpg *.png *.bmp)")  #Select file
            self.image = cv2.resize(cv2.imread(fname[0]), (224, 224)) #Load file
            self.label_image.setPixmap(QPixmap(fname[0]))
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    
    def btn_1_clicked(self):
        inference_cat = glob(os.path.join("Dataset_CvDl_Hw2_Q5/inference_dataset/Cat", "*.jpg"))
        inference_dog = glob(os.path.join("Dataset_CvDl_Hw2_Q5/inference_dataset/Dog", "*.jpg"))
        image_cat = plt.imread(inference_cat[0])
        image_dog = plt.imread(inference_dog[0])

        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image_cat)
        plt.title("Cat")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(image_dog)
        plt.title("Dog")
        plt.axis('off')
        plt.show()

    def btn_2_clicked(self):
        image = plt.imread("figure/class distribution.png")
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def btn_3_clicked(self):
        self.model.summary()
    
    def btn_4_clicked(self):
        image = plt.imread("figure/accuary comparison.png")
        plt.figure()
        plt.imshow(image)
        plt.axis('off')
        plt.show()
    
    def btn_5_clicked(self):
        image = np.expand_dims(self.image, axis = 0)
        result = self.model.predict(image).reshape(-1)
        if result[0] > 0.5: self.label_prediction.setText("Dog")
        else: self.label_prediction.setText("Cat")

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())