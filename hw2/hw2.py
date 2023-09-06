from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import cv2
import numpy as np
from glob import glob
import os

import UI.hw2_ui as ui
import func

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.isPCA = False

        self.btn_load_video.clicked.connect(self.btn_load_video_clicked)
        self.btn_load_image.clicked.connect(self.btn_load_image_clicked)
        self.btn_load_floder.clicked.connect(self.btn_load_floder_clicked)

        self.btn_1_1.clicked.connect(self.btn_1_1_clicked)

        self.btn_2_1.clicked.connect(self.btn_2_1_clicked)
        self.btn_2_2.clicked.connect(self.btn_2_2_clicked)

        self.btn_3_1.clicked.connect(self.btn_3_1_clicked)
        
        self.btn_4_1.clicked.connect(self.btn_4_1_clicked)
        self.btn_4_2.clicked.connect(self.btn_4_2_clicked)
    
    def btn_load_video_clicked(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './',"Vedio files (*.mp4)")  #Select file
            self.label_load_video.setText(fname[0])

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    
    def btn_load_image_clicked(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './',"Image files (*.png)")  #Select file
            self.label_load_image.setText(fname[0])
        
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def btn_load_floder_clicked(self):
        try:
            floder_path = str(QFileDialog.getExistingDirectory(self, "Select Directory")) #Select floder
            self.label_load_floder.setText(floder_path)
            self.isPCA = False
        
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def video_wait(self, wait_time = 10):
        k = cv2.waitKey(wait_time) & 0xff
        if k == 27:
            return True
        return False

    def btn_1_1_clicked(self):
        try:
            ave, std = func.create_gaussian(self.label_load_video.text(), 25)
            capture = cv2.VideoCapture(self.label_load_video.text())
            w, h = ave.shape

            while(capture.isOpened()):
                ret, image = capture.read()
                if not ret: break

                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                ret, mask = cv2.threshold(abs(gray - ave) - std, 0, 255, cv2.THRESH_BINARY)
                mask = mask.astype("uint8")

                foreground = cv2.bitwise_and(image, image, mask=mask)
                cv2.imshow("video", np.hstack((image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), foreground)))
                if self.video_wait(): break
                
            capture.release()
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
        finally:
            cv2.destroyAllWindows()
    
    def btn_2_1_clicked(self):
        try:
            capture = cv2.VideoCapture(self.label_load_video.text())
            ret, image = capture.read()
            if not ret: return
            
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            points = func.detect_blob(gray)

            func.draw_blob(image, points)

            func.imshow("Keypoints", image)
            cv2.waitKey(0)
            
            capture.release()
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
        finally:
            cv2.destroyAllWindows()
    
    def btn_2_2_clicked(self):
        try:
            capture = cv2.VideoCapture(self.label_load_video.text())
            func.flow(capture)
            capture.release()
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def btn_3_1_clicked(self):
        try:
            capture = cv2.VideoCapture(self.label_load_video.text())
            logo = cv2.imread(self.label_load_image.text())
            while(capture.isOpened()):
                ret, image = capture.read()
                if not ret: break

                result = func.homography(logo, image)
                result = np.hstack([image, result])
                func.imshow("Perspective Transform", result)
                if self.video_wait(): break
            capture.release()
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
        finally:
            cv2.destroyAllWindows()
    
    def pca_reconstruction(self):
        if self.isPCA: return

        images = func.load_jpg(self.label_load_floder.text())

        reconstruction, self.error = func.recon_pca(images)

        self.pca_result = np.hstack((images, reconstruction))
        self.isPCA = True

    def btn_4_1_clicked(self):
        try:
            self.pca_reconstruction()

            for image in self.pca_result:
                cv2.imshow("PCA reconstruction", image)
                if self.video_wait(500): break
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
        finally:
            cv2.destroyAllWindows()

    def btn_4_2_clicked(self):
        try:
            self.pca_reconstruction()
            print("max error: ", np.max(self.error))
            print("min error: ", np.min(self.error))
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())