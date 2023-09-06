from lib2to3.pytree import Base
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import ui.hw1_ui as ui

import func

import cv2

import numpy as np
import os
from glob import glob

class Main(QMainWindow, ui.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        #Store images in dictionary.
        self.images = dict()
        self.images['Image1'] = None #Left image of stereo
        self.images['Image2'] = None #Right image of stereo
        
        self.initial() #reset all param

        #Image Load function
        self.btn_load_floder.clicked.connect(self.LoadFloder)
        self.btn_load1.clicked.connect(self.Load1)
        self.btn_load2.clicked.connect(self.Load2)

        #Question 1 button link
        self.btn_1_1.clicked.connect(self.btn_1_1_clicked)
        self.btn_1_2.clicked.connect(self.btn_1_2_clicked)
        self.btn_1_3.clicked.connect(self.btn_1_3_clicked)
        self.btn_1_4.clicked.connect(self.btn_1_4_clicked)
        self.btn_1_5.clicked.connect(self.btn_1_5_clicked)

        #Question 1 button link
        self.btn_2_1.clicked.connect(self.btn_2_1_clicked)
        self.btn_2_2.clicked.connect(self.btn_2_2_clicked)

        #Question 3 button link
        self.btn_3_1.clicked.connect(self.btn_3_1_clicked)

        #Question 4 button link
        self.btn_4_1.clicked.connect(self.btn_4_1_clicked)
        self.btn_4_2.clicked.connect(self.btn_4_2_clicked)
    
    def imshow(self, fname, image):
        cv2.namedWindow(fname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(fname, 800, 600)
        cv2.imshow(fname, image)

    def initial(self):
        self.isCalibrated = False
        self.images['chess'] = [] #Original images in floder.
        self.images['chess_corner'] = [] #Calibration after corner detection.
        self.images['chess_result'] = [] #Combine of origin and undistorted image.
        self.comboBox_1_3.clear()

    def LoadFloder(self):
        try:
            self.initial() #Clear all parameters which are related with floder

            # Select Directory
            floder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

            '''
                Load All .bmp in directory.
            '''
            fnames = glob(os.path.join(floder, '*.bmp'))
            for fname in fnames:
                self.images['chess'].append(cv2.imread(fname))
            # add item in comboBox which user can select.
            for i in range(len(self.images['chess'])):
                self.comboBox_1_3.addItem(str(i + 1), i)

            '''
                Load alphabet liberary in directory.
            '''
            if os.path.exists(os.path.join(floder, "Q2_lib")): #chec the floder contain alphabet liberary
                self.fs = cv2.FileStorage(os.path.join(floder, 'Q2_lib/alphabet_lib_onboard.txt'), cv2.FILE_STORAGE_READ)
                self.fs_v = cv2.FileStorage(os.path.join(floder, 'Q2_lib/alphabet_lib_vertical.txt'), cv2.FILE_STORAGE_READ)
        
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def Load1(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './',"Image files (*.jpg *.png *.bmp)")  #Select file
            self.images['Image1'] = cv2.imread(fname[0]) #Load file
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def Load2(self):
        try:
            fname = QFileDialog.getOpenFileName(self, 'Open file', './',"Image files (*.jpg *.png *.bmp)") #Select file
            self.images['Image2'] = cv2.imread(fname[0]) #Load fle
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def calibrate(self):
        if self.isCalibrated: return #Camera is calibrated or not
        elif len(self.images['chess']) == 0: raise BaseException("Calibration fail: No input images") #Chec images are loaded success
        else: self.isCalibrated = True
        try:
            print("\033[0;33m[Wait] Calibrating\033[0;37m")
            self.images['chess_corner'] = []
            self.mtx, self.dist, self.rvecs, self.tvecs = func.calibration(self.images['chess'], self.images['chess_corner'])

            #calculation the undistortion image
            func.undistort(self.images['chess'], self.images['chess_result'], self.mtx, self.dist)

            print("\033[0;32m[Done] Calibrated\033[0;37m")
        except BaseException as e:
            self.isCalibrated = False
            raise BaseException(e)

    def btn_1_1_clicked(self):
        try:
            self.calibrate()

            for image in self.images['chess_corner']:
                self.imshow("Corners", image)
                cv2.waitKey(500)

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

        cv2.destroyAllWindows()

    def btn_1_2_clicked(self):
        try:
            self.calibrate()

            print("Intrinsic:\n", self.mtx)

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def btn_1_3_clicked(self):
        try:
            self.calibrate()

            text = int(self.comboBox_1_3.currentText()) - 1 #Get the text in combox.

            rmtx = np.zeros((3,3))
            cv2.Rodrigues(self.rvecs[text], rmtx)
            print("Extrinsic Matrix:\n", np.hstack((rmtx, self.tvecs[text])))

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def btn_1_4_clicked(self):
        try:
            self.calibrate()

            print("Distortion Matrix:\n", self.dist)

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    
    def btn_1_5_clicked(self):
        try:
            self.calibrate()

            for image in self.images['chess_result']:
                self.imshow("Distorted V.S Undistorted", image)
                cv2.waitKey(500)
            
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
        
        cv2.destroyAllWindows()

    def ar(self, fs):
        text = self.lineEdit.text() #Get the text in lineEdit
        if not text.isupper(): raise BaseException("Only input uppercase alphabet") #Check the input format

        images = func.augmented_reality(self.images['chess'], text, fs, self.rvecs, self.tvecs, self.mtx, self.dist)

        for image in images:
            self.imshow("AR", image)
            cv2.waitKey(1000)

        cv2.destroyAllWindows()

    def btn_2_1_clicked(self):
        try:
            self.calibrate()
            self.ar(self.fs)
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    
    def btn_2_2_clicked(self):
        try:
            self.calibrate()
            self.ar(self.fs_v)

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    
    def mouse_clicked(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            image = self.images['Image2'].copy()
            #The disparity map is 16-fixed-point which create from StereoBM_create.
            d = int(self.disparity[y][x] / 16) #d is -1 ~ 255
            if d == -1: return #not find the corressponding point.
            cv2.circle(image, (x - d, y), radius=10, color=(0, 255, 0), thickness=-1)
            self.imshow('Right', image)

    def btn_3_1_clicked(self):
        try:
            self.disparity, norm_disparity = func.Stereo(self.images['Image1'], self.images['Image2'])
            
            self.imshow('Disparity Map', norm_disparity)
            self.imshow('Left', self.images['Image1'])
            self.imshow('Right', self.images['Image2'])
            cv2.setMouseCallback('Left', self.mouse_clicked)

            cv2.waitKey()

        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

        cv2.destroyAllWindows()

    def btn_4_1_clicked(self):
        try:
            image, kp, des = func.sift_detect(self.images['Image1'])
            
            self.imshow("SIFT", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

    def btn_4_2_clicked(self):
        try:
            image = func.sift_matcher(self.images['Image1'], self.images['Image2'])
            
            self.imshow("Match", image)
            cv2.waitKey()
            cv2.destroyAllWindows()
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Main()
    window.show()
    sys.exit(app.exec_())