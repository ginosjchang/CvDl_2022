# cvdl2022_hw1
## how to use
1.  For question 1~4 ```make``` or ```python hw1.py```` to run the user interface.
2.  For question 5, ```python func.py``` will start training process and ```make dl``` or ```python dl.py``` to run the user interface.Remaber to rename youre model to ```model/model.h5```.
## Question 1 camera Calibration
### 1.1 Corner Detection
1. Find and draw the corners on the chessboard for each image.
2. Click button "1.1" to show each picture 0.5 seconds.
### 1.2 Find the Intrinsic Matrix
1. Find the intrinsic matrix
2. Click button "1.2" and then show the result on the console.
### 1.3 Find the Extrinsic Matrix
1. Find the extinsic mtrix of the chessboard for each of the 15 images.
2. Clicck button "1.3" and then show the result on the console window.
### 1.4 Find the Distortion Matrix
1. FInd the distortion matrix
2. Click button "1.4" to show the result on the console window.
### 1.5 Show the undistorted result
1. Undistort the chessboard imges.
2. Show distorted and undistorted images.
## Question 2 Augmented Reality
1. Clibrate 5 images to get intrinsic, distortion and extrinsic prameterss.
2. Input a "WOrd" less than 6 char in English in the textEdit box.
3. Deive the shape of the "Word" by using the provided library.
4. Show the "Word" on the chessboards images (1.bmp to 5.bmp).
5. Show the "Word" vertically on the chessboards images (1.bmp to 5.bmp).
6. CLick the button to show the "Word" on the picture. Show each picture for 1 second (totl 5 imaages).
## Question 3 Stereo Disparity Map
### 3.1 Stereo Disparity Map
1. Find the dispaarity map/image based on Left and Right stereo images.
### 3.2 Checking the Disparity Value
1. Click at left images and draw the correspoinding dot at right image.
## Question 4 SIFT (Scale-Invarriaant Feature Transform)
### 4.1 Keypoints
1. Load image 1 traffics.png (click "Load image 1").
2. Based on SIFT algorithm, find keypoints on traffics.png.
3. Then save and draw the keypoints of traffics.png as figuer 1.
### 4.2 Mtched Keypoints
1. Load Image 1 (traffics.png) and Image 2 (ambulance.png).
2. Based on SIFT algorithm, find the keypoints and descriptors at image 1 and image 2 (same as question 4.1) nd match the most related between descriptorrs 1 and descriptors 2.
3. Save and draw the matched feature points between two image, traffic.png and ambulance.png, show the results as figure 2.
### Training Cifr10 Classifier Using VGG19
### 5.1 Load Cifar10 and Random Show 9 Images with Label
### 5.2 Load Model and Show Model Structure.
### 5.3 Show Data AAugmentation Result.
### 5.4 SHow Accuracy and Loss
### 5.5 Inference
