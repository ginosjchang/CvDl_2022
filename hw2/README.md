# cvdl2022_hw2
## How to use
1. For question 1~4 ```make``` or ```python hw2.py``` to run the user interface.
2. For question 5, ```python train.py``` will start training process and ```make q5``` or ```python hw2_5.py``` to run the user interface.Remaber to rename youre model to model/model.h5.
## Question 1 Background Subtaction
1. Click the button "1.1 Background Subtraction" to show  windows: Origin Mask Foreground video.
2. Use first 25 frames to build the background model.
3. Do not use OpenCV function: cv2.createbackgroundSubtractor()
## Question 2 Optical Flow
### 2.1 Perspective Transform
1. Detect the 7 blue circles and display 1 image (first frame only). with them marked with aa red square bounding box and cross mark.
### 2.2 Video tracking
1. Track the 7 center points on the whole video uinsg OpenCV function.
2. Display the trajectory of each of the 7 tracking points throughout the video.
## Question 3 Perspective Transform
1. Click the button to load video.mp4 and logo.png.
2. Click the button to show result video.
## Question 4 PCA
### 4.1 Image Reconstruction
  Using PCA (Principal components analysis) to do dimension reduction and then reconstruct it back. CLick button "4.1 Image Reconstruction" and show original and reconstructed images.
### 4.2 Compute the reconstuction error
  Computing the reconstruction error (RE) for each logo.
## Quesstion 5 Trrain a Cat-Dog Classifier Using ResNet50
### 5.1 Load the dataset and resize images.
### 5.2 Plot class distribution of training dataset
### 5.3 Show the structure of ResNet50 model.
### 5.4 Set up 2 kinds of loss functions to train 2 ResNet50 models.
### 5.5 Compaare the accuracies of 2 ResNet50 models on validation dataset.
### 5.6 Use the better-trained model to run inference and show the predicted class label.
