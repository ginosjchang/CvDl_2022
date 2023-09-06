from glob import glob
import cv2
import numpy as np
import os

AR_offset = [[6, 5], [3, 5], [0, 5],
            [6, 2], [3, 2], [0, 2]] #The posiont of words on chess board

#numDisparities is the number of depth layer. The 256 layer is from -1 ~ 255.
#The picture will be seperate in several block which determined by the parameter blockSize
stereo = cv2.StereoBM_create(numDisparities=256, blockSize=25)

sift = cv2.SIFT_create()

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

learning_rate = 1e-5
batch_size = 8
epoch_num = 30

def calibration(images, corner_images):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) #termination criteria

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((8*11,3), np.float32)
    objp[:,:2] = np.mgrid[0:11,0:8].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for image in images:
        corner_image = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (11, 8), None) #Find the chess board corners

        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)

            cv2.drawChessboardCorners(corner_image, (11,8), corners2, ret) #Draw and display the corners
            corner_images.append(corner_image)
        else:
            raise BaseException("Calibration fail: Can't find corner")

    #Calibration.
    #mtx is intrisnic metrix. dist is distortion metrix. rvecs is rotation vectors. tvecs ar translation vetors.
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return mtx, dist, rvecs, tvecs

def undistort(images, undist_images, mtx, dist):
    for image in images:
        h, w = image.shape[:2]
        dst = cv2.undistort(image, mtx, dist, None, mtx)
        undist_images.append(np.hstack((image, dst)))

def augmented_reality(images, word, fs, rvecs, tvecs, mtx, dist):
    objpoint = []
    ar_images = []

    for i in range(len(word)):
        refpts = fs.getNode(word[i]).mat().reshape(-1, 3).astype("float32") #Get the alphabet 3d coordinate.
        for refpt in refpts:
            objpoint.append([refpt[0] + AR_offset[i][0], refpt[1] + AR_offset[i][1], refpt[2]]) #translaton to the corressponding position.

    for i in range(len(images)):
        imgpts, jac = cv2.projectPoints(np.array(objpoint), rvecs[i], tvecs[i], mtx, dist) #project 3d to 2d
        imgpts = imgpts.astype("int").reshape(-1, 2)
        image = images[i].copy()
        for j in range(0, len(imgpts), 2):
            image = cv2.line(image, tuple(imgpts[j]), tuple(imgpts[j + 1]), (0, 0, 255), 5) #draw the line on pictures.
        ar_images.append(image)
    return ar_images

def Stereo(left_image, right_image):
    gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

    disparity = stereo.compute(gray_left, gray_right) #Compute the disparity map.
    norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity, norm

def sift_detect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # find the keypoints and descriptors with SIFT
    kp, des = sift.detectAndCompute(gray, None)
    img = image.copy()

    cv2.drawKeypoints(gray, kp, img)

    return img, kp, des

def sift_matcher(image1, image2):
    _, kp1, des1 = sift_detect(image1)
    _, kp2, des2 = sift_detect(image2)

    matches = flann.knnMatch(des1, des2, k=2)
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    image = cv2.drawMatchesKnn(image1, kp1, image2, kp2, matches, None, **draw_params)
    return image

def DataLoader():
    import tensorflow as tf
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    train_y = tf.keras.utils.to_categorical(train_y, 10)
    test_y = tf.keras.utils.to_categorical(test_y, 10)
    return (train_x, train_y), (test_x, test_y)

def augmentation(image, label):
    import torchvision
    from PIL import Image
    aug_image = []
    aug_label = []

    for i in range(len(image)):
        im = Image.fromarray(image[i]) #Convert to PIL
        aug_image.append(image[i])
        aug_label.append(label[i])
        aug_image.append(np.array(torchvision.transforms.RandomRotation(degrees=30)(im)))
        aug_label.append(label[i])
        aug_image.append(np.array(torchvision.transforms.RandomResizedCrop(size=224)(im)))
        aug_label.append(label[i])
        aug_image.append(np.array(torchvision.transforms.RandomHorizontalFlip()(im)))
        aug_label.append(label[i])
    
    return np.array(aug_image), np.array(aug_label)

def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01

def plt_history(history):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Traing', 'Testing'], loc='lower right')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()


if __name__ == '__main__':
    import datetime
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + current_time
    model_name = 'models/Adam' + current_time +'.h5'

    import tensorflow as tf

    (train_image, train_label), (test_image, test_label) = DataLoader() #Load Cifar10 Data

    aug_image, aug_label = augmentation(train_image, train_label) #Create augmentation
    
    vgg19 = tf.keras.applications.VGG19(weights=None, input_shape = (32, 32, 3), classes=10) #Get VGG19 structure.

    #Select optimizer
    sgd = tf.keras.optimizers.SGD(lr=1e-2, momentum=0.9, nesterov=True)
    adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    vgg19.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy']) #Compile model with loss and optimizer

    #Callback function
    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #training
    vgg19.fit(x=aug_image, y=aug_label, batch_size=batch_size, epochs=epoch_num, callbacks=[change_lr, tensorboard_callback], validation_data=(test_image, test_label))

    vgg19.save(model_name)