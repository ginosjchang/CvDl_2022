import cv2
import numpy as np
import os
from glob import glob
from sklearn.decomposition import PCA

## Blob detector params
blobParams = cv2.SimpleBlobDetector_Params()
# Filter by Area.
blobParams.filterByArea = True
blobParams.minArea = 35
blobParams.maxArea = 90
# Filter by Circularity
blobParams.filterByCircularity = True
blobParams.minCircularity = 0.8
# Filter by Convexity
blobParams.filterByConvexity = True
blobParams.minConvexity = 0.8
# Filter by Inertia
blobParams.filterByInertia = True
blobParams.minInertiaRatio = 0.5
# Create detector
blobDetector = cv2.SimpleBlobDetector_create(blobParams)

## ArUCode param
# arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# arucoParams = cv2.aruco.DetectorParameters_create()
arucoParams = cv2.aruco.DetectorParameters()

def load_jpg(path):
    fnames = glob(os.path.join(path, '*.jpg')) #Get all .bmp file in the direction
    
    images = []

    for fname in fnames:
        image = cv2.imread(fname)
        images.append(image)

    return np.array(images)

def imshow(fname, image):
    cv2.namedWindow(fname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(fname, 800, 600)
    cv2.imshow(fname, image)

def create_gaussian(vedio_name, frame_num):
    capture = cv2.VideoCapture(vedio_name)

    train_data = []
    for i in range(frame_num):
        if(capture.isOpened()):
            ret, image = capture.read()
            if ret == True:
                train_data.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        else:
            break
    
    train_data = np.array(train_data)

    ave = np.average(train_data, axis = 0)
    std = np.std(train_data, axis = 0) * 5
    std[std < 25] = 25
    
    capture.release()
    return ave, std

def detect_blob(frame):
    keypoints = blobDetector.detect(frame)
    points = []
    for pts in keypoints:
        pt = pts.pt
        points.append(np.array([pt[0], pt[1]]))
    return np.array(points).reshape(-1, 1, 2)

def draw_blob(frame, points, color=(0, 0, 255)):
    pts = points.reshape(-1, 2).astype("uint32")
    for pt in pts:
        cv2.rectangle(frame, (pt[0] - 6, pt[1] - 6), (pt[0] + 6, pt[1] + 6), color)
        cv2.line(frame, (pt[0] - 6, pt[1]), (pt[0] + 6, pt[1]), color )
        cv2.line(frame, (pt[0], pt[1] - 6), (pt[0], pt[1] + 6), color )

def flow(capture, color = (0, 255, 255)):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Take first frame and find corners in it
    ret, old_frame = capture.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = detect_blob(old_gray)
    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    while(capture.isOpened()):
        ret, frame = capture.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0.astype("float32"), None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color, 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color, -1)
        img = cv2.add(frame, mask)
        imshow('frame', img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    cv2.destroyAllWindows()

def detect_aruco(image):
    result = np.zeros((4, 2))
    corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
    corners = np.reshape(corners, (-1, 4, 2))
    ids = np.array(ids).flatten()

    for i in range(len(ids)):
        try:
            id = ids[i]
            result[id-1][0] = corners[i][id-1][0]
            result[id-1][1] = corners[i][id-1][1]
        except BaseException as e:
            print("\033[0;31m[ERROR] {0}\033[0;37m".format(e))
    return result.astype("float32")

def homography(logo, image):
    w, h = image.shape[:2]
    # Check the shape of logo same as video image
    if logo.shape[:2] != image.shape[:2]: 
        logo = cv2.resize(logo, (h, w), interpolation=cv2.INTER_AREA)
    # Find corresponding points of corners.
    logo_corners = np.array(([[0, 0], [h-1, 0], [h-1, w-1], [0, w-1]])).astype("float32")
    aruco_corners = detect_aruco(image)
    # find homography to 
    mtx, _ = cv2.findHomography(logo_corners, aruco_corners)
    # rotation and translation the logo
    logo_warp = cv2.warpPerspective(logo, mtx, (h, w))
    # create the mask to subtracte the foreground of video
    logo_warp_gray = cv2.cvtColor(logo_warp, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_warp_gray, 1, 255, cv2.THRESH_BINARY_INV)
    image_foreground = cv2.bitwise_and(image, image, mask = mask)

    # Combine two image
    return cv2.add(image_foreground, logo_warp)

def recon_pca(inputs, n_components = 27):
    n, w, h, c = inputs.shape

    pca = PCA(n_components = n_components, copy = True, whiten = False)

    trans_pca = pca.fit_transform(inputs.reshape(n, -1).astype("uint32"))

    inv = pca.inverse_transform(trans_pca)

    reconstruct = cv2.normalize(inv, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U).reshape(-1, w, h, c).astype("uint8")

    error = np.zeros((n))

    for i in range(n):
        gray1 = cv2.cvtColor(inputs[i], cv2.COLOR_BGR2GRAY).astype(np.uint16)
        gray2 = cv2.cvtColor(reconstruct[i], cv2.COLOR_BGR2GRAY).astype(np.uint16)
        error[i] = np.sqrt(np.sum(np.square(gray1 - gray2)))

    print("PCA variance ratio: ", np.sum(pca.explained_variance_ratio_))

    return reconstruct, error.astype("uint32")