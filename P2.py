import cv2
import numpy as np

# detect shape and transform it to a square

cap = cv2.VideoCapture(-1) # 0 should work, but sometimes fails on linux
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)
width, height = 640, 480

def preProcess(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, 200, 200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=2)
    imgThres = cv2.erode(imgDial, kernel, iterations=1)
    return imgThres

def getWarp(img, biggest):
    orderpts = reorder(biggest)
    pts1 = np.float32(orderpts)
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (width, height))[20:img.shape[0]-20, 20:img.shape[1]-20]

def reorder(myPoints):
    myPoints = myPoints.reshape((4,2))
    newPoints = np.zeros((4,1,2), np.int32)
    add = myPoints.sum(1)
    newPoints[0] = myPoints[np.argmin(add)]
    newPoints[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints,axis=1)
    newPoints[1] = myPoints[np.argmin(diff)]
    newPoints[2] = myPoints[np.argmax(diff)]
    return newPoints

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    #contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #RETR_TREE, cv2.CHAIN_APPROX_SIMPLE

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 5000:
            #cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            if len(approx) == 4 and area > maxArea:
                biggest = approx
    cv2.drawContours(imgContour, biggest, -1, (255, 0, 0), 20)
    return biggest

def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    if rowsAvailable:
        for x in range (0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0,0), None,  scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2RGB)
        imageBlank = np.zeros((imgArray[0][0].shape[0], imgArray[0][0].shape[1], 3), np.uint8)
        hor = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2RGB)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

img = cv2.imread('Resources/document.png')
while True:
    #success, img = cap.read()
    img = cv2.resize(img, (width, height))
    imgContour = img.copy()
    imgThres = preProcess(img)
    biggest = getContours(imgThres)
    if biggest.size != 0:
        imgWarped = getWarp(img, biggest)
        imgArray = ([img,imgContour], [imgThres, imgWarped])
    else:
        imgArray = ([img,imgThres])
    imgStacked = stackImages(0.8,imgArray)
    cv2.imshow("r", imgStacked)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.destroyAllWindows()
        break
