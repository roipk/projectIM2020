
import glob

import cv2
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt
import imutils
from PIL import Image


imageDatabase = {}

class projectIM2020_q4:

    def get_database_images(self,path,end):
        image_listBW =[]
        image_listC = []
        for filename in glob.glob(path + '/*'+end):
            imbw = cv2.imread(filename, 0)
            imc = cv2.imread(filename, 3)
            image_listBW.append(imbw)
            image_listC.append(imc)
        return  image_listBW,image_listC


    def createImage(self, img):
        # img = cv2.imread(path, cv2.IMREAD_COLOR)
        # Convert to grayscale.
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.blur(gray, (3, 3))
        return gray_blurred

    def createCircles(self,img , cv,resolution,dr,param1,param2,minRadius,maxRadius):
        return cv2.HoughCircles(img, cv, resolution, dr, param1=param1, param2=param2, minRadius=minRadius,maxRadius=maxRadius)

    def drawCircle(self,originImg,PointsCircels,ColorCircle,ColorCenter):
        if PointsCircels is not None:

            # Convert the circle parameters a, b and r to integers.
            PointsCircels = np.uint16(np.around(PointsCircels))

            for pt in PointsCircels[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                # Draw the circumference of the circle.
                cv2.circle(originImg, (a, b), r, ColorCircle, 2)

                # Draw a small circle (of radius 1) to show the center.
                cv2.circle(originImg, (a, b), 1, ColorCenter, 3)


if __name__ =="__main__":
    ex4 = projectIM2020_q4()
    path = os.getcwd() + "\\ex1"
    imagesBW,imagesC = ex4.get_database_images(path,'.JPG')
    for i in range(3):#len(imagesBW)):
        imgbw = imagesBW[i]
        imgc = imagesC[i]
        plt.imshow(imgbw, cmap='gray')
        plt.show()
        # imgbw = imutils.resize(imgbw, width=400)
        # imgc = imutils.resize(imgc, width=400)
        imgbw = cv2.equalizeHist(imgbw)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        imgbw = clahe.apply(imgbw)
        plt.imshow(imgbw, cmap='gray')
        plt.show()

        # for x in range(80,160):
        #     for y in range(40,x//2):
        #         temp = imgc.copy()
        #         # plt.imshow(imgbw, cmap='gray')
        #         # plt.show()
        #         detected_circles = ex4.createCircles(imgbw, cv2.HOUGH_GRADIENT, 1, 20, x, y, 10, 50)
        #         # Draws those circles on the image
        #         if detected_circles is not None and len(detected_circles[0]) > 1:
        #             print(x,y , len(detected_circles[0]))
        #             ex4.drawCircle(temp, detected_circles, (255, 0, 0), (0, 0, 255))
        #             plt.imshow(temp, cmap='gray')
        #             plt.show()
        #         # else:
        #         #     print("not found")

        #
        # # originImg = view
        # # img = ex4.createImage(imgc)
        # plt.imshow(imgc)
        # plt.show()


