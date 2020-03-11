import glob
import cv2
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image


imageDatabase = {}

class projectIM2020_q2:

    def get_database_images(self,path,end):
        image_list =[]
        for filename in glob.glob(path + '/*'+end):
            im = cv2.imread(filename)
            image_list.append(im)
        return  image_list



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
    ex2 = projectIM2020_q2()
    path = os.getcwd() + "\\ex2"
    images = ex2.get_database_images(path,'.png')
    for view in images:
        originImg = view
        img = ex2.createImage(view)
        detected_circles = ex2.createCircles(img, cv2.HOUGH_GRADIENT, 1, 20, 82, 40, 20, 35)
        # Draws those circles on the image
        if detected_circles is not None and len(detected_circles[0]) == 1:
            ex2.drawCircle(originImg, detected_circles, (255, 0, 0), (0, 0, 255))
        plt.imshow(originImg)
        plt.show()

