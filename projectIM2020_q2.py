import cv2
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image


imageDatabase = {}

class projectIM2020_q2:



    def get_database_image(self):
        # dir = os.getcwd()+"\\projectIM2020-master" #mercury pc
        dir = os.getcwd() #roi pc
        a = []
        directory = "{}\{}".format(dir, 'ex2')
        if os.path.exists(directory):
            folders = list(os.walk(directory))
            os.chdir(directory)
            return  folders

    def find_circle(self, img):
        edges = cv2.Canny(img, 150, 100, apertureSize=3)
        # plt.imshow(edges), plt.xticks([]), plt.yticks([])
        # plt.show()
        circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=40, minRadius=6, maxRadius=40)
        circles = np.uint16(np.around(circles))
        # return  circles
        for i in circles[0, :]:

            # draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (255, 0, 0), 3)

        return img

    def createImage(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
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
                if (pt[2] == 31 and len(PointsCircels[0]) == 1 )  or (pt[2] == 39 and len(PointsCircels[0]) == 6) or (pt[2] == 34 and len(PointsCircels[0]) == 3 and pt[1] < 300) or (pt[2] == 23 and len(PointsCircels[0]) == 5) or (pt[2] == 38 and len(PointsCircels[0]) == 4):
                # if  20 < pt[2]:
                    print(pt[2])
                    a, b, r = pt[0], pt[1], pt[2]

                    # Draw the circumference of the circle.
                    cv2.circle(originImg, (a, b), r, ColorCircle, 2)

                    # Draw a small circle (of radius 1) to show the center.
                    cv2.circle(originImg, (a, b), 1, ColorCenter,3)


if __name__ =="__main__":
    ex2 = projectIM2020_q2()
    folders = ex2.get_database_image()

    for view in folders[0][2]:
        # view = folders[0][2][7]#(2,31.4), (4,39.4), (7,33.8), (8,22.7), (9,38.3)
        #Saves the original image
        originImg = cv2.imread(view, cv2.IMREAD_COLOR)
        img = ex2.createImage(view)
        detected_circles = ex2.createCircles(img, cv2.HOUGH_GRADIENT, 1, 20, 82, 40, 10, 40)
        print(detected_circles)
        ex2.drawCircle(originImg, detected_circles, (255, 0, 0), (0, 0, 255))
        plt.imshow(originImg)
        plt.show()
    #Creates a workable image for circle detection

    # for i in range(10,41):
    #     for k in range(i,41):
    #         img = ex2.createImage(view)
    #         img = img.copy()
    #         #Detects circles in an image
    #         # detected_circles = ex2.createCircles(img,cv2.HOUGH_GRADIENT,1,20,160,80,5,40)
    #
    #         detected_circles = ex2.createCircles(img,cv2.HOUGH_GRADIENT,1,20,82,40,i,k)
    #         if detected_circles is not None and len(detected_circles[0])>0:
    #             print(i,k)
    #             print(detected_circles)
    #
    #             #Draws those circles on the image
    #             ex2.drawCircle(originImg,detected_circles,(255,0,0),(0,0,255))
    #             plt.imshow(originImg)
    #             plt.show()




        # detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1=160, param2=80, minRadius=5,maxRadius=40)








        # img = cv2.imread(view)
        # # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # img1 = cv2.medianBlur(img, 5)
        # img_edge = ex2.find_edges(img)
        #
        # img_circles = ex2.find_circle(img_edge)
        # # # imgs = img + img_circles
        # # # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # # # imgs = gray + iom
        # plt.imshow(img_circles, cmap='gray'), plt.xticks([]), plt.yticks([])
        # plt.show()
