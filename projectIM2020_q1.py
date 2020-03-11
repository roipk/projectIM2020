import glob
import cv2
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image


imageDatabase = {}

class projectIM2020_q1:

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
        plt.imshow(gray_blurred,cmap='gray')
        plt.show()
        return gray_blurred

    def canny(self,img):
        canny =  cv2.Canny(img,50,150)
        plt.imshow(canny,cmap='gray')
        plt.show()
        return  canny

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













    def MS(self, img):
        f = np.fft.fft2(img)  # transform image with fourier
        fshift = np.fft.fftshift(f)
        # magniude_spctrume = 20 * np.log(np.abs(fshift))
        return fshift

    def mask(self, img, mask):
        height, width = img.shape
        img = np.zeros((height, width))
        if mask == 1:
            img[0:height, 0:width] = 1
            img[:height, width // 2 - 5:width // 2 + 5] = 0  # (B, G, R)
        if mask == 2:
            img[:height, :width] = 1
            img[height // 2 - 5:height // 2 + 5, 0:width] = 0  # (B, G, R)
        if mask == 3:
            img[:height, width // 2 - 5:width // 2 + 5] = 1  # (B, G, R)
        if mask == 4:
            img[height // 2 - 5:height // 2 + 5, 0:width] = 1  # (B, G, R)
        if mask == 5:
            img[height // 2 - 5:height // 2 + 5, width // 2 - 5:width // 2 + 5] = 1  # (B, G, R)
        if mask == 6:
            img[:height, :width] = 1
            img[height // 2 :height // 2 + 1, width // 2 :width // 2 +1] = 0  # (B, G, R)
        if mask == 7:
            img[:height, :width] = 1
            img[:height, width // 2 - 5:width // 2 + 5] = 0  # (B, G, R)
            img[height // 2 - 5:height // 2 + 5, 0:width] = 0  # (B, G, R)
        if mask == 8:
            img[:height, width // 2 - 5:width // 2 + 5] = 1  # (B, G, R)
            img[height // 2 - 5:height // 2 + 5, 0:width] = 1  # (B, G, R)

        return img

    def refersMS(self, filter):
        f = np.fft.ifft2(img)  # transform image with fourier
        # fshift = np.fft.fftshift(f)
        # magniude_spctrume = 20 * np.log(np.abs(f))
        return f

    def findWhit(self, img):
        height, width = img.shape
        newimg = np.zeros((height, width, 3))
        newimg[0:height, 0:width, 0] = 255
        newimg[0:height, 0:width, 1] = 255
        newimg[0:height, 0:width, 2] = 255

        for i in range(height):
            for j in range(width):
                if 70 < img[i][j] < 256:
                    # print(img[i][j])
                    newimg[i, j, 1] = 0
                    newimg[i, j, 2] = 0
        return newimg

    def addPic(self, img1, img2):
        for i in range(0, img1.shape[0]):
            for j in range(0, img1.shape[1]):
                if img2[i][j][0] == 255 and img2[i][j][1] == 0 and img2[i][j][2] == 0:
                    img1[i, j, 0] = 255
                    img1[i, j, 1] = 0
                    img1[i, j, 2] = 0
        return img1


if __name__ =="__main__":
    ex1 = projectIM2020_q1()
    path = os.getcwd() + "\\ex1"
    images = ex1.get_database_images(path,'.jpg')
    for view in images:
        img = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 23)
        kernel = np.ones((3, 3), np.uint8)
        canny = cv2.Canny(img,120,200)
        img = cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel)
        test = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        plt.imshow(test, cmap='gray')  # , plt.xticks([]), plt.yticks([])
        plt.show()



        # img = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY)
        # # edge = cv2.Canny(img,120,200)
        img = cv2.GaussianBlur(test,(7,7),0)
        img3d = view
        imgMS = ex1.MS(img)
        mask = ex1.mask(img, 7)
        ft = imgMS * mask
        revers = np.fft.ifft2(ft)
        findPix = ex1.findWhit(revers)
        i = ex1.addPic(img3d, findPix)
        plt.imshow(np.abs(i), cmap='gray')  # , plt.xticks([]), plt.yticks([])
        plt.show()




        # originImg = view
        # img = ex1.createImage(view)
        # img = ex1.canny(img)
        # detected_circles = ex1.createCircles(img, cv2.HOUGH_GRADIENT, 1, 20, 80, 42, 100, 200)
        # # Draws those circles on the image
        # # if detected_circles is not None and len(detected_circles[0]) == 1:
        # ex1.drawCircle(originImg, detected_circles, (255, 0, 0), (0, 0, 255))
        # plt.imshow(originImg)
        # plt.show()

