# Ex2a
# Roi Madmon - Roimd
# 203485164

import cv2
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt



class Ex2:
    def __init__(self):
        pass

    def addNoise(self, image, percent):
        percent /= 100
        row, col = image.shape
        mean = 0
        var = 500
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss * percent
        return noisy

    def gradient_intensity(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)
        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)
        G = np.hypot(Ix, Iy)
        D = np.arctan2(Iy, Ix)
        return (G, D)

    def round_angle(self, angle):
        angle = np.rad2deg(angle) % 180
        if (0 <= angle < 22.5) or (157 <= angle < 180):
            angle = 0
        elif 22.5 <= angle < 67.5:
            angle = 45
        elif 67.5 <= angle < 112.5:
            angle = 90
        elif 112.5 <= angle < 157.5:
            angle = 135
        return angle

    def suppression(self, img, D):
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        for i in range(M):
            for j in range(N):
                # find neighbour pixels to visit from the gradient directions
                where = self.round_angle(D[i, j])
                try:
                    if where == 0:
                        if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                            Z[i, j] = img[i, j]
                    elif where == 90:
                        if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                            Z[i, j] = img[i, j]
                    elif where == 135:
                        if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                            Z[i, j] = img[i, j]
                    elif where == 45:
                        if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                            Z[i, j] = img[i, j]
                except IndexError as e:
                    """ Todo: Deal with pixels at the image boundaries. """
                    pass
        return Z

    def threshold(self, img, t, T):
        # define gray value of a WEAK and a STRONG pixel
        cf = {
            'WEAK': np.int32(50),
            'STRONG': np.int32(255),
        }
        # get strong pixel indices
        strong_i, strong_j = np.where(img > T)
        # get weak pixel indices
        weak_i, weak_j = np.where((img >= t) & (img <= T))
        # get pixel indices set to be zero
        zero_i, zero_j = np.where(img < t)
        # set values
        img[strong_i, strong_j] = cf.get('STRONG')
        img[weak_i, weak_j] = cf.get('WEAK')
        img[zero_i, zero_j] = np.int32(0)
        return (img, cf.get('WEAK'))

    def tracking(self, img, weak, strong=255):
        M, N = img.shape
        for i in range(M):
            for j in range(N):
                if img[i, j] == weak:
                    # check if one of the neighbours is strong (=255 by default)
                    try:
                        if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong) or (img[i, j + 1] == strong) or (
                                img[i, j - 1] == strong) or (img[i + 1, j + 1] == strong) or (
                                img[i - 1, j - 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    def smoothImage(self, image):
        thmooth = cv2.GaussianBlur(image, (5, 5), 0)
        return thmooth

    def canny(self, image, noisePercrnt,low,high):

        noise = self.addNoise(image, noisePercrnt)
        img1 = self.smoothImage(noise)
        img2, D = self.gradient_intensity(img1)
        img3 = self.suppression(np.copy(img2), D)
        img4, weak = self.threshold(np.copy(img3), low, high)#20,120
        img5 = self.tracking(np.copy(img4), weak)
        return img5



    def setColor(self,originImg, img, R=0, G=0, B=0):
        for i in range(0, img.shape[0]):
            for j in range(0, img.shape[1]):
                if img[i, j] == 255:
                    originImg[i, j, 0] = R
                    originImg[i, j, 1] = G
                    originImg[i, j, 2] = B
        return originImg

    def findLines(self,img):
        height, width = img.shape
        img2 = np.zeros((height, width))
        for i in range(10,height-10):
            for j in range(10,width-10):
                found = True
                if img[i][j] == 255:
                    for k in range(5):
                        if self.getPixel(img,i,j,'d',k) != 255:
                            found = False
                        if self.getPixel(img,i+3,j+3,'d',k) != 255:
                            found = False
                    if found:
                        img2[i][j] = 255
        img2 = self.fixLine(img2)
        return img2

    def getPixel(self,img,i,j,side,moves):
        if side == "l":
            return img[i][j - moves]
        if side == "r":
            return img[i][j + moves]
        if side == "u":
            return img[i - moves][j]
        if side == "d":
            return img[i + moves][j]
        if side == "ld":
            return img[i + moves][j - moves]
        if side == "lu":
            return img[i - moves][j - moves]
        if side == "rd":
            return img[i + moves][j + moves]
        if side == "ru":
            return img[i - moves][j + moves]


    def fixLine(self,img):
        height, width = img.shape
        img2 = np.zeros((height, width))
        for i in range(0, height):
            for j in range(0, width):
                if img[i, j] == 255 and self.getPixel(img,i,j,'d',2) == 255:
                    for k in range(0,height):

                        if img2[k][j] != 255:
                            img2[k][j] = 255

        for i in range(5 , width-10):
            if img2[0][i] == 255 and img2[0][i+1] == 0 and img2[0][i+2] == 255:
                for k in range(height):
                    for u in range(5):
                        if img2[k][i+u] != 255:
                            img2[k][i+u] = 255
            if img2[0][i] == 0 and img2[0][i - 1] == 255 and img2[0][i +1] == 255:
                for k in range(height):
                    for u in range(-2,3):
                        if img2[k][i+u] != 255:
                            img2[k][i+u] = 255

        return img2



if __name__ == "__main__":
    img_path = os.getcwd() + '/sudoku-original.jpg'
    img = cv2.imread(img_path, 0)
    originImg = cv2.imread(img_path, 3)
    ex2a = Ex2()

    imgCanny = ex2a.canny(img, 30, 40, 120) #50/20/50 full sudoku // frame 40,50,100
    lines = ex2a.findLines(imgCanny)

    red = ex2a.setColor(originImg,lines,255,0,0)
    plt.imshow(red, cmap='gray')
    plt.title("Original Image ")
    plt.xticks([])
    plt.yticks([])
    plt.show()

