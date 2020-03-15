
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
            imbw = cv2.imread(filename)
            imc = cv2.imread(filename)
            image_listBW.append(imbw)
            image_listC.append(imc)
        return  image_listBW,image_listC





if __name__ =="__main__":
    ex4 = projectIM2020_q4()
    path = os.getcwd() + "\\ex1"
    imagesBW,imagesC = ex4.get_database_images(path,'.JPG')
    for i in range(3):#len(imagesBW)):
        # gray = imagesBW[i]
        # imgc = imagesC[i]

        img = imagesBW[i]

        img = imutils.resize(img, width=400)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)


        # gray = cv2.fastNlMeansDenoisingColored(gray, None, 10, 10, 7, 21)

        edges = cv2.Canny(gray, 130, 200)
        plt.imshow(edges,cmap="gray")
        plt.show()
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, maxLineGap=250)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 1)

        plt.imshow(img)
        plt.show()


