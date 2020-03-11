import cv2
import numpy as np
import os
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
from ex2a import Ex2



class projectIM2020_q1:
    dir = os.getcwd()
    a=[]
    directory = "{}\{}".format(dir, 'ex1')
    if os.path.exists(directory):
        folders = list(os.walk(directory))
        os.chdir(directory)
        for view in folders[0][2]:
            img = cv2.imread(view)
            row, col,m = img.shape
            newimg5 = img[650:row,350    :col]#5.jpg
            newimg14 = img[0:row-500, 460:col]  # 14.jpg
            newimg15 = img[560:row, 0:col-400]  # 15.jpg
            gray = cv2.cvtColor(newimg5, cv2.COLOR_BGR2GRAY)

            canny = cv2.Canny(gray,50,150,apertureSize = 3)
            corners = cv2.goodFeaturesToTrack(canny, 25, 0.01, 10)
            corners = np.int0(corners)

            for i in corners:
                x, y = i.ravel()
                cv2.circle(newimg, (x, y), 3, (255, 0, 0), 20)

            plt.imshow(newimg)
            plt.show()



            # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            #
            # lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            # for rho, theta in lines[0]:
            #     a = np.cos(theta)
            #     b = np.sin(theta)
            #     x0 = a * rho
            #     y0 = b * rho
            #     x1 = int(x0 + 1000 * (-b))
            #     y1 = int(y0 + 1000 * (a))
            #     x2 = int(x0 - 1000 * (-b))
            #     y2 = int(y0 - 1000 * (a))
            #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 40)
            # print(lines)
            # plt.imshow(img)  # , plt.xticks([]), plt.yticks([])
            # plt.show()




    #
    # for i in a:
    #     plt.imshow(i)  # , plt.xticks([]), plt.yticks([])
    #     plt.show()


if __name__ =="__main__":
    q1 = projectIM2020_q1()
