import os
from scipy import ndimage
from matplotlib import pyplot as plt
from PIL import Image
import cv2
import numpy as np
import imutils
import glob
import argparse

# from pyimagesearch.panorama import Stitcher

class projectIM2020_q3:


    def __init__(self):
        # determine if we are using OpenCV v3.X
        self.isv3 = imutils.is_cv3(or_better=True)

    def stitch(self, images, ratio=0.75, reprojThresh=4.0,
               showMatches=False):
        # unpack the images, then detect keypoints and extract
        # local invariant descriptors from them
        (imageB, imageA) = images
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # match features between the two images
        M = self.matchKeypoints(kpsA, kpsB,
                                featuresA, featuresB, ratio, reprojThresh)
        # if the match is None, then there aren't enough matched
        # keypoints to create a panorama

        if M is None:
            return (None,None)
        # otherwise, apply a perspective warp to stitch the images
        # together
        (matches, H, status) = M
        if len(matches) <= 31:
            return (None, None)

        result = cv2.warpPerspective(imageA, H,(imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # check to see if the keypoint matches should be visualized
        if showMatches:
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,status)
            # return a tuple of the stitched image and the
            # visualization
            return (result, vis)
        # return the stitched image
        return (result, None)

    def detectAndDescribe(self, image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.Canny(gray, 50, 200, apertureSize=3)
        # check to see if we are using OpenCV 3.X
        if self.isv3:
            # detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()

            (kps, features) = descriptor.detectAndCompute(image, None)
            # otherwise, we are using OpenCV 2.4.X
        else:
            # detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)
            # extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)
        # convert the keypoints from KeyPoint objects to NumPy
        # arrays
        kps = np.float32([kp.pt for kp in kps])
        # return a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
                       ratio, reprojThresh):
        # compute the raw matches and initialize the list of actual
        # matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []
        # loop over the raw matches
        for m in rawMatches:
            # ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))
        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])
            # compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                             reprojThresh)
            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)
        # otherwise, no homograpy could be computed
        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # initialize the output visualization image
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB
        return vis



    def get_database_images(self,path,end):
        image_list =[]
        for filename in glob.glob(path + '/*'+end):
            im = cv2.imread(filename)
            image_list.append(im)
        return  image_list


    def addBlackToImage(self,image1,image2):

        if image1.shape>image2.shape:
            tempImage = np.zeros(image1.shape,dtype=np.uint8)
            for i in range(0,tempImage.shape[0]):
                for j in range(0, tempImage.shape[1]):
                    try:
                        if image2[i,j] is not None:
                            tempImage[i,j] = image2[i,j]
                    except:
                        pass
        else:
            tempImage = np.zeros(image2.shape, dtype=np.uint8)
            for i in range(0, tempImage.shape[0]):
                for j in range(0, tempImage.shape[1]):
                    try:
                        if image1[i, j] is not None:
                            tempImage[i, j] = image1[i, j]
                    except:
                        pass

        return tempImage










if __name__ =="__main__":

    ex3 = projectIM2020_q3()
    path1 = os.getcwd() + "\\ex2" #databases
    path2 = os.getcwd() + "\\ex3" #half images
    images1 = ex3.get_database_images(path1,'.png')
    images2 = ex3.get_database_images(path2,'.png')
    for img1 in images1:
        for img2 in images2:
            imageA = img1
            imageB = img2
            imageA = imutils.resize(imageA, width=400)
            imageB = imutils.resize(imageB, width=400)
            newImage = ex3.addBlackToImage(imageA,imageB)

            if imageA.shape == newImage.shape:
                imageB = newImage

            else:
                imageA = imageB
                imageB = newImage

            # stitch the images together to create a panorama
            stitcher =  projectIM2020_q3()
            (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
            if vis is not None:
                plt.imshow(vis)
                plt.xticks([])
                plt.yticks([])
                plt.show()




