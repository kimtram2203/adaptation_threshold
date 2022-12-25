import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

imageData = ['anh1.jpg', 'anh2.jpg', 'anh3.png', 'anh4.jpg', 'anh5.png', 'anh6.jpg']
images = []
titles = ['Mean Thresholding', 'Gaussian Thresholding']


def image_partitioning(file_name):
    img = cv.imread('./assets/' + file_name, 0)
    img = cv.medianBlur(img, 5)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    images.append(th2)
    images.append(th3)


def run_data():
    for i in range(len(imageData)):
        image_partitioning(imageData[i])
    print(len(images))
    for j in range(len(images)):
        plt.subplot(len(images) // 2, 2, j + 1), plt.imshow(images[j], 'gray')
        plt.xticks([]), plt.yticks([])

                
run_data()
