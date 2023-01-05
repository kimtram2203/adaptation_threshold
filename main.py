import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt

imageData = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg', 'image7.jpg',
             'image8.jpg', 'image9.jpg', 'image10.jpg']
imageName = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']
thresh_mean = []
thresh_gaussian = []
origin = []


def image_partitioning(file_name):
    img = cv.imread('./assets/' + file_name, 0)
    origin.append(img)
    img2 = cv.medianBlur(img, 5)
    th2 = cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img2, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
    thresh_mean.append(th2)
    thresh_gaussian.append(th3)


def show_images(dataset: []):
    for i in range(len(dataset)):
        plt.subplot(2, len(dataset) // 2, i + 1), plt.imshow(dataset[i], 'gray')
        plt.title(imageName[i], fontdict={'fontsize': 12})
        plt.subplots_adjust(hspace=0.25)
        plt.xticks([]), plt.yticks([])
    plt.show()


def run_data():
    for i in range(len(imageData)):
        image_partitioning(imageData[i])
    show_images(origin)
    show_images(thresh_mean)
    show_images(thresh_gaussian)


run_data()
