import cv2 as cv
from matplotlib import pyplot as plt

## tập ảnh
imageData = ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', 'image6.jpg', 'image7.jpg',
             'image8.jpg', 'image9.jpg', 'image10.jpg']

## tập tên ảnh
imageName = ['image1', 'image2', 'image3', 'image4', 'image5', 'image6', 'image7', 'image8', 'image9', 'image10']

## output threshold mean
thresh_mean = []

## output threshold gaussion
thresh_gaussian = []

## output origin
origin = []


## Hàm phân ngưỡng một ảnh
def image_partitioning(file_name):
    ## đọc ảnh gốc, sau đó add vào tập origin
    image_origin = cv.imread('./assets/' + file_name, 0)
    origin.append(image_origin)

    ## Chuyển đổi ảnh gốc thành ảnh xám
    image_grey = cv.medianBlur(image_origin, 5)

    ## chuyển đổi ảnh xám theo phương pháp Adaptive Threshold Mean, đưa ảnh này vào tập thresh_mean
    image_thresh_mean = cv.adaptiveThreshold(image_grey, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 2)
    thresh_mean.append(image_thresh_mean)

    ## chuyển đổi ảnh xám theo phương pháp Adaptive Threshold Gaussion, đưa ảnh này vào tập thresh_gaussion
    image_thresh_gaussion = cv.adaptiveThreshold(image_grey, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11,
                                                 2)
    thresh_gaussian.append(image_thresh_gaussion)


## hàm show tập ảnh output bằng thư viện matplotlib
## dataset (tập output): origin, thresh_mean, thresh_gaussian
## title (title của tập ảnh): Origin Image, Adaptive Thresholding Mean, Adaptive Thresholding Gaussion
def show_images(dataset: [], title):
    ## lặp qua tập ảnh output
    for i in range(len(dataset)):
        ## Sắp xếp thứ tu các ảnh
        plt.subplot(2, len(dataset) // 2, i + 1), plt.imshow(dataset[i], 'gray')
        ## thêm title cho ảnh
        plt.title(imageName[i], fontdict={'fontsize': 12})
        ## điều chỉnh khoảng cách giữa các ảnh (chiều cao)
        plt.subplots_adjust(hspace=0.25)

    ## thêm title hiển thị cho tập ảnh
    plt.suptitle(title)
    ## show màn hình tập ảnh
    plt.show()


## lặp qua tập ảnh imageData(data_set)
for i in range(len(imageData)):
    ## gọi hàm phân ngưỡng ảnh
    image_partitioning(imageData[i])

## show tập ảnh gốc
show_images(origin, title='Origin Image')
## show tập ảnh Adaptive Thresholding Mean
show_images(thresh_mean, title='Adaptive Thresholding Mean')
## show tập ảnh Adaptive Thresholding Gaussion
show_images(thresh_gaussian, title='Adaptive Thresholding Gaussion')
