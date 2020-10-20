from logging import FATAL
import easyocr
import os.path as osp
from PIL import Image
import numpy as np
import cv2
from numpy.core.records import ndarray

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,3)
 
#thresholding
def thresholding(image, ths, inv=False):
    if inv:
        w = cv2.THRESH_BINARY_INV
    else:
        w = cv2.THRESH_BINARY
    if ths==0:
        working = w + cv2.THRESH_OTSU
    else:
        working = w
    img =  cv2.threshold(image, ths, 255, working)
    # img =  cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)
    print(img[0])
    return img[1]

def adaptiveThresholding(image, size=51, C=0, inv=1):
    if inv:
        working = cv2.THRESH_BINARY_INV
    else:
        working = cv2.THRESH_BINARY
    img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, working, size,C)
    return img

#dilation
def dilate(image, size):
    kernel = np.ones(size,np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image, size):
    kernel = np.ones(size,np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

#template matching
def match_template(image, template):
    return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED) 

def remove_dot(img, num):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    min_size = num  

    #your answer image
    img2 = np.zeros((output.shape))
    #for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] >= min_size:
            img2[output == i + 1] = 255

    return img2.astype(np.uint8)

# import 
reader = easyocr.Reader(['ko','en']) # need to run only once to load model into memory
path = 'E:\\강서구 pdf 이미지\\강서구 종합보고서_1214_-872.jpg'
img = np.array(Image.open(path))
# cv2.imshow('img', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img = get_grayscale(img)
img = cv2.resize(img, dsize=(0,0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
imgs = []

sharpening_1 = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
sharpening_2 = np.array([[-1, -1, -1, -1, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, 2, 9, 2, -1],
                         [-1, 2, 2, 2, -1],
                         [-1, -1, -1, -1, -1]]) / 9.0
sharpening_3 = np.array([[-1, -1, -1, -1, -1, -1, -1],
                         [-1,  0,  0,  10,  0,  0, -1],
                         [-1,  0,  10, 20, 10,  0, -1],
                         [-1,  10, 20, 79, 20, 10, -1],
                         [-1,  0,  10, 20, 10,  0, -1],
                         [-1,  0,  0,  10,  0,  0, -1],
                         [-1, -1, -1, -1, -1, -1, -1]]) -2
sharpening_4 = np.array([[-1, -1, -1, -1, -1, -1, -1],
                         [-1,  0,  0,  0,  0,  0, -1],
                         [-1,  0, 10, 10, 10,  0, -1],
                         [-1,  0, 10, 60, 10,  0, -1],
                         [-1,  0, 10, 10, 10,  0, -1],
                         [-1,  0,  0,  0,  0,  0, -1],
                         [-1, -1, -1, -1, -1, -1, -1]]) -2
sharpening_5 = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1],
                         [-1,  0,  0,  0,  0,  0,  0,  0, -1],
                         [-1,  0,  1,  1,  1,  1,  1,  0, -1],
                         [-1,  0,  1,  5,  5,  5,  1,  0, -1],
                         [-1,  0,  1,  5, 57,  5,  1,  0, -1],
                         [-1,  0,  1,  5,  5,  5,  1,  0, -1],
                         [-1,  0,  1,  1,  1,  1,  1,  0, -1],
                         [-1,  0,  0,  0,  0,  0,  0,  0, -1],
                         [-1, -1, -1, -1, -1, -1, -1, -1, -1]])
sharpening_8 = np.array([[-1, -1, -1, -1, -1, -1, -1], #best
                         [-1,  0,  0,  10,  0,  0, -1],
                         [-1,  0,  10,  1,  10,  0, -1],
                         [-1,  10,  1,  39, 1,  10, -1],
                         [-1,  0,  10,  1,  10,  0, -1],
                         [-1,  0,  0,  10,  0,  0, -1],
                         [-1, -1, -1, -1, -1, -1, -1]]) -1
sharpening = sharpening_5
print(sharpening.sum())
sharpening = (sharpening /sharpening.sum()) if sharpening.sum()!=0 else sharpening
print(sharpening)
# img = cv2.fastNlMeansDenoisingColored(img, 3, 3, 7, 14)
h, w = img.shape
img = cv2.vconcat([cv2.equalizeHist(img[:h//3, :w//3]), cv2.equalizeHist(img[:h//3, (-w//3+1):])])
img = cv2.filter2D(img, -1, sharpening)
# img = cv2.medianBlur(img, 3)
img = cv2.GaussianBlur(img, (3,3), 0)
# img = remove_noise(img)
# img = adaptiveThresholding(img, 15, -15, inv=True)
img = adaptiveThresholding(img, 13, -30, inv=False)
imgs.append(img)
img = dilate(img, (3,3))
img = erode(img, (3,3))
imgs.append(img)
img = erode(img, (2,2))
imgs.append(img)
img = remove_dot(img, 10)
imgs.append(img)
img = dilate(img, (3,3))
imgs.append(img)
# img = cv2.GaussianBlur(img, (3,3), 0)
# img = cv2.blur(img, (3,3))
# img = cv2.medianBlur(img, 3)
# imgs.append(img)
# img = thresholding(img, 245)
# img = thresholding(img, 114, inv=False)
# img = thresholding(img, 245, inv=True)
# imgs.append(img)
# img = dilate(img, (4,4))
# img = erode(img, (4,4))
# img1 = erode(img, (1,3))
# img2 = dilate(img1, (1,3))
# img = cv2.medianBlur(img, 3)
# img = thresholding(img)
# img = cv2.medianBlur(img, 3)
# img = thresholding(img)
# img = cv2.medianBlur(img, 5)
# img = thresholding(img)
# img1 = erode(img, 2)
# img2 = dilate(img1, 3)
# imgs.append(img)
# img = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
# imgs.append(img)
img = cv2.hconcat(imgs)
# img = canny(img)
# se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se1)
# se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# img = cv2.morphologyEx(img, cv2.MORPH_OPEN, se2)

# for ths in np.arange(0.1,1,0.1):
#     ths = np.round(ths, 1)
#     for con in np.arange(0.1,1,0.1):
#         con = np.round(con, 1)
#         # result = reader.readtext(img, detail=0, contrast_ths=ths, adjust_contrast=con, allowlist='0123456789.m:-하관정주행박진형()HP상/역로김원경', text_threshold=0.5, mag_ratio=1)
#         result = reader.readtext(img, detail=0, contrast_ths=ths, adjust_contrast=con,allowlist='0123456789.m:-하관정주행HP상/역로', text_threshold=0.5, width_ths=0.7)
#         print(ths, con, result)
result = reader.readtext(img, detail=0, allowlist='0123456789.m:-하관정주행HP상/역로', text_threshold=0.7, width_ths=0.7)
# result = reader.readtext(img, detail=0, allowlist='0123456789.m:-하관정주행박진형()HP상/역로김원경', text_threshold=0.5, width_ths=0.7)
print(result)
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

path = 'E:/capture/16-00030 (8-27-2020 1-30-14 PM)/16-00030 0316.jpg'
img = np.array(Image.open(path))
result = reader.readtext(img, detail=0, allowlist='0123456789.m:-하관정주행박진형()HP상/역로김원경', text_threshold=0.5, width_ths=0.7)
# result = reader.readtext(img, detail=0)
print(result)