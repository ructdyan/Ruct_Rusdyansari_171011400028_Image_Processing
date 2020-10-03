# This is the final project for ENGI 7854 - Image Processing
# written by Matthew Hiscock (201535705) and Jillian Breau (201624079).
# The goal of this project is to get an image of text ready 
# for optical character recognition to be used in a math equation solver.

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def pre_process(img_file):

    original_img = cv.imread(img_file)

    gray_img = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)

    gaussian_img = cv.GaussianBlur(gray_img, (3, 3), 0)

    adaptive_thresh_img = cv.adaptiveThreshold(gaussian_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 51, 11)

    averaged_img = cv.blur(adaptive_thresh_img, (5, 5))

    global_thresh_img = cv.threshold(averaged_img, 170, 255, cv.THRESH_BINARY)[1]

    inv_img = cv.bitwise_not(global_thresh_img)

    skew_corrected_img = skew_correct(inv_img)

    closed_img = cv.morphologyEx(skew_corrected_img, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    images = [original_img,        gray_img,           gaussian_img,
              adaptive_thresh_img, averaged_img,       global_thresh_img,
              inv_img,             skew_corrected_img, closed_img]
    img_titles = ["Original",           "Grayscale",       "Smooth with Gauss",
                  "Adaptive Threshold", "Smooth with Avg", "Global Threshold",
                  "Invert",             "Correct Skew",    "Close"] 

    fig_9_steps = plt.figure(figsize=(12, 9))
    fig_9_steps.suptitle(img_file + " Processing Steps")
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(img_titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    fig_output_image = plt.figure(figsize=(12, 9))
    plt.imshow(closed_img, 'gray')
    plt.title(img_file + " Final Output Image")
    plt.xticks([]),plt.yticks([])
    plt.show()


def skew_correct(skewed_img):
    coords = np.column_stack(np.where(skewed_img > 0))
    angle = cv.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    height, width = skewed_img.shape[:2]
    center = (width // 2, height // 2)

    matrix = cv.getRotationMatrix2D(center, angle, 1.0)
    unskewed_img = cv.warpAffine(skewed_img, matrix, (width, height), flags = cv.INTER_CUBIC, borderMode = cv.BORDER_REPLICATE)
    return unskewed_img


def main():
    
    img_file_lst = ["equation0.jpg", "equation1.jpg", "equation2.jpg", "equation3.jpg"]

    for img_file in img_file_lst:
        pre_process(img_file)


if __name__ == "__main__":
    main()