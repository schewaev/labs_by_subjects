# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:02:19 2026

@author: schewaev
"""

import sys
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
sys.path.append('../')
from utility import util

image1 = cv.imread('../images/lab3.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

shape = rgb_image1.shape
src_points = np.float32([
    [0, 255],
    [130, 0],
    [410, 255],
    [542, 0] 
])

dst_points = np.float32([
    [0, 0],
    [shape[0], 0],
    [0, shape[1]],
    [shape[0], shape[1]]
])

def rotation(img):
    (h, w) = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    rotation_matrix = cv.getRotationMatrix2D(center, 90, 0.5)
    rotated = cv.warpAffine(img, rotation_matrix, (w, h))
    return rotated

M = cv.getPerspectiveTransform(src_points, dst_points)
warped_image = cv.warpPerspective(rgb_image1, M, (shape[0], shape[1]))
warped_image = rotation(warped_image)
warped_image = warped_image[210:330, :]

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(17, 4))

plt.subplot(gs[0])
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.imshow(rgb_image1)

plt.subplot(gs[1])
plt.title('Выпрямленное изображение')
plt.xticks([]), plt.yticks([])
plt.imshow(warped_image)

contrast = 1.2
brightness = 50
bright_contrast_image = cv.convertScaleAbs(warped_image, alpha=contrast, beta=brightness)

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(17, 4))

plt.subplot(gs[0])
plt.title('Выпрямленное изображение')
plt.xticks([]), plt.yticks([])
plt.imshow(warped_image)

plt.subplot(gs[1])
plt.title('Изображение с повышенной яркостью')
plt.xticks([]), plt.yticks([])
plt.imshow(bright_contrast_image)