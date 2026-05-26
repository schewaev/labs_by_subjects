# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 17:50:02 2026

@author: schewaev
"""

import sys
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

plt.rcParams["figure.figsize"] = [6, 4]

image1 = cv.imread('../images/lab4.png')
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
hsv_image1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

range = [0, 256]

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.hist(gray_image1.reshape(-1), 256, range)
plt.show()

# Определение порогов для полос
lower_dark = 122
upper_dark = 131
lower_bright = 230
upper_bright = 240

# Создание маски для темных полос
mask_dark = cv.inRange(gray_image1, lower_dark, upper_dark)
# Создание маски для светлых полос
mask_bright = cv.inRange(gray_image1, lower_bright, upper_bright)

# Объединение масок
mask = cv.bitwise_or(mask_dark, mask_bright)

# Морфологические операции для расширения маски
kernel = np.ones((2, 2), np.uint8)
mask_dilated = cv.dilate(mask, kernel, iterations=1)

# Инвертирование маски для использования в inpainting
mask_inv = cv.bitwise_not(mask_dilated)

# Восстановление изображения с использованием inpaint
restored_image = cv.inpaint(gray_image1, mask_dilated, inpaintRadius=2, flags=cv.INPAINT_TELEA)

plt.figure(figsize=(15, 8))
gs = plt.GridSpec(1, 3)

titles = ['Исходное изображение', 'Маска полос', 'Восстановленное изображение']
images = [gray_image1, mask_dilated, restored_image]

for i in np.arange(len(images)):
    plt.subplot(gs[i])
    plt.xticks([]), plt.yticks([])
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')

plt.show()