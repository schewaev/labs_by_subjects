# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:02:41 2026

@author: schewaev
"""

import numpy as np
import cv2 as cv

import sys
sys.path.append('../')
from utility import util
import matplotlib.pyplot as plt

"""##### 2.1.2.3.2 Эквализация изображения

Другой популярной процедурой адаптивной гистограммной обработки изображения является *эквализация* изображения. Эквализация также отображает реальный яркостный диапазон данного изображения (от минимального до максимального значения интенсивности) на диапазон [0, 255]. При этом обеспечивается "выравнивание" числа пикселов изображения, имеющих различные значения яркости. Математическая форма данного преобразования имеет вид
$$
{\rm LUT}[i] = 255 \cdot  \frac{\sum\limits_{j=1}^i \textrm{Hist}[j]} {\sum\limits_{j=1}^{255} \textrm{Hist}[j]},
$$
В тех случаях, когда в диапазоне [Imin,Imax] более или менее равномерно присутствуют все градации, визуальный эффект от эквализации трудно отличить от эффекта нормализации. Однако в случае, когда значительная часть градаций яркости отсутствует, эквализация позволяет более равномерно использовать диапазон [0,255] для более контрастного отображения присутствующих на изображении градаций. Визуально это выглядит как "проявление" большего количества ранее не заметных на изображении деталей и контуров.

**Задание: реализуйте данный вид эквализации самостоятельно**
"""

histSize = [256]

image1 = cv.imread('../images/winter_cat.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

# Гистограмма изображения
hist, bins = np.histogram(gray_image1.flatten(), 256, [0, 256])

# Кумулятивная сумма гистограммы
cdf = hist.cumsum()

# Нормирование кумулятивной функции распределения
cdf_normalized = cdf * 255 / cdf[-1]

# Линейная интерполяция по куммулятивной сумме для получения нового изображения
result_image = np.interp(gray_image1.flatten(), bins[:-1], cdf_normalized).reshape(gray_image1.shape)

# Визуализация до и после эквализации
gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.subplot(gs[1])
plt.imshow(result_image, cmap='gray')
plt.subplot(gs[2])
plt.hist(gray_image1.reshape(-1), 256, [0, 256])
plt.subplot(gs[3])
plt.hist(result_image.reshape(-1), 256, [0, 256])
plt.show()