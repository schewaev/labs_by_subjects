# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 07:12:55 2026

@author: schewaev
"""

import cv2  
import numpy as np
import matplotlib.pyplot as plt

# 1 Поиск шаблона на изображении

# Загружаем изображение
rgb_img = cv2.imread('lab7.png') 
plt.figure()
plt.imshow(cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB))

# Преобразуем изображение в оттенки серого 
gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
  
# Загружаем шаблон  
template = cv2.imread('lab7_template.png')
plt.figure()
plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))

# 2 Для снижения влияния яркости изображения или шаблона на результат, 
# рекомендуется делать поиск не по исходным изображениям, а по выделенным
# из них границам

# Границы вычисляем на изображении в оттенках серого
imgG = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)

# Для выделения границ используем фильтр Собеля в двух направлениях
x = cv2.Sobel(imgG,cv2.CV_16S,1,0)  
y = cv2.Sobel(imgG,cv2.CV_16S,0,1)  

# берем модуль от результата применения фильтра Собеля
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)  
  
# объединяем "вертикальные" и "горизонтальные" границы в одно изображение
dstI = cv2.addWeighted(absX,0.5,absY,0.5,0)

plt.subplot(2,1,1),plt.imshow(imgG,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,1,2),plt.imshow(dstI,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])


# Такие же преобразования делаем и с шаблоном, но изменяем яркость шаблона
tmpG = 2*cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
x = cv2.Sobel(tmpG,cv2.CV_16S,1,0)  
y = cv2.Sobel(tmpG,cv2.CV_16S,0,1)  
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)  
dstT = cv2.addWeighted(absX,0.5,absY,0.5,0)

plt.subplot(3,1,1),plt.imshow(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY),cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,2),plt.imshow(tmpG,cmap = 'gray')
plt.title('Changed'), plt.xticks([]), plt.yticks([])
plt.subplot(3,1,3),plt.imshow(dstT,cmap = 'gray')
plt.title('Sobel'), plt.xticks([]), plt.yticks([])

# Считаем размеры шаблона
w, h = tmpG.shape
 
# Вычисляем метрику схожести  
res = cv2.matchTemplate(dstI,dstT,cv2.TM_CCOEFF_NORMED)  
plt.figure()
plt.imshow(res, cmap='jet')
plt.colorbar()

#  Отбираем максимумы и строим результат на графике
threshold = 0.35
loc = np.where(res >= threshold)  
plot_img = rgb_img.copy()
for pt in zip(*loc[::-1]):
    cv2.rectangle(plot_img, pt,(pt[0] + w, pt[1] + h),(0,255,255), 8)  

plt.figure()
plt.imshow(cv2.cvtColor(plot_img, cv2.COLOR_BGR2RGB))
