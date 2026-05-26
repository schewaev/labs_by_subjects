import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import segmentation_utils
from matplotlib.colors import hsv_to_rgb

image = cv.imread('../images/lab5.png')
image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

image_hsv_norm = image_hsv / 255.0
plt.imshow(hsv_to_rgb(image_hsv_norm))

# Координаты начальных точек на малине
seeds = [(450, 470), (130, 340), (310, 530), (220, 390), (255, 550), (140, 435), (120, 420), (330, 450), (310, 500)]
threshold = 0.15  # Настройте порог для выделения малины

x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))

# Выполняем разрастание областей
segmented_region = segmentation_utils.region_growingHSV(image_hsv_norm, seeds, threshold)

# Применяем морфологические операции закрытия и открытия для удаления шумов
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv.morphologyEx(segmented_region, cv.MORPH_CLOSE, kernel)
mask_cleaned = cv.morphologyEx(mask_cleaned, cv.MORPH_OPEN, kernel)

# Применяем маску
result = cv.bitwise_and(image_rgb, image_rgb, mask=mask_cleaned)

# Отображаем результат
plt.figure(figsize=(15, 20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(image_rgb)
plt.title("Исходное изображение")
plt.subplot(1, 2, 2)
plt.imshow(result)
plt.title("Выделение малины методом разрастания областей")
plt.show()
