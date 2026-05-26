import numpy as np
import cv2 as cv

import sys
sys.path.append('../')
from utility import util
import matplotlib.pyplot as plt

"""## 1 Бинаризация

В обработке изображений часто используется процедура пороговой бинаризации -- разбиения изображения на две области, одна из которых содержит все пиксели со значением ниже некоторого порога, а другая содержи все пиксели со значением выше этого порога.
Оптимальная пороговая сегментация основана на приближении гистограммы изображения к некоторой кривой с использованием весовых сумм двух или более вероятностей интенсивности с нормальным распределением. Тогда порог - это набор ближайших уровней яркости, соответствующих минимуму вероятности между максимумами двух или более нормальных распределений.


<img src="../content/binarization_examle.png" width="800"/>

Примеры бинаризации изображений будем рассматривать на изображении хлорелл под микроскопом

"""

image1 = cv.imread('../images/bloodcells.png')
image2 = cv.imread('../images/chlorella.png')
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
hsv_image1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
rgb_image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)
hsv_image2 = cv.cvtColor(image2, cv.COLOR_BGR2HSV)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

channels = [0]
histSize = [256]
range = [0, 256]

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.imshow(gray_image2, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[2])
plt.hist(gray_image1.reshape(-1), 256, range)
plt.subplot(gs[3])
plt.hist(gray_image2.reshape(-1), 256, range)
plt.show()

"""

### 1.1 Бинаризация полутоновых изображений (пороговая фильтрация).

Рассмотрим простую бинаризацию на примере полутонового изображения.

Бинаризация полутоновых изображений осуществляется просто. Для каждого пикселя применяется одно и то же пороговое значение. Если значение пикселя меньше порогового значения, оно имеет значение 0, в противном случае — максимальное значение. В библиотеке OpenCV есть функция для бинаризации изображений cv.threshold(), для применения пороговых значений. Функция принимает несколько параметров:
- image -- изображение, к которому применяется бинаризация;
- threshold -- пороговое значение;
- maxval -- максимальное значение, которое присваивается значениям пикселей, превышающим пороговое значение;
- type -- тип порога.

OpenCV предоставляет различные типы пороговых значений:
- cv.THRESH_BINARY
$$
\begin{equation*}
out(x, y) =
 \begin{cases}
   maxval \; &\textit{if image(x, y) > threshold } \\
   0 \; &\textit{иначе}
 \end{cases}
\end{equation*}
$$
- cv.THRESH_BINARY_INV
$$
\begin{equation*}
out(x, y) =
 \begin{cases}
   0 \; &\textit{if image(x, y) > threshold } \\
   maxval \; &\textit{иначе}
 \end{cases}
\end{equation*}
$$
- cv.THRESH_TRUNC
$$
\begin{equation*}
out(x, y) =
 \begin{cases}
   threshold \; &\textit{if image(x, y) > threshold } \\
   image(x,y) \; &\textit{иначе}
 \end{cases}
\end{equation*}
$$
- cv.THRESH_TOZERO
$$
\begin{equation*}
out(x, y) =
 \begin{cases}
   image(x, y) \; &\textit{if image(x, y) > threshold } \\
   0 \; &\textit{иначе}
 \end{cases}
\end{equation*}
$$
- cv.THRESH_TOZERO_INV
$$
\begin{equation*}
out(x, y) =
 \begin{cases}
   0 \; &\textit{if image(x, y) > threshold } \\
   image(x, y) \; &\textit{иначе}
 \end{cases}
\end{equation*}
$$

Подробнее по типам смотри [документацию](https://docs.opencv.org/4.x/d7/d1b/group__imgproc__misc.html#gaa9e58d2860d4afa658ef70a9b1115576)

"""

threshold = 150
image = gray_image1

ret, thresh1 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
ret, thresh2 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY_INV)
ret, thresh3 = cv.threshold(image, threshold, 255, cv.THRESH_TRUNC)
ret, thresh4 = cv.threshold(image, threshold, 255, cv.THRESH_TOZERO)
ret, thresh5 = cv.threshold(image, threshold, 255, cv.THRESH_TOZERO_INV)
titles = ['Grayscale Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [image, thresh1, thresh2, thresh3, thresh4, thresh5]
plt.figure(figsize=(15, 8))
for i in np.arange(len(images)):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

"""### Задание: реализуйте пороговую фильтрацию при помощи NumPy."""


# Установка порога
threshold = 150
maxval = 255

# Пороговая фильтрация с использованием OpenCV
ret, cv_thresh = cv.threshold(gray_image1, threshold, maxval, cv.THRESH_BINARY)

# Собственная реализация пороговой фильтрации с использованием NumPy
def numpy_threshold(image, threshold, maxval):
    binary_image = np.where(image > threshold, maxval, 0).astype(np.uint8)
    return binary_image

numpy_thresh = numpy_threshold(gray_image1, threshold, maxval)

# Отображение результатов
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv_thresh, cmap='gray', vmin=0, vmax=255)
plt.title('OpenCV THRESH_BINARY')
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(numpy_thresh, cmap='gray', vmin=0, vmax=255)
plt.title('NumPy THRESH_BINARY')
plt.xticks([])
plt.yticks([])

plt.show()



"""
### 1.2 Бинаризация Оцу (Otsu's Binarization)

Для определения оптимального порога бинаризации предложено большое количество различных подходов. Наиболее удачным из них является подход Оцу, который предполагает не только определение оптимального порога бинаризации, но и вычисление некоторого критерия бимодальности, т.е оценку того, действительно ли исследуемая гистограмма содержит именно две моды (два выраженных пика).
Подробнее про метод Оцу и алгоритм его работы можно почитать [здесь](https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html) и [здесь](https://ru.wikipedia.org/wiki/%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%9E%D1%86%D1%83)

Для использования этого метода в opencv используется таже функция cv.threshold(), в которой в качестве дополнительного флага передается параметр cv.THRESH_OTSU. Пороговое значение может быть выбрано произвольным. Затем алгоритм находит оптимальное пороговое значение, которое возвращается в качестве первого значения кортежа.

Работу метода Оцу рассмотрим на примере:
"""

threshold = 150
ret1, thresh1 = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
ret2, thresh2 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
titles = ['Original Image', 'Global Thresholding (threshold = %d)' % threshold,
          "Otsu's Thresholding (Otsu's threshold = %d)" % ret2]
images = [image, thresh1, thresh2]

plt.figure(figsize=(15, 8))
for i in np.arange(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

"""### 1.3 Адаптивная бинаризация.

В простой полутоновой бинаризации в качестве порога используется одно значение. Но это может быть не во всех случаях, например, если изображение имеет разные условия освещения в разных областях. В этом случае может помочь адаптивное пороговое значение. Здесь алгоритм определяет порог для пикселя на основе небольшой области вокруг него. Таким образом, мы получаем разные пороги для разных областей одного и того же изображения, что дает лучшие результаты для изображений с различной освещенностью.

Помимо описанных выше параметров, метод `cv.adaptiveThreshold()` принимает три входных параметра:

- **AdaptiveMethod** решает, как вычисляется пороговое значение:
    + `cv.ADAPTIVE_THRESH_MEAN_C`: Пороговое значение представляет собой среднее значение площади окрестностей минус константа C.
    + `cv.ADAPTIVE_THRESH_GAUSSIAN_C`: Пороговое значение представляет собой гауссово-взвешенную сумму значений окрестностей минус константа C.
- **BlockSize** определяет размер области соседства, а C — константа, которая вычитается из средней или взвешенной суммы соседних пикселей.
"""

ret1, thresh1 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
thresh2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, 5)
thresh3 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, 5)
titles = ['Original Image', "Otsu's Thresholding (Otsu's threshold = %d)" % ret1,
          'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [image, thresh1, thresh2, thresh3]

plt.figure(figsize=(10, 8))
for i in np.arange(len(images)):
    plt.subplot(2, 2, i + 1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()

"""### 1.4 Сегментация многомодальных изображений (Мультипороговая гистограммная бинаризация)

<img src="../content/multimodal_segmentation.png" width="800"/>

Сегментация многомодальных изображений - специально разработанный для данного класса задач, метод статистического выделения мод позволяет оценивать количество и степень выраженности мод гистограммы, опираясь на соответствующий график статистической производной (функции локальной разделимости), представляющий собой график значений критерия Оцу, вычисляемых в локальном скользящем окне, согласованном по ширине с ожидаемой шириной моды гистограммы.
**Задание: реализуйте метод сегментации многомодальных изображений**
"""


from skimage.filters import threshold_multiotsu

range = [0, 256]

# Чтение изображения
image1 = cv.imread('../images/lenna.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

# Применение мультипороговой бинаризации с использованием `threshold_multiotsu`
num_thresholds = 3  # Количество порогов
thresholds = threshold_multiotsu(gray_image1, classes=num_thresholds + 1)

# Функция для сегментации изображения на основе нескольких порогов
def apply_multi_thresholds(image, thresholds):
    thresholds = np.sort(thresholds)
    segmented_image = np.zeros_like(image)

    for i, t in enumerate(thresholds):
        segmented_image[image > t] = (i + 1) * (255 // (len(thresholds) + 1))

    segmented_image[image > thresholds[-1]] = 255

    return segmented_image

# Применение порогов
segmented_image = apply_multi_thresholds(gray_image1, thresholds)

# Отображение результатов
titles = ['Original Image', f'Multi Otsu Segmentation\nThresholds: {thresholds}']
images = [gray_image1, segmented_image]

del range

plt.figure(figsize=(10, 5))
for i in range(len(images)):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.xticks([])
    plt.yticks([])
plt.show()



"""### 1.5 Цветовая сегментация изображений

Известно, что цветные цифровые изображения представляют собой совокупность трех цветовых плоскостей, каждая из которых характеризует одну независимую составляющую цвета, представленную в том же формате, что и обычное 8-битное полутоновое изображение. Следовательно, все описанные процедуры обработки полутоновых изображений в яркостной области могут быть обобщены и на случай обработки цветных изображений. 
Специфика же здесь связана прежде всего с различными цветовыми моделями, позволяющими по-разному работать с разными цветовыми и другими составляющими изображения.
"""

image3 = cv.imread('../images/pencils.png')
rgb_image3 = cv.cvtColor(image3, cv.COLOR_BGR2RGB)
hsv_image3 = cv.cvtColor(rgb_image3, cv.COLOR_RGB2HSV)
h, s, v = cv.split(hsv_image3)

low_h = 75
high_h = 85

mask = cv.inRange(h, low_h, high_h)
result = cv.bitwise_and(rgb_image3, rgb_image3, mask=mask)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(rgb_image3)
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.imshow(mask, cmap='gray')
plt.title('Маска')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[2])
plt.hist(h.reshape(-1), np.max(h), [np.min(h), np.max(h)])
plt.vlines(low_h, 0, 5000, 'r'), plt.vlines(high_h, 0, 5000, 'r')
plt.title('Гистограмма hue слоя')
plt.subplot(gs[3])
plt.imshow(result)
plt.title('Изображение с пикселями выделенного цвета')
plt.show()




# Изменим стандартный размер графиков matplotlib
plt.rcParams["figure.figsize"] = [6, 4]

"""# 1. Метрики качества. Среднеквадратическая ошибка (MSE). Пиковое отношение сигнал/шум (PSNR).

Метрики качества служат для измерения схожести/различия между двумя изображениями.

## 1.1 PSNR
PSNR (peak signal-to-noise ratio, отношение пикового уровня сигнала к шуму) - наиболее часто используемая количественная метрика для измерения уровня искажений при работе с изображениями.

PSNR наиболее часто используется для измерения уровня искажений при сжатии изображений.

$$
PSNR = 10 \cdot log_{10}\frac{MAX^2_I}{MSE},
$$
где $MAX_I$ максимально возможное значение пикселя. Для 8-битного изображения - 255;
$$
MSE = \frac{1}{mn} \sum\limits_{i=0}^{m-1} \sum\limits_{j=0}^{n-1} |I(i,j) - K(i,j)|^2.
$$
Когда два изображения одинаковы, MSE даст ноль, что приведет к недопустимой операции деления на ноль в формуле PSNR. В этом случае PSNR не определен, и нам нужно будет рассматривать этот случай отдельно. Переход на логарифмическую шкалу происходит потому, что значения пикселей имеют очень широкий динамический диапазон. Готовая функция выглядит так:
"""

def getPSNR(I1, I2):
    s1 = cv.absdiff(I1, I2)  #|I1 - I2|
    s1 = np.float32(s1)  # cannot make a square on 8 bits
    s1 = s1 * s1  # |I1 - I2|^2
    sse = s1.sum()  # sum elements per channel
    if sse <= 1e-10:  # sum channels
        return 0  # for small values return zero
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr

"""Обычно значения результатов составляют от 30 до 50 (дБ). Если изображения значительно отличаются, вы получите гораздо более низкие (15 и ниже).

Проблема PSNR в том, что на практике она может не совпадать с субъективной оценкой качества изображения (видео) человеком. Это может привести, например, к неверному выводу о превосходстве одного кодека над другим.

## 1.2 SSIM

SSIM (structure similarity, структурное сходство) точнее учитывает особенности восприятия изображения и видео человеком.

SSIM метрика рассчитана на различные размеры окна. Разница между двумя окнами ${\displaystyle x}$ и ${\displaystyle y}$ имеющими одинаковый размер N×N:

$$
{\text{SSIM}}(x,y)={\frac  {(2\mu _{x}\mu _{y}+c_{1})(2\sigma _{{xy}}+c_{2})}{(\mu _{x}^{2}+\mu _{y}^{2}+c_{1})(\sigma _{x}^{2}+\sigma _{y}^{2}+c_{2})}},
$$
где:
- $\mu _{x}$ — среднее x,
- $\mu _{y}$ — среднее y,
- $\sigma_{x}^{2}$ — дисперсия x,
- $\sigma_{y}^{2}$ — дисперсия y,
- $\sigma_{xy}$ — ковариация x и y,
- $c_{1}=(k_{1}L)^{2}, \; c_{2}=(k_{2}L)^{2}$ — две переменных:
    + L — динамический диапазон пикселей (обычно $2^\text{bits per pixel}-1)$,
    + $k_{1}$=0,01 и $k_{2}$=0,03 — константы.

Для 8 битного изображения:
- $c_{1} = (0,01 * 2^8 - 1)^2 = 6,5025;$
- $c_{2} = (0,03 * 2^8 - 1)^2 = 58,5225.$

SSIM вычисляется для каждого канала в отдельности.
"""

def getSSIM(i1, i2):
    C1 = 6.5025  # only for 8-bit images
    C2 = 58.5225  # only for 8-bit images
    # INITS
    I1 = np.float32(i1)  # cannot calculate on one byte large values
    I2 = np.float32(i2)
    I2_2 = I2 * I2  # I2^2
    I1_2 = I1 * I1  # I1^2
    I1_I2 = I1 * I2  # I1 * I2
    # END INITS
    # PRELIMINARY COMPUTING
    mu1 = cv.GaussianBlur(I1, (11, 11), 1.5)
    mu2 = cv.GaussianBlur(I2, (11, 11), 1.5)
    mu1_2 = mu1 * mu1
    mu2_2 = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_2 = cv.GaussianBlur(I1_2, (11, 11), 1.5)
    sigma1_2 -= mu1_2
    sigma2_2 = cv.GaussianBlur(I2_2, (11, 11), 1.5)
    sigma2_2 -= mu2_2
    sigma12 = cv.GaussianBlur(I1_I2, (11, 11), 1.5)
    sigma12 -= mu1_mu2
    t1 = 2 * mu1_mu2 + C1
    t2 = 2 * sigma12 + C2
    t3 = t1 * t2  # t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))
    t1 = mu1_2 + mu2_2 + C1
    t2 = sigma1_2 + sigma2_2 + C2
    t1 = t1 * t2  # t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))
    ssim_map = cv.divide(t3, t1)  # ssim_map =  t3./t1;
    ssim = cv.mean(ssim_map)  # mssim = average of ssim map
    ssim = ssim[:3]
    return ssim

"""Посчитаем PSNR и SSIM для изображений lenna.png и его ухудшенного варианта lenna_bad.png:"""

image1 = cv.imread('../images/lenna.png')
image2 = cv.imread('../images/lenna_bad.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
gray_image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
rgb_image2 = cv.cvtColor(image2, cv.COLOR_BGR2RGB)

# rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)

gs = plt.GridSpec(1, 2)
plt.figure(figsize=(10, 4))
plt.subplot(gs[0])
plt.imshow(rgb_image1)
plt.title('lenna.png')
plt.xticks([]), plt.yticks([])
plt.subplot(gs[1])
plt.imshow(rgb_image2)
plt.title('lenna_bad.png')
plt.xticks([]), plt.yticks([])

print('PSNR = ', util.getPSNR(image1, image2))
print('SSIM = ', util.getSSIM(image1, image2))

"""# 2. Фильтрация изображений. Ранговая нелинейная фильтрация. Выделение объектов.

## 2.1 Задача фильтрации изображений. Модели шумов.
Под задачей "фильтрации изображений" в широком смысле понимают любые процедуры обработки изображений, при которых на вход процедуры подается (одно) растровое изображение, и на выходе также формируется растровое изображение. Такие процедуры типа (один растровый вход, один растровый выход) называют *фильтрами*.

Однако чаще под "фильтрацией" в более узком смысле понимают так называемую *помеховую фильтрацию*, или фильтрацию изображений от "шума". Задача помеховой фильтрации, сводится к тому, чтобы путем некоторой обработки наблюдаемого реального изображения как можно лучше "очистить его от шума", то есть получить изображение, наиболее близкое по своим характеристикам к исходному "незашумленному" изображению.

Для изучения методов фильтрации мы будем изображения зашумлять. Для этого будем использовать разные модели шумов:
- модель "соль и перец", под которым понимают замещение значения пикселя на 0 с вероятностью p и -- на 1 с вероятностью q

- модель аддитивного гауссовского шума, который описывается простой формулой:
$ Im^{\prime}[x,y] = Im[x,y] + N[0,\sigma] $,
где $N(\mu, \sigma)$ - нормальное распределение, $\mu$ - математическое ожидание, а $\sigma$ - среднеквадратическое отклонение.

"""

gs = plt.GridSpec(1, 3)
plt.figure(figsize=(17, 4))

plt.subplot(gs[0])
plt.title('Исходное изображение')
plt.xticks([]), plt.yticks([])
plt.imshow(gray_image1, cmap='gray')

plt.subplot(gs[1])
plt.xticks([]), plt.yticks([])
noisy_image = util.add_salt_and_peper_noise(gray_image1, 0.2)
psnr = util.getPSNR(gray_image1, noisy_image)
ssim = util.getSSIM(gray_image1, noisy_image)
plt.title(f'Изображение зашумленное шумом соль и перец \n PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')
plt.imshow(noisy_image, cmap='gray')

plt.subplot(gs[2])
plt.xticks([]), plt.yticks([])
noisy_image = util.add_gauss_noise(gray_image1, 0, 0.2)
psnr = util.getPSNR(gray_image1, noisy_image)
ssim = util.getSSIM(gray_image1, noisy_image)
plt.title(f'Изображение зашумленное гаусовским шумом \n PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')
plt.imshow(noisy_image, 'gray')
plt.show()

"""## 2.2 Нелинейная фильтрация полутоновых изображений

Значения отсчётов внутри окна фильтра сортируются в порядке возрастания (убывания); и значение, находящееся в середине упорядоченного списка, поступает на выход фильтра. В случае чётного числа отсчётов в окне выходное значение фильтра равно среднему значению двух отсчётов в середине упорядоченного списка. Окно перемещается вдоль фильтруемого сигнала, и вычисления повторяются.

Медианная фильтрация — эффективная процедура обработки сигналов, подверженных воздействию импульсных помех.

"""

image1 = cv.imread('../images/lenna.png')
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

sap_image1 = util.add_salt_and_peper_noise(gray_image1, 0.2)
median_image1 = cv.medianBlur(sap_image1, 3)
median_image2 = cv.medianBlur(sap_image1, 5)

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(8, 10))
plt.subplot(gs[0])
plt.imshow(gray_image1, cmap='gray')
plt.title('Исходное изображение')

plt.subplot(gs[1])
plt.imshow(sap_image1, cmap='gray')
psnr = util.getPSNR(gray_image1, sap_image1)
ssim = util.getSSIM(gray_image1, sap_image1)
plt.title(f'Изображение зашумленное \n шумом соль и перец \n PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')

plt.subplot(gs[2])
plt.imshow(median_image1, cmap='gray')
psnr = util.getPSNR(gray_image1, median_image1)
ssim = util.getSSIM(gray_image1, median_image1)
plt.title(f'Восстановленное изображение \n '
          f'медианным фильтром 3х3 \n'
          f'PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')

plt.subplot(gs[3])
plt.imshow(median_image2, cmap='gray')
psnr = util.getPSNR(gray_image1, median_image2)
ssim = util.getSSIM(gray_image1, median_image2)
plt.title(f'Восстановленное изображение \n '
          f'медианным фильтром 5х5 \n'
          f'PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')

plt.show()

"""Как видно из примера, медианная фильтрация является крайне эффективным методом фильтрации шумов.

# 3. Линейная фильтрация изображений в пространственной области.

Вся линейная фильтрация выражается через операцию свертки изображения с ядром фильтра. Выражение свертки записывается следующей формулой:

$$
g(x,y) = \omega \ast f(x,y) = \sum\limits_{dx=-a}^a \sum\limits_{dy=-b}^b \omega(dx,dy)f(x+dx,y+dy)
$$

где:
- $g(x,y)$ - отфильтрованное изображение;
- $f(x,y)$ - исходное изображение;
- $\omega$ - ядро фильтра.

Ядро фильтра представляет собой матрицу значений, которые определяют операцию над изображением.

## 3.1 Сглаживание изображения

**Скользящее среднее.**

Простейшим видом линейной фильтрации в пространственной области является скользящее среднее. Результатом фильтрации является математические ожидания, вычисленные по всем пикселям окна. Математически это эквивалентно свертке с ядром, все элемены которого равны 1/n, где n - количество элементов матрицы. Например, ядро 3х3 имеет следующий вид:

$$
\frac{1}{9}
 \begin{pmatrix}
  1 & 1 & 1 \\
  1 & 1 & 1 \\
  1 & 1 & 1
 \end{pmatrix}
$$

OpenCV предоставляет функцию cv.filter2D() для свертки ядра с изображением.

**Гауссиан.**


Повысить устойчивость результатов фильтрации на краях областей можно, если придать более близким точкам окрестности большее влияние на окончательный результат, чем дальним. Примером реализации этой идеи для окна размера 3×3 является маска:

$$
\frac{1}{16}
 \begin{pmatrix}
  1 & 2 & 1 \\
  2 & 4 & 2 \\
  1 & 2 & 1
 \end{pmatrix}
$$

Такая маска называется гауссовой. Соответственно, и использующий ее линейный фильтр также называется гауссовым. Используя дискретные приближения двумерной гауссовой функции, можно получить и другие гауссовы ядра большего размера. Обратите внимание на то, что сглаживающие или фильтрующие маски линейных фильтров должны иметь сумму всех элементов, равную 1.

OpenCV предоставляет функцию cv.filter2D() для свертки ядра с изображением.
"""

kernel55 = np.ones((5, 5), np.float32) / 25
kernel77 = np.ones((7, 7), np.float32) / 49

noisy_image = util.add_gauss_noise(gray_image1, 0, 0.15)

filtered_image1 = cv.filter2D(noisy_image, -1, kernel55)
filtered_image2 = cv.filter2D(noisy_image, -1, kernel77)
gaussian_image1 = cv.GaussianBlur(noisy_image, (7, 7), 0)
gaussian_image2 = cv.GaussianBlur(noisy_image, (15, 15), 0)

# вывод
gs = plt.GridSpec(2, 3)
plt.figure(figsize=(15, 12))

plt.subplot(gs[0, 0])
plt.xticks([]), plt.yticks([])
plt.title('Исходное изображение')
plt.imshow(gray_image1, cmap='gray')

plt.subplot(gs[1, 0])
plt.xticks([]), plt.yticks([])
plt.imshow(noisy_image, cmap='gray')
psnr = util.getPSNR(gray_image1, noisy_image)
ssim = util.getSSIM(gray_image1, noisy_image)
plt.title(f'Изображение зашумленное гауссовским шумом \n PSNR = {psnr:.3f} \n SSIM = {ssim:.3f}')

plt.subplot(gs[0, 1])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат средней линейной \n фильрации с ядром 5х5 \n '
          f'PSNR = {util.getPSNR(gray_image1, filtered_image1):.3f} \n '
          f'SSIM = {util.getSSIM(gray_image1, filtered_image1):.3f}')
plt.imshow(filtered_image1, 'gray')

plt.subplot(gs[0, 2])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат средней линейной \n фильрации с ядром 7х7 \n '
          f'PSNR = {util.getPSNR(gray_image1, filtered_image2):.3f} \n '
          f'SSIM = {util.getSSIM(gray_image1, filtered_image2):.3f}')
plt.imshow(filtered_image2, 'gray')

plt.subplot(gs[1, 1])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат гауссовской  \n фильрации с ядром 7х7 \n '
          f'PSNR = {util.getPSNR(gray_image1, gaussian_image1):.3f} \n '
          f'SSIM = {util.getSSIM(gray_image1, gaussian_image1):.3f}')
plt.imshow(gaussian_image1, 'gray')

plt.subplot(gs[1, 2])
plt.xticks([]), plt.yticks([])
plt.title(f'Результат гауссовской  \n фильрации с ядром 15х15 \n '
          f'PSNR = {util.getPSNR(gray_image1, gaussian_image2):.3f} \n '
          f'SSIM = {util.getSSIM(gray_image1, gaussian_image2):.3f}')
plt.imshow(gaussian_image2, 'gray')

plt.show()

"""Как видно по изображениям линейные фильтры хоть и позволяют несколько отфильтровывать шумы за счет сглаживания, при этом размывая изображение, однако они не столь эффективны против импульсных шумов по сравнению с нелинейными ранговыми фильтрами."""



"""
# 5. Математическая морфология Серра и методы анализа изображений

Математическая морфология Серра (ударение на последний слог) позволяет осуществлять обработку изображений с учетом формы и размера имеющихся на изображении областей. Морфологические операторы Серра позволяют: выделять или удалять на изображениях мелко- и среднеразмерные объекты заданной формы и размера, а также фильтровать (сглаживать) форму крупноразмерных объектов.

## 5.1. Операции сужения (erosion), расширения (dilation), закрытия (closing) и открытия (opening).

Базовыми операциями математической морфологии Серра являются: дилатация (расширение) и эрозия (сжатие) изображения X структурирующим элементом B. На этих базовых операциях основаны также операции открытия и закрытия.
Рассмотрим геометрический смысл операторов математической морфологии на примере обработки искусственного изображения:
"""

image = cv.imread('../images/binary_cross.png')
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
noise_image = util.add_salt_and_peper_noise(image, 0.02)
plt.imshow(image, 'gray')

kernel = np.ones((3, 3), np.uint8)
dilation = cv.dilate(noise_image, kernel, iterations=1)
erosion = cv.erode(noise_image, kernel, iterations=1)
opening = cv.morphologyEx(noise_image, cv.MORPH_OPEN, kernel)
closing = cv.morphologyEx(noise_image, cv.MORPH_CLOSE, kernel)
closeAndOpen = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel)

plt.figure(figsize=(15, 8))
plt.subplot(231)
plt.imshow(noise_image, 'gray')
plt.subplot(232)
plt.imshow(dilation, 'gray')
plt.subplot(233)
plt.imshow(erosion, 'gray')
plt.subplot(234)
plt.imshow(opening, 'gray')
plt.subplot(235)
plt.imshow(closing, 'gray')
plt.subplot(236)
plt.imshow(closeAndOpen, 'gray')
plt.show()