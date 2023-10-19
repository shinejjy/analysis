import cv2
import numpy as np
from matplotlib import pyplot as plt


class Process:
    def __init__(self, kernel_size=(3, 3)):
        if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
            raise ValueError("Kernel size must be odd in both dimensions.")
        self.kernel_size = kernel_size
        self.kernel = np.ones(kernel_size, np.uint8)

    def erode(self, img):
        pad_size = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        img_pad = np.pad(img, pad_size, constant_values=(255, 255))  # 周围填充最大值
        eroded_image = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                eroded_image[i][j] = np.min(img_pad[i: i + self.kernel_size[0], j: j + self.kernel_size[1]].flatten())
        return eroded_image

    def dilate(self, img):
        pad_size = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)
        img_pad = np.pad(img, pad_size, constant_values=(0, 0))  # 周围填充最小值
        dilated_image = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                dilated_image[i][j] = np.max(img_pad[i: i + self.kernel_size[0], j: j + self.kernel_size[1]].flatten())
        return dilated_image

    def open(self, img):
        eroded_image = self.erode(img)
        opened_image = self.dilate(eroded_image)
        return opened_image

    def close(self, img):
        dilated_image = self.dilate(img)
        closed_image = self.erode(dilated_image)
        return closed_image

    def gradient(self, img):
        dilated_image = self.dilate(img)
        eroded_image = self.erode(img)
        gradient_image = np.abs(dilated_image - eroded_image)
        return gradient_image


def main():
    # 读取图像
    image = cv2.imread('newspaper.jpg')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_image = cv2.resize(gray_image, (500, 500))

    process = Process()
    eroded_image = process.erode(gray_image)
    dilated_image = process.dilate(gray_image)
    opened_image = process.open(gray_image)
    closed_image = process.close(gray_image)
    gradient_image = process.gradient(gray_image)

    # 显示原始图像
    plt.figure(figsize=(8, 6))
    plt.imshow(gray_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.show()

    # 显示腐蚀后的图像
    plt.figure(figsize=(8, 6))
    plt.imshow(eroded_image, cmap='gray')
    plt.title('Eroded Image')
    plt.axis('off')
    plt.show()

    # 显示膨胀后的图像
    plt.figure(figsize=(8, 6))
    plt.imshow(dilated_image, cmap='gray')
    plt.title('Dilated Image')
    plt.axis('off')
    plt.show()

    # 显示开运算后的图像
    plt.figure(figsize=(8, 6))
    plt.imshow(opened_image, cmap='gray')
    plt.title('Opened Image')
    plt.axis('off')
    plt.show()

    # 显示闭运算后的图像
    plt.figure(figsize=(8, 6))
    plt.imshow(closed_image, cmap='gray')
    plt.title('Closed Image')
    plt.axis('off')
    plt.show()

    # 显示梯度图像
    plt.figure(figsize=(8, 6))
    plt.imshow(gradient_image, cmap='gray')
    plt.title('Gradient Image')
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
