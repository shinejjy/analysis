import cv2
import numpy as np
from matplotlib import pyplot as plt


def main():
    # Read the image
    image = cv2.imread('newspaper.jpg', cv2.IMREAD_GRAYSCALE)

    # Initialize the kernel and the number of iterations
    kernel = np.ones((3, 3), np.uint8)
    num_iterations = 21

    eroded_image = image.copy()
    dilated_image = image.copy()
    opened_image = image.copy()
    closed_image = image.copy()

    # Create a subplot to display multiple images
    fig, axes = plt.subplots(4, 5, figsize=(15, 10))

    for i in range(num_iterations):
        eroded_image = cv2.erode(eroded_image, kernel, iterations=1)
        dilated_image = cv2.dilate(dilated_image, kernel, iterations=1)
        opened_image = cv2.morphologyEx(opened_image, cv2.MORPH_OPEN, kernel, iterations=1)
        closed_image = cv2.morphologyEx(closed_image, cv2.MORPH_CLOSE, kernel, iterations=1)

        if i % 5 == 0:
            axes[0, i // 5].imshow(eroded_image, cmap='gray')
            axes[0, i // 5].set_title(f'Eroded {i + 1}')
            axes[0, i // 5].axis('off')
            axes[1, i // 5].imshow(dilated_image, cmap='gray')
            axes[1, i // 5].set_title(f'Dilated {i + 1}')
            axes[1, i // 5].axis('off')
            axes[2, i // 5].imshow(opened_image, cmap='gray')
            axes[2, i // 5].set_title(f'opened {i + 1}')
            axes[2, i // 5].axis('off')
            axes[3, i // 5].imshow(closed_image, cmap='gray')
            axes[3, i // 5].set_title(f'closed {i + 1}')
            axes[3, i // 5].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
