import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # read and change to RGB
    city_hall = cv2.imread('SFCityHall.png')
    city_hall = cv2.cvtColor(city_hall, cv2.COLOR_BGR2RGB)

    # plot image
    fig, axes = plt.subplots(4, 1)
    axes[0].imshow(city_hall)

    # gray scale
    gray = cv2.cvtColor(city_hall, cv2.COLOR_RGB2GRAY)
    axes[1].imshow(gray, cmap='gray')

    # create sobel filter
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    # apply filter
    filtered = cv2.filter2D(gray, -1, sobel_x)
    axes[2].imshow(filtered, cmap='gray')

    # thresshold
    retval, binary_image = cv2.threshold(filtered, 100, 255, cv2.THRESH_BINARYs)

    plt.imshow(city_hall)
    plt.show()
