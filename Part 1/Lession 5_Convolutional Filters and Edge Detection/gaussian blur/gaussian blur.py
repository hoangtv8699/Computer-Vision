import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # read and change to RGB
    city_hall = cv2.imread('../creating a filter/SFCityHall.png')
    city_hall = cv2.cvtColor(city_hall, cv2.COLOR_BGR2RGB)

    # plot image
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(city_hall)

    # gray scale
    gray = cv2.cvtColor(city_hall, cv2.COLOR_RGB2GRAY)
    axes[1].imshow(gray, cmap='gray')

    # gaussian blur
    gaus = cv2.GaussianBlur(gray, (5,5), 0)
    axes[2].imshow(gaus, cmap='gray')

    plt.show()
