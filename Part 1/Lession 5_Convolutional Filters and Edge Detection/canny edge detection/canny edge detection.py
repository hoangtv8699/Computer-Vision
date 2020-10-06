import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # read and change to RGB
    sunflower = cv2.imread('sunflower.jpg')
    sunflower = cv2.cvtColor(sunflower, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(sunflower, cv2.COLOR_RGB2GRAY)

    # create fig
    fig, axes = plt.subplots(1, 3)
    axes[0].imshow(sunflower)
    axes[1].imshow(gray, cmap='gray')

    # define thresshold
    lower = 120
    upper = 240

    # canny edge detection
    canny = cv2.Canny(gray, lower, upper)
    axes[2].imshow(canny, cmap='gray')

    f, axes = plt.subplots(1, 2)

    # wide and tight thresshold
    wide = cv2.Canny(gray, 30, 100)
    tight = cv2.Canny(gray, 100, 180)

    # show
    axes[0].imshow(wide, cmap='gray')
    axes[0].set_title('wide')
    axes[1].imshow(tight, cmap='gray')
    axes[1].set_title('tight')
    plt.show()