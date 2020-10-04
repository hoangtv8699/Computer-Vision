import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    water_balloon = cv2.imread('water-balloon.jpg')
    water_balloon_RBG = cv2.cvtColor(water_balloon, cv2.COLOR_BGR2RGB)
    water_balloon_HSV = cv2.cvtColor(water_balloon, cv2.COLOR_BGR2HSV)

    # create plot
    fig, axes = plt.subplots(4, 3)

    axes[0][0].imshow(water_balloon_RBG)
    axes[0][1].set_visible(False)
    axes[0][2].set_visible(False)

    # RBG channels
    r = water_balloon_RBG[:, :, 0]
    g = water_balloon_RBG[:, :, 1]
    b = water_balloon_RBG[:, :, 2]
    # plot R G B
    axes[1][0].imshow(r, cmap='gray')
    axes[1][1].imshow(g, cmap='gray')
    axes[1][2].imshow(b, cmap='gray')

    # HSV channels
    h = water_balloon_HSV[:, :, 0]
    s = water_balloon_HSV[:, :, 1]
    v = water_balloon_HSV[:, :, 2]
    # plot H S V
    axes[2][0].imshow(h, cmap='gray')
    axes[2][1].imshow(s, cmap='gray')
    axes[2][2].imshow(v, cmap='gray')

    # mask
    lower_pink = np.array([180, 0, 100])
    upper_pink = np.array([255, 255, 230])
    pink_filter = cv2.inRange(water_balloon_RBG, lower_pink, upper_pink)

    lower_hue = np.array([160, 0, 0])
    upper_hue = np.array([180, 255, 255])
    hue_filter = cv2.inRange(water_balloon_HSV, lower_hue, upper_hue)

    # implement filter
    RGB_copy = np.copy(water_balloon_RBG)
    HSV_copy = np.copy(water_balloon_RBG)

    RGB_copy[pink_filter == 0] = [0, 0, 0]
    HSV_copy[hue_filter == 0] = [0, 0, 0]

    axes[3][0].imshow(RGB_copy, cmap='gray')
    axes[3][1].imshow(HSV_copy, cmap='gray')
    axes[3][2].set_visible(False)

    plt.show()