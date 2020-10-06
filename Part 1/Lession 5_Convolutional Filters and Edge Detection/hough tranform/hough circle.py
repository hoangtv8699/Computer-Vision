import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # read and change to RGB
    farm = cv2.imread('round_farms.jpg')
    farm = cv2.cvtColor(farm, cv2.COLOR_BGR2RGB)

    # create fig
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(farm)

    # define thresshold
    lower = 50
    upper = 100

    # canny edge detection
    gray = cv2.cvtColor(farm, cv2.COLOR_BGR2GRAY)

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    circles_im = np.copy(farm)

    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, 1,
                               minDist=45,
                               param1=100,
                               param2=16,
                               minRadius=20,
                               maxRadius=40)

    # convert circles into expected type
    circles = np.uint16(np.around(circles))
    # draw each one
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(circles_im, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(circles_im, (i[0], i[1]), 2, (0, 0, 255), 3)

    axes[1].imshow(circles_im)

    plt.show()