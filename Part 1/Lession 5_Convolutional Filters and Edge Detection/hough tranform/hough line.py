import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # read and change to RGB
    phone = cv2.imread('phone.jpg')
    phone = cv2.cvtColor(phone, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(phone, cv2.COLOR_RGB2GRAY)

    # create fig
    fig, axes = plt.subplots(2, 2)
    axes[0][0].imshow(phone)
    axes[0][1].imshow(gray, cmap='gray')

    # define thresshold
    lower = 50
    upper = 100

    # canny edge detection
    canny = cv2.Canny(gray, lower, upper)
    axes[1][0].imshow(canny, cmap='gray')

    # define hough tranform parameter
    rho = 1
    theta = np.pi/180
    thresshold = 50
    min_line_length = 100
    max_line_gap = 5

    line_image = np.copy(phone)

    # hough line
    lines = cv2.HoughLinesP(canny, rho, theta, thresshold, np.array([]), min_line_length, max_line_gap)
    for line in lines:
        print(line)
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    axes[1][1].imshow(line_image)

    plt.show()