import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # load in color image for face detection
    image = cv2.imread('waffle.jpg')

    # convert to RBG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # create plt
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(image)

    # copy image
    image_copy = np.copy(image)

    # detect conner
    gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)

    # thesshold
    thresshold = 0.1*dst.max()

    # copy image
    corners_image = np.copy(image)

    for j in range(0, dst.shape[0]):
        for i in range(0, dst.shape[1]):
            if dst[j,i] > thresshold:
                cv2.circle(corners_image, (i, j), 2, (0, 255, 0), 1)

    axes[1].imshow(corners_image)
    plt.show()