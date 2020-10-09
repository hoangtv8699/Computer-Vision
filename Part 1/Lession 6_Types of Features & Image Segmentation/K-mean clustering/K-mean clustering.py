import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # load in color image for face detection
    image = cv2.imread('monarch.jpg')

    # convert to RBG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_copy = np.copy(image)

    # reshape image to 2d array pixel of 3 color and float type
    pixel_vals = image_copy.reshape((-1, 3))
    pixel_vals = np.float32(pixel_vals)

    # k-mean implementation
    k = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data to 8bit value
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape image to original dimention
    segmented_image = segmented_data.reshape(image_copy.shape)
    labels_reshape = labels.reshape(image_copy.shape[0], image_copy.shape[1])

    plt.imshow(segmented_image)
    plt.show()