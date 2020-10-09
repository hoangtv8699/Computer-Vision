import numpy as np
import matplotlib.pyplot as plt
import cv2


def orientations(contours):
    """
    Orientation
    :param contours: a list of contours
    :return: angles, the orientations of the contours
    """

    # Create an empty list to store the angles in
    # Tip: Use angles.append(value) to add values to this list
    angles = []

    for contour in contours:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
        angles.append(angle)

    return angles


def left_hand_crop(image, selected_contour):
    """
    Left hand crop
    :param image: the original image
    :param selectec_contour: the contour that will be used for cropping
    :return: cropped_image, the cropped image around the left hand
    """
    # Make a copy of the image to crop
    cropped_image = np.copy(image)

    # find rectangle
    x, y, w, h = cv2.boundingRect(selected_contour)

    # crop
    cropped_image = cropped_image[y : y + h, x : x + w]
    return cropped_image


if __name__ == '__main__':
    # load in color image for face detection
    image = cv2.imread('hand.jpg')

    # convert to RBG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # convert to binary image
    retval, binary = cv2.threshold(gray, 230, 255, cv2.THRESH_BINARY_INV)
    # plt.imshow(binary, cmap='gray')

    # find contours
    retval, contours, hierachy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # draw contours
    all_contours = cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    # plt.imshow(all_contours)

    angles = orientations(contours)
    print('Angles of each contour (in degrees): ' + str(angles))
    crop_hand = left_hand_crop(image, contours[0])
    plt.imshow(crop_hand)
    plt.show()