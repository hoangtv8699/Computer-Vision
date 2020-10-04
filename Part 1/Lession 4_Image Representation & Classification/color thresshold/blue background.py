import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    # read image and change to RGB
    image = cv2.imread("blue backgound.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print('this image is: ', type(image), 'with demention: ', image.shape)
    # plt.imshow(image_rgb)
    # plt.show()

    # define thresshold
    lower_blue = np.array([0, 0, 150])
    upper_blue = np.array([50, 70, 255])

    # find blue background
    mask = cv2.inRange(image_rgb, lower_blue, upper_blue)
    # plt.imshow(mask, cmap='gray')
    # plt.show()

    # mask image
    masked_image = np.copy(image_rgb)
    masked_image[mask != 0] = [0, 0, 0]
    # plt.imshow(masked_image)
    # plt.show()

    # read background image
    background_image = cv2.imread('stock-photography-slider.jpg')
    background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
    crop_background = background_image[0:342, 0:608]

    # mask background
    crop_background[mask == 0] = [0, 0, 0]
    # plt.imshow(crop_background)
    # plt.show()

    # add background
    new_image = masked_image + crop_background
    plt.imshow(new_image)
    plt.show()