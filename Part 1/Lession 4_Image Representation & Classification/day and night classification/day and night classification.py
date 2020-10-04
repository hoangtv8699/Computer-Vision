import helpers
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random


# Find the average Value or brightness of an image
def avg_brightness(rgb_image):
    # Convert image to HSV
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # Add up all the pixel values in the V channel
    sum_brightness = np.sum(hsv[:, :, 2])

    area = 600 * 1100.0
    # and the sum calculated above
    avg = sum_brightness / area

    return avg


def estimate_label(rgb_image):
    # extract avg brightness feature
    avg = avg_brightness(rgb_image)

    # use avg brightness to predict label (1, 0)
    predicted_label = 0
    thresshold = 100
    if avg > thresshold:
        predicted_label = 1
    return predicted_label


def get_misclassified_images(test_images):
    # Track misclassified images by placing them into a list
    misclassified_images_labels = []

    # Iterate through all the test images
    # Classify each image and compare to the true label
    for image in test_images:

        # Get true data
        im = image[0]
        true_label = image[1]

        # Get predicted label from your classifier
        predicted_label = estimate_label(im)

        # Compare true and predicted labels
        if (predicted_label != true_label):
            # If these labels are not equal, the image has been misclassified
            misclassified_images_labels.append((im, predicted_label, true_label))

    # Return the list of misclassified [image, predicted_label, true_label] values
    return misclassified_images_labels


if __name__ == '__main__':
    # Image data directories
    image_dir_training = "day_night_images/training/"
    image_dir_test = "day_night_images/test/"

    # Load training data
    IMAGE_LIST = helpers.load_dataset(image_dir_training)

    # Select an image and its label by list index
    image_index = 0
    selected_image = IMAGE_LIST[image_index][0]
    selected_label = IMAGE_LIST[image_index][1]

    # Display image and data about it
    # plt.imshow(selected_image)
    # plt.show()
    print("Shape: " + str(selected_image.shape))
    print("Label: " + str(selected_label))

    # Standardize all training images
    STANDARDIZED_LIST = helpers.standardize(IMAGE_LIST)

    # Select an image by index
    image_num = 0
    selected_image = STANDARDIZED_LIST[image_num][0]
    selected_label = STANDARDIZED_LIST[image_num][1]

    # Display image and data about it
    # plt.imshow(selected_image)
    # plt.show()
    print("Shape: " + str(selected_image.shape))
    print("Label [1 = day, 0 = night]: " + str(selected_label))

    image_num = 0
    test_im = STANDARDIZED_LIST[image_num][0]
    test_label = STANDARDIZED_LIST[image_num][1]

    # Convert to HSV
    hsv = cv2.cvtColor(test_im, cv2.COLOR_RGB2HSV)

    # Print image label
    print('Label: ' + str(test_label))

    # HSV channels
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # # Plot the original image and the three channels
    # f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 10))
    # ax1.set_title('Standardized image')
    # ax1.imshow(test_im)
    # ax2.set_title('H channel')
    # ax2.imshow(h, cmap='gray')
    # ax3.set_title('S channel')
    # ax3.imshow(s, cmap='gray')
    # ax4.set_title('V channel')
    # ax4.imshow(v, cmap='gray')

    # As an example, a "night" image is loaded in and its avg brightness is displayed
    image_num = 190
    test_im = STANDARDIZED_LIST[image_num][0]

    avg = avg_brightness(test_im)
    print('Avg brightness: ' + str(avg))
    # plt.imshow(test_im)
    # plt.show()

    # Load test data
    TEST_IMAGE_LIST = helpers.load_dataset(image_dir_test)

    # Standardize the test data
    STANDARDIZED_TEST_LIST = helpers.standardize(TEST_IMAGE_LIST)

    # Shuffle the standardized test data
    random.shuffle(STANDARDIZED_TEST_LIST)

    MISCLASSIFIED = get_misclassified_images(STANDARDIZED_TEST_LIST)

    # Accuracy calculations
    total = len(STANDARDIZED_TEST_LIST)
    num_correct = total - len(MISCLASSIFIED)
    accuracy = num_correct / total

    print('Accuracy: ' + str(accuracy))
    print("Number of misclassified images = " + str(len(MISCLASSIFIED)) + ' out of ' + str(total))

    num = 0
    test_mis_im = MISCLASSIFIED[num][0]
    plt.imshow(test_mis_im)
    plt.show()
    print(str(MISCLASSIFIED[num][1]))