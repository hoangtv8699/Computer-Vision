import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
from models import Net

import cv2

if __name__ == '__main__':
    # load in color image for face detection
    image = cv2.imread('images/obamas.jpg')

    # switch red and blue color channels
    # --> by default OpenCV assumes BLUE comes first, not RED as in many images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # plot the image
    # fig = plt.figure(figsize=(9, 9))
    # plt.imshow(image)
    # plt.show()

    # load in a haar cascade classifier for detecting frontal faces
    face_cascade = cv2.CascadeClassifier('../../detector_architectures/haarcascade_frontalface_default.xml')

    # run the detector
    # the output here is an array of detections; the corners of each detection box
    # if necessary, modify these parameters until you successfully identify every face in a given image
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    # make a copy of the original image to plot detections on
    image_with_detections = image.copy()

    # loop over the detected faces, mark the image where each face is found
    for (x, y, w, h) in faces:
        # draw a rectangle around each detected face
        # you may also need to change the width of the rectangle drawn depending on image resolution
        cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)

    fig = plt.figure(figsize=(9, 9))

    plt.imshow(image_with_detections)
    plt.show()

    # load saved model
    net = Net()
    net.load_state_dict(torch.load('saved_models/keypoints_model_1.pt'))
    print(net.eval())

    image_copy = np.copy(image)

    # loop over the detected faces from your haar cascade
    for (x, y, w, h) in faces:
        # Select the region of interest that is the face in the image
        face = image_copy[y:y + h, x:x + w]

        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        ## TODO: Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]

        ## TODO: Rescale the detected face to be the expected square size for your CNN (224x224, suggested)

        ## TODO: Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)

        ## TODO: Make facial keypoint predictions using your loaded, trained network
        ## perform a forward pass to get the predicted facial keypoints

        ## TODO: Display each detected face and the corresponding keypoints

        # plot the image
        fig = plt.figure(figsize=(9, 9))
        plt.imshow(image)
        plt.show()