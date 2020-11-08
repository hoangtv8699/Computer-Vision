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

    # # make a copy of the original image to plot detections on
    # image_with_detections = image.copy()
    #
    # # loop over the detected faces, mark the image where each face is found
    # for (x, y, w, h) in faces:
    #     # draw a rectangle around each detected face
    #     # you may also need to change the width of the rectangle drawn depending on image resolution
    #     cv2.rectangle(image_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 3)
    #
    # fig = plt.figure(figsize=(9, 9))
    #
    # plt.imshow(image_with_detections)
    # plt.show()

    # load saved model
    net = Net()
    net.load_state_dict(torch.load('saved_models/keypoints_model_5_ep.pt'))
    print(net.eval())

    image_copy = np.copy(image)

    # loop over the detected faces from your haar cascade
    for (x, y, w, h) in faces:
        # Select the region of interest that is the face in the image
        face = image_copy[y - int(0.25 * h):y + int(1.25 * h), x - int(0.25 * w):x + int(1.25 * w)]

        origin_face = np.copy(face)
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        # Normalize the grayscale image so that its color range falls in [0,1] instead of [0,255]
        face = face / 255.0

        # Rescale the detected face to be the expected square size for your CNN (224x224, suggested)
        intput_size = (224, 224)
        face = cv2.resize(face, intput_size)
        # Reshape the numpy image shape (H x W x C) into a torch image shape (C x H x W)
        face = np.reshape(face, (1, 1, 224, 224))

        face_torch = torch.from_numpy(face)
        face_torch = face_torch.type(torch.FloatTensor)

        # Make facial keypoint predictions using your loaded, trained network
        # perform a forward pass to get the predicted facial keypoints
        output_pts = net(face_torch)
        print(output_pts)
        # reshape to 1 x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)
        output_pts = output_pts.detach().numpy()
        # undo normalization of keypoints
        output_pts = output_pts * 50.0 + 100
        print(output_pts)
        # Display each detected face and the corresponding keypoints
        plt.imshow(origin_face)
        plt.scatter(output_pts[0, :, 0], output_pts[0, :, 1], s=20, marker='.', c='m')
        plt.show()