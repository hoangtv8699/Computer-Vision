import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == '__main__':
    # load in color image for face detection
    image = cv2.imread('multi_faces.jpg')

    # convert to RBG
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.imshow(image)

    # convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    plt.figure(figsize=(20, 10))
    plt.imshow(gray, cmap='gray')

    # load in cascade classifier
    face_cascade = cv2.CascadeClassifier('../../../detector_architectures/haarcascade_frontalface_default.xml')

    # run the detector on the grayscale image
    faces = face_cascade.detectMultiScale(gray, 4, 6)

    # print out the detections found
    print('We found ' + str(len(faces)) + ' faces in this image')
    print("Their coordinates and lengths/widths are as follows")
    print('=============================')
    print(faces)

    img_with_detections = np.copy(image)  # make a copy of the original image to plot rectangle detections ontop of

    # loop over our detections and draw their corresponding boxes on top of our original image
    for (x, y, w, h) in faces:
        # draw next detection as a red rectangle on top of the original image.
        # Note: the fourth element (255,0,0) determines the color of the rectangle,
        # and the final argument (here set to 5) determines the width of the drawn rectangle
        cv2.rectangle(img_with_detections, (x, y), (x + w, y + h), (255, 0, 0), 5)

        # display the result
    plt.figure(figsize=(20, 10))
    plt.imshow(img_with_detections)
    plt.show()