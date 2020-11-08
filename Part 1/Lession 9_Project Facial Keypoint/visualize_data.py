# import the required libraries
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import cv2
from load_dataset import FacialKeypointsDataset, Rescale, RandomCrop, Normalize, ToTensor
from torchvision import transforms, utils


def show_keypoints(image, key_pts):
    """Show image with keypoints"""
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')


if __name__ == '__main__':
    key_pts_frame = pd.read_csv('data/training_frames_keypoints.csv')

    n = 0
    image_name = key_pts_frame.iloc[n, 0]
    key_pts = key_pts_frame.iloc[n, 1:].to_numpy()
    key_pts = key_pts.astype('float').reshape(-1, 2)

    print('Image name: ', image_name)
    print('Landmarks shape: ', key_pts.shape)
    print('First 4 key pts: {}'.format(key_pts[:4]))

    # print out some stats about the data
    print('Number of images: ', key_pts_frame.shape[0])

    # # select an image by index in our data frame
    # n = 0
    # image_name = key_pts_frame.iloc[n, 0]
    # key_pts = key_pts_frame.iloc[n, 1:].to_numpy()
    # key_pts = key_pts.astype('float').reshape(-1, 2)
    #
    # plt.figure(figsize=(5, 5))
    # show_keypoints(mpimg.imread(os.path.join('data/training/', image_name)), key_pts)
    # plt.show()
    #
    # Construct the dataset
    face_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                          root_dir='data/training/')
    #
    # # print some stats about the dataset
    # print('Length of dataset: ', len(face_dataset))
    #
    # num_to_display = 3
    #
    # for i in range(num_to_display):
    #     # # define the size of images
    #     # fig = plt.figure(figsize=(20, 10))
    #
    #     # randomly select a sample
    #     rand_i = np.random.randint(0, len(face_dataset))
    #     sample = face_dataset[rand_i]
    #
    #     # print the shape of the image and keypoints
    #     print(i, sample['image'].shape, sample['keypoints'].shape)
    #
    #     ax = plt.subplot(1, num_to_display, i + 1)
    #     ax.set_title('Sample #{}'.format(i))
    #
    #     # Using the same display function, defined earlier
    #     show_keypoints(sample['image'], sample['keypoints'])
    #
    # plt.show()
    #
    # test out some of these transforms
    # rescale = Rescale(100)
    # crop = RandomCrop(50)
    # composed = transforms.Compose([Rescale(250),
    #                                RandomCrop(224)])
    #
    # # apply the transforms to a sample image
    # test_num = 500
    # sample = face_dataset[test_num]
    #
    # fig = plt.figure()
    # for i, tx in enumerate([rescale, crop, composed]):
    #     transformed_sample = tx(sample)
    #
    #     ax = plt.subplot(1, 3, i + 1)
    #     plt.tight_layout()
    #     ax.set_title(type(tx).__name__)
    #     show_keypoints(transformed_sample['image'], transformed_sample['keypoints'])
    #
    # plt.show()
    #
    # define the data tranform
    # order matters! i.e. rescaling should come before a smaller crop
    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)

    # print some stats about the transformed data
    print('Number of images: ', len(transformed_dataset))

    # make sure the sample tensors are the expected size
    for i in range(5):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())

