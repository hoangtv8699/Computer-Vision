import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# the dataset we created in Notebook 1 is copied in the helper file `data_load.py`
from load_dataset import FacialKeypointsDataset
# the transforms we defined in Notebook 1 are in the helper file `data_load.py`
from load_dataset import Rescale, RandomCrop, Normalize, ToTensor

from models import Net


# test the model on a batch of test images
def net_sample_output(test_loader, net):
    # iterate through the test dataset
    for i, sample in enumerate(test_loader):

        # get sample data: images and ground truth keypoints
        images = sample['image']
        key_pts = sample['keypoints']

        # convert images to FloatTensors
        images = images.type(torch.cuda.FloatTensor)

        # forward pass to get net output
        output_pts = net(images)

        # reshape to batch_size x 68 x 2 pts
        output_pts = output_pts.view(output_pts.size()[0], 68, -1)

        # break after first image is tested
        if i == 0:
            return images, output_pts, key_pts


def show_all_keypoints(image, predicted_key_pts, gt_pts=None):
    """Show image with predicted keypoints"""
    # image is grayscale
    plt.imshow(image, cmap='gray')
    plt.scatter(predicted_key_pts[:, 0], predicted_key_pts[:, 1], s=20, marker='.', c='m')
    # plot ground truth points as green pts
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    plt.figure(figsize=(10, 10))
    for i in range(batch_size):
        ax = plt.subplot(1, batch_size, i + 1)

        # un-transform the image data
        image = test_images[i].data  # get the image from it's wrapper
        image = image.cpu().numpy()  # convert to numpy array from a Tensor
        image = np.transpose(image, (1, 2, 0))  # transpose to go from torch to numpy image

        # un-transform the predicted key_pts data
        predicted_key_pts = test_outputs[i].data
        predicted_key_pts = predicted_key_pts.cpu().numpy()
        # undo normalization of keypoints
        predicted_key_pts = predicted_key_pts * 50.0 + 100

        # plot ground truth points for comparison, if they exist
        ground_truth_pts = None
        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts * 50.0 + 100

        # call show_all_keypoints
        show_all_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)

        plt.axis('off')

    plt.show()


def train_net(n_epochs, criterion, optimizer):
    # prepare the net for training
    net.train()
    # to track the loss as the network trains
    loss_over_time = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        ep_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image']
            key_pts = data['keypoints']

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.cuda.FloatTensor)
            images = images.type(torch.cuda.FloatTensor)

            # forward pass to get outputs
            output_pts = net(images)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            ep_loss += loss.item()
            if batch_i % 10 == 9:  # print every 10 batches
                avg_loss = running_loss/10
                loss_over_time.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                running_loss = 0.0
        lr_scheduler.step(ep_loss/len(train_loader))
    print('Finished Training')
    return loss_over_time


if __name__ == '__main__':
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(device)
    net = Net()
    net.to(device)
    print(net)

    # define the data_transform
    data_transform = transforms.Compose([Rescale(250),
                                         RandomCrop(224),
                                         Normalize(),
                                         ToTensor()])

    # create the transformed dataset
    transformed_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=data_transform)
    test_dataset = FacialKeypointsDataset(csv_file='data/test_frames_keypoints.csv',
                                          root_dir='data/test/',
                                          transform=data_transform)

    print('Number of images: ', len(transformed_dataset))

    # iterate through the transformed dataset and print some stats about the first few samples
    for i in range(4):
        sample = transformed_dataset[i]
        print(i, sample['image'].size(), sample['keypoints'].size())

    # load training data in batches
    batch_size = 10

    train_loader = DataLoader(transformed_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    # test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)
    #
    # # print out the dimensions of the data to see if they make sense
    # print(test_images.data.size())
    # print(test_outputs.data.size())
    # print(gt_pts.size())
    #
    # visualize_output(test_images, test_outputs, gt_pts)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=5, min_lr=0.0001, verbose=True)

    n_epochs = 100  # start small, and increase when you've decided on your model structure and hyperparams

    training_loss = train_net(n_epochs, criterion, optimizer)

    model_dir = 'saved_models/'
    model_name = 'keypoints_model_5_ep.pt'

    # after training, save your model parameters in the dir 'saved_models'
    torch.save(net.state_dict(), model_dir + model_name)

    plt.plot(training_loss)
    plt.xlabel('1000\'s of batches')
    plt.ylabel('loss')
    plt.ylim(0, 2.5)  # consistent scale
    plt.show()

    # get a sample of test data again
    test_images, test_outputs, gt_pts = net_sample_output(test_loader, net)

    print(test_images.data.size())
    print(test_outputs.data.size())
    print(gt_pts.size())
    visualize_output(test_images, test_outputs, gt_pts)

