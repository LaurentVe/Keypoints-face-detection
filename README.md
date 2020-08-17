# Keypoints-face-detection
detection of 68 facial keypoints from images using Deep CNN

This project is the first project from Udacity's Computer Vision Nanodegree.

The objective is to train a deep neural network to detect and predict 68 facial keypoints from a picture. Facial keypoints include points around the eyes, nose, and mouth on a face and are used in many applications. These applications include: facial tracking, facial pose recognition, facial filters, and emotion recognition. Your completed code should be able to look at any image, detect faces, and predict the locations of facial keypoints on each face.

[](/asset/key_pts_example.png)

The CNN takes any images as an input, pre-process them (converting to grayscale and 224x224) and outputs 68 coordinate tuples, one for each keypoint. The training is performed on a set of images taken from [YouTube Faces Dataset](https://www.cs.tau.ac.il/~wolf/ytfaces/). This facial keypoints dataset consists of 5770 color images. All of these images are separated into either a training or a test set of data.

The project comprises 4 notebooks:
- Notebook 1 : Loading and Visualizing the Facial Keypoint Data
- Notebook 2 : Defining and Training a Convolutional Neural Network (CNN) to Predict Facial Keypoints
- Notebook 3 : Facial Keypoint Detection Using Haar Cascades and your Trained CNN
- Notebook 4 : Fun Filters and Keypoint Uses

The CNN model architecture is defined in the `models.py` file.
The architecture I used was inpired by this [paper](https://arxiv.org/pdf/1710.00977.pdf).

