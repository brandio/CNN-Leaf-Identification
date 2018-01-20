# Author : Branden Strochinsky

# This script formats the LeafSnap data for a cnn and saves/loads it from file.
# Usage : Download the LeafSnap dataset and make sure the path is setup correctly in the script.

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
from random import *
from matplotlib import pyplot as plt
from scipy import misc as scipy_misc
import os

IMAGE_SIZE = 50
species_dict = {"Ulmus pumila" : 0, "Prunus virginiana" : 1, "Cryptomeria japonica": 2, "Aesculus pavi": 3, "Styrax japonica": 4,
                "Quercus muehlenbergii": 5, "Prunus sargentii": 6, "Juglans cinerea": 7,
                "Salix nigra": 8, "Koelreuteria paniculata" : 9}

# This should point to the Leafsnap dataset!!
dir_path = os.getcwd() + "\\leafs"

class LeafImage:
    def __init__(self,id,path,segmentedPath,species,source):
        self.path = path
        self.id = id
        self.segmentedPath = segmentedPath
        self.source = source
        species_array = np.zeros(10)
        species_array[species_dict[species]] = 1.0
        self.species = species_array
        self.set_image()

    def set_image(self):
        path = ('\\' + self.path.replace('/','\\'))
        segmented_path = ('\\' + self.segmentedPath.replace('/','\\'))

        image = mpimg.imread((dir_path + path))
        image.setflags(write=True)
        segmentedImage = mpimg.imread((dir_path + segmented_path))

        plt.imshow(image)
        plt.show()
        plt.close()

        plt.imshow(segmentedImage)
        plt.show()
        plt.close()

        image = scipy_misc.imresize(image, (segmentedImage.shape[0], segmentedImage.shape[1]), interp='bilinear')

        for x in range(segmentedImage.shape[0]):
            for y in range(segmentedImage.shape[1]):
                if segmentedImage[x,y] < 0.9:
                    image[x,y,0] = 255
                    image[x, y, 1] = 255
                    image[x, y, 2] = 255

        image = scipy_misc.imresize(image,(IMAGE_SIZE,IMAGE_SIZE),interp='bilinear')

        self.rotated = 0
        self.mirrored = 0
        rotatedImage = image
        if randint(0, 100) > 50:
            for i in range(randint(1,3)):
                rotatedImage = np.rot90(image)
            self.rotated_image = rotatedImage.flatten()
            self.rotated = 1

        if randint(0,100) > 50:
            mirroredImage = image
            axis = randint(0,1)
            mirroredImage = np.flip(mirroredImage,axis)
            self.mirrored_image = mirroredImage.flatten()
            self.mirrored = 1
        self.image = image.flatten()


def load_leaf_data():
    train_array = np.load('train_images1.npy')
    test_array = np.load('test_images1.npy')
    train_labels_array = np.load('train_labels1.npy')
    test_labels_array =  np.load('test_labels1.npy')

    train_dataset = {"images" : train_array, "labels" : train_labels_array}
    test_dataset = {"images": test_array, "labels": test_labels_array}
    dataset = (train_dataset, test_dataset)
    return dataset

def format_leaf_data():
    print("loading and processing leaf data")
    fileName = 'D:\machineLearning\leafs\leafsnap-dataset-images.txt'
    file = open(fileName, 'r')
    text = file.read()
    file.close()
    text = text.replace("\n","\t")
    leafImagesText = text.split("\t")
    index = 5
    train_leaf_images = []
    train_leaf_labels = []
    test_leaf_images = []
    test_leaf_labels = []
    validate_leaf_images = []
    validate_leaf_labels = []
    halfway = False
    while index < len(leafImagesText) - 4:
        if(index > len(leafImagesText)/2 and halfway == False):
            print("halfway done")
            halfway = True
        if leafImagesText[index + 3] in species_dict and leafImagesText[index + 4] == 'field':
            leaf = LeafImage(leafImagesText[index], leafImagesText[index + 1], leafImagesText[index + 2],
                      leafImagesText[index + 3], leafImagesText[index + 4])

            random_int = randint(0,100)
            if random_int > 80:
                if random_int > 101:
                    validate_leaf_images.append(leaf.image)
                    validate_leaf_labels.append(leaf.species)
                else:
                    test_leaf_images.append(leaf.image)
                    test_leaf_labels.append(leaf.species)
                    if(leaf.rotated):
                        test_leaf_images.append(leaf.rotated_image)
                        test_leaf_labels.append(leaf.species)
                    if(leaf.mirrored):
                        test_leaf_images.append(leaf.mirrored_image)
                        test_leaf_labels.append(leaf.species)
            else:
                train_leaf_images.append(leaf.image)
                train_leaf_labels.append(leaf.species)
                if(leaf.rotated):
                    train_leaf_images.append(leaf.rotated_image)
                    train_leaf_labels.append(leaf.species)
                if(leaf.mirrored):
                    train_leaf_images.append(leaf.mirrored_image)
                    train_leaf_labels.append(leaf.species)

        index = index + 5

    train_array = np.array(train_leaf_images)
    train_labels_array = np.array(train_leaf_labels)

    test_array = np.array(test_leaf_images)
    test_labels_array = np.array(test_leaf_labels)

    validate_array = np.array(validate_leaf_images)
    validate_labels_array = np.array(validate_leaf_labels)


    train_dataset = {"images" : train_array, "labels" : train_labels_array}
    test_dataset = {"images": test_array, "labels": test_labels_array}
    validate_dataset = {"images": validate_array, "labels": validate_labels_array}
    dataset = (train_dataset, test_dataset)

    # Where the data gets save to.... if these names change change the names in load_leaf_data
    np.save('train_images1.npy', train_array)
    np.save('train_labels1.npy', train_labels_array)
    np.save('test_images1.npy', test_array)
    np.save('test_labels1.npy', test_labels_array)

    print("datasets created")
    return dataset

#Uncomment and run this to format the LeafSnap
#format_leaf_data()