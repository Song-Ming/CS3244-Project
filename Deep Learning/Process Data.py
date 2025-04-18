import numpy as np
import pandas as pd
import os
import shutil
import random
from PIL import Image
import albumentations as A
import csv

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(600)
wd = os.getcwd()
img_dir = os.path.join(wd,'images')

class WeedsDataset():
    def __init__(self, labels, img_dir):
        self.labels = pd.read_csv(labels)
        self.img_dir = img_dir
        self.classes = self.labels[['Label','Species']].drop_duplicates().sort_values(by = 'Label').reset_index(drop = True)['Species']
        self.data = []
        self.train = []
        self.validation = []
        self.test = []
        
    # Load dataset
    def __loaddata__(self):
        for row in self.labels.itertuples():
            filename = row.Filename
            label = row.Label
            img_path = os.path.join(self.img_dir, filename)
            image = np.asarray(Image.open(img_path))
            self.data.append([image,label,filename])
        del self.labels
        print('Data has been loaded')

    # Split into train/valid/test
    def __split__(self):
        random.shuffle(self.data)
        n = len(self.data)
        n_train = round(n * 0.8)
        n_test = round(n * 0.1)
        
        self.train = self.data[:n_train]
        self.test = self.data[n_train:n_train + n_test]
        self.valid = self.data[n_train + n_test:]
        del self.data
        
        print('Data has been split')
        print('Size of training set is {}'.format(len(self.train)))
        print('Size of validation set is {}'.format(len(self.valid)))
        print('Size of test set is {}'.format(len(self.test)))
                
    # Augment training data
    def __augment__(self,frac,transform):
        n = round(len(self.train)*frac/2)
        sample1 = random.sample(self.train,n)
        sample2 = random.sample(self.train,n)
        
        for i in range(len(sample1)):
            transformed = transform(image = sample1[i][0])
            self.train.append([transformed['image'],sample1[i][1],sample1[i][2]])   
        for i in range(len(sample2)):
            transformed = transform(image = sample2[i][0])
            self.train.append([transformed['image'],sample2[i][1],sample2[i][2]])   
    
        print('Data has been augmented')
        print('Size of training set is {}'.format(len(self.train)))

    # Create folders for saved data, delete existing files
    def __folders__(self):
        if os.path.exists(os.path.join(wd,'Sets')):
            shutil.rmtree(os.path.join(wd,'Sets'))
        os.makedirs(os.path.join(wd,'Sets','train'))
        os.makedirs(os.path.join(wd,'Sets','valid'))
        os.makedirs(os.path.join(wd,'Sets','test'))
            
        if os.path.exists(os.path.join(wd,'Labels')):
            shutil.rmtree(os.path.join(wd,'Labels'))
        os.makedirs(os.path.join(wd,'Labels'))

    # Save images and labels
    def __save__(self):
        lst = [['Filename','Label','Class','Original name']]
        for i in range(len(self.train)):
            image = Image.fromarray(self.train[i][0])
            label = self.train[i][1]
            filename = self.train[i][2]
            path = os.path.join(wd,'Sets','train')
            image.save(os.path.join(path,'image_{}.jpg'.format(i)))
            lst.append(['image_{}.jpg'.format(i),label,self.classes[label],filename])
        with open(os.path.join(wd,'Labels','train.csv'), 'w', newline='') as traincsv:
            writer = csv.writer(traincsv)
            writer.writerows(lst)

        lst = [['Filename','Label','Class','Original name']]
        for i in range(len(self.valid)):
            image = Image.fromarray(self.valid[i][0])
            label = self.valid[i][1]
            filename = self.valid[i][2]
            path = os.path.join(wd,'Sets','valid')
            image.save(os.path.join(path,'image_{}.jpg'.format(i)))
            lst.append(['image_{}.jpg'.format(i),label,self.classes[label],filename])
        with open(os.path.join(wd,'Labels','valid.csv'), 'w', newline='') as validcsv:
            writer = csv.writer(validcsv)
            writer.writerows(lst)

        lst = [['Filename','Label','Class','Original name']]
        for i in range(len(self.test)):
            image = Image.fromarray(self.test[i][0])
            label = self.test[i][1]
            filename = self.test[i][2]
            path = os.path.join(wd,'Sets','test')
            image.save(os.path.join(path,'image_{}.jpg'.format(i)))
            lst.append(['image_{}.jpg'.format(i),label,self.classes[label],filename])
        with open(os.path.join(wd,'Labels','test.csv'), 'w', newline='') as testcsv:
            writer = csv.writer(testcsv)
            writer.writerows(lst)
        print('Data has been saved')

weeds = WeedsDataset(os.path.join(wd,'Labels.csv'),img_dir)
weeds.__loaddata__()
weeds.__split__()
transform = A.Compose([A.RandomRotate90(p=1),
                       A.HorizontalFlip(p=0.5),
                       A.Affine(scale=(0.8,1.2),keep_ratio=True,p=1),
                       A.ColorJitter(brightness=0.4,contrast=0.4,saturation=0.2,hue=0,p=1)])
weeds.__augment__(1,transform)
weeds.__folders__()
weeds.__save__()