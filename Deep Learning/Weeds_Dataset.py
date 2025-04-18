import os
import pandas as pd
import torch
import torchvision

class WeedsDataset():
    def __init__(self, labels_dir, img_dir, transform, setting):
        self.train_labels = pd.read_csv(os.path.join(labels_dir,'train.csv'))
        self.valid_labels = pd.read_csv(os.path.join(labels_dir,'valid.csv'))
        self.test_labels = pd.read_csv(os.path.join(labels_dir,'test.csv'))
        self.img_dir = img_dir
        self.classes = None
        self.counts = None
        self.train = []
        self.valid = []
        self.test = []
        self.setting = setting
        self.transform = transform
        
    # Load dataset
    def __loaddata__(self):
        self.classes = self.test_labels[['Label','Class']].drop_duplicates().sort_values(by = 'Label').reset_index(drop = True)['Class']
        if self.setting == 'train':
            self.counts = {'train':{},'valid':{}}
            for key in self.classes.keys():
                self.counts['train'][key] = 0
                self.counts['valid'][key] = 0
            for row in self.train_labels.itertuples():
                filename = row.Filename
                label = row.Label
                self.counts['train'][row.Label] += 1
                img_path = os.path.join(self.img_dir,'train',filename)
                image = torchvision.io.read_image(img_path)
                self.train.append([image,label])   
            for row in self.valid_labels.itertuples():
                filename = row.Filename
                label = row.Label
                self.counts['valid'][row.Label] += 1
                img_path = os.path.join(self.img_dir,'valid',filename)
                image = torchvision.io.read_image(img_path)
                self.valid.append([image,label])  
            del self.train_labels,self.valid_labels,self.test_labels
            print('Training data has been loaded') 
        
        elif self.setting == 'test':
            self.counts = {'test':{}}
            for key in self.classes.keys():
                self.counts['test'][key] = 0
            for row in self.test_labels.itertuples():
                filename = row.Filename
                label = row.Label
                self.counts['test'][row.Label] += 1
                img_path = os.path.join(self.img_dir,'test',filename)
                image = torchvision.io.read_image(img_path)
                self.test.append([image,label])
            del self.train_labels,self.valid_labels,self.test_labels
            print('Test data has been loaded')

    def __apply__(self):
        if self.setting == 'train':
            for i in range(len(self.train)):
                image = self.transform(self.train[i][0])
                label = torch.tensor(self.train[i][1])
                self.train[i] = [image,label]
            for i in range(len(self.valid)):
                image = self.transform(self.valid[i][0])
                label = torch.tensor(self.valid[i][1])
                self.valid[i] = [image,label]
            print('Data has been processed')
        elif self.setting == 'test':
            for i in range(len(self.test)):
                image = self.transform(self.test[i][0])
                label = torch.tensor(self.test[i][1])
                self.test[i] = [image,label]
            print('Data has been processed')