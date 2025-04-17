# CS3244 Group 2 Project: Identifying 8 Types of Weeds

## Group Members
<ul>
  <li> Lean Ze She Andre (A0257880R) </li>
  <li> Muhammad Rusyaidi B Zullkafli (A0252282J) </li>
  <li> Sim Zi Ying (A0258709N) </li>
  <li> Tan Ze Qi Christian Lambayong (A0252452L) </li>
  <li> Tiffany Choong (A0257678H) </li>
  <li> Wong Song Ming (A0251855Y) </li>
</ul>


## Project Overview
This project aims to use machine learning techniques to automatically identify 8 types of weeds native to Australia. The dataset contains images of these weeds, as well as images with no weeds, and each image is labeled accordingly. <br>

The project will explore both deep learning (Transformer and CNN) and traditional machine learning approaches (SVM, Random Forest, KNN) for classifying images into respective weed categories.

## Dataset
The dataset used for this project contains images of 8 different types of weeds, as well as images where no weeds are present. The dataset is publicly available and comes from the DeepWeeds research paper published by Scientific Reports. <br>

<ul>
  <li> Dataset Source: https://github.com/AlexOlsen/DeepWeeds </li>
</ul>

## Motivation
Weeds pose a significant threat to agricultural productivity, biodiversity, and native ecosystems. In Australia, the spread of invasive weed species can negatively impact farming operations and land management. This project addresses the need for an automated weed detection system, which can help farmers reduce labor and time spent on weed management, ultimately improving productivity and reducing costs.

## Approach
Our approach to solving this problem involves the following steps:
<ol>
  <li> Data Augmentation: Enhancing the dataset with transformations like rotations and flipping to improve model generalization. </li>
  <li> Image Classification: Implementing models for identifying weeds in images. </li>
  <li> Model Evaluation: Comparing deep learning-based models (CNN, Transformer) with traditional machine learning models (SVM, Random Forest, KNN). </li>
</ol>

## Evaluation
To evaluate model performance, we will use the following metrics:
<ul>
  <li> Accuracy: The overall classification accuracy of the model. </li>
  <li> F1 Score: The harmonic mean of precision and recall for each class, especially useful given the class imbalance in the dataset. </li>
  <li> Training Time: Time taken for model inference, as real-time weed detection is essential. </li>
</ul>

## Deep Learning
### Data Preparation
<ul>
  <li> Directory Setup follows the structure as defined in directory_order file. </li>
  <li> Script: Process Data.ipynb 
    <li> Splits the dataset into training, validation and test sets with a fixed random seed for reproducibility. </li>
    <li> Applies offline data augmentation using Albumentations. </li>
    <li> Saves the processed data into: 
      <li> Weeds/Sets </li>
      <li> Weeds/Labels </li>
    </li>
</li>
</ul>

Place the images, labels and ipynb files according to the Directory order file. Process Data splits and augments the data with a set seed and saves the resulting training, validation and test sets in Weeds/Sets and Weeds/Labels. Swin.ipynb and Convnext.ipynb contain the training and evaluation of the Swin Transformer small and Convnext small models respectively. To speed up the process, the entire dataset is loaded into memory so the program will take about 18-20 GB of RAM. The records folder contains the training and validation loss of each epoch of training for the models. The optimal weights for each model is stored on google drive at 
https://drive.google.com/drive/folders/1KAnvMIL_II11WkVqymGF3nXKtiPBoY2i?usp=drive_link.

For both Swin and Convnext, the pretrained (imagenet) pytorch implementations were used as a base for finetuning. The classifier was replaced by a MLP with 9 output features representing the 9 classes. Most of the model was frozen except for the upper few layers. The total number of trainable parameters for both was around 14 million out of a total of 49 million. Swin Transformer achieved 96.23% top-1 accuracy while Convnext achieved 95.43% top-1 accuracy on the test set. The lowest F1 score for Swin was 0.9082 while the lowest F1 score for Convnext was 0.8609, both for class Chinee Apple. Total inference time at batch size 30 for 720 images was 2.516s and 1.882s respectively.
