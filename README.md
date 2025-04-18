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
  <li> Inference Time: Time taken for model inference, as real-time weed detection is essential. </li>
</ul>

## Deep Learning
### Data Preparation
<ul>
  <li> Place the labels and images from the DeepWeeds repository into a folder together with the py files.
  <li> Process_Data splits and processes the dataset into training, validation, and test sets using a fixed seeed for reproduciblity. </li>
  <li> Data augmentation (Albumentations) is applied during this. </li>
  <li> The resulting sets are saved in Folder/Sets for image data, and Folder/Labels for class labels. The folders are automatically created. </li>
</ul>

### Model Training & Evaluation
<ul>
  <li> Swin_Training and Convnext_Training train the respective models using Pytorch SGD and cross entropy loss. </li>
  <li> The best checkpoints are automatically saved to Folder/Models/Folder_name while the summary data is saved to Folder/runs/Folder_name. </li>
  <li> Evaluation evaluates the best epochs of both the Swin and Convnext models. </li>
  <li> Training_Functions, Evaluation_Functions, Weeds_Dataset, Convnext_Model and Swin_Model contains the functions and classes needed for the above 3 files to run. </li>
</ul>

The entire training/test data is loaded into memory so the training files take about 20 GB RAM while the evaluation file takes about 4 GB RAM.

### Model Weights
The optimal trained model weights are stored on Google Drive: https://drive.google.com/drive/folders/1KAnvMIL_II11WkVqymGF3nXKtiPBoY2i?usp=drive_link.

### Model Setup
We used pre-trained ImageNet Pytorch implementations of Swin Transformer and ConvNeXt as base models, and replaced the original classifer with a custom Multi-Layer Perceptron (MLP) outputting 9 classes. <br>

We adopted partial fine-tuning where the lower layers of the models were frozen. Each model had around 49 million parameters total. Convnext was modified from the original Pytorch implementation with the addition of 3 squeeze and excitation blocks.

### Performance Metric
For Swin Transformer, we noted that the model achieved 97.03% top-1 accuracy, lowest F1 score of 0.9390, and an inference time of 6.403s. <br>

For ConvNeXt, we noted that the model achieved 95.34% top-1 accuracy, lowest F1 score of 0.8955 and an inference time of 5.176s. <br>

Both timings were done on an NVIDIA RTX 3070 GPU and AMD Ryzen 5 5600X CPU.

