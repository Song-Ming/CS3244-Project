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
This project aims to develop a system for automated weed detection using image classification techniques to identify 8 types of weeds native to Australia. The dataset contains images of these weeds, as well as images with no weeds, and each image is labeled accordingly. <br>

The project will explore both deep learning (Swin Model and CNN) and traditional machine learning approaches (SVM, Random Forest, KNN) for classifying images into respective weed categories.

## Dataset
The dataset used for this project contains images of 8 different types of weeds, as well as images where no weeds are present. The dataset is publicly available and comes from a research paper published by Scientific Reports. The data can be considered reliable as it comes from the authors' official GitHub repository. <br>

<ul>
  <li> Dataset Source: https://github.com/AlexOlsen/DeepWeeds </li>
  <li> Dataset Link: https://www.tensorflow.org/datasets/catalog/deep_weeds </li>
</ul>

## Motivation
Weeds pose a significant threat to agricultural productivity, biodiversity, and native ecosystems. In Australia, the spread of invasive weed species can negatively impact farming operations and land management. This project addresses the need for an automated weed detection system, which can help farmers reduce labor and time spent on weed management, ultimately improving productivity and reducing costs.

## Approach
Our approach to solving this problem involves the following key steps:
<ol>
  <li> Image Classification: Implementing models for identifying weeds in images. </li>
  <li> Data Augmentation: Enhancing the dataset with transformations like RandomCrop, ColorJitter, and Gaussian Noise to improve model generalization. </li>
  <li> Model Evaluation: Comparing deep learning-based models (CNN, Transformer) with traditional machine learning models (SVM, Random Forest, KNN). </li>
</ol>

## Evaluation
To evaluate model performance, we will use the following metrics:
<ul>
  <li> Accuracy: The overall classification accuracy of the model. </li>
  <li> F1 Score: The harmonic mean of precision and recall, especially useful given the class imbalance in the dataset. </li>
  <li> Training Time: Time taken for model training, as real-time weed detection is essential. </li>
</ul>
We will use micro-averaging to handle the class imbalance and ensure fair evaluation across multiple classes.

## Resources
<ol>
  <li> PyTorch Documentation on Transformation and Augmentation: https://pytorch.org/vision/main/transforms.html </li>
</ol>
