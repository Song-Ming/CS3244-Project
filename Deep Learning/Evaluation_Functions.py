import time
import math
import numpy as np
from Training_Functions import get_batch
import torch
from sklearn.metrics import confusion_matrix

def metrics(TP,FP,FN):
    if TP > 0:
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1  = TP / (TP + 0.5 * (FP + FN))
    else:
        precision = 0
        recall = 0
        F1  = 0
    return precision, recall, F1

def evaluate(model, dataset, batch_size, classes):
    start = time.perf_counter()
    total = 0
    correct = 0
    batches = math.floor(len(dataset) / batch_size)
    current_index = 0
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for i in range(batches):
        inputs, labels = get_batch(dataset, current_index, batch_size)
        current_index += batch_size
        prediction = torch.argmax(model(inputs),dim = 1)
        
        correct += sum(prediction == labels).item()
        total += batch_size

        all_preds.extend(prediction.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    remainder = len(dataset) % batch_size
    if remainder != 0:
        inputs, labels = get_batch(dataset, current_index, remainder)
        prediction = torch.argmax(model(inputs),dim = 1)
        
        correct += sum(prediction == labels).item()
        total += remainder

        all_preds.extend(prediction.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total

    # Compute Metrics for individual classes
    num_classes = len(classes) 
    conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    print(conf_matrix)

    for i in range(num_classes):
        TP = conf_matrix[i, i]  # True Positives
        FP = np.sum(conf_matrix[:, i]) - TP  # False Positives
        FN = np.sum(conf_matrix[i, :]) - TP  # False Negatives
        precision,recall,F1 = metrics(TP, FP, FN)
        print(f"Precision for class {classes[i]}: {precision:.4f}")
        print(f"Recall for class {classes[i]}: {recall:.4f}")
        print(f"F1-score for class {classes[i]}: {F1:.4f}")
    print('Total Accuracy is {}%'.format(accuracy))
    
    end = time.perf_counter()
    print('Time taken is {}'.format(end - start))