import os
import time
import math
import random
import torch
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda')
wd = os.getcwd()
model_path = os.path.join(wd,'Models')

def get_batch(dataset, current_index, batch):
    images,labels = zip(*dataset[current_index:current_index+batch])
    return torch.stack(images, dim = 0).to(device), torch.stack(labels).to(device)
    
def train_one_epoch(model, dataset, batch_size, optimizer, loss_fn):
    random.shuffle(dataset.train)
    batches = math.floor(len(dataset.train)/batch_size)
    current_index = 0
    running_loss = 0

    for i in range(batches):
        inputs,labels = get_batch(dataset.train, current_index, batch_size)
        current_index += batch_size
        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().item()

    remainder = len(dataset.train)%batch_size
    if remainder != 0:
        inputs,labels = get_batch(dataset.train, current_index, remainder)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.cpu().item()
        batches += 1
    
    return running_loss/batches

def validate(model, dataset, batch_size, loss_fn):
    batches = math.floor(len(dataset.valid)/batch_size)
    current_index = 0
    running_loss = 0
    
    for i in range(batches):
        inputs,labels = get_batch(dataset.valid, current_index, batch_size)
        current_index += batch_size
        output = model(inputs)
        running_loss += loss_fn(output, labels).cpu().item()

    remainder = len(dataset.valid)%batch_size
    if remainder != 0:
        inputs,labels = get_batch(dataset.valid, current_index, remainder)
        output = model(inputs)
        running_loss += loss_fn(output, labels).cpu().item()
        batches += 1
    
    return running_loss/batches

def train(epochs, model, dataset, batch_size, folder_name, patience, optimiser, loss_fn, scheduler):
    best_vloss = 1000
    current_vloss = 0
    epoch_counter = 1
    no_improve = 0
    resets = 0
    writer = SummaryWriter(log_dir = os.path.join(wd,'runs',folder_name))
    start = time.perf_counter()
    if not os.path.exists('{}\{}'.format(model_path,folder_name)):
        os.makedirs('{}\{}'.format(model_path,folder_name))
        
    for epoch in range(epochs):
        print('EPOCH {}:'.format(epoch_counter))
        model.train()
        train_loss = train_one_epoch(model, dataset, batch_size, optimiser, loss_fn)
        model.eval()
        with torch.no_grad():
            current_vloss = validate(model, dataset, batch_size, loss_fn)

        print('LOSS train {} valid {}'.format(train_loss, current_vloss))
        print('LR is {}%'.format(scheduler.get_last_lr()))
        writer.add_scalar("Loss/train", train_loss, epoch_counter)
        writer.add_scalar("Loss/valid", current_vloss, epoch_counter)

        if current_vloss <= best_vloss:
                best_vloss = current_vloss
                print('New best validation loss')
                path = '{}\\{}\\model_{}'.format(model_path, folder_name, epoch_counter)
                torch.save(model.state_dict(), path)
                no_improve = 0
        else:
            no_improve += 1
            
        if no_improve > patience:
            no_improve = 0
            if resets == 3:
                return
            else:
                resets += 1
            model.load_state_dict(torch.load(path, weights_only = True))
            print('Reset to best checkpoint')
            writer.add_text('Resets', 'Reset to previous checkpoint after epoch {}'.format(epoch_counter))
        scheduler.step(current_vloss)
        
        epoch_counter += 1

    end = time.perf_counter()
    writer.flush()
    writer.close()
    print('Average time taken per epoch is {}s'.format((end - start)/epochs))
    return

def compute_weights(dataset):
    counts = dataset.counts['train']
    total = sum(counts.values())
    n_samples = len(dataset.train)
    n_classes = len(dataset.classes)
    output = torch.zeros(n_classes)
    for i in range(n_classes):
        weight = n_samples/(n_classes*counts[i])
        output[i] = weight
    return output

