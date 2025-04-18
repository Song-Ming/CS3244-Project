import pickle
from Evaluation_Functions import *
from Weeds_Dataset import *
from Convnext_Model import model as convnext_model,transform as convnext_transform
from Swin_Model import model as swin_model,transform as swin_transform

# Set Directories
wd = os.getcwd()
labels_dir = os.path.join(wd,'Labels')
img_dir = os.path.join(wd,'Sets')
model_path = os.path.join(wd,'Models')
            
c_dataset = WeedsDataset(labels_dir,img_dir,convnext_transform,'test')
c_dataset.__loaddata__()
c_dataset.__apply__()
            
s_dataset = WeedsDataset(labels_dir,img_dir,swin_transform,'test')
s_dataset.__loaddata__()
s_dataset.__apply__()

# Load checkpoints
best_path_c = os.path.join(model_path,'convnext','model_35')
best_path_d = os.path.join(model_path,'swin','model_47')
convnext_model.load_state_dict(torch.load(best_path_c, weights_only = True))
swin_model.load_state_dict(torch.load(best_path_d, weights_only = True))

# Evaluate
print('Convnext Model Results:')
with torch.no_grad():
    evaluate(convnext_model,c_dataset.test,32,c_dataset.classes)
print('Swin Model Results:')
with torch.no_grad():
    evaluate(swin_model,s_dataset.test,32,s_dataset.classes)