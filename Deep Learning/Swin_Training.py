from Weeds_Dataset import *
from Training_Functions import *
from Swin_Model import *

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
set_seed(600)

wd = os.getcwd()
labels_dir = os.path.join(wd,'Labels')
img_dir = os.path.join(wd,'Sets')
model_path = os.path.join(wd,'Models')

weeds = WeedsDataset(labels_dir,img_dir,transform,'train')
weeds.__loaddata__()
weeds.__apply__()

patience = 8
weights = compute_weights(weeds)
print(weights)
loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
optimiser = torch.optim.SGD(model.parameters(), lr = 0.0006, momentum = 0.8, weight_decay = 0.00001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor = 0.5, patience = patience)

train(50,model,weeds,32,'swin',patience,optimiser,loss_fn,scheduler)




