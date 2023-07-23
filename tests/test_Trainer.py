import torch, torch.nn as nn, sys, pathlib
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

import torch.optim.lr_scheduler as lrsched
from torch.optim import Optimizer
from torch.utils.data import DataLoader,Subset
# Import mnist for tests
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms as t
from src.torchenhanced import Trainer, DevModule, ConfigModule
import wandb, os

curfold = pathlib.Path(__file__).parent

class LinSimple(ConfigModule):
    def __init__(self, hidden = 28*28, out =10):
        config = locals()
        del config['self']
        del config['__class__']

        super().__init__(config)

        self.layer = nn.Linear(hidden, out)
    
    def forward(self, x):
        return self.layer(x)
    
class LinearTrainer(Trainer):

    def __init__(self, run_name: str = None, project_name: str = None, state_save_loc=None,model_save_loc=None):

        super().__init__(LinSimple(), run_name=run_name, project_name=project_name,state_save_loc=state_save_loc,model_save_loc=model_save_loc)

        self.dataset =Subset(MNIST(os.path.join(curfold,'data'),download=True,transform=t.ToTensor()),range(100))
    
    def get_loaders(self, batch_size):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=True), None
    
    def process_batch(self, batch_data, data_dict: dict, **kwargs):
        x, y = batch_data
        x = x.reshape((x.shape[0],-1))
        
        pred = self.model(x) # (B,10)
        loss = F.cross_entropy(pred,y,reduction='mean') 

        if(data_dict['time']%data_dict['batch_log']==data_dict['batch_log']-1):
            wandb.log({'loss':loss.item()}, step=data_dict['time'])
        
        return loss, data_dict

# FOR MANUAL TESTING, COULDN'T FIGURE OUT HOW TO AUTOMATE IT
# trainer = LinearTrainer(run_name='test_broken_question', project_name='AnewDawn', state_save_loc=os.path.join(curfold), model_save_loc=os.path.join(curfold))
# trainer.load_state(os.path.join(curfold,'LinSimple_state/test_broken_question'))
# trainer.train_epochs(epochs=5, batch_size=4, batch_log=500, save_every=1, val_log=2)

def test_Trainer_config():
    ma = LinSimple(hidden=32,out=15)

    config = ma.config

    assert config == {'hidden':32, 'out':15, 'name':'LinSimple'}, f"Invalid config : {config}"

# Probably need to add more unit_tests...

