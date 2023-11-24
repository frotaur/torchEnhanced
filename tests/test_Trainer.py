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
import  os,time

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
    def __init__(self, run_name: str = None, project_name: str = None, state_save_loc=None,reach_plateau=100,run_config={}):
        model = LinSimple()
        opti = torch.optim.Adam(model.parameters(),lr=1e-3)
        schedo = lrsched.LinearLR(opti,start_factor=0.01,end_factor=1,total_iters=reach_plateau)

        super().__init__(model,optim=opti,scheduler=schedo, run_name=run_name, 
                         project_name=project_name,state_save_loc=state_save_loc,run_config=run_config)

        self.dataset =Subset(MNIST(os.path.join(curfold,'data'),download=True,transform=t.ToTensor()),range(100))
    
    def get_loaders(self, batch_size, num_workers=0):
        self.loss_val = []
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=False), DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
    
    def process_batch(self, batch_data,**kwargs):
        x, y = batch_data
        x = x.reshape((x.shape[0],-1))
        
        pred = self.model(x) # (B,10)
        loss = F.cross_entropy(pred,y,reduction='mean') 

        # assert self.stepnum==data_dict['stepnum'], f"Stepnum mismatch {self.stepnum} vs {data_dict['stepnum']}"
        # assert self.step_log==data_dict['step_log'], f"Step_log mismatch {self.step_log} vs {data_dict['step_log']}"
        # assert self.epoch == data_dict['epoch'], f"Epoch mismatch {self.epoch} vs {data_dict['epoch']}"
        # assert self.batchnum == data_dict['batchnum'], f"Batchnum mismatch {self.batchnum} vs {data_dict['batchnum']}"

        if(self.do_batch_log):
            self.logger.log({'lossme/train':loss.item()},commit=False)
            self.logger.log({'other/lr':self.scheduler.get_last_lr()[0]},commit=False)
        
        return loss
    
    def process_batch_valid(self, batch_data):
        x, y = batch_data
        x = x.reshape((x.shape[0],-1))
        
        pred = self.model(x) # (B,10)
        loss = F.cross_entropy(pred,y,reduction='mean') 

        self.loss_val.append(loss.item())

        return loss
    
    def valid_log(self):
        self.logger.log({'lossme/valid':sum(self.loss_val)/len(self.loss_val)},commit=False)
        self.loss_val = []
        

# FOR MANUAL TESTING, COULDN'T FIGURE OUT HOW TO AUTOMATE IT
#EPOCHS :
# trainer = LinearTrainer(run_name='test_epochs', project_name='test_torchenhanced', 
#                         state_save_loc=os.path.join(curfold),reach_plateau=200, run_config={'manamajeff':True})
# trainer.change_lr(1e-4)
# if(os.path.exists(os.path.join(curfold,'test_torchenhanced','state','test_epochs.state'))):
#     trainer.load_state(os.path.join(curfold,'test_torchenhanced','state','test_epochs.state'))
# time.sleep(2)

# trainer.train_epochs(epochs=100, batch_size=10, step_log=30, save_every=60,aggregate=2,batch_sched=True)

# #STEPS :
# trainer = LinearTrainer(run_name='test_steps', project_name='test_torchenhanced', state_save_loc=os.path.join(curfold),reach_plateau=1500)

# trainer.change_lr(1e-4)
# if(os.path.exists(os.path.join(curfold,'test_torchenhanced','state','test_steps.state'))):
#     trainer.load_state(os.path.join(curfold,'test_torchenhanced','state','test_steps.state'))

# time.sleep(2)
# trainer.train_steps(steps=1000, batch_size=10, step_log=30, save_every=60,aggregate=2, valid_every=100)


def test_resume_train():
    trainer = LinearTrainer(run_name='test_resume_train', project_name='test_torchenhanced', state_save_loc=os.path.join(curfold),reach_plateau=1500)
    trainer.change_lr(1e-4)
    trainer.train_steps(steps=1000, batch_size=200, step_log=30, save_every=60,aggregate=2, valid_every=100)
    trainer.save_state()
    print(f'Finished at batches : {trainer.batches}, epochs : {trainer.epochs}, steps : {trainer.steps_done}')
    
    # Test with resuming
    trainer2 = LinearTrainer(run_name='test_with_resume', project_name='test_torchenhanced', state_save_loc=os.path.join(curfold),reach_plateau=1500)
    trainer2.change_lr(1e-4)
    trainer2.load_state(os.path.join(curfold,'test_torchenhanced','state','test_resume_train.state')) # Load the state
    trainer2.train_steps(steps=200, batch_size=10, step_log=30, save_every=3000,aggregate=2, valid_every=100, resume_batches=True)

    # Test without resuming
    trainer3 = LinearTrainer(run_name='test_without_resume', project_name='test_torchenhanced', state_save_loc=os.path.join(curfold),reach_plateau=1500)
    trainer3.change_lr(1e-4)
    trainer3.load_state(os.path.join(curfold,'test_torchenhanced','state','test_resume_train.state')) # Load the state
    trainer3.train_steps(steps=200, batch_size=10, step_log=30, save_every=3000,aggregate=2, valid_every=100)

    assert trainer2.batches == trainer3.batches, f"Batch mismatch : {trainer2.batches} vs {trainer.batches}"
    assert trainer2.epochs == trainer3.epochs, f"Epoch mismatch : {trainer2.epochs} vs {trainer.epochs}"
    assert trainer2.steps_done == trainer3.steps_done, f"Step mismatch : {trainer2.steps_done} vs {trainer.steps_done}"

    # Look on wandb to see if they are the same


def test_Save_Weights():
    lintra = LinearTrainer(run_name='test_save_weights', project_name='AnewDawn', state_save_loc=os.path.join(curfold))
    lintra.save_state()

    Trainer.save_model_from_state(state_path=os.path.join(curfold,'AnewDawn','state','test_save_weights.state'),
                                 save_dir=os.path.join(curfold,'AnewDawn'),name='testJEFF')

    assert os.path.isfile(os.path.join(curfold,'AnewDawn','testJEFF.pt')), "Weights not found"
    assert os.path.isfile(os.path.join(curfold,'AnewDawn','testJEFF.config')), "Config not found"

def test_Trainer_config():
    ma = LinSimple(hidden=32,out=15)
    config = ma.config
    assert config == {'hidden':32, 'out':15, 'name':'LinSimple'}, f"Invalid config : {config}"


# Probably need to add more unit_tests...
