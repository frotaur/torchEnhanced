import torch.nn as nn
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path
from tqdm import tqdm


class DevModule(nn.Module):
    """
        Extremely small wrapper for nn.Module.
        Simply adds a method device() that returns
        the current device the module is on. Changes if
        self.to(...) is called.
    """
    def __init__(self):
        super().__init__()

        self.register_buffer('_devtens',torch.empty(0))
    

    def device(self):
        return self._devtens.device



class Trainer(DevModule):
    """
        Mother class used to train models, exposing a host of useful functions.

        Parameters :
        model : nn.Module, model to be trained
        model_save_loc : str or None(default), folder in which to save the raw model weights
        state_save_loc : str or None(default), folder in which to save the training state, 
        used to resume training.
        device : torch.device, device on which to train the model
        writer_name : str, for tensorboard, name of the training session
    """

    def __init__(self, model : nn.Module, model_save_loc=None,state_save_loc=None,device='cpu', writer_name = None):
        super().__init__()
        self.model = model 
        self.model.to(device)

        if(model_save_loc is None) :
            self.model_save_loc = os.path.join(Path(__file__).parent,f"{self.model.__class__.__name__}_weights")
        else :
            self.model_save_loc = model_save_loc
        
        if(state_save_loc is None) :
            self.state_save_loc = os.path.join(Path(__file__).parent,f"{self.model.__class__.__name__}_state")
        else :
            self.state_save_loc = state_save_loc

        for direc in [self.model_save_loc,self.state_save_loc]:
            os.makedirs(direc,exist_ok=True)
        

        # Session hash, the date to not overwrite sessions
        self.session_hash = datetime.now().strftime('%H-%M_%d_%m')
        if(writer_name is None):
            writer_name= os.path.join(Path(__file__).parent,'runs',self.session_hash)
        else :
            writer_name = os.path.join(Path(__file__).parent,'runs',writer_name)
        
        self.writer = SummaryWriter(writer_name)


    def load_state(self,state_path):
        """
            Loads trainer minimal trainer state (model,session_hash).
            If other values need to be added, should be done after calling super.load_state.

            params : 
            state_path : str, location of the sought-out state_dict

            returns : the loaded state_dict with remaining parameters
        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load state')
        state_dict = torch.load(state_path)
        self.model.load_state_dict(state_dict['model'])
        del state_dict['model']
        self.session_hash = state_dict['session']
        del state_dict['session']

        return state_dict


    def save_state(self,state_dict,name=None,unique=False):
        """
            Saves trainer state.
            Params : 
            state_dict : dict, contains at least the following key-values:
                - 'model' : contains model.state_dict
                - 'session' : contains self.session_hash
            Additionally, can contain logging info like last loss, epoch number, and others.
            name : str, name of the save file, overrides automatic one
            unique : bool, if to generate unique savename (with date)
            
        """

        if (name is None):
            name=f"{self.model.__class__.__name__}_state_{self.session_hash}.pt"
        if (unique):
            name=name+'_'+datetime.now().strftime('%H-%M_%d_%m')
        saveloc = os.path.join(self.state_save_loc,name)
        torch.save(state_dict,saveloc)

        print('Saved training state')

    def save_model(self, name=None):
        """
            Saves model weights onto trainer model_save_loc.
        """
        if (name is None):
            name=f"{self.model.__class__.__name__}_{datetime.now().strftime('%H-%M_%d_%m')}.pt"

        saveloc = os.path.join(self.model_save_loc,name)
        
        torch.save(self.model.state_dict(), saveloc)
        try :
            torch.save(self.model.get_config(), os.path.join(self.model_save_loc,name[:-3]+'.config'))
        except Exception as e:
            print(f'''Problem when trying to get configuration of model : {e}. Make sure model.get_config()
                  is defined.''')
            raise e

        print(f'Saved checkpoint : {name}')


class Trainer(nn.Module):
    """
        Mother class used to train models, exposing a host of useful functions.

        Parameters :
        model : Model to be trained
        optim : Optimizer to be used. ! Must be initialized
        with the model parameters ! Default : AdamW with 1e-3 lr.
        scheduler : Scheduler to be used. Can be provided only if using
        non-default optimizer. Must be initialized with 
        aforementioned optimizer. Default : warmup for 4 epochs from 1e-6.
        model_save_loc : str or None(default), folder in which to save the raw model weights
        state_save_loc : str or None(default), folder in which to save the training state, 
        used to resume training.
        device : torch.device, device on which to train the model
        writer_name : str, for tensorboard, name of the training session
    """

    def __init__(self, model : nn.Module, optim :Optimizer =None, scheduler : lrsched._LRScheduler =None, 
                 model_save_loc=None,state_save_loc=None,device='cpu', writer_name = None):
        super().__init__()
        self.model = model 
        self.model.to(device)

        self.register_buffer('device_tens',torch.empty(1,device=device))

        if(model_save_loc is None) :
            self.model_save_loc = os.path.join(Path(__file__).parent,f"{self.model.__class__.__name__}_weights")
        else :
            self.model_save_loc = model_save_loc
        if(state_save_loc is None) :
            self.state_save_loc = os.path.join(Path(__file__).parent,f"{self.model.__class__.__name__}_state")
        else :
            self.state_save_loc = state_save_loc
        
        if(optim is None):
            self.optim = torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        else :
            self.optim = optim

        if(scheduler is None):
            self.scheduler = lrsched.LinearLR(self.optim,start_factor=0.05,total_iters=4)
        else :
            self.scheduler = scheduler

        for direc in [self.model_save_loc,self.state_save_loc]:
            os.makedirs(direc,exist_ok=True)
        

        # Session hash, the date to not overwrite sessions
        self.session_hash = datetime.now().strftime('%H-%M_%d_%m')
        if(writer_name is None):
            writer_name= os.path.join(Path(__file__).parent,'runs',self.session_hash)
        else :
            writer_name = os.path.join(Path(__file__).parent,'runs',writer_name)
        
        self.writer = SummaryWriter(writer_name)


    def load_state(self,state_path):
        """
            Loads trainer minimal trainer state (model,session_hash).
            If other values need to be added, should be done after calling super.load_state.

            params : 
            state_path : str, location of the sought-out state_dict

            returns : the loaded state_dict with remaining parameters
        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load state')
        state_dict = torch.load(state_path)
        self.model.load_state_dict(state_dict['model'])
        del state_dict['model']
        self.session_hash = state_dict['session']
        del state_dict['session']
        self.optim.load_state_dict(state_dict['optim'])
        del state_dict['optim']
        self.scheduler.load_state_dict(state_dict['scheduler'])
        del state_dict['scheduler']

        return state_dict


    def save_state(self,state_dict,name=None,unique=False):
        """
            Saves trainer state.
            Params : 
            state_dict : dict, contains at least the following key-values:
                - 'model' : contains model.state_dict
                - 'session' : contains self.session_hash
                - 'optim' :optimizer
                - 'scheduler : scheduler
            Additionally, can contain logging info like last loss, epoch number, and others.
            name : str, name of the save file, overrides automatic one
            unique : bool, if to generate unique savename (with date)
        """

        if (name is None):
            name=f"{self.model.__class__.__name__}_state_{self.session_hash}.pt"
        if (unique):
            name=name+'_'+datetime.now().strftime('%H-%M_%d_%m')
        saveloc = os.path.join(self.state_save_loc,name)
        torch.save(state_dict,saveloc)

        print('Saved training state')

    def save_model(self, name=None):
        """
            Saves model weights onto trainer model_save_loc.
        """
        if (name is None):
            name=f"{self.model.__class__.__name__}_{datetime.now().strftime('%H-%M_%d_%m')}.pt"

        saveloc = os.path.join(self.model_save_loc,name)
        
        torch.save(self.model.state_dict(), saveloc)
        try :
            torch.save(self.model.get_config(), os.path.join(self.model_save_loc,name[:-3]+'.config'))
        except Exception as e:
            print(f'''Problem when trying to get configuration of model : {e}. Make sure model.get_config()
                  is defined.''')
            raise e

        print(f'Saved checkpoint : {name}')


    def process_batch(batch_data,data_dict : dict):
        """
            Redefine this in sub-classes. Should return the loss, as well as 
            the data_dict (potentially updated). Can do logging and other things 
            optionally. Loss is automatically logged, so no need to worry about it. 

            params:
            batch_data : whatever is returned by the dataloader
            data_dict : Dictionary containing necessary data, mainly
            for logging. Always contains the following key-values :
                - time : float variable giving a value to the progression of the batches
                - batchnum : current batch number
                - batch_log : batch interval in which we should log
                - totbatch : total number of batches.
            data_dict can be modified to store running values, or any other value that might
            be important later. If data_dict is updated, this will persist through the next iteration
            and call of process_batch.

            Returns : 2-uple, (loss, data_dict)
        """
        raise NotImplementedError('process_batch should be implemented in Trainer sub-class')

    def get_loaders(batch_size):
        """
            Builds the dataloader needed for training and validation.
            Should be re-implemented in subclass.

            Params :
            batch_size

            Returns :
            2-uple, (trainloader, validloader)
        """
        raise NotImplementedError('get_loaders should be redefined in Trainer sub-class')

    def epoch_log(data_dict):
        """
            To be implemented in sub-class. Does the logging at the epoch level,
            is called every epoch. Data_dict has (at least) key-values :
                - time : float variable giving a value to the progression of the batches
                - batchnum : current batch number
                - batch_log : batch interval in which we should log
                - totbatch : total number of batches.
            And any number of additional values, depending on what process_batch does.
        """
        raise NotImplementedError('batch_log should be redefined in Trainer sub-class')

    def train_epochs(self,epochs : int,*,save_every:int=50,batch_log:int=None,batch_size:int=32,aggregate:int=1,load_from:str=None,unique:bool=False):
        """
            Trains for specified epoch number. Very basic training loop. This method should generally be re-implemented in full,
            to cater to the specifics of the problem.
            Params :
            epochs : number of epochs to train for
            save_every : saves trainer state every 'save_every' epochs
            batch_log : If not none, will also log every batch_log batches, in addition to each epoch
            batch_size : -
            aggregate : how many batches to aggregate (effective batch_size is aggreg*batch_size)
            load_from : path to a trainer state_dict. Loads the state
                of the trainer from file, then continues training the specified
                number of epochs.
            unique : if True, do not overwrites previous save states.
        """
        if(os.path.isfile(str(load_from))):
            # Loads the trainer state
            self.load_state(load_from)
        
        train_loader,valid_loader = self.get_loaders(batch_size)
        self.model.train()
        epoch=self.scheduler.last_epoch
        print('Number of batches/epoch : ',len(train_loader))
        data_dict={}
        data_dict['batch_log']=batch_log
        for ep_incr in tqdm(range(epochs)):
            epoch_loss,batch_log_loss,batchnum,n_aggreg=(0,0,0,0)
            totbatch = len(train_loader)

            data_dict['epoch']=epoch
            data_dict['totbatch']=totbatch

            for batchnum,batch_data in tqdm(enumerate(train_loader),total=totbatch) :
                n_aggreg+=1
                # Process the batch according to the model.
                data_dict['batchnum']=batchnum
                data_dict['time']=(epoch-1)*totbatch//batch_log+batchnum//batch_log

                loss, data_dict = self.process_batch(batch_data,data_dict)

                loss=loss/aggregate # Rescale loss if aggregating.
                loss.backward()

                epoch_loss+=loss.item()*aggregate
                batch_log_loss+=loss.item()*aggregate
                
                if(batchnum%batch_log==batch_log-1):
                    self.writer.add_scalar('loss/train',batch_log_loss,data_dict['time'])
                    batch_log_loss=0

                if(n_aggreg%aggregate==aggregate-1):
                    n_aggreg=0
                    self.optim.step()
                    self.optim.zero_grad()

            self.scheduler.step()

            # Log data
            self.writer.add_scalar('ep-loss/train',epoch_loss,data_dict['time'])
            self.epoch_log(data_dict)
            
            if(valid_loader is not None):

                with torch.no_grad():
                    self.model.eval()
                    val_loss=0
                    
                    accupix =0

                    for (batchnum,batch_data) in enumerate(valid_loader):
                        loss, data_dict = self.process_batch_valid(batch_data)
                        val_loss+=loss.item()

            # Log validation data
            self.writer.add_scalar('ep-loss/valid',val_loss,data_dict['time'])
            self.valid_log(data_dict)

            self.writer.flush()

            self.model.train()
            epoch+=1

            if ep_incr%save_every==0 :
                state = dict(optim=self.optim.state_dict(),scheduler=self.scheduler.state_dict()
                     ,model=self.model.state_dict(),session=self.session_hash,model_config=self.model.get_config())
                self.save_state(state,name=self.run_name,unique=unique)

    def __del__(self):
        self.writer.close()

    def device(self):
        return self.device_tens.device

    def __del__(self):
        self.writer.close()

