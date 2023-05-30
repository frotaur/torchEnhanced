import torch.nn as nn
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from pathlib import Path



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

    def train_epochs(self):
        raise NotImplementedError('Sub-classes of Trainer should implement train_epochs')
    
    def device(self):
        return self.device_tens.device

    def __del__(self):
        self.writer.close()

