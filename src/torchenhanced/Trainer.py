import torch.nn as nn
import torch, wandb, os
import torch.optim.lr_scheduler as lrsched
from torch.optim import Optimizer
from datetime import datetime
from tqdm import tqdm
from .modules import DevModule, ConfigModule


class Trainer(DevModule):
    """
        Mother class used to train models, exposing a host of useful functions.
        Should be subclassed to be used, and the following methods should be redefined :
            - process_batch, mandatory
            - get_loaders, mandatory
            - epoch_log, optional
            - valid_log, optional
            - process_batch_valid, mandatory if validation is used (i.e. get_loaders returns 2 loaders)
        For logging, use wandb.log, which is already initialized. One should be logged in into the wandb
        account to make the logging work. See wandb documentation for info on logging.
            

        Parameters :
        model : Model to be trained
        optim : Optimizer to be used. ! Must be initialized
        with the model parameters ! Default : AdamW with 1e-3 lr.
        scheduler : Scheduler to be used. Can be provided only if using
        non-default optimizer. Must be initialized with aforementioned 
        optimizer. Default : warmup for 4 epochs from 1e-6.
        state_save_loc : str or None(default), folder in which to store data 
        pertaining to training, such as the training state, wandb folder and model weights.
        device : torch.device, device on which to train the model
        run_name : str, for wandb and saves, name of the training session
        project_name : str, name of the project in which the run belongs
    """

    def __init__(self, model : nn.Module, optim :Optimizer =None, scheduler : lrsched._LRScheduler =None, 
                 state_save_loc=None,device:str ='cpu', run_name :str = None,project_name :str = None):
        super().__init__()
        
        self.to(device)
        self.model = model.to(device)

        
        if(state_save_loc is None) :
            self.data_fold = os.path.join('.',project_name)
            self.state_save_loc = os.path.join(self.data_fold,"state")
            self.model_save_loc = os.path.join(self.data_fold,"weights")
        else :
            self.data_fold = os.path.join(state_save_loc,project_name)#
            
            self.state_save_loc = os.path.join(state_save_loc,project_name,"state")
            self.model_save_loc = os.path.join(state_save_loc,project_name,"weights")
        
        os.makedirs(self.data_fold,exist_ok=True)
        if(optim is None):
            self.optim = torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        else :
            self.optim = optim

        if(scheduler is None):
            self.scheduler = lrsched.LinearLR(self.optim,start_factor=0.05,total_iters=4)
        else :
            self.scheduler = scheduler
        

        # Session hash, the date to not overwrite sessions
        self.session_hash = datetime.now().strftime('%H-%M_%d_%m')
        if(run_name is None):
            self.run_name = self.session_hash
            run_name= os.path.join('.','runs',self.session_hash)
        else :
            self.run_name=run_name
            run_name = os.path.join('.','runs',run_name)
        
        # Basic config, when sub-classing can add dataset and such
        # Maybe useless
        self.run_config = dict(model=self.model.__class__.__name__,
                               lr_init=self.optim.param_groups[0]['lr'],
                               scheduler = self.scheduler.__class__.__name__)
        
        self.run_id = wandb.util.generate_id() # For restoring the run
        self.project_name = project_name
        
        # Universal attributes for logging purposes
        self.stepnum = 0
        self.batchnum = None
        self.step_log = None
        self.totbatch = None

    def change_lr(self, new_lr):
        """
            Changes the learning rate of the optimizer.
            Might clash with scheduler ?
        """

        for g in self.optim.param_groups:
            g['lr'] = new_lr
        

    def load_state(self,state_path : str):
        """
            Loads trainer minimal trainer state (model,session_hash,optim,scheduler).

            params : 
            state_path : str, location of the sought-out state_dict

        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load state')
        state_dict = torch.load(state_path,map_location=self.device)
        if(self.model.config != state_dict['model_config']):
            print('WARNING ! Loaded model configuration and state model_config\
                  do not match. This may generate errors.')
            
        self.model.load_state_dict(state_dict['model'])
        self.session_hash = state_dict['session']
        self.optim.load_state_dict(state_dict['optim'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.run_id = state_dict['run_id']
        # Maybe I need to load also the run_name, we'll see

        print('LOAD OF SUCCESSFUL !')


    def save_state(self,epoch:int = None):
        """
            Saves trainer state. Describe by the following dictionary :

            state_dict : dict, contains at least the following key-values:
                - 'model' : contains model.state_dict
                - 'session' : contains self.session_hash
                - 'optim' :optimizer
                - 'scheduler : scheduler
                - 'model_config' : json allowing one to reconstruct the model.
                - 'run_id' : id of the run, for wandb

            If you want a more complicated state, training_epoch should be overriden.

            Args :
            epoch : int, if not None, will append the epoch number to the state name.
        """
        os.makedirs(self.state_save_loc,exist_ok=True)

        # Create the state
        try :
            model_config = self.model.config
        except AttributeError as e:
            print(f'''Error while fetching model config ! 
                    Make sure model.config is defined. (see ConfigModule doc).
                    Continuing, but might generate errors while loading/save models)''')
            model_config = None

        state = dict(optim=self.optim.state_dict(),scheduler=self.scheduler.state_dict()
            ,model=self.model.state_dict(),session=self.session_hash,model_config=model_config,
            name=self.model.__class__.__name__, run_id=self.run_id)

        name = self.run_name
        if (epoch is not None):
            name=name+'_'+f'{epoch}'

        name = name + '.state'
        saveloc = os.path.join(self.state_save_loc,name)
        torch.save(state,saveloc)

        print('Saved training state')


    @staticmethod
    def save_model_from_state(state_path:str,save_dir:str='.',name:str=None):
        """
            Extract model weights and configuration, and saves two files in the specified directory,
            the weights (.pt) and a .config file containing the model configuration, which can be loaded
            as a dictionary with torch.load.

            Args :
            state_path : path to the trainer state
            save_dir : directory in which to save the model
            name : name of the model, if None, will be model_name_date.pt
        """
        namu, config, weights = Trainer.config_from_state(state_path,device='cpu')

        if (name is None):
            name=f"{namu}_{datetime.now().strftime('%H-%M_%d_%m')}"
        name=name+'.pt'
        os.makedirs(save_dir,exist_ok=True)
        saveloc = os.path.join(save_dir,name)
        
        torch.save(weights, saveloc)

        torch.save(config, os.path.join(save_dir,name[:-3]+'.config'))

        print(f'Saved weights of {name} at {save_dir}/{name}  !')


    @staticmethod
    def config_from_state(state_path: str, device: str=None):
        """
            Given the path to a trainer state, returns a tuple (config, weights)
            for the saved model. The model can then be initialized by using config 
            as its __init__ arguments, and load the state_dict from weights.

            Args :
            state_path : path of the saved trainer state
            device : device on which to load. Default one if None specified

            returns: 3-uple
            model_name : str, the saved model class name
            config : dict, the saved model config
            weights : torch.state_dict, the model's state_dict

        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load config from it')
        
        if(device is None):
            state_dict = torch.load(state_path)
        else :
            state_dict = torch.load(state_path,map_location=device)

        config = state_dict['model_config']
        model_name = state_dict['name']
        weights = state_dict['model']

        return model_name,config,weights


    def process_batch(self,batch_data,data_dict : dict = None,**kwargs):
        """
            Redefine this in sub-classes. Should return the loss, as well as 
            the data_dict (potentially updated). Can do logging and other things 
            optionally. Loss is automatically logged, so no need to worry about it. 

            Args :
            batch_data : whatever is returned by the dataloader
            data_dict : DEPRECATED ! Avoid using it. Use class attributes instead. 
            Default class attributes, automatically maintained by the trainer, are :
                - self.batchnum : current validation mini-batch number
                - self.step_log : number of steps (minibatches) interval in which we should log 
                - self.totbatch : total number of validation minibatches.
                - self.epoch : current epoch

            Returns : 2-uple, (loss, data_dict)
        """
        raise NotImplementedError('process_batch should be implemented in Trainer sub-class')

    def process_batch_valid(self,batch_data, data_dict : dict = None, **kwargs):
        """
            Redefine this in sub-classes. Should return the loss, as well as 
            the data_dict (potentially updated). There should be NO logging done
            inside this function, only in valid_log. Proper use should be to collect the data
            to be logged in a class attribute, and then log it in valid_log (to log once per epoch)
            Loss is automatically logged, so no need to worry about it. 

            Args :
            batch_data : whatever is returned by the dataloader
            data_dict : DEPRECATED ! Avoid using it. Use class attributes instead. 
            Default class attributes, automatically maintained by the trainer, are :
                - self.batchnum : current validation mini-batch number
                - self.step_log : number of steps (minibatches) interval in which we should log 
                - self.totbatch : total number of validation minibatches.
                - self.epoch : current epoch

            Returns : 2-uple, (loss, data_dict)
        """
        raise NotImplementedError('process_batch_valid should be implemented in Trainer sub-class')


    def get_loaders(self,batch_size, num_workers=0):
        """
            Builds the dataloader needed for training and validation.
            Should be re-implemented in subclass.

            Args :
            batch_size

            Returns :
            2-uple, (trainloader, validloader)
        """
        raise NotImplementedError('get_loaders should be redefined in Trainer sub-class')

    def epoch_log(self,data_dict :dict = None):
        """
            To be (optionally) implemented in sub-class. Does the logging 
            at the epoch level, is called every epoch. Only log using commit=False,
            because of sync issues with the epoch x-axis.

            Args :
            data_dict : DEPRECATED ! Avoid using it. Use class attributes instead. 
            Default class attributes, automatically maintained by the trainer, are :
                - self.batchnum : current validation mini-batch number
                - self.step_log : number of steps (minibatches) interval in which we should log 
                - self.totbatch : total number of validation minibatches.
                - self.epoch : current epoch
        """
        pass

    def valid_log(self,data_dict = None):
        """
            To be (optionally) implemented in sub-class. Does the logging 
            at the epoch level, is called every epoch. Only log using commit=False,
            because of sync issues with the epoch x-axis.


            Args :
            data_dict : DEPRECATED ! Avoid using it. Use class attributes instead. 
                Default class attributes, automatically maintained by the trainer, are :
                    - self.batchnum : current validation mini-batch number
                    - self.step_log : number of steps (minibatches) interval in which we should log 
                    - self.totbatch : total number of validation minibatches.
                    - self.epoch : current epoch
        """
        pass

    def train_epochs(self,epochs : int,*,batch_sched:bool=False,save_every:int=50,
                     backup_every: int=None,step_log:int=None,batch_log:int=None,
                     batch_size:int=32,num_workers:int=0,aggregate:int=1, load_from:str=None,
                     batch_tqdm:bool=True,
                     **kwargs):
        """
            Trains for specified epoch number. This method trains the model in a basic way,
            and does very basic logging. At the minimum, it requires process_batch and 
            process_batch_valid to be overriden, and other logging methods are optionals.

            data_dict can be used to carry info from one batch to another inside the same epoch,
            and can be used by process_batch* functions for logging of advanced quantities.
            Params :
            epochs : number of epochs to train for
            batch_sched : if True, scheduler steps (by a lower amount) between each batch.
            Not that this use is deprecated, so it is recommended to keep False. For now, 
            necessary for some Pytorch schedulers (cosine annealing).
            save_every : saves trainer state every 'save_every' epochs
            step_log : If not none, will also log every step_log minibatches, in addition to each epoch
            batch_log : same as step_log, DEPRECATED
            batch_size : batch size
            num_workers : number of workers in dataloader
            aggregate : how many batches to aggregate (effective batch_size is aggreg*batch_size)
            load_from : path to a trainer state_dict. Loads the state
                of the trainer from file, then continues training the specified
                number of epochs.
        """

        # Maybe remove this option ? Probably better just to load manually
        if(os.path.isfile(str(load_from))):
            # Loads the trainer state
            self.load_state(load_from)
        elif(load_from is not None) :
            print(f'Specified "load_from" directory {load_from} non-existent; \
                  continuing with model from scratch.')
    
        # Initiate logging
        wandb.init(name=self.run_name,project=self.project_name,config=self.run_config,
                   id = self.run_id,resume='allow',dir=self.data_fold)
        
        # Define the custom x axis metric, epochs
        wandb.define_metric("epochs")
        # For all plots, we plot against the epoch by default
        wandb.define_metric("*", step_metric='epochs')


        
        train_loader,valid_loader = self.get_loaders(batch_size,num_workers=num_workers)
        validate = valid_loader is not None

        self.model.train()
        self.epoch=self.scheduler.last_epoch
        print('Number of batches/epoch : ',len(train_loader))
        data_dict={}

        if(step_log is None):
            step_log=batch_log # FOR BACKWARD COMPATIBILITY, TO BE DEPRECATED
        
        data_dict['batch_log']=step_log
        data_dict['step_log']=step_log # to be deprecated 

        self.step_log = step_log
        step_loss=[]

        
        for ep_incr in tqdm(range(epochs)):
            epoch_loss,batchnum,n_aggreg=[[],0,0]
            
            data_dict=self._reset_data_dict(data_dict)
            data_dict['epoch']=self.epoch
            self.totbatch = len(train_loader)
            data_dict['totbatch']=self.totbatch

            # Iterate with or without tqdm
            if(batch_tqdm):
                iter_on=tqdm(enumerate(train_loader),total=self.totbatch)
            else :
                iter_on=enumerate(train_loader)

            # Epoch of Training
            for batchnum,batch_data in iter_on :
                n_aggreg+=1
                # Process the batch according to the model.
                self.batchnum=batchnum
                data_dict['batchnum']=self.batchnum
                self.stepnum=(self.epoch)*self.totbatch+batchnum
                data_dict['stepnum']=self.stepnum
                loss, data_dict = self.process_batch(batch_data,data_dict)
                
                epoch_loss.append(loss.item())
                step_loss.append(loss.item())

                loss=loss/aggregate # Rescale loss if aggregating.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

                
                if(self.stepnum%step_log==step_log-1):
                    wandb.log({'steploss/train':sum(step_loss)/len(step_loss)},commit=False)
                    wandb.log({'epochs': self.epoch+batchnum/self.totbatch},commit=True) # Fractional epoch
                    step_loss=[]

                if(n_aggreg%aggregate==aggregate-1):
                    n_aggreg=0
                    self.optim.step()
                    self.optim.zero_grad()
                if(batch_sched):
                    self.scheduler.step(self.epoch+batchnum/self.totbatch)

            if(not batch_sched):
                self.scheduler.step()
            
            # Epoch of validation
            if(validate):
                with torch.no_grad():
                    self.model.eval()
                    val_loss=[]
                    data_dict['totbatch'] = len(valid_loader)
                    self.totbatch = len(valid_loader) # For now we use same totbatch for train and valid, might wanna change that in the future
                    for (batchnum,batch_data) in enumerate(valid_loader):
                        self.batchnum=batchnum
                        data_dict['batchnum']=self.batchnum
                        
                        loss, data_dict = self.process_batch_valid(batch_data,data_dict)
                        val_loss.append(loss.item())

                self.model.train()
    
            self.epoch+=1 # Epoch is finished

            # Log training at epoch level
            wandb.log({'loss/train':sum(epoch_loss)/len(epoch_loss)},commit=False)
            self.epoch_log(data_dict)

            if(validate):
                # Log validation data
                wandb.log({'loss/valid':sum(val_loss)/len(val_loss)},commit=False)
                self.valid_log(data_dict)

            # Add the x-axis tick and commit
            wandb.log({'epochs': self.epoch},commit=True)


            if ep_incr%save_every==save_every-1 :
                self.save_state()
            
            if backup_every is not None:
                if ep_incr%backup_every==backup_every-1 :
                    self.save_state(epoch=self.epoch)

        wandb.finish()

    def _reset_data_dict(self,data_dict):
        keys = list(data_dict.keys())
        for k in keys:
            if k not in ['epoch','batch_log', 'step_log'] :
                del data_dict[k]
        # Probably useless to return
        return data_dict

