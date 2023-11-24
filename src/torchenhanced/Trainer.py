import torch.nn as nn, math
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
        
        Use train_epochs OR train_steps, according to whether you would like to train at epoch level or at batch number level.
        Loading a state trained with train_epochs and using it in train_steps will cause unexpected behavior, and vice-versa.

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
        run_config : dict, dictionary of hyperparameters (any). Will be viewable in wandb.
    """

    def __init__(self, model : nn.Module, optim :Optimizer =None, scheduler : lrsched._LRScheduler =None,*, 
                 state_save_loc=None,device:str ='cpu', run_name :str = None,project_name :str = None,
                 run_config : dict = {}):
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
        
        self.run_config = dict(model=self.model.__class__.__name__,
                               **run_config)
        
        self.run_id = wandb.util.generate_id() # For restoring the run
        self.project_name = project_name
        
        # Universal attributes for logging purposes
        self.stepnum = 0 # number of steps in current training instance
        self.batchnum = None # same as stepnum, DEPRECATED

        self.batches = 0 # number of total batches ever
        self.steps_done = 0 # number of total steps ever TODO (now its same as batches, will become optimizer steps later)
        self.epochs = 0 # number of total epochs ever
        self.samples = 0 # number of total samples ever

        self.step_log = None # number of steps between each log
        self.totbatch = None # total number of batches in one epoch for this training instance
        self.do_batch_log = False 

        # Used for logging instead of wandb.log, useful if wandb not imported
        self.logger = None
    
    def change_lr(self, new_lr):
        """
            Changes the learning rate of the optimizer.
            Might clash with scheduler ?
        """

        for g in self.optim.param_groups:
            g['lr'] = new_lr
        

    def load_state(self,state_path : str, strict: bool=True):
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
        assert self.model.class_name == state_dict['model_name'], f'Loaded model {state_dict["model_name"]} mismatch with current: {self.model.class_name}!'
        assert self.optim.__class__.__name__ == state_dict['optim_name'], f'Loaded optimizer : {state_dict["optim_name"]} mismatch with current: {self.optim.__class__.__name__} !'
        assert self.scheduler.__class__.__name__ == state_dict['scheduler_name'], f'Loaded scheduler : {state_dict["scheduler_name"]} mismatch with current: {self.optim.__class__.__name__} !'

        self.model.load_state_dict(state_dict['model_state'],strict=strict)
        self.session_hash = state_dict['session']
        self.optim.load_state_dict(state_dict['optim_state'])
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        self.run_id = state_dict['run_id']
        self.steps_done = state_dict.get('steps_done',0)
        self.batches = state_dict.get('batches',0)

        self.epochs = state_dict.get('epochs',0)
        self.samples = state_dict.get('samples',0)
        self.run_config = state_dict.get('run_config',{'model':self.model.__class__.__name__})
        # Maybe I need to load also the run_name, we'll see

        print('LOAD OF SUCCESSFUL !')
        print('loaded dict : ', state_dict['batches'])

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
                - 'steps_done' : only applicable in case of step training, number of steps done
                - 'samples' : number of samples seen
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

        state = dict(
        optim_state=self.optim.state_dict(),scheduler_state=self.scheduler.state_dict(),model_state=self.model.state_dict(),
        model_name=self.model.class_name,optim_name=self.optim.__class__.__name__,scheduler_name=self.scheduler.__class__.__name__,
        model_config=model_config,session=self.session_hash,run_id=self.run_id, steps_done=self.steps_done,epochs=self.epochs,
        samples=self.samples, batches=self.batches,run_config=self.run_config
        )

        name = self.run_name
        if (epoch is not None):
            os.makedirs(os.path.join(self.state_save_loc,'backups'),exist_ok=True)
            name=os.path.join('backups',name+'_'+f'{epoch:.2f}')

        name = name + '.state'
        saveloc = os.path.join(self.state_save_loc,name)
        torch.save(state,saveloc)

        print(f'Saved training state at {datetime.now().strftime("%H-%M_%d_%m")}')
        print(f'At save : ', self.batches)

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
        namu, config, weights = Trainer.model_config_from_state(state_path,device='cpu')

        if (name is None):
            name=f"{namu}_{datetime.now().strftime('%H-%M_%d_%m')}"
        name=name+'.pt'
        os.makedirs(save_dir,exist_ok=True)
        saveloc = os.path.join(save_dir,name)
        
        torch.save(weights, saveloc)

        torch.save(config, os.path.join(save_dir,name[:-3]+'.config'))

        print(f'Saved weights of {name} at {save_dir}/{name}  !')

    @staticmethod
    def opti_names_from_state(state_path: str,device='cpu'):
        """
            Given the path to a trainer state, returns a 2-tuple (opti_config, scheduler_config),
            where each config is a tuple of the name of the optimizer, and its state_dict.
            Usually useful only if you forgot which optimizer you used, but load_state should
            be used instead usually.
            
            Args :
            state_path : path of the saved trainer state
            device : device on which to load state

            Returns :
            2-uple, (optim_config, scheduler_config), where *_config = (name, state_dict)

            Example of use :
            get name from opti_config[0]. Use it with eval (or hardcoded) to get the class,
            instanciante : 
            optim = torch.optim.AdamW(model.parameters(),lr=1e-3)
            optim.load_state_dict(opti_config[1])
        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load config from it')

        state_dict = torch.load(state_path,map_location=device)
        opti_name = state_dict['optim_name']
        opti_state = state_dict['optim_state']
        sched_name = state_dict['sched_name']
        sched_state = state_dict['sched_state']

        return (opti_name,opti_state),(sched_name,sched_state)

    @staticmethod
    def config_from_state(state_path: str,device: str=None):
        """
            Given the path to a trainer state, returns a tuple (config, weights)
            for the saved model. The model can then be initialized by using config 
            as its __init__ arguments, and load the state_dict from weights.

            Args :
            state_path : path of the saved trainer state
            device : device on which to load. Default one if None specified

            returns: 3-uple
            model_name : str, the saved model class name
            config : dict, the saved model config (instanciate with element_name(**config))
            state_dict : torch.state_dict, the model's state_dict (load with .load_state_dict(weights))

        """
        print('WARNING : Deprecated, will be removed in next version')
        print('For model config, use model_config_from_state instead')
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load config from it')
        
        if(device is None):
            state_dict = torch.load(state_path)
        else :
            state_dict = torch.load(state_path,map_location=device)

        config = state_dict['model_config']
        model_name = state_dict['model_name']
        weights = state_dict['model_state']

        return model_name,config,weights

    @staticmethod
    def model_config_from_state(state_path: str,device: str=None):
        """
            Given the path to a trainer state, returns a tuple (config, weights)
            for the saved model. The model can then be initialized by using config 
            as its __init__ arguments, and load the state_dict from weights.

            Args :
            state_path : path of the saved trainer state
            device : device on which to load. Default one if None specified

            returns: 3-uple
            model_name : str, the saved model class name
            config : dict, the saved model config (instanciate with element_name(**config))
            state_dict : torch.state_dict, the model's state_dict (load with .load_state_dict(weights))

        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load config from it')
        
        if(device is None):
            state_dict = torch.load(state_path)
        else :
            state_dict = torch.load(state_path,map_location=device)

        config = state_dict['model_config']
        model_name = state_dict['model_name']
        weights = state_dict['model_state']

        return model_name,config,weights
    
    @staticmethod
    def run_config_from_state(state_path: str,device: str=None):
        """
            Given the path to a trainer state, returns the run_config dictionary.

            Args :
            state_path : path of the saved trainer state
            device : device on which to load. Default one if None specified

            returns: dict, the run_config dictionary
        """
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load config from it')
        
        if(device is None):
            state_dict = torch.load(state_path)
        else :
            state_dict = torch.load(state_path,map_location=device)

        return state_dict['run_config']

    def process_batch(self,batch_data,**kwargs):
        """
            Redefine this in sub-classes. Should return the loss, as well as 
            the data_dict (potentially updated). Can do logging and other things 
            optionally. Loss is automatically logged, so no need to worry about it. 

            Args :
            batch_data : whatever is returned by the dataloader
            Default class attributes, automatically maintained by the trainer, are :
                - self.batchnum : current validation mini-batch number
                - self.step_log : number of steps (minibatches) interval in 
                        which we should log. (PREFER USING do_batch_log instead)
                - self.do_batch_log : whether we should log this batch or not
                - self.totbatch : total number of validation minibatches.
                - self.epoch : current epoch
                - self.samples : number of samples seen
            Returns : 2-uple, (loss, data_dict)
        """
        raise NotImplementedError('process_batch should be implemented in Trainer sub-class')

    def process_batch_valid(self,batch_data, **kwargs):
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
                    Minibatch logging in valid is not recommended, since it is not synchronized with the epoch x-axis.
                - self.totbatch : total number of validation minibatches.
                - self.epoch : current epoch
                - self.samples : number of samples seen

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

    def epoch_log(self):
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
                - self.samples : number of samples seen
        """
        pass

    def valid_log(self):
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
                    
                    - self.samples : number of samples seen

        """
        pass
    
    def train_init(self,**kwargs):
        """
            Can be redefined for doing stuff just at the beginning of the training,
            for example, freezing weights, preparing some extra variables, or anything really.
            Not mandatory, it is called at the very beginnig of train_epochs/train_steps. The
            dictionary 'train_init_params' is passed as parameter. As such, it can take
            any combination of parameters.
        """
        pass

    def train_epochs(self,epochs : int,batch_size:int,*,batch_sched:bool=False,save_every:int=50,
                     backup_every: int=None,step_log:int=None,
                     num_workers:int=0,aggregate:int=1,
                     batch_tqdm:bool=True,train_init_params:dict={}):
        """
            Trains for specified epoch number. This method trains the model in a basic way,
            and does very basic logging. At the minimum, it requires process_batch and 
            process_batch_valid to be overriden, and other logging methods are optionals.

            data_dict can be used to carry info from one batch to another inside the same epoch,
            and can be used by process_batch* functions for logging of advanced quantities.
            Params :
            epochs : number of epochs to train for
            batch_size : batch size
            batch_sched : if True, scheduler steps (by a lower amount) between each batch.
            Not that this use is deprecated, so it is recommended to keep False. For now, 
            necessary for some Pytorch schedulers (cosine annealing).
            save_every : saves trainer state every 'save_every' epochs
            step_log : If not none, will also log every step_log minibatches, in addition to each epoch
            batch_log : same as step_log, DEPRECATED
            num_workers : number of workers in dataloader
            aggregate : how many batches to aggregate (effective batch_size is aggreg*batch_size)
            load_from : path to a trainer state_dict. Loads the state
                of the trainer from file, then continues training the specified
                number of epochs.
            train_init_params : Parameter dictionary passed as argument to train_init
        """
        
        # Initiate logging
        self.logger = wandb.init(name=self.run_name,project=self.project_name,config=self.run_config,
                   id = self.run_id,resume='allow',dir=self.data_fold)
        
        # Define the custom x axis metric, epochs
        self.logger.define_metric("batches",hidden=True)
        self.logger.define_metric("epochs",hidden=True)
        self.logger.define_metric("ksamples",hidden=True)
        # For all plots, we plot against the epoch by default
        self.logger.define_metric("*", step_metric='epochs')

        self.train_init(**train_init_params)
        
        train_loader,valid_loader = self.get_loaders(batch_size,num_workers=num_workers)
        validate = valid_loader is not None

        self.model.train()
        if(batch_sched):
            assert self.epochs-self.scheduler.last_epoch<1e-5, f'Epoch mismatch {self.epochs} vs {self.scheduler.last_epoch}'
        else:
            assert int(self.epochs)==self.scheduler.last_epoch, f'Epoch mismatch {self.epochs} vs {self.scheduler.last_epoch}'
        #Floor frac epochs, since we start at start of epoch, and also for the scheduler :
        self.epochs = int(self.epochs)
        print('Number of batches/epoch : ',len(train_loader))
        self.stepnum = 0 # This is the current instance number of steps, using for when to log save etc

        self.step_log = step_log
        step_loss=[]

        
        for ep_incr in tqdm(range(epochs)):
            epoch_loss,n_aggreg=[[],0]
            
            self.totbatch = len(train_loader)

            # Iterate with or without tqdm
            if(batch_tqdm):
                iter_on=tqdm(enumerate(train_loader),total=self.totbatch)
            else :
                iter_on=enumerate(train_loader)

            # Epoch of Training
            for batchnum,batch_data in iter_on :
                # Process the batch
                self.batchnum=batchnum
                epoch_loss, step_loss, n_aggreg = self._step_batch(batch_data,True,epoch_loss,step_loss,n_aggreg, aggregate, step_sched=False)
    
                if(batch_sched):
                    self.scheduler.step(self.epochs)

                self.stepnum+=1
                self.steps_done+=1
                self.batches+=1
                self.samples+=batch_size
                self.epochs+=1/self.totbatch

            self.epochs = round(self.epochs) # round to integer, should already be, but to remove floating point stuff
            
            if(not batch_sched):
                self.scheduler.step()
            else :
                self.scheduler.step(self.epochs)
            
            # Epoch of validation
            if(validate):
                self._validate(valid_loader,batch_tqdm)
                self.model.train()
                self.valid_log()
    

            # Log training at epoch level
            self.logger.log({'loss/train_epoch':sum(epoch_loss)/len(epoch_loss)},commit=False)
            self.epoch_log()
                
            self._update_x_axis(epoch_mode=True)
            
            # Save and backup when applicable
            self._save_and_backup(curstep=ep_incr,save_every=save_every,backup_every=backup_every)

        self.logger.finish()

    def train_steps(self,steps : int,batch_size:int,*,save_every:int=50,
                    backup_every: int=None, valid_every:int=1000,step_log:int=None,
                    num_workers:int=0,aggregate:int=1,pickup:bool=True,resume_batches:bool=False,
                    batch_tqdm:bool=True, train_init_params:dict={}):
        """
            Trains for specified number of steps(batches). This method trains the model in a basic way,
            and does very basic logging. At the minimum, it requires process_batch and 
            process_batch_valid to be overriden, and other logging methods are optionals. Epoch_log is not
            used in step level training.
            Note that the scheduler will be called AFTER EVERY MINIBATCH, i.e. after every step. Everything
            is logged by default against the number of steps, but the 'epochs' metric is also defined, and
            it depends on the size of the dataloader defined in get_loaders.

            Params :
            batch_size : batch size
            steps : number of steps (batches) to train for
            save_every : saves trainer state every 'save_every' epochs
            backup_every : saves trainer state without overwrite every 'backup_every' steps
            valid_every : validates the model every 'valid_every' steps
            step_log : If not none, used for logging every step_log steps. In process_batch,
            use self.do_step_log to know when to log. NOTE : if using aggregating, for step logging,
            we consider a step to be aggregate batches, not 'true' batches.
            num_workers : number of workers in dataloader
            aggregate : how many batches to aggregate (effective batch_size is aggreg*batch_size)
            pickup : if False, will train for exactly 'steps' steps. If True, will restart at the previous
            number of steps, and train until total number of steps is 'steps'. Basically, if true, picks up
            where it left off.
            resume_batches : if True, will resume training assuming the first self.batches on the dataloader
            are already done. Usually, use ONLY if dataloader does NOT shuffle.
            batch_tqdm : whether to use tqdm for the batch loop or not
            train_init_params : Parameter dictionary passed as argument to train_init
        """
    
        # Initiate logging
        self.logger = wandb.init(name=self.run_name,project=self.project_name,config=self.run_config,
                   id = self.run_id,resume='allow',dir=self.data_fold)
        
        # Define the custom x axis metrics
        self.logger.define_metric("epochs",hidden=True)
        self.logger.define_metric("batches",hidden=True)
        # For all plots, we plot against the batches by default, since we do step training
        self.logger.define_metric("*", step_metric='batches')

        self.train_init(**train_init_params)
        
        train_loader,valid_loader = self.get_loaders(batch_size,num_workers=num_workers)
        validate = valid_loader is not None

        self.totbatch = len(train_loader) # Number of batches in one epoch

        self.model.train()
        # _=self.scheduler.last_epoch # this is equal to self.steps_done
    

        print('Number of batches/epoch : ',len(train_loader))

        self.step_log = step_log
        step_loss=[]

        steps_completed = False
        if(pickup):
            self.stepnum = self.batches
        else:
            self.stepnum = 0 #Current instance stepnumber, used for when to log and stop training.

        while not steps_completed:
            iter_on=enumerate(train_loader)

            if(resume_batches):
                resume_batches=False # Only resume for the first epoch, not if we reach and and restart.
                tofastforward = (self.batches)%self.totbatch
                print(f'Fast forwarding {self.batches}%{self.totbatch}={tofastforward} batches')
                for _ in tqdm(range(tofastforward)):
                    # skip batches already done
                    next(iter_on)
                if(batch_tqdm):
                    iter_on=tqdm(iter_on,total=self.totbatch-tofastforward)
            else :
                if(batch_tqdm):
                    iter_on=tqdm(iter_on,total=self.totbatch)
    
            n_aggreg=0
            # Epoch of Training
            for batchnum,batch_data in iter_on :
                # Process the batch according to the model.
                self.batchnum=batchnum
                _, step_loss, n_aggreg = self._step_batch(batch_data,False,[],step_loss,n_aggreg, aggregate,step_sched=True)

                # Validation if applicable
                if(validate and self.batches%valid_every==valid_every-1):
                    self._validate(valid_loader,batch_tqdm)
                    self.valid_log()
                    self._update_x_axis(epoch_mode=False)
                    self.model.train()


                self.stepnum+=1
                self.steps_done+=1 # TO BE MODIFIED TO BE OPTIMIZER STEPS
                self.batches+=1
                self.samples+=batch_size
                self.epochs +=1/self.totbatch

                # TODO minor bug, when we resume we shift by one minibatch the saving schedule
                # Comes because the first save location is at %valid_every-1, so it last one less step since we start stepnum at 0
    	        # Save and backup on the basis of BATCHES, not steps. That way when we resume, everything is consistent
                self._save_and_backup(self.batches,save_every,backup_every)


            
                if(self.stepnum>=steps):
                    steps_completed=True
                    self._save_and_backup(1,save_every,backup_every)
                    break
            
            

        wandb.finish()

    def _update_x_axis(self,epoch_mode):
        """
            Adds and commits pending wandb.log calls, and adds the x-axis metrics,
            to use the correct defaults.

            Args:   
            epoch_mode : bool, whether default x-axis is epoch or not
        """
        # TODO remove the epoch_mode, and just use epochs as last one, it doesn't make a difference anyway
        # TODO make so that batches area actually batches, and steps_done are optimizer steps
        self.logger.log({'ksamples' : self.samples//1000},commit=False)

        if(epoch_mode):
            self.logger.log({'batches': self.steps_done},commit=False) # TO BE MODIFIED TO BE OPTIMIZER STEPS
            self.logger.log({'epochs': self.epochs},commit=True)
        else :
            self.logger.log({'epochs': self.epochs},commit=False)
            self.logger.log({'batches': self.steps_done},commit=True)

    def _step_batch(self, batch_data, epoch_mode, epoch_loss, step_loss, n_aggreg, aggregate, step_sched):
        """
            Internal function, makes one step of training given minibatch
        """
        n_aggreg+=1
        self.do_batch_log = self.stepnum%(self.step_log*aggregate)==0 #if aggregating, one 'batch' is actually aggregate batches

        loss = self.process_batch(batch_data)
        
        epoch_loss.append(loss.item())
        step_loss.append(loss.item())

        loss=loss/aggregate # Rescale loss if aggregating.
        loss.backward() # Accumulate gradients
        
        if(self.do_batch_log):
            self.logger.log({'loss/train_step':sum(step_loss)/len(step_loss)},commit=False)
            self._update_x_axis(epoch_mode=epoch_mode)
            step_loss=[]
            self.do_batch_log=False

        if(n_aggreg%aggregate==aggregate-1):
            n_aggreg=0
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

            self.optim.step()
            self.optim.zero_grad()
            if(step_sched):
                self.scheduler.step()
        
        return epoch_loss,step_loss,n_aggreg

    @torch.no_grad()
    def _validate(self,valid_loader, batch_tqdm)->None:
        self.model.eval()
        val_loss=[]
        t_totbatch = self.totbatch
        t_batchnum = self.batchnum

        self.totbatch = len(valid_loader) # For now we use same totbatch for train and valid, might wanna change that in the future
        if(batch_tqdm):
            print('------ Validation ------')
            iter_on=tqdm(enumerate(valid_loader),total=self.totbatch)
        else:
            iter_on=enumerate(valid_loader)

        for (v_batchnum,v_batch_data) in iter_on:
            self.batchnum=v_batchnum
            
            loss = self.process_batch_valid(v_batch_data)
            val_loss.append(loss.item())
        
        self.totbatch=t_totbatch
        self.batchnum=t_batchnum

        # Log validation data
        self.logger.log({'loss/valid':sum(val_loss)/len(val_loss)},commit=False)
    
    def _save_and_backup(self,curstep,save_every,backup_every):
        # We use curstep-1, to save at a moment consistent with the valid
        # And valid looks at curstep-1. (we updated curstep in between)
        if (curstep-1)%save_every==0 :
            self.save_state()
        
        if backup_every is not None:
            if (curstep-1)%backup_every==backup_every-1 :
                self.save_state(epoch=self.epochs)