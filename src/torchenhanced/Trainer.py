import torch.nn as nn, math
import torch, wandb, os
import torch.optim.lr_scheduler as lrsched
from torch.optim import Optimizer
from datetime import datetime, timezone, timedelta
from tqdm import tqdm


class Trainer:
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
        save_loc : str or None(default), folder in which to store data 
        pertaining to training, such as the training state, wandb folder and model weights.
        device : torch.device, device on which to train the model
        parallel : None or list[int,str], if None, no parallelization, if list, list of devices (int or torch.device) to parallelize on
        run_name : str, for wandb and saves, name of the training session
        project_name : str, name of the project in which the run belongs
        run_config : dict, dictionary of hyperparameters (any). Will be viewable in wandb.
        no_logging : bool, if True, will not log anything, and will not use wandb.
    """

    def __init__(self, model : nn.Module, optim :Optimizer =None, scheduler : lrsched._LRScheduler =None,*, 
                 save_loc=None,device:str ='cpu',parallel:list[int]=None, run_name :str = None,project_name :str = None,
                 run_config : dict = {}, no_logging = False):
        super().__init__()
        self.DEBUG_useitem = DEBUG_useitem
        
        self.parallel_train = parallel is not None
        self.parallel_devices = parallel
        if(self.parallel_train):
            # Go to GPU if parallel training
            device = self.parallel_devices[0]

        self.model = model.to(device)

        if(project_name is None):
            project_name = 'unnamed_project'

        if(save_loc is None) :
            self.data_fold = os.path.join('.',project_name)
            self.save_loc = os.path.join(self.data_fold,"state")
        else :
            self.data_fold = os.path.join(save_loc,project_name)#
            self.save_loc = os.path.join(save_loc,project_name,"state")

        
        os.makedirs(self.data_fold,exist_ok=True)
        if(optim is None):
            self.optim = torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        else :
            self.optim = optim

        if(scheduler is None):
            self.scheduler = lrsched.LinearLR(self.optim,start_factor=0.05,total_iters=4)
        else :
            self.scheduler = scheduler
        
        self.cooldown_data = {'do_cooldown':None, 'cooldown_steps':None, 'cooldown_started':None} # For built-in lr cooldown support

        # Session hash, the date to not overwrite sessions
        # Use time in UTC+2, because I live in France
        self.session_hash = datetime.now(timezone(timedelta(hours=2))).strftime('%H-%M_%d_%m')
        
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
        
        self.device=device
        # Universal attributes for logging purposes
        self.stepnum = 0 # number of steps in current training instance (+1 each optimizer step)
        self.batchnum = 0 # (+1 each batch. Equal to stepnum if no aggregation)

        self.batches = 0 # number of total batches ever (+1 each batch. Same a steps_done if no aggregation)
        self.steps_done = 0 # number of total steps ever (+1 each optimizer step)
        self.epochs = 0 # number of total epochs ever
        self.samples = 0 # number of total samples ever

        self.step_log = None # number of steps between each log
        self.totbatch = None # total number of batches in one epoch for this training instance
        self.do_step_log = False 

        # Used for logging instead of wandb.log, useful if wandb not imported
        self.logger = None
        self.logging = not no_logging

    def change_lr(self, new_lr):
        """
            Changes the learning rate of the optimizer.
            Might clash with scheduler ?
        """
        for g in self.optim.param_groups:
            g['lr'] = new_lr
        
    def cooldown_lr(self, cooldown_steps):
        """
            When called, starts a cooldown period for the learning rate.
            This is inspired by https://arxiv.org/pdf/2405.18392

            Args :
            cooldown_steps : int, number of steps for the cooldown. 
        """
        self.save_state(suffix='cdstart',epoch=self.epochs) # Snapshot before cooldown
        
        self.scheduler = lrsched.LinearLR(self.optim,1., 0.,total_iters=cooldown_steps)

        print(f'Cooldown period started, doing {cooldown_steps} steps, \
              i.e. {cooldown_steps/self.steps_done*100:.1f}% of total steps.')
        
        self.cooldown_data['cooldown_started']=True


    def save_state(self,epoch:int = None, suffix:str = None):
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
            epoch : int, if not None, will append the epoch number to the state name, and save in 'backups' folder.
            suffix : str, if not None, will be appended to the state name.
        """
        os.makedirs(self.save_loc,exist_ok=True)
        
        # Avoid saving the DataParallel
        saving_model = self.model.module if self.parallel_train else self.model
    
        # Create the state
        try :
            model_config = saving_model.config
        except AttributeError as e:
            raise AttributeError(f"Error while fetching model config ! Make sure model.config is defined. (see ConfigModule doc).")

        state = dict(
        optim_state=self.optim.state_dict(),scheduler_state=self.scheduler.state_dict(),model_state=saving_model.state_dict(),
        model_name=saving_model.class_name,optim_name=self.optim.__class__.__name__,scheduler_name=self.scheduler.__class__.__name__,
        model_config=model_config,session=self.session_hash,run_id=self.run_id, steps_done=self.steps_done,epochs=self.epochs,
        samples=self.samples, batches=self.batches,run_config=self.run_config, cooldown_data=self.cooldown_data
        )

        name = self.run_name
        if(suffix is not None):
            name = name+'_'+suffix

        if (epoch is not None):
            os.makedirs(os.path.join(self.save_loc,'backups'),exist_ok=True)
            name=os.path.join('backups',name+'_'+f'{epoch:.2f}')

        name = name + '.state'
        saveloc = os.path.join(self.save_loc,name)
        torch.save(state,saveloc)

        print(f'Saved training state at {datetime.now().strftime("%H-%M_%d_%m")}')
        print(f'At save,{self.steps_done/1000:.2f}k steps are done, i.e. {self.epochs:.4f} epochs.')

    def load_state(self,state_path : str, strict: bool=True):
        """
            Loads Trainer state, for restoring a run.

            params : 
            state_path : location of the sought-out state_dict
            strict: whether to load the state_dict in strict mode or not
        """

        if(isinstance(self.model, nn.DataParallel)):
            # Unwrap the model, since we saved the state_dict of the model, not the DataParallel
            self.model = self.model.module.to(self.device)
        
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load state')

        state_dict = torch.load(state_path,map_location=self.device)
        if(self.model.config != state_dict['model_config']):
            print('WARNING ! Loaded model configuration and state model_config\
                  do not match. This may generate errors.')
        
        assert self.model.class_name == state_dict['model_name'], f'Loaded model {state_dict["model_name"]} mismatch with current: {self.model.class_name}!'
        self.model.load_state_dict(state_dict['model_state'],strict=strict)
        assert self.optim.__class__.__name__ == state_dict['optim_name'], f'Loaded optimizer : {state_dict["optim_name"]} mismatch with current: {self.optim.__class__.__name__} !'
        self.optim.load_state_dict(state_dict['optim_state'])

        self.cooldown_data = state_dict.get('cooldown_data',self.cooldown_data)

        if(self.cooldown_data['cooldown_started']==True):
            print('Detected started cooldown, switching scheduler to cooldown.')
            self.scheduler = lrsched.LinearLR(self.optim,1., 0.,total_iters=1)# Placeholder data, will be updated in cooldown_lr

        assert self.scheduler.__class__.__name__ == state_dict['scheduler_name'], f'Loaded scheduler : {state_dict["scheduler_name"]} mismatch with current: {self.optim.__class__.__name__} !'
        self.scheduler.load_state_dict(state_dict['scheduler_state'])
        
        self.session_hash = state_dict['session']
        self.run_id = state_dict['run_id']
        self.steps_done = state_dict.get('steps_done',0)
        self.batches = state_dict.get('batches',0)

        self.epochs = state_dict.get('epochs',0)
        self.samples = state_dict.get('samples',0)
        self.run_config = state_dict.get('run_config',{'model':self.model.__class__.__name__})
        # Maybe I need to load also the run_name, we'll see
        
        # Reset the default step_loss, although shouldn't load stuff after a bit of training.
        self.step_loss = []

        print('Training state load successful !')
        print(f'Loaded state had {state_dict["epochs"]} epochs trained.')

    def load_model_from_state(self,state_path : str, strict: bool=True, force_config_match:bool=False):
        """
            Loads only model weights from state. Useful if you just want to load a 
            pretrained model to train it on a different dataset.
        """
        if(isinstance(self.model, nn.DataParallel)):
            # Unwrap the model, since we saved the state_dict of the model, not the DataParallel
            self.model = self.model.module.to(self.device)
        
        if(not os.path.exists(state_path)):
            raise ValueError(f'Path {state_path} not found, can\'t load state')

        state_dict = torch.load(state_path,map_location=self.device)
        if(self.model.config != state_dict['model_config']):
            if(force_config_match):
                raise ValueError(f'Loaded model configuration and state model_config\
                                 do not match. \n Model : {self.model.config} \n State : {state_dict["model_config"]}')
            else:
                print('WARNING ! Loaded model configuration and state model_config\
                    do not match. This may generate errors.')
            
        assert self.model.class_name == state_dict['model_name'], f'Loaded model {state_dict["model_name"]} mismatch with current: {self.model.class_name}!'

        self.model.load_state_dict(state_dict['model_state'],strict=strict)

        print('Model load successful !')
        print(f'Loaded model had {state_dict["epochs"]} epochs trained.')

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
    def model_config_from_state(state_path: str,device: str=None):
        """
            Given the path to a trainer state, returns a 3-uple (model_name,config, weights)
            for the saved model. The model can then be initialized by using config 
            as its __init__ arguments, and load the state_dict from weights.

            Args :
            state_path : path of the saved trainer state
            device : device on which to load. Previous one if None specified

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

    def process_batch(self,batch_data):
        """
            Redefine this in sub-classes. Should return the loss. Batch_data will be on 'cpu' most of the
            time, except if you dataset sets a specific device. Can do logging and other things 
            optionally. Loss is automatically logged, so no need to worry about it. 
            Use self.model to access the model.

            Args :
            batch_data : whatever is returned by the dataloader
            Default class attributes, automatically maintained by the trainer, are :
                - self.device : current model device
                - self.stepnum : current step number since last training/epoch start
                - self.do_step_log : whether we should log this batch or not
                - self.totbatch : total number of minibatches in one epoch.
                - self.epochs: current epoch
                - self.samples : number of samples seen
                - self.steps_done : number of steps done since the beginning of training
            Returns : 2-uple, (loss, data_dict)
        """
        raise NotImplementedError('process_batch should be implemented in Trainer sub-class')

    def process_batch_valid(self,batch_data):
        """
            Redefine this in sub-classes. Should return the loss, as well as 
            the data_dict (potentially updated). Use self.model to access the model (it is already in eval mode).
            Batch_data will be on 'cpu' most of the time, except if you dataset sets a specific device. 
            There should be NO logging done inside this function, only in valid_log.
            Proper use should be to collect the data to be logged in a class attribute,
            and then log it in valid_log (to log once per epoch). Loss is automatically 
            logged, so no need to worry about it. 

            Args :
            batch_data : whatever is returned by the dataloader
            Default class attributes, automatically maintained by the trainer, are :
                - self.device : current model device
                - self.batchnum : current validation mini-batch number
                - self.totbatch : total number of validation minibatches.
                - self.epochs: current epoch
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
            Default class attributes, automatically maintained by the trainer, are :
                - self.device : current model device
                - self.stepnum : current step number since last training/epoch start
                - self.do_step_log : whether we should log this batch or not
                - self.totbatch : total number of minibatches in one epoch.
                - self.epochs: current epoch
                - self.samples : number of samples seen
                - self.steps_done : number of steps done since the beginning of training
        """
        pass

    def valid_log(self):
        """
            To be (optionally) implemented in sub-class. Does the logging 
            at the epoch level, is called every epoch. Only log using commit=False,
            because of sync issues with the epoch x-axis.


            Args :
            Default class attributes, automatically maintained by the trainer, are :
                - self.batchnum : current validation mini-batch number
                - self.totbatch : total number of validation minibatches.
                - self.epochs: current epoch
                - self.samples : number of samples seen
                - self.steps_done : number of steps done since the beginning of training
        """
        pass
    
    def train_init(self,**kwargs):
        """
            Can be redefined for doing stuff just at the beginning of the training,
            for example, freezing weights, preparing some extra variables, or anything really.
            Not mandatory, it is called at the very beginnig of train_epochs/train_steps. The
            dictionary 'train_init_params' is passed as parameter list. As such, it can take
            any combination of parameters.
        """
        pass

    def train_epochs(self,epochs : int,batch_size:int,*,batch_sched:bool=False,save_every:int=50,
                     backup_every: int=None,step_log:int=None,
                     num_workers:int=0,aggregate:int=1,
                     batch_tqdm:bool=True,train_init_params:dict={}):
        """
            # TODO : GET IT UP TO SPEED WITH TRAIN_STEPS
            Trains for specified epoch number. This method trains the model in a basic way,
            and does very basic logging. At the minimum, it requires process_batch and 
            process_batch_valid to be overriden, and other logging methods are optionals.

            data_dict can be used to carry info from one batch to another inside the same epoch,
            and can be used by process_batch* functions for logging of advanced quantities.
            Params :
            epochs : number of epochs to train for
            batch_size : batch size
            batch_sched : if True, scheduler steps at each optimizer step. If used, take care
            to choose scheduler parameters accordingly. If False, will step scheduler at each epoch.
            save_every : saves trainer state every 'save_every' EPOCHS
            backup_every : saves trainer state without overwrite every 'backup_every' EPOCHS
            step_log : If not none, will also log every step_log optim steps, in addition to each epoch
            num_workers : number of workers in dataloader
            aggregate : how many batches to aggregate (effective batch_size is aggreg*batch_size)
            batch_tqdm : if True, will use tqdm for the batch loop, if False, will not use tqdm
            train_init_params : Parameter dictionary passed as argument to train_init
        """
        print('WARNING : train_epochs is not up to date, prefer using train_steps for now. Will be \
              update in the future, if I find it is useful.')
        # Initiate logging
        if(self.logging):
            self._init_logger()
            # For all plots, we plot against the epoch by default
            self.logger.define_metric("*", step_metric='epochs')

        self.train_init(**train_init_params)
        
        train_loader,valid_loader = self.get_loaders(batch_size,num_workers=num_workers)
        validate = valid_loader is not None
        
        self.totbatch = len(train_loader)
        assert self.totbatch > aggregate, f'Aggregate ({aggregate}) should be smaller than number of batches \
                                            in one epoch ({self.totbatch}), otherwise we never step !'
        self.model.train()
        if(self.parallel_train):
            print('Parallel training on devices : ',self.parallel_devices)
            self.model = nn.DataParallel(self.model,device_ids=self.parallel_devices)
        
        if(batch_sched):
            assert self.epochs-self.scheduler.last_epoch<1e-5, f'Epoch mismatch {self.epochs} vs {self.scheduler.last_epoch}'
        else:
            assert int(self.epochs)==self.scheduler.last_epoch, f'Epoch mismatch {self.epochs} vs {self.scheduler.last_epoch}'
        
        #Floor frac epochs, since we start at start of epoch, and also for the scheduler :
        self.epochs = int(self.epochs)
        print('Number of batches/epoch : ',len(train_loader))
        self.stepnum = 0 # This is the current instance number of steps

        self.step_log = step_log
        self.step_loss=[]
        
        
        for ep_incr in tqdm(range(epochs)):
            self.epoch_loss=[]
            n_aggreg = 0

            # Iterate with or without tqdm
            if(batch_tqdm):
                iter_on=tqdm(enumerate(train_loader),total=self.totbatch)
            else :
                iter_on=enumerate(train_loader)

            # Epoch of Training
            for batchnum,batch_data in iter_on :
                # Process the batch
                self.batchnum=batchnum
                n_aggreg = self._step_batch(batch_data,n_aggreg, aggregate, step_sched=batch_sched)

                self.samples+=batch_size
                self.epochs+=1/self.totbatch
                # NOTE: Not great, but batches and steps update in _step_batch by necessity

            self.epochs = round(self.epochs) # round to integer, should already be, but to remove floating point stuff
            
            if(not batch_sched):
                self.scheduler.step()
            
            # Epoch of validation
            if(validate):
                self._validate(valid_loader)
                if(self.logging):
                    self.valid_log()
                self.model.train()
    

            # Log training loss at epoch level
            if(self.logging):
                self.logger.log({'loss/train_epoch':sum(self.epoch_loss)/len(self.epoch_loss)},commit=False)
                self.epoch_log()
                    
                self._update_x_axis()
            
            # Save and backup when applicable
            self._save_and_backup(curstep=ep_incr,save_every=save_every,backup_every=backup_every)

        self.save_state() # Save at the end of training
        if(self.logging):
            self.logger.finish()

    def train_steps(self,steps : int,batch_size:int,*,save_every:int=50,
                    backup_every: int=None, valid_every:int=1000,step_log:int=None,
                    num_workers:int=0,aggregate:int=1,pickup:bool=True,resume_batches:bool=False, 
                    train_init_params:dict={}, cooldown_finish:bool|float=False, cooldown_now:bool=False):
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
            steps : number of steps (optim calls) to train for
            save_every : saves trainer state every 'save_every' epochs
            backup_every : saves trainer state without overwrite every 'backup_every' steps
            valid_every : validates the model every 'valid_every' steps
            step_log : If not none, used for logging every step_log optim steps. In process_batch,
            use self.do_step_log to know when to log. 
            num_workers : number of workers in dataloader
            aggregate : how many batches to aggregate (effective batch_size is aggreg*batch_size)
            pickup : if False, will train for exactly 'steps' more steps. If True, will restart at the previous
            number of steps, and train until TOTAL number of steps is 'steps'. Useful for resuming training,
            if you want to train for a certain specific number of steps. In both cases, the training resumes
            where it left off, only difference is how many MORE steps it will do.
            resume_batches : if True, will resume training assuming the first self.batches on the dataloader
            are already done. Usually, use ONLY if dataloader does NOT shuffle.
            train_init_params : Parameter dictionary passed as argument to train_init
            cooldown_finish : if True or float in [0,1], will finish with a lr cooldown period.
            If float, will cooldown for that fraction of the total steps, otherwise 10%.
            cooldown_now : if True, immediately start a lr cooldown period.
        """
        
        # LR cooldown stuff : 
        update_cooldown = (self.cooldown_data['cooldown_started'] is None) or (self.cooldown_data['cooldown_started'] is False)
        cooldown_wanted = cooldown_now or bool(cooldown_finish)
        total_steps_wanted = steps if pickup else self.steps_done+steps

        if(update_cooldown):
            # No cooldown data, update with the provided one
            self.cooldown_data['do_cooldown'] = cooldown_wanted # yes either if finish, or now cooldown
            self.cooldown_data['cooldown_started'] = False

            percent_cooldown = 0.1 if isinstance(cooldown_finish,bool) else cooldown_finish

            if(cooldown_now):
                cooldown_step_start = 0 # Ensures it registers as started
                steps_remaining = max(0,total_steps_wanted-self.steps_done)
                self.cooldown_data['cooldown_steps'] = int(self.steps_done*percent_cooldown) # start immediately
                # Set to do exactly cooldown steps
                steps = self.cooldown_data['cooldown_steps']
                pickup = False # No pickup, we do exactly the cooldown steps
                self.cooldown_lr(cooldown_steps=self.cooldown_data['cooldown_steps']) # Start cooldown
            elif(bool(cooldown_finish)):
                self.cooldown_data['cooldown_steps'] = int(total_steps_wanted*percent_cooldown) # start at end
                cooldown_step_start = total_steps_wanted - self.cooldown_data['cooldown_steps'] 
            else :
                cooldown_step_start = None # No cooldown
        elif(self.cooldown_data['cooldown_started'] and cooldown_wanted):
            print('WARNING : Cooldown already started, ignoring provided cooldown parameters.')

        # Initiate logging
        if(self.logging):
            self._init_logger()
            # For all plots, we plot against the batches by default, since we do step training
            self.logger.define_metric("*", step_metric='steps')

        self.train_init(**train_init_params)
        
        train_loader,valid_loader = self.get_loaders(batch_size,num_workers=num_workers)
        validate = valid_loader is not None

        self.totbatch = len(train_loader) # Number of batches in one epoch

        assert self.totbatch >= aggregate, f'Aggregate ({aggregate}) should be smaller than number of batches \
                                            in one epoch ({self.totbatch}), otherwise we never step !'
        self.model.train()

        if(self.parallel_train):
            print('Parallel training on devices : ',self.parallel_devices)
            self.model = nn.DataParallel(self.model,device_ids=self.parallel_devices)
        
        print(f'Number of batches/epoch : {len(train_loader)/1000:.2f}k')

        self.step_log = step_log
        self.step_loss=[]
        self.epoch_loss = None

        steps_completed = False
        if(pickup):
            self.stepnum = self.steps_done # Pick up where we left off
        else:
            self.stepnum = 0 # Stepnum used for logging, and when to stop. This means, we train for a further 'steps' steps.

        while not steps_completed:
            iter_on=enumerate(train_loader)

            if(resume_batches):
                resume_batches=False # Only resume for the first epoch, not if we reach and and restart.
                tofastforward = (self.batches)%self.totbatch
                print(f'Fast forwarding {self.batches}%{self.totbatch}={tofastforward} batches')
                for _ in tqdm(range(tofastforward)):
                    # skip batches already done
                    next(iter_on)
                iter_on=tqdm(iter_on,total=self.totbatch-tofastforward)
            else :
                iter_on=tqdm(iter_on,total=self.totbatch)
    
            n_aggreg=0
            # Epoch of Training
            for batchnum,batch_data in iter_on :
                # Process the batch according to the model.
                self.batchnum=batchnum
                n_aggreg = self._step_batch(batch_data,n_aggreg, aggregate,step_sched=True)

                just_stepped = (n_aggreg==0)
                # Validation if applicable
                # We WONT validate at the start, since first steps_done with just_stepped is 1
                if(validate and self.steps_done%valid_every==0 and just_stepped):
                    self._validate(valid_loader)
                    if(self.logging):
                        self.valid_log()
                        self._update_x_axis()
                    self.model.train()

                self.samples+=batch_size
                self.epochs +=1/self.totbatch  # NOTE: Not great, but batches and steps update in _step_batch by necessity


                if(just_stepped):
                    # n_aggreg = 0 whenever we just stepped, so we can check for saving.
                    self._save_and_backup(self.steps_done,save_every,backup_every)

                if(self.stepnum>=steps):
                    steps_completed=True
                    self._save_and_backup(1,save_every,backup_every)
                    break
                
                start_cooldown = self.cooldown_data['do_cooldown'] and \
                                 self.cooldown_data['cooldown_started']==False and \
                                 self.steps_done>=cooldown_step_start # NOTE :maybe put == ? should be okay either way
                
                if(start_cooldown):
                    self.cooldown_lr(cooldown_steps=self.cooldown_data['cooldown_steps'])
        
        self.save_state() # Save at the end of training
        if(self.logging):
            self.logger.finish()

    def _update_x_axis(self):
        """
            Adds and commits pending wandb.log calls, and adds the x-axis metrics,
            to use the correct defaults.

            Args:   
            epoch_mode : bool, whether default x-axis is epoch or not
        """

        self.logger.log({'ksamples' : self.samples//1000},commit=False)
        self.logger.log({'epochs': self.epochs},commit=False)
        self.logger.log({'batches': self.batches},commit=False)
        self.logger.log({'steps': self.steps_done},commit=True)

    def _step_batch(self, batch_data, n_aggreg, aggregate, step_sched):
        """
            Internal function, makes one step of training given minibatch
        """
        # Compute loss, and custom batch logging
        loss = self.process_batch(batch_data)
        
        # Update default logging 
        self.step_loss.append(loss.detach())
        if(self.epoch_loss is not None):
            self.epoch_loss.append(loss.detach())

        # Do default logging
        if(self.do_step_log):
            if(self.logging):
                self.logger.log({'loss/train_step':torch.mean(torch.stack(self.step_loss)).item()},commit=False)

                self._update_x_axis()
            self.step_loss=[]
    
        loss=loss/aggregate # Rescale loss if aggregating.
        loss.backward() # Accumulate gradients
        
        self.batches+=1
        n_aggreg+=1

        if(n_aggreg%aggregate==0):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)

            self.optim.step()
            self.optim.zero_grad()

            if(step_sched):
                self.scheduler.step()

            ## Update the step number
            self.stepnum+=1
            self.steps_done+=1
            n_aggreg=0

        # we won't log at stepnum=0, since we see it at self.stepnum=1 first, but that's ok
        self.do_step_log = (n_aggreg==0 and ((self.stepnum)%self.step_log)==0) if self.step_log else False

        return n_aggreg

    @torch.no_grad()
    def _validate(self,valid_loader)->None:
        self.model.eval()
        val_loss=[]
        t_totbatch = self.totbatch
        t_batchnum = self.batchnum

        self.totbatch = len(valid_loader) # For now we use same totbatch for train and valid, might wanna change that in the future
        print('------ Validation ------')
        iter_on=tqdm(enumerate(valid_loader),total=self.totbatch)

        for (v_batchnum,v_batch_data) in iter_on:
            self.batchnum=v_batchnum
            
            loss = self.process_batch_valid(v_batch_data)
            val_loss.append(loss.detach())
        
        self.totbatch=t_totbatch
        self.batchnum=t_batchnum
        
        # Log validation data
        if(self.logging):
            self.logger.log({'loss/valid':torch.mean(torch.stack(val_loss)).item()},commit=False)
        
    def _init_logger(self):
        """ Initiate the logger, and define the custom x axis metrics """
        self.logger = wandb.init(name=self.run_name,project=self.project_name,config=self.run_config,
                   id = self.run_id,resume='allow',dir=self.data_fold)
        
        self.logger.define_metric("epochs",hidden=True)
        self.logger.define_metric("steps",hidden=True)
        self.logger.define_metric("ksamples",hidden=True)
        self.logger.define_metric("batches",hidden=True)

    def _save_and_backup(self,curstep,save_every,backup_every):
        # First time we check, curstep=1, so with this check we don't save at the beginning
        if curstep%save_every==0 :
            self.save_state()
        
        if backup_every is not None:
            if curstep%backup_every==0 :
                self.save_state(epoch=self.epochs)