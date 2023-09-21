import torch.optim.lr_scheduler as lrsched
from torch.optim.optimizer import Optimizer
import math

class _enable_get_lr_call:

    def __init__(self, o):
        self.o = o

    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False

class CosineWarmup(lrsched.CosineAnnealingWarmRestarts):
    """
        LR-scheduler with cosine learning rate, and a warmup phase additionally.
        Either input values for number of BATCHES, and .step() each batch, or input
        values for EPOCHS, and .step(epoch+batch/total_batches) each batch. This is less
        than ideal in my opinion, but the Trainer integrates that behavior (for now)

        Parameters :
        optimizer : torch.optim.Optimizer
            Optimizer on which the learning rate will act.
        warmup_steps : int
            Number of warmup steps.
        T_0 : int
            Cosine (initial) period in number of steps
        lr_init : float
            Value of lr at start of warmup
        lr_shrink : float (0.,1.)
            Multiplies lr at each reset, shrinking it
        T_mult : float (1.,infty)
            How much the period is multiplied at each restart (every period)
        eta_min : float
            minimum lr that is reached during oscillations
    """

    def __init__(self, optimizer: Optimizer, warmup_steps: int,T_0:int,lr_shrink:float=0.75, lr_init:float=1e-7,T_mult=1, eta_min=0, last_epoch=-1, verbose=False) -> None:
        self.warmup_steps=warmup_steps
        self.t=last_epoch
        self.lr_init = lr_init
        self.lr_shrink = lr_shrink
        self.lr_shrink_i = 1
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch, verbose)

        

    def get_lr(self):
        if(self.t<=self.warmup_steps):
            # Warming up
            return [self.lr_init+(base_lr-self.lr_init)*self.t/self.warmup_steps for base_lr in self.base_lrs]
        else :
            # It's warm. T_cur should have correct translated value.
            return [self.lr_shrink_i*(self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2)
                for base_lr in self.base_lrs]

    def step(self, epoch = None):
        """
            Almost equivalent to original cosineAnnealing step, but with warmup.
        """
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        if epoch is None:
            epoch = self.last_epoch + 1
        
        if epoch < 0:
            raise ValueError("Expected non-negative epoch, but got {}".format(epoch))
    
        # Update values using closed-form
        self.t=epoch
        if self.t>self.warmup_steps:
            t_eff = self.t-self.warmup_steps
            if t_eff >= self.T_0:
                if self.T_mult == 1:
                    n = t_eff//self.T_i
                    self.T_cur = t_eff % self.T_0
                    self.T_i=self.T_0
                else:
                    n = int(math.log((t_eff / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.T_cur = t_eff - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
                self.lr_shrink_i = self.lr_shrink**n
            else:
                self.T_i = self.T_0
                self.t=epoch
                self.T_cur = self.t-self.warmup_steps
    
        self.last_epoch = math.floor(epoch)



        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group['lr'] = lr
                self.print_lr(self.verbose, i, lr, epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]
