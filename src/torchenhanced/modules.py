import torch.nn as nn
import torch


class DevModule(nn.Module):
    """
        Extremely small wrapper for nn.Module.
        Simply adds a method device() that returns
        the current device the module is on. Changes if
        self.to(...) is called.

        args :
        config : Dictionary that contains the key:value pairs needed to 
        instantiate the model (essentially the arguments of the __init__ method).
    """
    def __init__(self):
        super().__init__()

        self.register_buffer('_devtens',torch.empty(0))

    @property
    def device(self):
        return self._devtens.device

    @property
    def paranum(self):
        return sum(p.numel() for p in self.parameters())
    
    @property
    def config(self):
        """
            Returns a json-serializable dict containing the config of the model.
            Essentially a key-value dictionary of the init arguments of the model.
            Should be redefined in sub-classes.
        """
        return self._config


class ConfigModule(DevModule):
    """
        Same as DevModule, but with a config property that
        stores the necessary data to reconstruct the model.
        Use preferably over DevModule, especially with use with Trainer.

        args :
        config : Dictionary that contains the key:value pairs needed to 
        instantiate the model (i.e. the argument values of the __init__ method)
    """
    def __init__(self, config:dict):
        super().__init__()

        self._config = config
        self._config['name'] = self.__class__.__name__

    @property
    def config(self):
        """
            Returns a json-serializable dict containing the config of the model.
            Essentially a key-value dictionary of the init arguments of the model.
            Should be redefined in sub-classes.
        """
        return self._config

