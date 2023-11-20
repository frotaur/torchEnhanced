import torch.nn as nn
import torch


class DevModule(nn.Module):
    """
        Extremely small wrapper for nn.Module.
        Simply adds a method device() that returns
        the current device the module is on. Changes if
        self.to(...) is called.

        args :
        device : optional, default 'cpu'. Device to initialize the module on.
    """
    def __init__(self, device:str='cpu'):
        super().__init__()

        self.register_buffer('_devtens',torch.empty(0, device=device))

    @property
    def device(self):
        return self._devtens.device

    @property
    def paranum(self):
        return sum(p.numel() for p in self.parameters())


class ConfigModule(DevModule):
    """
        Same as DevModule, but with a config property that
        stores the necessary data to reconstruct the model.
        Use preferably over DevModule, especially with use with Trainer.

        args :
        config : Dictionary that contains the key:value pairs needed to 
        instantiate the model (i.e. the argument values of the __init__ method)
    """
    def __init__(self, config:dict, device:str='cpu'):
        super().__init__(device=device)

        self._config = config

        self.class_name = self.__class__.__name__ # Use this instead if, at time of saving, you need the name.
        # self._config['name'] = self.__class__.__name__ # Decided against using name in config, it defeats the purpose.

    @property
    def config(self):
        """
            Returns a json-serializable dict containing the config of the model.
            Essentially a key-value dictionary of the init arguments of the model.
            Should be redefined in sub-classes.
        """
        return self._config

