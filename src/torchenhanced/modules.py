import torch.nn as nn
import torch
import inspect
import os


class DevModule(nn.Module):
    """
    Extremely small wrapper for nn.Module.
    Simply adds a method device() that returns
    the current device the module is on. Changes if
    self.to(...) is called.

    args :
    device : optional, default 'cpu'. Device to initialize the module on.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()

        self.register_buffer("_devtens", torch.empty(0, device=device))

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

    Currently INCOMPATIBLE with the use of *args in __init__. **kwargs are fine.

    Args :
    device : optional, default 'cpu'. Explicitely provide if no argument 'device' is present in __init__.
    config : deprecated, only for compatibility. Do not use.
    """

    def __init__(self, config=None, device="cpu"):
        frame = inspect.currentframe().f_back
        args, _, kwarg_name, local_vars = inspect.getargvalues(frame)
        # Return a dictionary excluding 'self'
        self._config = {arg: local_vars[arg] for arg in args if arg != "self"}
        if kwarg_name is not None:
            self._config.update(local_vars[kwarg_name])  # Add **kwargs

        if "device" in self._config.keys():
            device = self._config["device"]

        super().__init__(device=device)

        # Use this if, at time of saving, you need the name.
        self.class_name = self.__class__.__name__

    @property
    def config(self):
        """
        Returns a json-serializable dict containing the config of the model.
        Essentially a key-value dictionary of the init arguments of the model.
        Should be redefined in sub-classes.
        """
        return self._config

    def save_config(self, save_folder: str, save_name: str = None):
        """
        Saves the full configuration of the model, including
        constructor arguments, and weights. Can be loaded with
        load_config.

        Args:
            save_loc : Path to save the configuration to.
            save_name : Name of the file to save the configuration to.
            If None, saves with the class name.
        """
        full_dict = {"config": self.config, "state_dict": self.state_dict()}
        torch.save(
            full_dict,
            os.path.join(save_folder, save_name + ".config" if save_name is not None else self.class_name + ".config"),
        )

    def load_config(self, config_path: str, device: str = "cpu"):
        """
        Loads the configuration of the model from a file.

        Args:
            config_path : Path to the configuration file, saved with 'save_config'.
            device : Device to initialize the model on.
        """
        saved_data = torch.load(config_path, weights_only=True, map_location=device)
        self.load_state_dict(saved_data["state_dict"])
        assert self._config == saved_data["config"], "Configuration mismatch. Model and saved config do not match."

    @classmethod
    def from_config(cls, config_path: str, device: str = "cpu"):
        """
        Returns an instance of the model with the given configuration.

        Args:
            config_path: Path to the configuration file, saved with 'save_config'.
            device: Device to initialize the model on.
        """
        saved_data = torch.load(config_path, weights_only=True, map_location=device)
        config = saved_data["config"]
        config["device"] = device  # Update device to current

        # Create instance of whatever class this method was called on
        instance = cls(**config)
        instance.load_state_dict(saved_data["state_dict"])
        return instance

    def load_weights(self, weights_loc: str, strict: bool = True):
        """
        Loads weights from a file into the model, mapping them to current device.

        Args:
        weights_loc :  Path to the file containing the weights.
        strict : If True, loads only if the model has exactly the same keys as the weights.
        """
        self.load_state_dict(torch.load(weights_loc, weights_only=True,map_location=self.device), strict=strict)

    def save_weights(self, weights_loc: str):
        """
        Saves the weights of the model to a file.

        Args:
        weights_loc :  Path to the file to save the weights to.
        """
        torch.save(self.state_dict(), weights_loc)
