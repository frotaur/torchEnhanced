# TorchEnhanced
Wrappers for pytorch stuff I use on the daily. Basically a minimal 'pytorch lightning', I was just not aware it existed at the time of creation.


## Basic Usage
Install with `pip install torchenhanced`.  

Here we describe how to use at a basic level the different components included in `torchenhanced`. There are many unrelated things it helps to do, so we dedicate a section to each.

### Improved nn.Module
`torchenhanced` defines two new classes which are meant as stand-in for `nn.Module`.

**DevModule**
Import with `from torchenhanced import DevModule`.
`DevModule` is short for 'DeviceModule'. It is a `nn.Module`, but has an additional attribute `device`, which helps keeps track of the current device it is on.

Use it just like `nn.Module`, except it needs to be initialized with a device :

```
    class MyModule(DevModule):
        def __init__(hidden, device='cpu'):
            super().__init__(device)
            layer = nn.Linear(hidden,hidden,device=self.device)
```

Works just [STILL WIP]


