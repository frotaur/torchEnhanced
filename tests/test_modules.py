import torch,sys,pathlib, torch.nn as nn, os
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

curpath = pathlib.Path(__file__).parent
from src.torchenhanced.modules import DevModule, ConfigModule


def test_device_initialization():
    module = DevModule(device='cpu')
    assert str(module.device) == 'cpu', 'Problem in DevModule device initialization'

    if torch.cuda.is_available():
        module_cuda = DevModule(device='cuda')
        assert str(module_cuda.device).startswith('cuda'), 'Problem in DevModule device initialization'

def test_paranum_empty_module():
    module = DevModule()
    assert module.paranum==0, 'Problem in DevModule paranum'


def make_test_configmodule():
    class TestConfigModule(ConfigModule):
        def __init__(self, hidden = 28*28, out =10):
            config = {'hidden':hidden, 'out':out}

            super().__init__(config)

            self.layer = nn.Linear(hidden, out)
        
        def forward(self, x):
            return self.layer(x)

    testmodule = TestConfigModule()
    
    return testmodule

def test_configmodule_config():
    module = make_test_configmodule()
    config = module.config
    assert config['hidden']==28*28, 'Problem in ConfigModule config'
    assert config['out']==10, 'Problem in ConfigModule config'

def test_configmodule_weights():
    module = make_test_configmodule()

    module.save_weights(os.path.join(curpath,'test_torchenhanced/test_weights.pt'))

    module.load_weights(os.path.join(curpath,'test_torchenhanced/test_weights.pt'))

    if( torch.cuda.is_available() ):
        module.to('cuda')
    
    module.load_weights(os.path.join(curpath,'test_torchenhanced/test_weights.pt'))

    assert module.paranum>0, 'Problem in ConfigModule load_weights'

# test_configmodule_weights()