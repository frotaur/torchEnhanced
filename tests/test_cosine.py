import torch,sys,pathlib
sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
from src.torchenhanced.ml.optim import CosineWarmup
import matplotlib.pyplot as plt

def test_cosine():
    pass #(for manual thingy only)
    # opti = torch.optim.SGD([torch.zeros(1)],lr=0.1)
    # sched = CosineWarmup(optimizer=opti, warmup_steps=10, T_0=100,
    #                      lr_shrink=0.8,lr_init=1e-3,T_mult=1,eta_min=1e-3)
    
    # lrs = []
    # epochs = []
    # for ep in range(600):
    #     # SImulate 600 epochs
    #     for batch in range(400):
    #         # Simulate batch
    #         sched.step(ep+batch/400)
    #         epochs.append(ep+batch/400)
    #         lrs.append(sched.get_last_lr())
    # # Plot lrs against epochs
    # plt.plot(epochs,lrs)
    # plt.show()

# test_cosine()