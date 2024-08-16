from test_trainer import LinSimple, LinearTrainer
from pathlib import Path
import os, torch, shutil
from torch.optim.lr_scheduler import LinearLR
curfold = Path(__file__).parent
    

def remove_saves():
    shutil.rmtree(os.path.join(curfold,'test_torchenhanced'),ignore_errors=True)

def test_cooldown_finish():
    remove_saves()
    lintra = LinearTrainer(run_name='test_lr_cd', project_name='test_torchenhanced',
                           reach_plateau=200, save_loc=os.path.join(curfold))
    
    lintra.train_steps(steps=1000, step_log=1, batch_size=4, cooldown_finish=True)

    assert lintra.scheduler.get_last_lr()[0] == 0., f"Cooldown_finish failed : scheduler last lr : {lintra.scheduler.get_last_lr()[0]}"
    assert isinstance(lintra.scheduler, LinearLR), f"Cooldown_finish failed : scheduler type : {type(lintra.scheduler)}"

def test_cooldown_custom_steps():
    remove_saves()
    lintra = LinearTrainer(run_name='test_lr_custom_cd', project_name='test_torchenhanced',
                           reach_plateau=200, save_loc=os.path.join(curfold))
    
    lintra.train_steps(steps=1000, step_log=1, batch_size=4, cooldown_finish=0.2)

    assert lintra.scheduler.get_last_lr()[0] == 0., f"Cooldown_custom failed : scheduler last lr : {lintra.scheduler.get_last_lr()[0]}"
    assert isinstance(lintra.scheduler, LinearLR), f"Cooldown_custom failed : scheduler type : {type(lintra.scheduler)}"

def test_cooldown_now():
    remove_saves()
    lintra = LinearTrainer(run_name='test_lr_cd_now', project_name='test_torchenhanced',
                           reach_plateau=200, save_loc=os.path.join(curfold))
    
    lintra.train_steps(steps=500, step_log=1, batch_size=4) # train 500 steps with nocooldown

    lintra.load_state(os.path.join(curfold,'test_torchenhanced','state','test_lr_cd_now.state')) # Load the finished state

    lintra.train_steps(steps=500, step_log=1, batch_size=4, pickup=False,cooldown_now=True) # train 500 steps with cooldown

    assert lintra.scheduler.get_last_lr()[0] == 0., f"Cooldown_now failed : scheduler last lr : {lintra.scheduler.get_last_lr()[0]}"
    assert isinstance(lintra.scheduler, LinearLR), f"Cooldown_now failed : scheduler type : {type(lintra.scheduler)}"

def test_cooldown_pickup():
    remove_saves()
    lintra = LinearTrainer(run_name='test_lr_cd_pickup', project_name='test_torchenhanced',
                           reach_plateau=200, save_loc=os.path.join(curfold))
    
    lintra.train_steps(steps=500, step_log=1, batch_size=4) # train 500 steps with nocooldown

    lintra.load_state(os.path.join(curfold,'test_torchenhanced','state','test_lr_cd_pickup.state')) # Load the finished state

    lintra.train_steps(steps=800, step_log=1, batch_size=4, pickup=True,cooldown_finish=True) 

    assert lintra.scheduler.get_last_lr()[0] == 0., f"Cooldown_pickup failed : scheduler last lr : {lintra.scheduler.get_last_lr()[0]}"
    assert isinstance(lintra.scheduler, LinearLR), f"Cooldown_pickup failed : scheduler type : {type(lintra.scheduler)}"

def test_cooldown_nopickup():
    remove_saves()
    lintra = LinearTrainer(run_name='test_lr_cd_nopickup', project_name='test_torchenhanced',
                           reach_plateau=200, save_loc=os.path.join(curfold))
    
    lintra.train_steps(steps=500, step_log=1, batch_size=4) # train 500 steps with nocooldown

    lintra.load_state(os.path.join(curfold,'test_torchenhanced','state','test_lr_cd_nopickup.state')) # Load the finished state

    lintra.train_steps(steps=500, step_log=1, batch_size=4, pickup=False,cooldown_finish=True) 

    assert lintra.scheduler.get_last_lr()[0] == 0., f"Cooldown_nopickup failed : scheduler last lr : {lintra.scheduler.get_last_lr()[0]}"
    assert isinstance(lintra.scheduler, LinearLR), f"Cooldown_nopickup failed : scheduler type : {type(lintra.scheduler)}"

if __name__	== '__main__':
    test_cooldown_finish()
    test_cooldown_custom_steps()
    test_cooldown_now()
    test_cooldown_pickup()
    test_cooldown_nopickup()