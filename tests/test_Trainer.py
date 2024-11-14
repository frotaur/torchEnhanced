import torch, torch.nn as nn, sys, pathlib

sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())

import torch.optim.lr_scheduler as lrsched
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset

# Import mnist for tests
from torchvision.datasets import MNIST
import torch.nn.functional as F
from torchvision import transforms as t
from src.torchenhanced import Trainer, DevModule, ConfigModule
import os, time, shutil

curfold = pathlib.Path(__file__).parent


class LinSimple(ConfigModule):
    def __init__(self, hidden=28 * 28, out=10):
        config = locals()
        del config["self"]
        del config["__class__"]

        super().__init__(config)

        self.layer = nn.Linear(hidden, out)
        # Initialize all weights to 1, bias to 0:
        self.layer.weight.data.fill_(1)
        self.layer.bias.data.fill_(0)

    def forward(self, x):
        return self.layer(x)


class LinearTrainer(Trainer):
    def __init__(
        self,
        run_name: str = None,
        project_name: str = None,
        save_loc=None,
        reach_plateau=100,
        run_config={},
        parallel=None,
        device="cpu",
        no_logging=False,
    ):
        model = LinSimple()
        opti = torch.optim.Adam(model.parameters(), lr=1e-3)
        schedo = lrsched.LinearLR(opti, start_factor=0.01, end_factor=1, total_iters=reach_plateau)

        super().__init__(
            model,
            optim=opti,
            scheduler=schedo,
            run_name=run_name,
            project_name=project_name,
            save_loc=save_loc,
            run_config=run_config,
            parallel=parallel,
            device=device,
            no_logging=no_logging,
        )

        self.dataset = Subset(MNIST(os.path.join(curfold, "data"), download=True, transform=t.ToTensor()), range(100))

    def get_loaders(self, batch_size, num_workers=0):
        self.loss_val = []
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=False), DataLoader(
            self.dataset, batch_size=batch_size, shuffle=False
        )

    def process_batch(self, batch_data, **kwargs):
        x, y = batch_data
        x = x.to(self.device)
        y = y.to(self.device)

        x = x.reshape((x.shape[0], -1))

        pred = self.model(x)  # (B,10)
        loss = F.cross_entropy(pred, y, reduction="mean")

        # assert self.stepnum==data_dict['stepnum'], f"Stepnum mismatch {self.stepnum} vs {data_dict['stepnum']}"
        # assert self.step_log==data_dict['step_log'], f"Step_log mismatch {self.step_log} vs {data_dict['step_log']}"
        # assert self.epoch == data_dict['epoch'], f"Epoch mismatch {self.epoch} vs {data_dict['epoch']}"
        # assert self.batchnum == data_dict['batchnum'], f"Batchnum mismatch {self.batchnum} vs {data_dict['batchnum']}"

        if self.do_step_log and self.logging:
            self.logger.log({"lossme/train": loss.item()}, commit=False)
            self.logger.log({"other/lr": self.scheduler.get_last_lr()[0]}, commit=False)

        return loss

    def process_batch_valid(self, batch_data):
        x, y = batch_data
        x = x.to(self.device)
        y = y.to(self.device)
        x = x.reshape((x.shape[0], -1))

        pred = self.model(x)  # (B,10)
        loss = F.cross_entropy(pred, y, reduction="mean")

        self.loss_val.append(loss.item())

        return loss

    def valid_log(self):
        self.logger.log({"lossme/valid": sum(self.loss_val) / len(self.loss_val)}, commit=False)
        self.loss_val = []


def test_parallel():
    # Not much happening here, just test that we have no errors
    # EPOCHS :
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)
    trainer = LinearTrainer(
        run_name="test_ep_paral",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=200,
        run_config={"manamajeff": True},
        parallel=["cuda:0"],
        device="cpu",
    )
    trainer.change_lr(1e-4)

    assert trainer.parallel_train, "Trainer not parallelized"

    if os.path.exists(os.path.join(curfold, "test_torchenhanced", "state", "test_ep_paral.state")):
        trainer.load_state(os.path.join(curfold, "test_torchenhanced", "state", "test_ep_paral.state"))
    time.sleep(2)

    trainer.train_epochs(epochs=20, batch_size=10, step_log=50, save_every=10, aggregate=2, batch_sched=True)

    # STEPS :
    trainer = LinearTrainer(
        run_name="test_steps_paral",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
        parallel=["cuda:0"],
        device="cpu",
    )

    trainer.change_lr(1e-4)
    if os.path.exists(os.path.join(curfold, "test_torchenhanced", "state", "test_steps_paral.state")):
        trainer.load_state(os.path.join(curfold, "test_torchenhanced", "state", "test_steps_paral.state"))

    time.sleep(2)
    trainer.train_steps(steps=200, batch_size=10, step_log=50, save_every=50, aggregate=2, valid_every=100)


def test_aggregate():
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)

    trainer = LinearTrainer(
        run_name="test_aggregate", project_name="test_torchenhanced", save_loc=os.path.join(curfold), reach_plateau=200
    )

    trainer.train_steps(steps=200, batch_size=10, step_log=30, save_every=60, aggregate=2, valid_every=100)

    assert trainer.batches == 400, f"Batch mismatch : {trainer.batches} vs expected 400"
    assert trainer.steps_done == 200, f"Step mismatch : {trainer.steps_done} vs expected 200"


def test_aggregate_visual_compare():
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)

    trainer = LinearTrainer(
        run_name="compare_aggreg", project_name="test_torchenhanced", save_loc=os.path.join(curfold), reach_plateau=200
    )

    trainer.train_steps(steps=600, batch_size=10, step_log=20, save_every=60, aggregate=2, valid_every=100)

    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)

    trainer = LinearTrainer(
        run_name="compare_no_aggreg",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=200,
    )

    trainer.train_steps(steps=600, batch_size=20, step_log=20, save_every=60, aggregate=1, valid_every=100)

    print(f'Compare runs "compare_aggreg" and "compare_noaggreg" on wandb')
    print(
        f'Train Losses won"t match exactly (because for aggregate, we have more datapoints), but validation should be equal'
    )


def test_get_model_from_state():
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)

    trainer = LinearTrainer(
        run_name="test_get_model",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
    )

    trainer.train_steps(steps=100, batch_size=10, step_log=5, save_every=60, aggregate=2, valid_every=100)
    trainer.save_state()
    print(f"Finished at batches : {trainer.batches}, epochs : {trainer.epochs}, steps : {trainer.steps_done}")
    model_old = trainer.model

    model_new = LinearTrainer.get_model_from_state(
        constructor=LinSimple, state_file=os.path.join(curfold, "test_torchenhanced", "state", "test_get_model.state")
    )

    assert model_old.config == model_new.config, f"Config mismatch : {model_old.config} vs {model_new.config}"
    for (name1, param1), (name2, param2) in zip(model_old.state_dict().items(), model_new.state_dict().items()):
        # Check parameter names match
        assert name1 == name2, f"Parameter names don't match: {name1} vs {name2}"

        # Check parameter values match using torch.allclose
        assert torch.allclose(param1, param2), f"Parameter values don't match for {name1}"


def test_autoresume():
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)
    trainer = LinearTrainer(
        run_name="test_resume_train",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
    )

    trainer.train_steps(
        steps=100, batch_size=10, step_log=5, save_every=60, aggregate=1, valid_every=100, load_state=True
    )
    trainer.save_state()
    print(f"Finished at batches : {trainer.batches}, epochs : {trainer.epochs}, steps : {trainer.steps_done}")

    trainer2 = LinearTrainer(
        run_name="test_resume_train",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
    )

    trainer2.train_steps(
        steps=100, batch_size=10, step_log=5, save_every=60, aggregate=1, valid_every=100, load_state=True, pickup=False
    )
    trainer2.save_state()
    print(f"Finished at batches : {trainer2.batches}, epochs : {trainer2.epochs}, steps : {trainer2.steps_done}")

    assert trainer2.steps_done == 200, f"Auto resume failed, should have been 200 steps, got {trainer2.steps_done}"


def test_resume_train():
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)

    trainer = LinearTrainer(
        run_name="test_resume_train",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
    )
    trainer.change_lr(1e-4)
    trainer.train_steps(steps=200, batch_size=10, step_log=5, save_every=60, aggregate=2, valid_every=100)
    trainer.save_state()
    print(f"Finished at batches : {trainer.batches}, epochs : {trainer.epochs}, steps : {trainer.steps_done}")

    # Test with resuming
    trainer2 = LinearTrainer(
        run_name="test_with_resume",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
    )
    trainer2.change_lr(1e-4)
    trainer2.load_state(
        os.path.join(curfold, "test_torchenhanced", "state", "test_resume_train.state")
    )  # Load the state
    trainer2.train_steps(
        steps=150, batch_size=10, step_log=5, save_every=3000, aggregate=2, valid_every=100, resume_batches=True
    )

    # Test without resuming
    trainer3 = LinearTrainer(
        run_name="test_without_resume",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=1500,
    )
    trainer3.change_lr(1e-4)
    trainer3.load_state(
        os.path.join(curfold, "test_torchenhanced", "state", "test_resume_train.state")
    )  # Load the state
    trainer3.train_steps(
        steps=150, batch_size=10, step_log=5, save_every=3000, aggregate=2, valid_every=100, resume_batches=False
    )

    assert trainer2.batches == trainer3.batches, f"Batch mismatch : {trainer2.batches} vs {trainer.batches}"
    assert trainer2.epochs == trainer3.epochs, f"Epoch mismatch : {trainer2.epochs} vs {trainer.epochs}"
    assert trainer2.steps_done == trainer3.steps_done, f"Step mismatch : {trainer2.steps_done} vs {trainer.steps_done}"

    # Look on wandb to see if they are the same


def test_save_weights():
    shutil.rmtree(os.path.join(curfold, "test_torchenhanced"), ignore_errors=True)

    lintra = LinearTrainer(
        run_name="test_save_weights", project_name="test_torchenhanced", save_loc=os.path.join(curfold)
    )
    lintra.save_state()

    Trainer.save_model_from_state(
        state_path=os.path.join(curfold, "test_torchenhanced", "state", "test_save_weights.state"),
        save_dir=os.path.join(curfold, "test_torchenhanced"),
        name="testJEFF",
    )

    assert os.path.isfile(os.path.join(curfold, "test_torchenhanced", "testJEFF.pt")), "Weights not found"
    assert os.path.isfile(os.path.join(curfold, "test_torchenhanced", "testJEFF.config")), "Config not found"


def test_Trainer_config():
    ma = LinSimple(hidden=32, out=15)
    config = ma.config
    assert config == {"hidden": 32, "out": 15}, f"Invalid config : {config}, should be {{'hidden':32, 'out':15}}"


def test_no_logging():
    trainee = LinearTrainer(
        run_name="test_no_logging", project_name="test_torchenhanced", save_loc=os.path.join(curfold), no_logging=True
    )
    trainee.train_steps(steps=200, batch_size=10, step_log=30, save_every=60, aggregate=2, valid_every=100)


# Probably need to add more unit_tests...
if __name__ == "__main__":
    pass
    # # test_save_weights()
    # # test_parallel()
    # # test_aggregate()
    # # test_resume_train()
    # # test_no_logging()
    # # test_aggregate_visual_compare()
    # # test_get_model_from_state()
    # test_autoresume()
