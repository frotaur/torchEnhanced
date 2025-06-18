import torch, torch.nn as nn, sys, pathlib, os

sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
import torch.optim.lr_scheduler as lrsched
from torch.utils.data import DataLoader
from src.torchenhanced import Trainer, ConfigModule

curfold = pathlib.Path(__file__).parent


class LinSimple(ConfigModule):
    def __init__(self, hidden=1, out=1):
        super().__init__()

        self.layer = nn.Linear(hidden, out)

    def forward(self, x):
        return self.layer(x)


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.data = torch.arange(0, 100).reshape((100, 1)).float()
        self.targets = torch.arange(1, 101).reshape((100, 1)).float()

    def __getitem__(self, index):
        # print('Dispensing item nb: ',index)
        return self.data[index], self.targets[index]  # (1,1), (1,1)

    def __len__(self):
        return len(self.data)


class LinearTrainer(Trainer):
    def __init__(self, run_name: str = None, project_name: str = None, save_loc=None, reach_plateau=100, run_config={}):
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
        )

        self.dataset = SimpleDataset()  # Numbers from 0 to 99
        self.loss_val = []

    def get_loaders(self, batch_size, num_workers=0):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=False), None

    def process_batch(self, batch_data, **kwargs):
        x, y = batch_data  # (B,1), (B,1)
        print(f"Processing batch {self.batches} : {x}")
        pred = self.model(x)  # (B,1)
        loss = ((pred - y) ** 2).mean()  # MSE

        if self.do_step_log:
            self.logger.log({"lossme/train": loss.item()}, commit=False)
            self.logger.log({"other/lr": self.scheduler.get_last_lr()[0]}, commit=False)

        return loss

    def process_batch_valid(self, batch_data):
        x, y = batch_data  # (B,1), (B,1)

        pred = self.model(x)  # (B,1)
        loss = ((pred - y) ** 2).mean()  # MSE

        self.loss_val.append(loss.item())

        return loss

    def valid_log(self):
        self.logger.log({"lossme/valid": sum(self.loss_val) / len(self.loss_val)}, commit=False)
        self.loss_val = []


def test_resume_train():
    """
    Testing is not automatic... Gotta look at the output, whether it does resume at the correct batch
    """
    trainer = LinearTrainer(
        run_name="test_saveforresume",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=7,
    )
    print("-----------------INITIAL TRAINING-----------------")
    trainer.train_steps(steps=29, batch_size=1, step_log=1, save_every=5, aggregate=2, valid_every=1)
    # print(f'Finished at batches : {trainer.batches}, epochs : {trainer.epochs}, steps : {trainer.steps_done}')

    # Test without resuming
    trainer3 = LinearTrainer(
        run_name="test_without_resume",
        project_name="test_torchenhanced",
        save_loc=os.path.join(curfold),
        reach_plateau=7,
    )
    print("-----------------WITHOUT RESUMING TRAINING-----------------")
    trainer3.load_state(
        os.path.join(curfold, "test_torchenhanced", "state", "test_saveforresume.state")
    )  # Load the state
    trainer3.train_steps(
        steps=5, batch_size=1, step_log=1, save_every=5, aggregate=2, valid_every=1, resume_batches=False, pickup=False
    )

    # Test with resuming
    trainer2 = LinearTrainer(
        run_name="test_with_resume", project_name="test_torchenhanced", save_loc=os.path.join(curfold), reach_plateau=7
    )
    print("-----------------RESUMING TRAINING-----------------")
    trainer2.load_state(
        os.path.join(curfold, "test_torchenhanced", "state", "test_saveforresume.state")
    )  # Load the state
    trainer2.train_steps(
        steps=29, batch_size=1, step_log=1, save_every=5, aggregate=2, valid_every=1, resume_batches=True, pickup=True
    )


# if __name__ == "__main__":
#     test_resume_train()
