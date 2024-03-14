import torch.nn as nn
from torch import optim
import lightning as L

from models import DeepDLinear, DeepLinear, DeepNLinear


class LinearModel(L.LightningModule):
    def __init__(
        self,
        model_name,
        seq_len,
        pred_len,
        num_layer=1,
        enc_in=1,
        individual=False,
        log_grad=False,
        learning_rate=1e-5,
        **kwargs,
    ):
        super().__init__()
        model_dict = {
            "DeepDLinear": DeepDLinear.Model,
            "DeepLinear": DeepLinear.Model,
            "DeepNLinear": DeepNLinear.Model,
        }
        self.model_name = model_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_layer = num_layer
        self.enc_in = enc_in
        self.individual = individual
        self.learning_rate = learning_rate
        self.log_grad = log_grad

        self.model = model_dict[self.model_name](
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            num_layer=self.num_layer,
            enc_in=self.enc_in,
            individual=self.individual,
            **kwargs,
        )
        self.criterion = nn.MSELoss()

        self.validation_step_losses = []
        self.training_step_losses = []
        self.test_step_losses = []

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        x = x.float()
        y = y.float()

        outputs = self.model(x)
        outputs = outputs[:, :, :1]
        y = y[:, :, :1]
        loss = self.criterion(outputs, y)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)

        self.training_step_losses.append(loss.item())
        self.log("train_loss_step", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        x = x.float()
        y = y.float()
        return self.model(x)

    def on_train_epoch_end(self):
        losses = self.training_step_losses
        avg_loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalars("loss", {"train": avg_loss}, self.current_epoch)
        self.log("train_loss", avg_loss, prog_bar=True, logger=True)
        self.training_step_losses.clear()

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)

        self.validation_step_losses.append(loss.item())

    def on_validation_epoch_end(self):
        losses = self.validation_step_losses
        avg_loss = sum(losses) / len(losses)
        self.logger.experiment.add_scalars("loss", {"val": avg_loss}, self.current_epoch)
        self.log("val_loss", avg_loss, prog_bar=True, logger=True)
        self.validation_step_losses.clear()

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch)

        self.test_step_losses.append(loss.item())

    def on_test_epoch_end(self) -> None:
        losses = self.test_step_losses
        avg_loss = sum(losses) / len(losses)
        self.log("test_loss", avg_loss, prog_bar=True, logger=True)
        self.test_step_losses.clear()

    def on_after_backward(self):
        if self.log_grad:
            global_step = self.global_step
            for name, param in self.named_parameters():
                self.logger.experiment.add_histogram(name, param, global_step)
                if param.requires_grad:
                    self.logger.experiment.add_histogram(f"{name}_grad", param.grad, global_step)

    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.learning_rate)
        # sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        # sch = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

        return opt
