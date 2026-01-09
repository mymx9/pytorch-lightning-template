from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models import DinoV3Base


class DinoV3BaseModule(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 2,
        freeze_backbone: bool = True,
        image_size: int = 224,
        hidden_dim: int = 256,
        dropout_rate: float = 0.3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.backbone = DinoV3Base(
            freeze_backbone=freeze_backbone,
            image_size=image_size,
        )
        
        output_dim = self.backbone.get_output_dim()
        
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        metric = Accuracy(task="multiclass", num_classes=num_classes)
        self.train_acc = metric.clone()
        self.val_acc = metric.clone()
        self.test_acc = metric.clone()
        
        loss_metric = MeanMetric()
        self.train_loss = loss_metric.clone()
        self.val_loss = loss_metric.clone()
        self.test_loss = loss_metric.clone()
        
        self.val_acc_best = MaxMetric()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
    
    def on_train_start(self):
        self.val_acc_best.reset()
    
    def model_step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y
    
    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def on_train_epoch_end(self):
        pass
    
    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute())
    
    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.model_step(batch)
        
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        
        return {"loss": loss, "preds": preds, "targets": targets}
    
    def on_test_epoch_end(self):
        pass
    
    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch
        logits = self.forward(x)
        preds = torch.argmax(logits, dim=1)
        return preds


if __name__ == "__main__":
    m = DinoV3BaseModule()
    print(m)
