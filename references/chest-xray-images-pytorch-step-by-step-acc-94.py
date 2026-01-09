!pip install pytorch-lightning
!pip install albumentations timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet34
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import time
import random
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pprint
import timm
import pytorch_lightning as pl
import logging
import albumentations as A
from albumentations.pytorch import ToTensorV2

h = {
    "num_epochs": 10,
    "batch_size": 64,
    "image_size": 224,
    "fc1_size": 512,
    "lr": 0.001,
    "model": "efficientnetv2",
    "scheduler": "CosineAnnealingLR10",
    "balance": True,
    "early_stopping_patience": float("inf"),
    "use_best_checkpoint": False
}

class CustomImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, is_valid_file=None):
        self.dataset = datasets.ImageFolder(root, is_valid_file=is_valid_file)
        self.transform = transform
        self.targets = self.dataset.targets

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.transform:
            image = self.transform(image=np.array(image))["image"] / 255.0
        return image, label

    def __len__(self):
        return len(self.dataset)

class PneumoniaDataModule(pl.LightningDataModule):
    def __init__(self, h, data_dir):
        super().__init__()
        self.h = h
        self.data_dir = data_dir

    def setup(self, stage=None):
        data_transforms_train_alb = A.Compose([
            A.Rotate(limit=20),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1, p=1),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0, rotate_limit=0, p=0.5),
            A.Perspective(scale=(0.05, 0.15), keep_size=True, p=0.5),
            A.Resize(height=h["image_size"], width=h["image_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])        

        data_transforms_val_alb = A.Compose([
            A.Resize(self.h["image_size"], self.h["image_size"]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])  

        val_split = 0.2
        train_filenames, val_filenames = self._split_file_names(self.data_dir+"train/", val_split)

        # Load the datasets
        self.train_dataset = CustomImageFolder(self.data_dir+"train/", transform=data_transforms_train_alb, is_valid_file=lambda x: x in train_filenames)
        self.val_dataset = CustomImageFolder(self.data_dir+"train/", transform=data_transforms_val_alb, is_valid_file=lambda x: x in val_filenames)    
        self.test_dataset = CustomImageFolder(self.data_dir+"test/", transform=data_transforms_val_alb, is_valid_file=lambda x: self._is_image_file(x))
    
    def train_dataloader(self):
        if self.h["balance"]:
            sampler = self._create_weighted_sampler(self.train_dataset)
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.h["batch_size"], sampler=sampler, num_workers=4)
        else:
            return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.h["batch_size"], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.h["batch_size"], num_workers=4)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.h["batch_size"], num_workers=4)

    def _extract_patient_ids(self, filename):
        patient_id = filename.split('_')[0].replace("person", "")
        return patient_id

    def _is_image_file(self, file_path):
        return file_path.lower().endswith((".jpeg", ".jpg", ".png"))

    def _split_file_names(self, input_folder, val_split_perc):
        # Pneumonia files contain patient id, so we group split them by patient to avoid data leakage
        pneumonia_patient_ids = set([self._extract_patient_ids(fn) for fn in os.listdir(os.path.join(input_folder, 'PNEUMONIA'))])
        pneumonia_val_patient_ids = random.sample(list(pneumonia_patient_ids), int(val_split_perc * len(pneumonia_patient_ids)))

        pneumonia_val_filenames = []
        pneumonia_train_filenames = []

        for filename in os.listdir(os.path.join(input_folder, 'PNEUMONIA')):
            if self._is_image_file(filename):
                patient_id = self._extract_patient_ids(filename)
                if patient_id in pneumonia_val_patient_ids:
                    pneumonia_val_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))
                else:
                    pneumonia_train_filenames.append(os.path.join(input_folder, 'PNEUMONIA', filename))

        # Normal (by file, no patient information in file names)
        normal_filenames  = [os.path.join(input_folder, 'NORMAL', fn) for fn in os.listdir(os.path.join(input_folder, 'NORMAL'))]
        normal_filenames = [filename for filename in normal_filenames if self._is_image_file(filename)]
        normal_val_filenames = random.sample(normal_filenames, int(val_split_perc * len(normal_filenames)))
        normal_train_filenames = list(set(normal_filenames)-set(normal_val_filenames))

        train_filenames = pneumonia_train_filenames + normal_train_filenames
        val_filenames = pneumonia_val_filenames + normal_val_filenames

        return train_filenames, val_filenames        


    def _create_weighted_sampler(self, dataset):
        targets = dataset.targets
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        weights = [class_weights[label] for label in targets]
        sampler = WeightedRandomSampler(weights, len(weights))
        return sampler

class PneumoniaModel(pl.LightningModule):
    def __init__(self, h):
        super().__init__()
        self.h = h
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        metrics = {"val_loss": loss, "val_acc": acc}
        self.log_dict(metrics, on_epoch=True, on_step=True, prog_bar=True)
        return metrics        

    def on_test_epoch_start(self):
        self.test_outputs = []

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        preds = torch.argmax(outputs, dim=1)    
        self.test_outputs.append({"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels})
        return {"test_loss": loss, "test_acc": acc, "preds": preds, "labels": labels}

    def on_test_epoch_end(self):
        test_loss_mean = torch.stack([x["test_loss"] for x in self.test_outputs]).mean()
        test_acc_mean = torch.stack([x["test_acc"] for x in self.test_outputs]).mean()

        self.test_predicted_labels = torch.cat([x["preds"] for x in self.test_outputs], dim=0).cpu().numpy()
        self.test_true_labels = torch.cat([x["labels"] for x in self.test_outputs], dim=0).cpu().numpy()
        #Todo: remove f1 calculation from here
        f1 = f1_score(self.test_true_labels, self.test_predicted_labels)

        self.test_f1 = f1
        self.test_acc = test_acc_mean.cpu().numpy() #Todo - fix it

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.h["lr"])
        scheduler_dic = self._configure_scheduler(optimizer)

        if (scheduler_dic["scheduler"]):
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler_dic
            }            
        else:
            return optimizer

    def _configure_scheduler(self, optimizer):
        scheduler_name = self.h["scheduler"]
        lr = self.h["lr"]
        if (scheduler_name==""):
            return {
                "scheduler": None
            }
        if (scheduler_name=="CosineAnnealingLR10"):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=h["num_epochs"], eta_min=lr*0.1) #*len(train_loader) if "step"
            return {
                "scheduler": scheduler,
                "interval": "epoch"
            }
        if (scheduler_name=="ReduceLROnPlateau5"):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
            return {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_loss",
                "strict": True
            }
        print ("Error. Unknown scheduler name '{scheduler_name}'")
        return None

    def _create_model(self):
        if (self.h["model"]=="efficientnetv2"):
            return timm.create_model("tf_efficientnetv2_b0", pretrained=True, num_classes=2)
        if (self.h["model"]=="fc"):
            return nn.Sequential(
                nn.Flatten(),
                nn.Linear(3 * self.h["image_size"] * self.h["image_size"], self.h["fc1_size"]),
                nn.ReLU(),
                nn.Linear(self.h["fc1_size"], 2)
            )
        if (self.h["model"]=="cnn"):
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(64 * (self.h["image_size"] // 8) * (self.h["image_size"] // 8), 512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, 2)
            )
        if (self.h["model"]=="resnet34"):
            model = resnet34(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, 2)
            return model       

class InfoPrinterCallback(pl.Callback):
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"CPU cores: {os.cpu_count()}, Device: {device}, GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"CPU cores: {os.cpu_count()}, Device: {device}")

        # Print hyperparameters for records
        print("Hyperparameters:")
        pprint.pprint(h, indent=4)

    def setup(self, trainer, pl_module, stage):
        self.start_time = time.time()     

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip the sanity check
        if trainer.sanity_checking:
            return

        epoch = trainer.current_epoch
        total_epochs = trainer.max_epochs

        elapsed_time = time.time() - self.start_time
        avg_time_per_epoch = elapsed_time / (epoch + 1)
        avg_time_per_epoch_min, avg_time_per_epoch_sec = divmod(avg_time_per_epoch, 60)

        remaining_epochs = total_epochs - epoch - 1
        remaining_time = remaining_epochs * avg_time_per_epoch
        remaining_time_min, remaining_time_sec = divmod(remaining_time, 60)

        print(f"Epoch {epoch + 1}/{total_epochs}: ", end="")

        if "val_loss" in trainer.callback_metrics:
            validation_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
            #self.validation_losses.append(validation_loss)            
            print(f"Validation Loss = {validation_loss:.4f}", end="")
        else:
            print(f"Validation Loss not available", end="")

        if "train_loss_epoch" in trainer.logged_metrics:
            train_loss = trainer.logged_metrics["train_loss_epoch"].cpu().numpy()
            print(f", Train Loss = {train_loss:.4f}", end="")
        else:
            print(f", Train Loss not available", end="")

        print(f", Epoch Time: {avg_time_per_epoch_min:.0f}m {avg_time_per_epoch_sec:02.0f}s, Remaining Time: {remaining_time_min:.0f}m {remaining_time_sec:02.0f}s")

    def plot_losses(self):
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()   

class PlotTestConfusionMatrixCallback(pl.Callback):
    def on_test_end(seld, trainer, pl_module):
        cm = confusion_matrix(pl_module.test_true_labels, pl_module.test_predicted_labels)
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
        disp.plot()
        plt.show()   

class PlotTrainingLogsCallback(pl.Callback):
    def __init__(self):
        self.validation_losses = []
        self.training_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        if "train_loss_epoch" in trainer.logged_metrics:
            train_loss = trainer.logged_metrics["train_loss_epoch"].cpu().numpy()
            self.training_losses.append(train_loss)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        if "val_loss" in trainer.callback_metrics:
            validation_loss = trainer.callback_metrics["val_loss"].cpu().numpy()
            self.validation_losses.append(validation_loss)            

    def on_fit_end(self, trainer, pl_module):
        plt.plot(self.training_losses, label="Training Loss")
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()   

def check_solution(h, verbose):
    pneumonia_data = PneumoniaDataModule(h, "/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/")
    pneumonia_model = PneumoniaModel(h)

    # Callbacks
    info_printer = InfoPrinterCallback()

    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=h["early_stopping_patience"],
        verbose=True,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="model_checkpoints",
        monitor="val_loss",
        verbose=True,
    )

    callbacks = [info_printer, early_stopping, checkpoint_callback]
    if (verbose):
        callbacks.append(PlotTestConfusionMatrixCallback())
        callbacks.append(PlotTrainingLogsCallback())

    trainer = pl.Trainer(
        max_epochs=h["num_epochs"],
        accelerator="auto",
        callbacks=callbacks,
        log_every_n_steps=1,
        fast_dev_run=False
    )

    trainer.fit(pneumonia_model, datamodule=pneumonia_data)

    if (h["use_best_checkpoint"]):
        #Debug lines
        trainer.test(pneumonia_model, datamodule=pneumonia_data)
        print(f"Last: F1= {pneumonia_model.test_f1:.4f}, Acc= {pneumonia_model.test_acc:.4f}")

        best_model_path = checkpoint_callback.best_model_path
        best_model = PneumoniaModel.load_from_checkpoint(best_model_path, h=h)
        pneumonia_model = best_model

    trainer.test(pneumonia_model, datamodule=pneumonia_data)
    print(f"Best: F1= {pneumonia_model.test_f1:.4f}, Acc= {pneumonia_model.test_acc:.4f}")


    return pneumonia_model.test_f1, pneumonia_model.test_acc


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

f1_array = np.array([])
accuracy_array = np.array([])
start_time = time.time()

repeats = 5
for i in range(repeats):
    print("===============================================")
    print(f"Running solution {i+1}/{repeats}")
    f1, accuracy = check_solution(h, verbose=(i==0))
    print(f"F1 = {f1:.2f}, accuracy = {accuracy:.2f} ")
    f1_array = np.append(f1_array, f1)
    accuracy_array = np.append(accuracy_array, accuracy) 

# Calculate elapsed time and remaining time
repeat_time = (time.time() - start_time) / repeats
repeat_time_min, repeat_time_sec = divmod(repeat_time, 60)

# Printing final results
print("Results")
print(f"F1: {np.mean(f1_array):.1%} (+-{np.std(f1_array):.1%})")
print(f"Accuracy: {np.mean(accuracy_array):.1%} (+-{np.std(accuracy_array):.1%})")
print(f"Time of one solution: {repeat_time_min:.0f}m {repeat_time_sec:.0f}s")
print(f" | {np.mean(f1_array):.1%} (+-{np.std(f1_array):.1%}) | {np.mean(accuracy_array):.1%} (+-{np.std(accuracy_array):.1%}) | {repeat_time_min:.0f}m {repeat_time_sec:.0f}s")

# Print hyperparameters for reminding what the final data is for
print("Hyperparameters:")
pprint.pprint(h, indent=4)