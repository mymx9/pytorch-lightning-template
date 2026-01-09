import os
import random
from typing import Callable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class ChestXRayDataset(Dataset):
    """Chest X-Ray binary classification dataset.
    
    Dataset structure:
        train/
            NORMAL/
                *.jpeg
            PNEUMONIA/
                *.jpeg
        test/
            NORMAL/
                *.jpeg
            PNEUMONIA/
                *.jpeg
    
    Labels:
        NORMAL: 0
        PNEUMONIA: 1
    """
    
    def __init__(
        self,
        root_dir: str,
        filenames: list[str],
        transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize dataset.
        
        Args:
            root_dir: Root directory of the dataset
            filenames: List of image file paths
            transform: Optional transform to be applied on images
            is_valid_file: Optional function to filter valid files
        """
        self.root_dir = root_dir
        self.filenames = filenames
        self.transform = transform
        self.is_valid_file = is_valid_file
        
        self.targets = self._load_targets()
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get item by index.
        
        Args:
            index: Index of the item
            
        Returns:
            Tuple of (image, label) where:
                - image: Tensor of shape (C, H, W)
                - label: Integer class label (0 or 1)
        """
        image_path = self.filenames[index]
        image = Image.open(image_path).convert('RGB')
        
        label = self._extract_label(image_path)
        
        if self.transform:
            image = self.transform(image=np.array(image))["image"]
        
        return image, label
    
    def _extract_label(self, image_path: str) -> int:
        """Extract label from image path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            0 for NORMAL, 1 for PNEUMONIA
        """
        if 'NORMAL' in image_path.upper():
            return 0
        elif 'PNEUMONIA' in image_path.upper():
            return 1
        else:
            raise ValueError(f"Unknown label in path: {image_path}")
    
    def _load_targets(self) -> list[int]:
        """Load targets for weighted sampling.
        
        Returns:
            List of labels for all samples
        """
        targets = []
        for filename in self.filenames:
            label = self._extract_label(filename)
            targets.append(label)
        return targets


class ChestXRayBinaryDataModule(pl.LightningDataModule):
    """LightningDataModule for Chest X-Ray binary classification.
    
    A DataModule implements 6 key methods:
        def prepare_data(self):
            # Download/verify data
        def setup(self, stage):
            # Load and split data
        def train_dataloader(self):
            # Return train dataloader
        def val_dataloader(self):
            # Return validation dataloader
        def test_dataloader(self):
            # Return test dataloader
        def predict_dataloader(self):
            # Return predict dataloader
    
    Key features:
        - Patient-level split for PNEUMONIA to avoid data leakage
        - Random split for NORMAL
        - Weighted sampling for class imbalance
        - Data augmentation for training set
        - Flexible configuration via YAML
    
    Docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """
    
    def __init__(
        self,
        data_dir: str = "/root/autodl-fs/data/chest_xray",
        train_val_split: float = 0.2,
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        image_size: int = 224,
        balance: bool = True,
    ):
        """Initialize Chest X-Ray DataModule.
        
        Args:
            data_dir: Root directory of the dataset
            train_val_split: Validation split ratio (0.2 = 20%)
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            pin_memory: Whether to use pinned memory
            image_size: Target image size for resizing
            balance: Whether to use weighted sampling for class imbalance
        """
        super().__init__()
        self.save_hyperparameters()
        
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        
        self.train_dataset: Dataset = None
        self.val_dataset: Dataset = None
        self.test_dataset: Dataset = None
    
    @property
    def num_classes(self) -> int:
        """Return the number of classes (binary classification)."""
        return 2
    
    def prepare_data(self):
        """Verify data directory structure."""
        train_dir = os.path.join(self.hparams.data_dir, 'train')
        test_dir = os.path.join(self.hparams.data_dir, 'test')
        
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Train directory not found: {train_dir}")
        if not os.path.exists(test_dir):
            raise FileNotFoundError(f"Test directory not found: {test_dir}")
        
        for subdir in ['NORMAL', 'PNEUMONIA']:
            if not os.path.exists(os.path.join(train_dir, subdir)):
                raise FileNotFoundError(f"Subdirectory not found: {os.path.join(train_dir, subdir)}")
    
    def setup(self, stage: str = None):
        """Load and split datasets.
        
        Key features:
            - Patient-level split for PNEUMONIA to avoid data leakage
            - Random split for NORMAL
            - Weighted sampling for class imbalance
        """
        if self.train_dataset and self.val_dataset and self.test_dataset:
            return
        
        train_dir = os.path.join(self.hparams.data_dir, 'train')
        test_dir = os.path.join(self.hparams.data_dir, 'test')
        
        train_filenames, val_filenames = self._split_train_val(
            train_dir, 
            self.hparams.train_val_split
        )
        
        test_filenames = self._get_test_filenames(test_dir)
        
        self.train_dataset = ChestXRayDataset(
            root_dir=train_dir,
            filenames=train_filenames,
            transform=self.train_transform,
        )
        
        self.val_dataset = ChestXRayDataset(
            root_dir=train_dir,
            filenames=val_filenames,
            transform=self.val_transform,
        )
        
        self.test_dataset = ChestXRayDataset(
            root_dir=test_dir,
            filenames=test_filenames,
            transform=self.val_transform,
        )
    
    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        if self.hparams.balance:
            sampler = self._create_weighted_sampler(self.train_dataset)
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                sampler=sampler,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
        else:
            return DataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.batch_size,
                shuffle=True,
                num_workers=self.hparams.num_workers,
                pin_memory=self.hparams.pin_memory,
            )
    
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
    
    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Return predict dataloader."""
        return self.test_dataloader()
    
    def teardown(self, stage: str = None):
        """Clean up after fit or test."""
        pass
    
    def _create_train_transform(self) -> Callable:
        """Create training data augmentation pipeline.
        
        Augmentations:
            - Rotation: ±20 degrees
            - Horizontal flip: 50% probability
            - Color jitter: brightness/contrast/saturation/hue
            - Shift/scale/rotate: small random transformations
            - Perspective: 5-15% scale
            - Resize: to target image size
            - Normalize: ImageNet statistics
        """
        return A.Compose([
            A.Rotate(limit=20, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.1,
                p=1.0,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0,
                rotate_limit=0,
                p=0.5,
            ),
            A.Perspective(
                scale=(0.05, 0.15),
                keep_size=True,
                p=0.5,
            ),
            A.Resize(
                height=self.hparams.image_size,
                width=self.hparams.image_size,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def _create_val_transform(self) -> Callable:
        """Create validation/test data transformation pipeline.
        
        Transformations:
            - Resize: to target image size
            - Normalize: ImageNet statistics
        """
        return A.Compose([
            A.Resize(
                height=self.hparams.image_size,
                width=self.hparams.image_size,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    
    def _split_train_val(
        self,
        train_dir: str,
        val_split_ratio: float,
    ) -> Tuple[list[str], list[str]]:
        """Split training data into train and validation sets.
        
        Key strategy:
            - PNEUMONIA: Split by patient ID to avoid data leakage
            - NORMAL: Random split by file
        
        Args:
            train_dir: Path to training directory
            val_split_ratio: Validation split ratio (e.g., 0.2 for 20%)
            
        Returns:
            Tuple of (train_filenames, val_filenames)
        """
        pneumonia_dir = os.path.join(train_dir, 'PNEUMONIA')
        normal_dir = os.path.join(train_dir, 'NORMAL')
        
        pneumonia_files = [
            f for f in os.listdir(pneumonia_dir)
            if f.lower().endswith(('.jpeg', '.jpg', '.png'))
        ]
        pneumonia_patient_ids = set([
            self._extract_patient_id(f) for f in pneumonia_files
        ])
        
        num_pneumonia_val = int(val_split_ratio * len(pneumonia_patient_ids))
        pneumonia_val_patient_ids = random.sample(
            list(pneumonia_patient_ids),
            num_pneumonia_val
        )
        
        pneumonia_train_files = []
        pneumonia_val_files = []
        
        for filename in pneumonia_files:
            patient_id = self._extract_patient_id(filename)
            if patient_id in pneumonia_val_patient_ids:
                pneumonia_val_files.append(os.path.join(pneumonia_dir, filename))
            else:
                pneumonia_train_files.append(os.path.join(pneumonia_dir, filename))
        
        normal_files = [
            f for f in os.listdir(normal_dir)
            if f.lower().endswith(('.jpeg', '.jpg', '.png'))
        ]
        normal_files = [os.path.join(normal_dir, f) for f in normal_files]
        
        num_normal_val = int(val_split_ratio * len(normal_files))
        normal_val_files = random.sample(normal_files, num_normal_val)
        normal_train_files = list(set(normal_files) - set(normal_val_files))
        
        train_filenames = pneumonia_train_files + normal_train_files
        val_filenames = pneumonia_val_files + normal_val_files
        
        return train_filenames, val_filenames
    
    def _get_test_filenames(self, test_dir: str) -> list[str]:
        """Get all test filenames.
        
        Args:
            test_dir: Path to test directory
            
        Returns:
            List of test image file paths
        """
        test_filenames = []
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_dir = os.path.join(test_dir, class_name)
            files = [
                os.path.join(class_dir, f)
                for f in os.listdir(class_dir)
                if f.lower().endswith(('.jpeg', '.jpg', '.png'))
            ]
            test_filenames.extend(files)
        
        return test_filenames
    
    def _extract_patient_id(self, filename: str) -> str:
        """Extract patient ID from PNEUMONIA filename.
        
        PNEUMONIA filenames format: person{patient_id}_bacteria_{index}.jpeg
        Example: person1_bacteria_1.jpeg -> patient_id = "1"
        
        Args:
            filename: Image filename
            
        Returns:
            Patient ID as string
        """
        basename = os.path.splitext(filename)[0]
        parts = basename.split('_')
        if len(parts) > 0 and parts[0].startswith('person'):
            return parts[0].replace('person', '')
        return basename
    
    def _create_weighted_sampler(self, dataset: Dataset) -> WeightedRandomSampler:
        """Create weighted sampler for class imbalance.
        
        Args:
            dataset: Dataset with targets attribute
            
        Returns:
            WeightedRandomSampler instance
        """
        targets = dataset.targets
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        sample_weights = [class_weights[label] for label in targets]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )


if __name__ == "__main__":
    dm = ChestXRayBinaryDataModule()
    dm.prepare_data()
    dm.setup()
    print(f"Train dataset size: {len(dm.train_dataset)}")
    print(f"Val dataset size: {len(dm.val_dataset)}")
    print(f"Test dataset size: {len(dm.test_dataset)}")
    
    for batch in dm.train_dataloader():
        print(f"Batch shape: {batch[0].shape}, Labels: {batch[1].shape}")
        break
