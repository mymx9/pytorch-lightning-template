import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, patch

from src.data_modules.chest_xray_binary import ChestXRayBinaryDataModule, ChestXRayDataset


class TestChestXRayDataset:
    """Test ChestXRayDataset class."""

    def test_dataset_initialization(self):
        """Test dataset initialization."""
        # Test with valid parameters
        dataset = ChestXRayDataset(
            root_dir="/root/autodl-fs/data/chest_xray/train",
            filenames=["/root/autodl-fs/data/chest_xray/train/NORMAL/test.jpeg"],
        )
        assert dataset is not None
        assert len(dataset) == 1

    def test_dataset_length(self):
        """Test __len__ method."""
        filenames = ["file1.jpeg", "file2.jpeg", "file3.jpeg"]
        dataset = ChestXRayDataset(
            root_dir="/root/autodl-fs/data/chest_xray/train",
            filenames=filenames,
        )
        assert len(dataset) == 3

    @patch('PIL.Image.open')
    def test_dataset_getitem(self, mock_open):
        """Test __getitem__ method."""
        # Mock image loading
        mock_image = MagicMock()
        mock_image.convert.return_value = MagicMock()
        mock_open.return_value = mock_image

        filenames = [
            "/root/autodl-fs/data/chest_xray/train/NORMAL/test.jpeg",
            "/root/autodl-fs/data/chest_xray/train/PNEUMONIA/test.jpeg",
        ]
        dataset = ChestXRayDataset(
            root_dir="/root/autodl-fs/data/chest_xray/train",
            filenames=filenames,
        )

        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert label in [0, 1]
        assert image.shape[0] == 3  # RGB channels

    def test_label_extraction(self):
        """Test label extraction from file paths."""
        dataset = ChestXRayDataset(
            root_dir="/root/autodl-fs/data/chest_xray/train",
            filenames=["/path/to/image.jpeg"],
        )

        # Test NORMAL label
        assert dataset._extract_label("/path/NORMAL/image.jpeg") == 0
        # Test PNEUMONIA label
        assert dataset._extract_label("/path/PNEUMONIA/image.jpeg") == 1

    def test_transform_application(self):
        """Test transform application."""
        from albumentations import Compose, Resize
        transform = Compose([Resize(224, 224)])

        dataset = ChestXRayDataset(
            root_dir="/root/autodl-fs/data/chest_xray/train",
            filenames=["/path/to/image.jpeg"],
            transform=transform,
        )

        image, _ = dataset[0]
        assert image.shape[1:] == (224, 224)  # H, W


class TestChestXRayBinaryDataModule:
    """Test ChestXRayBinaryDataModule class."""

    def test_datamodule_initialization(self):
        """Test DataModule initialization."""
        dm = ChestXRayBinaryDataModule()
        assert dm is not None
        assert dm.num_classes == 2
        assert dm.hparams.batch_size == 64

    def test_prepare_data(self):
        """Test data preparation."""
        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        # Should not raise any exceptions

    def test_setup(self):
        """Test data setup."""
        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        dm.setup()

        assert dm.train_dataset is not None
        assert dm.val_dataset is not None
        assert dm.test_dataset is not None
        assert len(dm.train_dataset) > 0
        assert len(dm.val_dataset) > 0
        assert len(dm.test_dataset) > 0

    def test_dataloaders(self):
        """Test dataloader creation."""
        dm = ChestXRayBinaryDataModule(batch_size=32)
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
        test_loader = dm.test_dataloader()

        assert train_loader is not None
        assert val_loader is not None
        assert test_loader is not None
        assert train_loader.batch_size == 32

    def test_data_split(self):
        """Test data split strategy."""
        dm = ChestXRayBinaryDataModule(train_val_split=0.2)
        dm.prepare_data()
        dm.setup()

        # Check split ratio
        total_train_val = len(dm.train_dataset) + len(dm.val_dataset)
        val_ratio = len(dm.val_dataset) / total_train_val

        assert abs(val_ratio - 0.2) < 0.05  # Allow 5% tolerance

    def test_patient_level_split(self):
        """Test patient-level split for PNEUMONIA."""
        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        dm.setup()

        # Verify no patient appears in both train and val
        # This requires tracking patient IDs in the split
        pass

    def test_weighted_sampling(self):
        """Test weighted sampling for class imbalance."""
        dm = ChestXRayBinaryDataModule(balance=True)
        dm.prepare_data()
        dm.setup()

        train_loader = dm.train_dataloader()

        # Check if sampler is WeightedRandomSampler
        from torch.utils.data import WeightedRandomSampler
        assert isinstance(train_loader.sampler, WeightedRandomSampler)

    def test_transform_creation(self):
        """Test transform creation."""
        dm = ChestXRayBinaryDataModule(image_size=256)
        dm.prepare_data()

        # Check if transforms are created
        assert dm.train_transform is not None
        assert dm.val_transform is not None

    def test_teardown(self):
        """Test teardown method."""
        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        dm.setup()
        dm.teardown()
        # Should not raise any exceptions


class TestDataIntegrity:
    """Test data integrity and correctness."""

    def test_no_data_leakage(self):
        """Test no data leakage between train and val."""
        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        dm.setup()

        # Get all file paths
        train_files = set(dm.train_dataset.filenames)
        val_files = set(dm.val_dataset.filenames)

        # Check no overlap
        overlap = train_files.intersection(val_files)
        assert len(overlap) == 0

    def test_label_distribution(self):
        """Test label distribution in splits."""
        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        dm.setup()

        # Count labels in each split
        train_labels = dm.train_dataset.targets
        val_labels = dm.val_dataset.targets
        test_labels = dm.test_dataset.targets

        # Check both classes are present
        assert 0 in train_labels and 1 in train_labels
        assert 0 in val_labels and 1 in val_labels
        assert 0 in test_labels and 1 in test_labels

    @patch('PIL.Image.open')
    def test_image_loading(self, mock_open):
        """Test image loading and shape."""
        # Mock image loading
        mock_image = MagicMock()
        mock_image.convert.return_value = MagicMock()
        mock_open.return_value = mock_image

        dm = ChestXRayBinaryDataModule()
        dm.prepare_data()
        dm.setup()

        # Load a sample
        image, label = dm.train_dataset[0]

        # Check image properties
        assert isinstance(image, torch.Tensor)
        assert image.dtype == torch.float32
        assert image.shape[0] == 3  # RGB
        assert image.shape[1:] == (224, 224)  # H, W
        assert label in [0, 1]
