"""
Test script for DinoV3Base model and multiclass data module.
"""
import os
import sys
import torch
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import DinoV3Base
from src.modules import DinoV3BaseModule
from src.data_modules import ChestXRayMulticlassDataModule


def test_dinov3_base_model():
    """Test DinoV3Base model loading and inference."""
    print("=" * 60)
    print("Testing DinoV3Base Model")
    print("=" * 60)
    
    try:
        model = DinoV3Base()
        print(f"✓ Model loaded from: {model.model_path}")
        print(f"✓ Output dimension: {model.get_output_dim()}")
        print(f"✓ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"✓ Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        test_input = torch.randn(2, 3, 224, 224)
        print(f"\n✓ Test input shape: {test_input.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(test_input)
            print(f"✓ Output shape: {output.shape}")
            assert output.shape == (2, 768), f"Expected shape (2, 768), got {output.shape}"
        
        print("\n✓ DinoV3Base model test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ DinoV3Base model test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov3_base_module():
    """Test DinoV3BaseModule for binary classification."""
    print("\n" + "=" * 60)
    print("Testing DinoV3BaseModule (Binary Classification)")
    print("=" * 60)
    
    try:
        module = DinoV3BaseModule(num_classes=2)
        print(f"✓ Module initialized with num_classes=2")
        print(f"✓ Total parameters: {sum(p.numel() for p in module.parameters()):,}")
        
        test_input = torch.randn(4, 3, 224, 224)
        print(f"\n✓ Test input shape: {test_input.shape}")
        
        module.eval()
        with torch.no_grad():
            logits = module(test_input)
            print(f"✓ Logits shape: {logits.shape}")
            assert logits.shape == (4, 2), f"Expected shape (4, 2), got {logits.shape}"
            
            preds = torch.argmax(logits, dim=1)
            print(f"✓ Predictions shape: {preds.shape}")
            print(f"✓ Predictions: {preds.numpy()}")
        
        print("\n✓ DinoV3BaseModule (binary) test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ DinoV3BaseModule (binary) test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dinov3_base_module_multiclass():
    """Test DinoV3BaseModule for multiclass classification."""
    print("\n" + "=" * 60)
    print("Testing DinoV3BaseModule (Multiclass Classification)")
    print("=" * 60)
    
    try:
        module = DinoV3BaseModule(num_classes=3)
        print(f"✓ Module initialized with num_classes=3")
        print(f"✓ Total parameters: {sum(p.numel() for p in module.parameters()):,}")
        
        test_input = torch.randn(4, 3, 224, 224)
        print(f"\n✓ Test input shape: {test_input.shape}")
        
        module.eval()
        with torch.no_grad():
            logits = module(test_input)
            print(f"✓ Logits shape: {logits.shape}")
            assert logits.shape == (4, 3), f"Expected shape (4, 3), got {logits.shape}"
            
            preds = torch.argmax(logits, dim=1)
            print(f"✓ Predictions shape: {preds.shape}")
            print(f"✓ Predictions: {preds.numpy()}")
            print(f"✓ Unique predictions: {np.unique(preds.numpy())}")
        
        print("\n✓ DinoV3BaseModule (multiclass) test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ DinoV3BaseModule (multiclass) test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step():
    """Test training step with synthetic data."""
    print("\n" + "=" * 60)
    print("Testing Training Step")
    print("=" * 60)
    
    try:
        module = DinoV3BaseModule(num_classes=3)
        module.train()
        
        batch_size = 4
        x = torch.randn(batch_size, 3, 224, 224)
        y = torch.randint(0, 3, (batch_size,))
        
        print(f"✓ Batch input shape: {x.shape}")
        print(f"✓ Batch labels: {y.numpy()}")
        
        loss, preds, targets = module.model_step((x, y))
        print(f"✓ Loss: {loss.item():.4f}")
        print(f"✓ Predictions: {preds.numpy()}")
        print(f"✓ Targets: {targets.numpy()}")
        
        assert loss.item() > 0, "Loss should be positive"
        assert preds.shape == (batch_size,), f"Expected shape ({batch_size},), got {preds.shape}"
        assert targets.shape == (batch_size,), f"Expected shape ({batch_size},), got {targets.shape}"
        
        print("\n✓ Training step test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Training step test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiclass_data_module():
    """Test ChestXRayMulticlassDataModule."""
    print("\n" + "=" * 60)
    print("Testing ChestXRayMulticlassDataModule")
    print("=" * 60)
    
    try:
        dm = ChestXRayMulticlassDataModule(
            data_dir="/root/autodl-fs/data/chest_xray",
            batch_size=8,
            num_workers=2,
            image_size=224,
        )
        
        print("✓ DataModule initialized")
        print(f"✓ Number of classes: {dm.num_classes}")
        
        dm.prepare_data()
        print("✓ Data preparation completed")
        
        dm.setup()
        print("✓ Data setup completed")
        print(f"✓ Train dataset size: {len(dm.train_dataset)}")
        print(f"✓ Val dataset size: {len(dm.val_dataset)}")
        print(f"✓ Test dataset size: {len(dm.test_dataset)}")
        
        train_loader = dm.train_dataloader()
        print(f"✓ Train dataloader created")
        
        batch = next(iter(train_loader))
        images, labels = batch
        print(f"✓ Batch shape: {images.shape}")
        print(f"✓ Labels shape: {labels.shape}")
        print(f"✓ Label distribution: {np.bincount(labels.numpy())}")
        
        assert images.shape[0] == 8, f"Expected batch size 8, got {images.shape[0]}"
        assert images.shape[1:] == (3, 224, 224), f"Expected image shape (3, 224, 224), got {images.shape[1:]}"
        assert labels.shape[0] == 8, f"Expected label batch size 8, got {labels.shape[0]}"
        
        print("\n✓ ChestXRayMulticlassDataModule test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ ChestXRayMulticlassDataModule test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_extraction():
    """Test label extraction from filenames."""
    print("\n" + "=" * 60)
    print("Testing Label Extraction")
    print("=" * 60)
    
    try:
        from src.data_modules.chest_xray_multiclass import ChestXRayMulticlassDataset
        
        test_cases = [
            ("/path/to/NORMAL/IM-0001-0001.jpeg", 0),
            ("/path/to/PNEUMONIA/person1_bacteria_1.jpeg", 1),
            ("/path/to/PNEUMONIA/person2_virus_1.jpeg", 2),
            ("/path/to/PNEUMONIA/person100_bacteria_50.jpeg", 1),
            ("/path/to/PNEUMONIA/person200_virus_100.jpeg", 2),
        ]
        
        dataset = ChestXRayMulticlassDataset(
            root_dir="/root/autodl-fs/data/chest_xray/train",
            filenames=[],
        )
        
        for filepath, expected_label in test_cases:
            label = dataset._extract_label(filepath)
            print(f"✓ {os.path.basename(filepath)} -> Label: {label} (Expected: {expected_label})")
            assert label == expected_label, f"Expected label {expected_label}, got {label}"
        
        print("\n✓ Label extraction test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Label extraction test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_different_resolutions():
    """Test model with different resolutions."""
    print("\n" + "=" * 60)
    print("Testing Different Resolutions")
    print("=" * 60)
    
    try:
        resolutions = [224, 384, 512]
        
        for resolution in resolutions:
            print(f"\nTesting resolution: {resolution}x{resolution}")
            
            module = DinoV3BaseModule(
                num_classes=3,
                image_size=resolution,
            )
            
            test_input = torch.randn(2, 3, resolution, resolution)
            module.eval()
            
            with torch.no_grad():
                logits = module(test_input)
                print(f"  ✓ Input shape: {test_input.shape}")
                print(f"  ✓ Output shape: {logits.shape}")
                assert logits.shape == (2, 3), f"Expected shape (2, 3), got {logits.shape}"
        
        print("\n✓ Different resolutions test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Different resolutions test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DinoV3Base Model and Multiclass Data Module Test Suite")
    print("=" * 60)
    
    results = {}
    
    results['DinoV3Base Model'] = test_dinov3_base_model()
    results['DinoV3BaseModule (Binary)'] = test_dinov3_base_module()
    results['DinoV3BaseModule (Multiclass)'] = test_dinov3_base_module_multiclass()
    results['Training Step'] = test_training_step()
    results['Label Extraction'] = test_label_extraction()
    results['Different Resolutions'] = test_different_resolutions()
    
    try:
        results['Multiclass Data Module'] = test_multiclass_data_module()
    except Exception as e:
        print(f"\n⚠ Multiclass Data Module test SKIPPED (data not available): {e}")
        results['Multiclass Data Module'] = None
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is True:
            print(f"✓ {test_name}: PASSED")
        elif result is False:
            print(f"✗ {test_name}: FAILED")
        else:
            print(f"⚠ {test_name}: SKIPPED")
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    total = len(results)
    
    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    
    if failed == 0:
        print("\n✓ All tests PASSED!")
        return 0
    else:
        print(f"\n✗ {failed} test(s) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
