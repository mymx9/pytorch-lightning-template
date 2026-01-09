# Chest X-Ray Binary Classification Dataset

## 数据集概述

Chest X-Ray数据集用于二分类任务：区分正常胸部X光片和肺炎胸部X光片。

### 数据集结构

```

/root/autodl-fs/data/chest_xray/
├── train/
│   ├── NORMAL/
│   │   ├── IM-0001-0001.jpeg
│   │   ├── IM-0002-0001.jpeg
│   │   └── ...
│   └── PNEUMONIA/
│       ├── person1_bacteria_1.jpeg
│       ├── person1_bacteria_2.jpeg
│       ├── person2_virus_1.jpeg
│       └── ...
└── test/
├── NORMAL/
│   ├── IM-0003-0001.jpeg
│   └── ...
└── PNEUMONIA/
├── person100_bacteria_1.jpeg
└── ...

```

### 数据集统计

- **类别**: 2 (NORMAL, PNEUMONIA)
- **训练集大小**: ~5,216张图像
- **测试集大小**: ~624张图像
- **图像格式**: JPEG
- **图像尺寸**: 可变（通常为1024×1024或更高）
- **颜色空间**: 灰度（但加载为RGB）

### 标签映射

- NORMAL: 0
- PNEUMONIA: 1

## 使用方法

### 基本使用

```python
from src.data_modules.chest_xray_binary import ChestXRayBinaryDataModule

# 创建DataModule
datamodule = ChestXRayBinaryDataModule(
    data_dir="/root/autodl-fs/data/chest_xray",
    train_val_split=0.2,
    batch_size=64,
    image_size=224,
    balance=True,
)

# 准备数据
datamodule.prepare_data()
datamodule.setup()

# 获取数据加载器
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
test_loader = datamodule.test_dataloader()
```

### 使用配置文件

```bash
# 使用配置文件训练
python src/main.py fit \
  -c configs/data/chest_xray_binary.yaml \
  -c configs/model/your_model.yaml \
  -c configs/default.yaml
```

## 数据预处理

### 训练集增强

训练集应用以下数据增强：

1. **旋转**: ±20度随机旋转
2. **水平翻转**: 50%概率
3. **颜色抖动**: 亮度、对比度、饱和度、色调（±0.1）
4. **位移缩放旋转**: 小幅度随机变换
5. **透视变换**: 5-15%缩放
6. **调整大小**: 统一调整为224×224
7. **归一化**: ImageNet统计量
   * mean: [0.485, 0.456, 0.406]
   * std: [0.229, 0.224, 0.225]
8. **转换为张量**: 使用ToTensorV2

### 验证/测试集处理

验证和测试集只进行：

1. **调整大小**: 统一调整为224×224
2. **归一化**: ImageNet统计量
   * mean: [0.485, 0.456, 0.406]
   * std: [0.229, 0.224, 0.225]
3. **无数据增强**: 保持原始数据分布
4. **转换为张量**: 使用ToTensorV2

### 归一化策略

**选择**: ImageNet统计量

**原因**:
- 预训练模型（如ResNet、EfficientNet）在ImageNet上训练
- 使用相同的归一化统计量可以更好地利用预训练权重
- 适用于医学图像的通用特征提取

## 数据集划分

### 划分策略

**核心原则**: 避免数据泄露（Data Leakage）

#### PNEUMONIA类别划分

**问题**: 文件名包含患者ID（如`person1_bacteria_1.jpeg`）

**策略**: 按患者ID分组划分

**步骤**:
1. 提取所有PNEUMONIA文件的患者ID
2. 随机选择20%的患者ID作为验证集
3. 将同一患者的所有图像分配到同一集合

**好处**: 避免同一患者的图像同时出现在训练集和验证集

#### NORMAL类别划分

**问题**: 文件名不包含患者信息

**策略**: 随机文件划分

**步骤**:
1. 获取所有NORMAL文件
2. 随机选择20%作为验证集
3. 剩余80%作为训练集

### 划分比例

- 训练集: 80%
- 验证集: 20%
- 测试集: 独立（使用test/目录）

### 类别不平衡处理

**问题**: PNEUMONIA样本通常比NORMAL样本多

**解决方案**: 加权随机采样（WeightedRandomSampler）

- 计算每个类别的权重：`weight = 1 / class_count`
- 为每个样本分配对应的类别权重
- 使用WeightedRandomSampler确保每个batch中类别平衡

## 注意事项

### 1. 数据泄露

**重要**: PNEUMONIA文件必须按患者ID划分，避免同一患者的图像同时出现在训练集和验证集。

### 2. 类别不平衡

**建议**: 使用`balance=True`参数启用加权采样，确保训练过程中类别平衡。

### 3. 图像尺寸

**注意**: 原始图像尺寸可变，统一调整为224×224。如需其他尺寸，修改`image_size`参数。

### 4. 内存使用

**建议**: 对于大batch_size，建议设置`pin_memory=True`和适当的`num_workers`以提高加载速度。

### 5. 随机种子

**建议**: 为可复现性，设置全局随机种子（如`seed_everything: 3407`）。

### 6. 数据验证

**建议**: 在训练前检查数据集完整性：

```python
datamodule = ChestXRayBinaryDataModule()
datamodule.prepare_data()
datamodule.setup()

print(f"训练集大小: {len(datamodule.train_dataset)}")
print(f"验证集大小: {len(datamodule.val_dataset)}")
print(f"测试集大小: {len(datamodule.test_dataset)}")
```

## 性能优化

### DataLoader配置建议

```python
# GPU训练
ChestXRayBinaryDataModule(
    batch_size=64,
    num_workers=4,
    pin_memory=True,
)

# CPU训练
ChestXRayBinaryDataModule(
    batch_size=32,
    num_workers=2,
    pin_memory=False,
)
```

### 数据加载优化

1. **num_workers**: 通常设置为CPU核心数
2. **pin_memory**: GPU训练时设置为True
3. **batch_size**: 根据GPU内存调整

## 故障排除

### 常见问题

1. **FileNotFoundError**: 检查`data_dir`路径是否正确
2. **数据泄露**: 检查PNEUMONIA患者ID划分是否正确
3. **类别不平衡**: 检查是否启用了`balance=True`
4. **内存不足**: 减小`batch_size`或`image_size`

## 参考资料

- [Kaggle Chest X-Ray Dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [Albumentations Documentation](https://albumentations.ai/docs/)
- [PyTorch Lightning DataModule](https://lightning.ai/docs/pytorch/latest/data/datamodule.html)
