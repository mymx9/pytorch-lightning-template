from typing import Optional

import torch
from torch import nn
from transformers import AutoModel, AutoImageProcessor


class DinoV3Base(nn.Module):
    def __init__(
        self,
        model_path: str = "/root/autodl-fs/model/facebook/dinov3-vitb16-pretrain-lvd1689m",
        freeze_backbone: bool = True,
        use_pooler_output: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        
        self.model_path = model_path
        self.freeze_backbone = freeze_backbone
        self.use_pooler_output = use_pooler_output
        self.image_size = image_size
        
        self.backbone = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        )
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.processor = AutoImageProcessor.from_pretrained(model_path)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=x)
        
        if self.use_pooler_output:
            return outputs.pooler_output
        else:
            return outputs.last_hidden_state
    
    def get_output_dim(self) -> int:
        if hasattr(self.backbone, 'config'):
            return self.backbone.config.hidden_size
        return 768
    
    def process_image(self, image) -> torch.Tensor:
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt")
            return inputs.pixel_values
        return image


if __name__ == "__main__":
    model = DinoV3Base()
    print(f"Model loaded from: {model.model_path}")
    print(f"Output dimension: {model.get_output_dim()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    test_input = torch.randn(2, 3, 224, 224)
    print(f"\nTest input shape: {test_input.shape}")
    
    output = model(test_input)
    print(f"Output shape: {output.shape}")
