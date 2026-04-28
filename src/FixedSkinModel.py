import torch
import torch.nn as nn
from torchvision import models

class FixedSkinModel(nn.Module):
    def __init__(self, num_classes=8):
        super(FixedSkinModel, self).__init__()
        
        # Load model without pretrained weights first
        self.model = models.efficientnet_b0(weights=None)
        
        # Then manually download and load weights without hash check
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth',
                map_location='cpu',
                check_hash=False  # Skip hash verification
            )
            self.model.load_state_dict(state_dict)
            print("✅ Pretrained weights loaded successfully!")
        except Exception as e:
            print(f"⚠️ Could not load pretrained weights: {e}")
            print("⚠️ Training from scratch instead")
        
        # Freeze all layers initially except the classifier head
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Classifier input features (1280 for B0)
        in_features = 1280
        
        # Replace the classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(p=0.2), # As requested in the prompt
            nn.Linear(256, num_classes)
            # No Sigmoid — BCEWithLogitsLoss handles it
        )
        
        # Variables for GradCAM
        self.gradients = None
        
        # Print total parameters and trainable parameters when model is initialized
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"FixedSkinModel Initialized! Total params: {total_params:,} | Trainable params: {trainable_params:,}")

    def freeze_backbone(self):
        """Freeze all backbone layers, only classifier will train"""
        for param in self.model.features.parameters():
            param.requires_grad = False
        print("✅ Backbone frozen - training classifier head only")
        
    def unfreeze_backbone(self, num_layers=20):
        """Unfreezes the last N layers of the backbone for fine-tuning."""
        print(f"Unfreezing the last {num_layers} parameter tensors of the backbone...")
        # Get all parameters in the features module
        params = list(self.model.features.parameters())
        # Unfreeze the last `num_layers` parameters
        unfreeze_params = params[-num_layers:] if num_layers < len(params) else params
        for param in unfreeze_params:
            param.requires_grad = True
            
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Model Update -> Total params: {total_params:,} | Trainable params: {trainable_params:,}")

    def get_feature_maps(self, x):
        """Forward hooks for Grad-CAM visualization later."""
        # Hook for gradients
        def save_gradient(grad):
            self.gradients = grad
            
        # Forward pass through features
        features = self.model.features(x)
        features.register_hook(save_gradient)
        
        # Forward pass through classifier
        pooled = self.model.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        output = self.model.classifier(pooled)
        
        return output, features

    def forward(self, x):
        return self.model(x)