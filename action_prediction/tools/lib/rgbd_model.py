#!/usr/bin/env python3
"""
RGB-D Action Recognition Model.

Unified model that takes 4-channel (RGB + Depth) input
and outputs action class predictions.
Supports 3D CNN and transformer video backbones.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

# Try importing torchvision for pretrained models
try:
    import torchvision.models.video as video_models
    from torchvision.models.video import r3d_18, R3D_18_Weights
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# Set torch hub checkpoint directory to local project root
# Torch Hub appends 'checkpoints/' automatically, so setting it to root will result in 'root/checkpoints/'
try:
    _project_root = Path(__file__).resolve().parents[2]
    torch.hub.set_dir(str(_project_root))
    print(f"Set torch hub dir to: {_project_root}")
except Exception as e:
    print(f"Warning: Could not set torch hub dir: {e}")


def _get_first_conv3d(model: nn.Module) -> Tuple[str, nn.Conv3d]:
    """Find the first Conv3d layer in a model."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d):
            return name, module
    raise ValueError("No Conv3d layer found in model.")


def _replace_first_conv3d(model: nn.Module, in_channels: int) -> None:
    """Replace the first Conv3d layer to accept in_channels."""
    name, old_conv = _get_first_conv3d(model)
    new_conv = nn.Conv3d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight[:, :3]
        if in_channels > 3:
            rgb_mean = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
            for i in range(3, in_channels):
                new_conv.weight[:, i:i+1] = rgb_mean
    # Replace in model by name
    parent = model
    parts = name.split('.')
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new_conv)


class RGBDActionModel(nn.Module):
    """
    RGB-D Action Recognition Model.
    
    Uses early fusion (4-channel input: RGB + normalized depth) with
    a video backbone for temporal modeling.
    
    Supports:
    - r3d_18 (baseline)
    - swin3d_t / swin3d_s / swin3d_b (best accuracy)
    - mvit_v2_s / mvit_v2_b (high accuracy)
    """
    
    def __init__(
        self,
        backbone: str = 'swin3d_b',
        in_channels: int = 4,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.5,
    ):
        """
        Initialize RGB-D action model.
        
        Args:
            backbone: Backbone architecture ('r3d_18', 'swin3d_b', 'mvit_v2_s', etc.)
            in_channels: Number of input channels (4 for RGB-D)
            num_classes: Number of output classes
            pretrained: Use pretrained weights (RGB channels only)
            dropout: Dropout ratio before final FC
        """
        super().__init__()
        
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Build backbone
        if backbone == 'r3d_18':
            if not HAS_TORCHVISION:
                raise ImportError("torchvision required for r3d_18 backbone")
            
            if pretrained:
                weights = R3D_18_Weights.KINETICS400_V1
                self.backbone = r3d_18(weights=weights)
            else:
                self.backbone = r3d_18(weights=None)
            
            # Modify first conv layer for 4-channel input
            self._adapt_first_conv(in_channels)

            # Get feature dimension
            self.feat_dim = self.backbone.fc.in_features

            # Replace classifier
            self.backbone.fc = nn.Identity()
        elif backbone in ('swin3d_t', 'swin3d_s', 'swin3d_b'):
            if not HAS_TORCHVISION:
                raise ImportError("torchvision required for Swin3D backbones")
            try:
                from torchvision.models.video import swin3d_t, swin3d_s
                try:
                    from torchvision.models.video import swin3d_b
                except Exception:
                    swin3d_b = None
            except Exception as e:
                raise ImportError(
                    "Swin3D not available in your torchvision version. "
                    "Upgrade torchvision (>=0.16)."
                ) from e

            weights = None
            if pretrained:
                try:
                    from torchvision.models.video import Swin3D_T_Weights, Swin3D_S_Weights
                    try:
                        from torchvision.models.video import Swin3D_B_Weights
                    except Exception:
                        Swin3D_B_Weights = None
                    if backbone == 'swin3d_t':
                        weights = Swin3D_T_Weights.DEFAULT
                    elif backbone == 'swin3d_b':
                        weights = Swin3D_B_Weights.DEFAULT if Swin3D_B_Weights is not None else None
                    else:
                        weights = Swin3D_S_Weights.DEFAULT
                except Exception:
                    weights = None

            if backbone == 'swin3d_t':
                self.backbone = swin3d_t(weights=weights)
            elif backbone == 'swin3d_b':
                if swin3d_b is None:
                    raise ImportError("swin3d_b not available; upgrade torchvision or use swin3d_s.")
                self.backbone = swin3d_b(weights=weights)
            else:
                self.backbone = swin3d_s(weights=weights)

            # Replace first conv for RGB-D
            _replace_first_conv3d(self.backbone, in_channels)

            # Get feature dimension from head
            self.feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        elif backbone in ('mvit_v2_s', 'mvit_v2_b'):
            if not HAS_TORCHVISION:
                raise ImportError("torchvision required for MViT backbones")
            try:
                from torchvision.models.video import mvit_v2_s
                try:
                    from torchvision.models.video import mvit_v2_b
                except Exception:
                    mvit_v2_b = None
            except Exception as e:
                raise ImportError(
                    "MViT V2 not available in your torchvision version. "
                    "Upgrade torchvision (>=0.16)."
                ) from e

            weights = None
            if pretrained:
                try:
                    from torchvision.models.video import MViT_V2_S_Weights
                    try:
                        from torchvision.models.video import MViT_V2_B_Weights
                    except Exception:
                        MViT_V2_B_Weights = None
                    if backbone == 'mvit_v2_b':
                        weights = MViT_V2_B_Weights.DEFAULT if MViT_V2_B_Weights is not None else None
                    else:
                        weights = MViT_V2_S_Weights.DEFAULT
                except Exception:
                    weights = None

            if backbone == 'mvit_v2_b':
                if mvit_v2_b is None:
                    raise ImportError("mvit_v2_b not available; upgrade torchvision or use mvit_v2_s.")
                self.backbone = mvit_v2_b(weights=weights)
            else:
                self.backbone = mvit_v2_s(weights=weights)

            _replace_first_conv3d(self.backbone, in_channels)
            self.feat_dim = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.feat_dim, num_classes),
        )
    
    def _adapt_first_conv(self, in_channels: int):
        """
        Adapt first conv layer for different number of input channels.
        
        For RGB-D (4 channels):
        - Copy RGB weights for first 3 channels
        - Initialize depth channel with zeros (or average of RGB)
        """
        old_conv = self.backbone.stem[0]
        
        # Create new conv with desired input channels
        new_conv = nn.Conv3d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        
        # Copy weights
        with torch.no_grad():
            if in_channels >= 3:
                # Copy RGB weights
                new_conv.weight[:, :3] = old_conv.weight[:, :3]
                
                # Initialize extra channels (depth) with average of RGB
                if in_channels > 3:
                    rgb_mean = old_conv.weight[:, :3].mean(dim=1, keepdim=True)
                    for i in range(3, in_channels):
                        new_conv.weight[:, i:i+1] = rgb_mean
            else:
                # Fewer channels than RGB
                new_conv.weight[:, :in_channels] = old_conv.weight[:, :in_channels]
            
            if old_conv.bias is not None:
                new_conv.bias = old_conv.bias
        
        # Replace conv
        self.backbone.stem[0] = new_conv
    
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, T, C, H, W) or (B, C, T, H, W)
            return_features: If True, return features before classifier
            
        Returns:
            logits: (B, num_classes)
            features: (B, feat_dim) if return_features=True
        """
        # Ensure correct shape: (B, C, T, H, W)
        if x.dim() == 5:
            if x.shape[2] == self.in_channels:
                # Input is (B, T, C, H, W), need (B, C, T, H, W)
                x = x.permute(0, 2, 1, 3, 4)
            # else: already (B, C, T, H, W)
        else:
            raise ValueError(f"Expected 5D input, got shape {x.shape}")
        
        # Extract features
        features = self.backbone(x)
        
        # Classify
        logits = self.head(features)
        
        if return_features:
            return logits, features
        return logits
    
    def predict(
        self,
        x: torch.Tensor,
        return_probs: bool = True,
    ) -> torch.Tensor:
        """
        Predict class probabilities or labels.
        
        Args:
            x: Input tensor
            return_probs: If True, return softmax probabilities
            
        Returns:
            probs (B, num_classes) or labels (B,)
        """
        logits = self.forward(x)
        
        if return_probs:
            return F.softmax(logits, dim=-1)
        else:
            return torch.argmax(logits, dim=-1)


class CausalRGBDModel(RGBDActionModel):
    """
    Causal version of RGB-D model for anticipation/early prediction.
    
    Uses only past frames for prediction (no future lookahead).
    This is important for real-time anticipation mode.
    """
    
    def __init__(
        self,
        backbone: str = 'r3d_18',
        in_channels: int = 4,
        num_classes: int = 8,
        pretrained: bool = True,
        dropout: float = 0.5,
        causal_window: int = 8,
    ):
        """
        Initialize causal RGB-D model.
        
        Args:
            causal_window: Number of past frames to use for prediction
        """
        super().__init__(backbone, in_channels, num_classes, pretrained, dropout)
        self.causal_window = causal_window
    
    def forward_causal(
        self,
        frame_buffer: torch.Tensor,
        current_idx: int = -1,
    ) -> torch.Tensor:
        """
        Causal forward pass using only past frames.
        
        Args:
            frame_buffer: Buffer of frames (B, T_buffer, C, H, W)
            current_idx: Index of current frame (-1 = last frame)
            
        Returns:
            logits: (B, num_classes)
        """
        if current_idx < 0:
            current_idx = frame_buffer.shape[1] + current_idx
        
        # Get causal window (past frames only)
        start_idx = max(0, current_idx - self.causal_window + 1)
        end_idx = current_idx + 1
        
        x = frame_buffer[:, start_idx:end_idx]
        
        # If not enough frames, pad with first frame
        if x.shape[1] < self.causal_window:
            pad_size = self.causal_window - x.shape[1]
            first_frame = x[:, :1].expand(-1, pad_size, -1, -1, -1)
            x = torch.cat([first_frame, x], dim=1)
        
        return self.forward(x)

        return self.forward(x)


class ONNXWrapper:
    """
    Wrapper for ONNX Runtime inference to match RGBDActionModel interface.
    """
    def __init__(self, model_path: str, device: str = 'cuda'):
        if not HAS_ONNX:
            raise ImportError("onnxruntime is required for ONNX models. Install with: pip install onnxruntime-gpu")
            
        self.device = device
        
        # Select provider
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if device == 'cpu':
            providers = ['CPUExecutionProvider']
            
        print(f"Loading ONNX model from {model_path} using {providers[0]}...")
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"Active ONNX Providers: {self.session.get_providers()}")
        
        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
    def predict(self, x: torch.Tensor, return_probs: bool = True) -> torch.Tensor:
        """
        Run inference on PyTorch tensor input.
        
        Args:
            x: Input tensor (B, T, C, H, W) - live inference format
            return_probs: If True, return softmax probabilities
        """
        # 1. Prepare Input: Ensure (B, C, T, H, W) layout for ONNX
        # Live inference typically provides (B, T, C, H, W)
        if isinstance(x, torch.Tensor):
            # Check if we need to permute: (B, T, C, H, W) -> (B, C, T, H, W)
            # If x.shape[2] is the channel dim (4) and x.shape[1] is time dim (16), permute
            if x.dim() == 5 and x.shape[2] == 4:  # Channels in dim 2 means (B, T, C, H, W)
                x = x.permute(0, 2, 1, 3, 4)
            # Convert to float32 numpy (ONNX standard)
            x_np = x.float().cpu().numpy()
        else:
            # Numpy input
            if x.ndim == 5 and x.shape[2] == 4:
                x_np = x.transpose(0, 2, 1, 3, 4)
            else:
                x_np = x
            x_np = x_np.astype('float32')
        
        # 2. Run Inference (ONNX Runtime handles GPU execution via CUDA EP)
        outputs = self.session.run([self.output_name], {self.input_name: x_np})
        logits_np = outputs[0]
        
        # 3. Process Output
        logits = torch.from_numpy(logits_np)
        # Output logits are on CPU, apply softmax here (fast for small tensor)
        if return_probs:
            return F.softmax(logits, dim=-1)
        
        return torch.argmax(logits, dim=-1)
    
    def eval(self):
        """Mock eval method."""
        pass
    
    def to(self, device):
        """Mock to method - provider is set at init."""
        return self
def load_model(
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda:0',
    **kwargs,
) -> RGBDActionModel:
    """
    Load RGB-D action model from config and/or checkpoint.
    
    Args:
        config_path: Path to config file (optional)
        checkpoint_path: Path to checkpoint (optional)
        device: Device to load model to
        **kwargs: Override config parameters
        
    Returns:
        Loaded model
    """
    # Check for ONNX
    if checkpoint_path and str(checkpoint_path).lower().endswith('.onnx'):
        return ONNXWrapper(checkpoint_path, device=device)

    # Default config
    config = {
        'backbone': 'swin3d_b',
        'in_channels': 4,
        'num_classes': 8,
        'pretrained': True,
        'dropout': 0.5,
    }
    
    # Load config file if provided
    if config_path:
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", config_path)
        cfg_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg_module)
        
        if hasattr(cfg_module, 'model'):
            config.update(cfg_module.model)
        if hasattr(cfg_module, 'data'):
            data_cfg = cfg_module.data
            if isinstance(data_cfg, dict) and data_cfg.get('use_delta', False):
                if config.get('in_channels', 4) == 4:
                    config['in_channels'] = 8
    
    # Override with kwargs
    config.update(kwargs)
    
    # Create model
    if config.get('causal', False):
        model = CausalRGBDModel(
            backbone=config['backbone'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained', True),
            dropout=config.get('dropout', 0.5),
            causal_window=config.get('causal_window', 8),
        )
    else:
        model = RGBDActionModel(
            backbone=config['backbone'],
            in_channels=config['in_channels'],
            num_classes=config['num_classes'],
            pretrained=config.get('pretrained', True),
            dropout=config.get('dropout', 0.5),
        )
    
    # Load checkpoint
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


if __name__ == '__main__':
    # Test model
    print("Testing RGBDActionModel...")
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create model
    model = RGBDActionModel(
        backbone='swin3d_b',
        in_channels=4,
        num_classes=8,
        pretrained=True,
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 16, 4, 224, 224).to(device)  # (B, T, C, H, W)
    
    with torch.no_grad():
        logits = model(x)
        probs = model.predict(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Output probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum(dim=-1)}")  # Should be 1.0
    
    # Test causal model
    print("\nTesting CausalRGBDModel...")
    causal_model = CausalRGBDModel(
        backbone='r3d_18',
        in_channels=4,
        num_classes=8,
        pretrained=False,  # Skip download for speed
        causal_window=8,
    ).to(device)
    
    # Simulate frame buffer
    buffer = torch.randn(2, 32, 4, 224, 224).to(device)
    
    with torch.no_grad():
        logits = causal_model.forward_causal(buffer, current_idx=15)
    
    print(f"Causal output shape: {logits.shape}")
    
    print("\n✅ RGBDActionModel works!")
