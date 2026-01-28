import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List, Tuple

class GradCAM:
    """
    Grad-CAM for 3D ResNets (Video Action Recognition).
    
    Computes class-discriminative localization maps using the gradients
    of the target concept flowing into the final convolutional layer.
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        """
        Args:
            model: The RGB-D model
            target_layer: The target layer to hook (e.g., model.backbone.layer4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        
        # Register hooks
        self.handles = []
        self.handles.append(
            target_layer.register_forward_hook(self._save_activations)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self._save_gradients)
        )
        
    def _save_activations(self, module, input, output):
        self.activations = output.detach()
        
    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output is a tuple, usually (tensor,)
        self.gradients = grad_output[0].detach()
        
    def __call__(
        self, 
        x: torch.Tensor, 
        class_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            class_idx: Target class index. If None, uses the predicted class.
            
        Returns:
            heatmap: (T, H, W) numpy array in range [0, 1]
        """
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(x)
        
        if class_idx is None:
            class_idx = torch.argmax(logits, dim=1).item()
            
        # Backward pass
        target = logits[0, class_idx]
        target.backward()
        
        # Compute weights: global average pooling of gradients
        # gradients shape: (B, C, T, H, W) (e.g., 1, 512, 4, 7, 7)
        weights = torch.mean(self.gradients, dim=(2, 3, 4), keepdim=True)
        
        # Weighted combination of activations
        # activations shape: (B, C, T, H, W)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        
        # ReLU
        cam = F.relu(cam)
        
        # Interpolate to input size (T, H, W)
        # Input x is (B, C, T, H, W)
        target_t = x.shape[2]
        target_h = x.shape[3]
        target_w = x.shape[4]
        
        cam = F.interpolate(
            cam, 
            size=(target_t, target_h, target_w), 
            mode='trilinear', 
            align_corners=False
        )
        
        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy() # (T, H, W)
        
        cam_min = cam.min()
        cam_max = cam.max()
        
        if cam_max - cam_min > 0:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

def apply_heatmap(
    img: np.ndarray, 
    heatmap: np.ndarray, 
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET
) -> np.ndarray:
    """
    Overlay heatmap on image.
    
    Args:
        img: (H, W, 3) BGR image
        heatmap: (H, W) float [0, 1]
        alpha: Opacity of heatmap
        
    Returns:
        Overlay image
    """
    # Convert heatmap to color
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Blend
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)
    return overlay
