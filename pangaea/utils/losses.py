import torch
from torch.nn import functional as F
import os


class WeightedCrossEntropy(torch.nn.Module):
    def __init__(self, ignore_index: int, distribution: list[float]) -> None:
        super(WeightedCrossEntropy, self).__init__()
        # Initialize the weights based on the given distribution
        self.weights = [1 / w for w in distribution]

        # Convert weights to a tensor and move to CUDA
        loss_weights = torch.Tensor(self.weights).to("cuda")
        self.loss = torch.nn.CrossEntropyLoss(
            ignore_index=ignore_index, weight=loss_weights
        )

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Compute the weighted cross-entropy loss
        return self.loss(logits, target)


class DICELoss(torch.nn.Module):
    def __init__(self, ignore_index: int) -> None:
        super(DICELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, logits, target):
        num_classes = logits.shape[1]

        # Convert logits to probabilities using softmax or sigmoid
        if num_classes == 1:
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)

        # Create a mask to ignore the specified index
        mask = target != self.ignore_index
        target = target.clone()
        target[~mask] = 0

        # Convert target to one-hot encoding if necessary
        if num_classes == 1:
            target = target.unsqueeze(1)
        else:
            target = F.one_hot(target, num_classes=num_classes)
            target = target.permute(0, 3, 1, 2)

        # Apply the mask to the target
        target = target.float() * mask.unsqueeze(1).float()
        intersection = torch.sum(probs * target, dim=(2, 3))
        union = torch.sum(probs + target, dim=(2, 3))

        # Compute the Dice score
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        valid_dice = dice_score[mask.any(dim=1).any(dim=1)]
        dice_loss = 1 - valid_dice.mean()  # Dice loss is 1 minus the Dice score

        return dice_loss

class SpectralSpatialLoss(torch.nn.Module):
    def __init__(
        self, ignore_index: int, edge_weight: float=0.5,
        schedule_type: str='linear'
    ) -> None:
        super(SpectralSpatialLoss, self).__init__()
        self.ignore_index = ignore_index
        self.edge_weight = edge_weight
        self.initial_edge_weight = edge_weight
        self.schedule_type = schedule_type

        # configure cuda
        local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device("cuda", local_rank)
    
    def _create_sobel_kernels(self, device, dtype):
        """Create Sobel kernels on the specified device and dtype"""
        sobel_x_arr = ([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
        sobel_x = torch.tensor(
            sobel_x_arr, dtype=dtype, device=self.device
        ).view(1, 1, 3, 3)
        
        sobel_y_arr = ([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
        sobel_y = torch.tensor(
            sobel_y_arr, dtype=dtype, device=self.device
        ).view(1, 1, 3, 3)
        
        return sobel_x, sobel_y
    
    def compute_edge_loss(self, logits, target):
        """Compute edge-based loss using Sobel operators."""
        edge_loss = 0.0
        device = logits.device
        dtype = logits.dtype

        # Ensure logits and target are 4D tensors
        logits = logits.unsqueeze(1)
        target = target.unsqueeze(1)
        
        # Create Sobel kernels directly on the input device with matching dtype
        sobel_x, sobel_y = self._create_sobel_kernels(device, dtype)
        
        # Apply Sobel X
        logits_edge_x = F.conv2d(logits, sobel_x, padding=1)
        target_edge_x = F.conv2d(target, sobel_x, padding=1)
        
        # Apply Sobel Y
        logits_edge_y = F.conv2d(logits, sobel_y, padding=1)
        target_edge_y = F.conv2d(target, sobel_y, padding=1)
        
        # Combine X and Y gradients (magnitude)
        logits_edge_grad = torch.sqrt(logits_edge_x**2 + logits_edge_y**2 + 1e-8)
        target_edge_grad = torch.sqrt(target_edge_x**2 + target_edge_y**2 + 1e-8)
        
        # Add L1 loss for this band's edges
        edge_loss += F.l1_loss(logits_edge_grad, target_edge_grad)
            
        return edge_loss / target.size(1) # Average over channels

    def forward(self, logits, target):
        """Compute the spectral-spatial loss."""
        # Spectral loss (L1 across all bands and pixels)
        spectral_loss = F.l1_loss(logits, target)
        
        # Spatial (edge) loss
        edge_loss = self.compute_edge_loss(logits, target)
        
        # Combined loss
        total_loss = spectral_loss + self.edge_weight * edge_loss
        
        return total_loss

    def get_loss_components(self, pred, target):
        """Return individual loss components for monitoring."""
        spectral_loss = F.l1_loss(pred, target)
        edge_loss = self.compute_edge_loss(pred, target)
        total_loss = spectral_loss + self.edge_weight * edge_loss
        
        return {
            'spectral_loss': spectral_loss.item(),
            'edge_loss': edge_loss.item(),
            'total_loss': total_loss.item()
        }  

    def __repr__(self):
        return f'SpectralSpatialLoss(edge_weight={self.edge_weight}, num_bands={self.num_bands})'

