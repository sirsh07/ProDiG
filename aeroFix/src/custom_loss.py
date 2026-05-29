import torch
import torch.nn.functional as F

def multi_resolution_loss(pred, target, weights = [0.25,0.25,0.25,0.25]):
    """
    Compute a multi-resolution loss between predictions and target.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, C, D, H, W).
        target (torch.Tensor): Target tensor of shape (B, C, D, H, W).
        weights (list): List of weights for each resolution level.
    """
    total_loss = 0.0
    h, w = pred.shape[2], pred.shape[3]
    
    for i, weight in enumerate(weights):
        if i == 0:
            pred_resized = pred
            target_resized = target
        else:
            
            pred_resized = torch.nn.functional.interpolate(pred, size=(h//((2 ** i)), w//((2 ** i))))
            target_resized = torch.nn.functional.interpolate(target, size=(h//((2 ** i)), w//((2 ** i))))
            
            
        loss = torch.nn.functional.mse_loss(pred_resized, target_resized, reduction="mean")
        total_loss += weight * loss
    return total_loss


def rand_masked_loss(pred, target, patch_size=64):
  
  loss_matrix =  torch.nn.functional.mse_loss(pred, target,reduction='none')

  b, c, h, w = loss_matrix.shape
  n = (h//patch_size) * (w//patch_size)

  mask = torch.cat([
      torch.zeros(n//2, dtype=torch.float32),
      torch.ones(n//2, dtype=torch.float32)
  ]).to(pred.device)
  perm = torch.randperm(n).to(pred.device)

  mask = mask[perm].view(h//patch_size,w//patch_size)

  w = torch.ones((1, 1, patch_size, patch_size), device=pred.device)
  mask = F.conv_transpose2d(mask.unsqueeze(0).unsqueeze(0), w, stride=patch_size, padding=0, output_padding=0)

  loss = 2 * (loss_matrix * mask)
  
  return loss.mean()


def _get_weight(target: torch.Tensor):
        # convert RGB to G
    rgb_to_gray_kernel = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
    target = torch.sum(
        target * rgb_to_gray_kernel.to(target.device), dim=1, keepdim=True
    )
    # initialize sobel kernel in x and y axis
    G_x = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    G_y = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    G_x = torch.tensor(G_x, dtype=target.dtype, device=target.device)[None]
    G_y = torch.tensor(G_y, dtype=target.dtype, device=target.device)[None]
    G = torch.stack((G_x, G_y))

    target = F.pad(target, (1, 1, 1, 1), mode="replicate")  # padding = 1
    grad = F.conv2d(target, G, stride=1)
    mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()

    n, c, h, w = mag.size()
    block_size = 2
    blocks = (
        mag.view(n, c, h // block_size, block_size, w // block_size, block_size)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    block_mean = (
        blocks.sum(dim=(-2, -1), keepdim=True)
        .tanh()
        .repeat(1, 1, 1, 1, block_size, block_size)
        .permute(0, 1, 2, 4, 3, 5)
        .contiguous()
    )
    block_mean = block_mean.view(n, c, h, w)
    weight_map = 1 - block_mean

    return weight_map



def weighted_mse_loss(pred, target):
  
#   loss_matrix =  torch.nn.functional.mse_loss(pred, target,reduction='none')

    with torch.no_grad():
        w = _get_weight((target + 1) / 2)
        
    loss_matrix =  torch.nn.functional.mse_loss(pred, target,reduction='none') * (1-w)
    
    
    return loss_matrix.mean()


        
def multi_resolution_weighted_loss(pred, target, weights = [0.25,0.25,0.25,0.25]):
    """
    Compute a multi-resolution weighted loss between predictions and target.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, C, D, H, W).
        target (torch.Tensor): Target tensor of shape (B, C, D, H, W).
        weights (list): List of weights for each resolution level.
    """
    total_loss = 0.0
    h, w = pred.shape[2], pred.shape[3]
    
    for i, weight in enumerate(weights):
        if i == 0:
            pred_resized = pred
            target_resized = target
        else:
            
            pred_resized = torch.nn.functional.interpolate(pred, size=(h//((2 ** i)), w//((2 ** i))))
            target_resized = torch.nn.functional.interpolate(target, size=(h//((2 ** i)), w//((2 ** i))))
            
            
        loss = weighted_mse_loss(pred_resized, target_resized)
        total_loss += weight * loss
        
        
    return total_loss







