import torch
import torch.nn as nn
import math
import numpy as np

class SRigLScheduler:
    """
    Structured RigL (SRigL) for dlshogi - TensorRT 2:4 Structured Sparsity Implementation
    
    Targeting NVIDIA Ampere+ Tensor Core acceleration with 2:4 fine-grained sparsity.
    
    Constraint:
    - Every block of 4 contiguous elements in the input channel dimension must have exactly 2 non-zeros.
    - Global sparsity is fixed at 50%.
    
    Algorithm:
    - Initialize with random 2:4 masks.
    - Periodically update masks based on "Magnitudes + Gradients":
      Score = |Weight| + alpha * |Gradient|
      For each block of 4, keep top-2 weights with highest scores.
    """
    
    def __init__(self, 
                 model,
                 update_freq=1000,
                 alpha=1.0,
                 start_step=0,
                 end_step=None,
                 device='cuda'):
        """
        Args:
            model: PyTorch model
            update_freq: Topology update frequency
            alpha: Weight for gradient contribution in score (Score = |W| + alpha * |G|)
            start_step: Step to start sparse updates
            end_step: Step to stop sparse updates (fix topology)
            device: Device for computations
        """
        self.model = model
        self.sparsity = 0.5  # Fixed for 2:4
        self.update_freq = update_freq
        self.alpha = alpha
        self.start_step = start_step
        self.end_step = end_step
        self.device = device
        
        # State
        self.masks = {}
        self.layers_to_prune = []
        self.initialized = False
        
        print(f"\n{'='*70}")
        print(f"SRigL - 2:4 Structured Sparsity for TensorRT")
        print(f"{'='*70}")
        print(f"Sparsity: 50% (2:4 fixed)")
        print(f"Update Frequency: {self.update_freq} steps")
        print(f"{'='*70}\n")
        
        self._initialize()
    
    def _initialize(self):
        """Initialize 2:4 masks."""
        # Identify Conv2d and Linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # TensorRT 2:4 requires K (input channels) to be divisible by 4
                if isinstance(module, nn.Conv2d):
                     # Conv2d weights: [Out, In, KH, KW]
                     # Sparsity is usually applied to the Input channel dimension
                    if module.in_channels % 4 != 0:
                        print(f"Skipping {name}: in_channels {module.in_channels} not divisible by 4")
                        continue
                elif isinstance(module, nn.Linear):
                    # Linear weights: [Out, In]
                    if module.in_features % 4 != 0:
                        print(f"Skipping {name}: in_features {module.in_features} not divisible by 4")
                        continue
                        
                self.layers_to_prune.append((name, module))
        
        if len(self.layers_to_prune) == 0:
            print("[WARNING] No layers suitable for 2:4 sparsity found!")
            return
        
        print("Initializing 2:4 masks...")
        
        for name, module in self.layers_to_prune:
            weight = module.weight.data
            mask = self._create_2_4_mask(weight) # Initialize with random 2:4 or magnitude based
            self.masks[name] = mask
            
            # Apply initial mask
            module.weight.data.mul_(mask)
            
            print(f"  {name}: 2:4 mask initialized.")
        
        self.initialized = True
        print("")
    
    def _create_2_4_mask(self, tensor, scores=None):
        """
        Create a 2:4 mask based on scores.
        If scores is None, uses random selection (or magnitude if tensor provided).
        tensor: Weight tensor [Out, In, ...]
        """
        with torch.no_grad():
            if scores is None:
                scores = torch.abs(tensor)
            
            # Reshape to identify blocks of 4 in the input dimension (dim 1)
            # Conv2d: [Out, In, KH, KW] -> [Out, In//4, 4, KH, KW] -> permute to put 4 at end
            # Linear: [Out, In] -> [Out, In//4, 4]
            
            original_shape = tensor.shape
            
            if tensor.dim() == 4: # Conv2d
                out_c, in_c, kh, kw = original_shape
                # Reshape to [Out * (In//4) * KH * KW, 4]
                # We want to select 2 out of every 4 along In dimension.
                # It's easiest to treat it as a collection of vectors of length 4.
                # However, TensorRT requires the 2:4 pattern specifically along the C_in dimension.
                # Structure: For each output channel and each spatial position, the input channel vector is sparse.
                
                # Reshape so that the dimension of size 4 is the last one
                # [Out, In, KH, KW] -> [Out, KH, KW, In] -> [Out, KH, KW, In//4, 4]
                temp = scores.permute(0, 2, 3, 1).reshape(out_c, kh, kw, in_c // 4, 4)
                
                # Find top 2 indices in the last dimension
                _, top_indices = torch.topk(temp, 2, dim=-1) # indices in range [0, 3]
                
                # Create mask
                mask_temp = torch.zeros_like(temp)
                mask_temp.scatter_(-1, top_indices, 1.0)
                
                # Restore shape
                # [Out, KH, KW, In//4, 4] -> [Out, KH, KW, In] -> [Out, In, KH, KW]
                mask = mask_temp.reshape(out_c, kh, kw, in_c).permute(0, 3, 1, 2)
                
            elif tensor.dim() == 2: # Linear
                out_f, in_f = original_shape
                # [Out, In//4, 4]
                temp = scores.reshape(out_f, in_f // 4, 4)
                
                _, top_indices = torch.topk(temp, 2, dim=-1)
                
                mask_temp = torch.zeros_like(temp)
                mask_temp.scatter_(-1, top_indices, 1.0)
                
                mask = mask_temp.reshape(original_shape)
                
            else:
                raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
                
            return mask

    def before_step(self, batch):
        pass # Not used in this simplified version
    
    def after_backward(self):
        """Apply mask to gradients to ensure sparse updates."""
        if not self.initialized:
            return
            
        with torch.no_grad():
            for name, module in self.layers_to_prune:
                if module.weight.grad is not None:
                    module.weight.grad.mul_(self.masks[name])

    def after_step(self, global_step):
        """Update masks and weights."""
        if not self.initialized:
            return
            
        # Enforce mask on weights (just in case optimizer violated it)
        with torch.no_grad():
            for name, module in self.layers_to_prune:
                module.weight.data.mul_(self.masks[name])
                
        # Topology Update
        if (self.end_step is None or global_step < self.end_step) and \
           (global_step >= self.start_step) and \
           (global_step % self.update_freq == 0):
            
            print(f"[SRigL] Updating 2:4 masks at step {global_step}")
            self._update_topology()
            
    def _update_topology(self):
        with torch.no_grad():
            for name, module in self.layers_to_prune:
                weight = module.weight
                if weight.grad is None:
                    continue
                    
                # Score = |Weight| + alpha * |Gradient|
                # Higher score = more important to keep/grow
                
                # Gradients for zero weights tell us where to grow.
                # Weights for non-zero tell us what to keep.
                
                score = torch.abs(weight) + self.alpha * torch.abs(weight.grad)
                
                # Re-calculate optimal 2:4 mask based on these scores
                new_mask = self._create_2_4_mask(weight, scores=score)
                
                # Update mask
                self.masks[name] = new_mask
                
                # Apply new mask to weight immediately
                # (Newly grown weights will be 0 initially, which is fine, they will learn next step)
                # (Pruned weights will become 0)
                weight.data.mul_(new_mask)
                
                # Reset optimizer state for pruned weights? 
                # RigL usually resets momentum for new weights. 
                # For simplicity, we skip complex optimizer state manipulation here, 
                # as 2:4 switches are frequent and local.
