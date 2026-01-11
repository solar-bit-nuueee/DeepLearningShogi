import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiagonalParam(nn.Module):
    """
    Parameterizes weights using diagonal offsets and values as per DynaDiag.
    Paper: "Dynamic Sparse Training of Diagonally Sparse Networks"
    
    NOTE: This implementation simulates DynaDiag using dense tensor operations.
    It reproduces the *learning dynamics* and *accuracy* of the method, but NOT the 
    training speedup. Real speedup requires custom CUDA kernels for BCSR conversion.
    """
    def __init__(self, out_features, in_features, sparsity=0.9, temperature=1.0):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.sparsity = sparsity
        self.temperature = temperature
        
        # 1. Diagonal Definition: Pseudo-diagonals based on max(M, N)
        # Appendix A: "modulo base for pseudo-diagonals is max(M, N)"
        self.num_candidates = max(out_features, in_features)
        
        # Determine number of active diagonals K based on sparsity
        # Eq 3 context: K = (1 - S) * M * N / min(M, N)
        total_params = out_features * in_features
        min_dim = min(out_features, in_features)
        target_active = int(total_params * (1.0 - sparsity))
        self.K = max(1, int(target_active / min_dim))
        
        # Importance weights alpha (learnable) - one per diagonal candidate
        self.alpha = nn.Parameter(torch.zeros(self.num_candidates))
        nn.init.normal_(self.alpha, mean=0.0, std=0.01)
        
        # Values V (learnable). 
        # Stored as dense tensor for PyTorch simulation.
        # Ideally, we only store active diagonals, but for simulation we mask a dense weight.
        self.values = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.values, a=math.sqrt(5))
        
        # Pre-compute indices for fast mask generation
        # We need (j - i) % num_candidates for every (i, j)
        rows = torch.arange(out_features).unsqueeze(1)
        cols = torch.arange(in_features).unsqueeze(0)
        self.register_buffer('diag_indices', (cols - rows) % self.num_candidates)
        
    def get_mask(self):
        """
        Computes the binary mask using Differentiable TopK (Eq 5).
        """
        # Eq 5: alpha_tilde = min(k * exp(a/T) / sum(...), 1)
        # We implement this soft selection.
        soft_scores = self.K * F.softmax(self.alpha / self.temperature, dim=0)
        soft_scores = torch.clamp(soft_scores, max=1.0)
        
        if self.training:
            # Forward: Use Hard TopK (Structure)
            # Backward: Gradients flow through Soft TopK (Selection) -> STE
            
            # 1. Hard TopK Mask
            _, topk_indices = torch.topk(self.alpha, self.K)
            
            # Create boolean mask for active diagonals
            # diag_indices shape: (out, in), values in [0, num_candidates-1]
            # active_diagonals shape: (num_candidates,)
            active_diagonals = torch.zeros(self.num_candidates, device=self.alpha.device, dtype=torch.bool)
            active_diagonals[topk_indices] = True
            
            hard_mask = active_diagonals[self.diag_indices].float()
            
            # 2. Soft Mask (Dense, for gradient approximation)
            # Expand soft_scores to (out, in)
            soft_mask = soft_scores[self.diag_indices]
            
            # 3. Straight-Through Estimator
            # y = hard + (soft - soft.detach())
            mask = hard_mask + (soft_mask - soft_mask.detach())
        else:
            # Inference: Strict Hard TopK
            _, topk_indices = torch.topk(self.alpha, self.K)
            active_diagonals = torch.zeros(self.num_candidates, device=self.alpha.device, dtype=torch.bool)
            active_diagonals[topk_indices] = True
            mask = active_diagonals[self.diag_indices].float()
            
        return mask

    def forward(self, x):
        pass

class DynaDiagLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, sparsity=0.9, temperature=1.0):
        super().__init__(in_features, out_features, bias)
        self.param_diag = DiagonalParam(out_features, in_features, sparsity, temperature)
        
        # Initialize values from original weights (if converting) or standard init
        # Note: We keep self.weight as a dummy or delete it. 
        # Safest is to delete it to ensure we don't accidentally use it.
        del self.weight
        
    def forward(self, input):
        mask = self.param_diag.get_mask()
        # Apply mask to the dense values
        masked_weight = self.param_diag.values * mask
        return F.linear(input, masked_weight, self.bias)
        
    def get_l1_reg(self):
        # L1 regularization on alpha (importance weights)
        return torch.sum(torch.abs(self.param_diag.alpha))

class DynaDiagConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 sparsity=0.9, temperature=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                         dilation, groups, bias, padding_mode)
        
        # Flatten kernel for diagonal application
        # Paper Appendix F.1 (CNNs): treats Conv weights as matrix
        # Usually (Out, In * Kh * Kw)
        self.flatten_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.param_diag = DiagonalParam(out_channels, self.flatten_in, sparsity, temperature)
        del self.weight
        
    def forward(self, input):
        mask = self.param_diag.get_mask()
        # Reshape mask/values to 4D for conv2d
        mask_4d = mask.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        weight_4d = self.param_diag.values.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        
        masked_weight = weight_4d * mask_4d
        
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            masked_weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def get_l1_reg(self):
        return torch.sum(torch.abs(self.param_diag.alpha))

class DynaDiagScheduler:
    """
    Manages temperature annealing for DynaDiag.
    """
    def __init__(self, model, temperature_init=10.0, temperature_final=0.1, total_steps=100000):
        self.model = model
        self.temp_init = temperature_init
        self.temp_final = temperature_final
        self.total_steps = total_steps
        self.current_step = 0
        
        self.diag_modules = []
        for m in model.modules():
            if isinstance(m, (DynaDiagLinear, DynaDiagConv2d)):
                self.diag_modules.append(m)
                
    def step(self):
        self.current_step += 1
        progress = min(1.0, self.current_step / self.total_steps)
        # Cosine annealing
        temp = self.temp_final + 0.5 * (self.temp_init - self.temp_final) * (1 + math.cos(math.pi * progress))
        
        for m in self.diag_modules:
            m.param_diag.temperature = temp

def convert_to_dynadiag(model, sparsity=0.9, exclude_first_layer=True):
    """
    Replaces Linear and Conv2d layers with DynaDiag equivalents.
    """
    def replace_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if exclude_first_layer and "input" in full_name.lower():
                continue
                
            if isinstance(child, nn.Linear):
                new_layer = DynaDiagLinear(child.in_features, child.out_features, 
                                           child.bias is not None, sparsity=sparsity)
                # Copy weights
                new_layer.param_diag.values.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                setattr(module, name, new_layer)
                
            elif isinstance(child, nn.Conv2d):
                if child.groups > 1: 
                    continue
                if child.kernel_size == (1, 1):
                    continue
                    
                new_layer = DynaDiagConv2d(child.in_channels, child.out_channels, child.kernel_size,
                                           child.stride, child.padding, child.dilation, child.groups,
                                           child.bias is not None, child.padding_mode, sparsity=sparsity)
                new_layer.param_diag.values.data = child.weight.data.clone().view(child.out_channels, -1)
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                setattr(module, name, new_layer)
            
            else:
                replace_layers(child, full_name)
                
    replace_layers(model)
    return model

def get_dynadiag_l1_reg(model):
    """Computes total L1 regularization loss for all alpha parameters."""
    l1_loss = 0.0
    for m in model.modules():
        if isinstance(m, (DynaDiagLinear, DynaDiagConv2d)):
            l1_loss += m.get_l1_reg()
    return l1_loss
