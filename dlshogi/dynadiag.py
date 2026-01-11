import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DiagonalParam(nn.Module):
    """
    Base class for diagonal parameters.
    Implements the core logic of W = Sum(TopK(alpha) * P * diag(V)).
    """
    def __init__(self, out_features, in_features, sparsity=0.9, temperature=1.0, 
                 trainable_topk=True):
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.sparsity = sparsity
        self.temperature = temperature
        
        # Determine number of diagonals K
        N_min = min(out_features, in_features)
        total_params = out_features * in_features
        # Ensure at least 1 diagonal
        self.K = max(1, int((1 - sparsity) * total_params / N_min))
        
        # Max possible diagonals = N_max + N_min - 1 ? 
        # Actually in the paper: K_total = max(M, N). The paper formulation (Eq 1) 
        # uses modular arithmetic, so there are effectively max(M, N) unique offsets
        # if we wrap around, or simpler: just N_min diagonals in a full matrix?
        # Re-reading paper Section 3.1: "K_total = max(M,N)"
        # "We generalize... to represent any matrix W... with K being the required number of diagonals"
        # The offset "off" can range to cover all diagonals.
        # Let's use the modular definition: j = (i + off) % N.
        # There are `in_features` possible offsets if N=in_features. 
        # Wait, usually for rectangular matrices, the number of "diagonals" in the modular sense is just `in_features`.
        # Because for each row i, there are `in_features` columns j.
        # We will use `in_features` as the number of possible candidates for offsets.
        self.num_candidates = in_features 
        
        # Importance weights alpha (learnable)
        self.alpha = nn.Parameter(torch.zeros(self.num_candidates))
        nn.init.normal_(self.alpha, mean=0.0, std=0.01)
        
        # Values V (learnable). 
        # We have K * N_min parameters. Or do we keep all V for all candidates?
        # The paper says: "W_K = Sum(alpha_j * P_j * diag(V_j))"
        # To make it fully differentiable/learnable, we probably keep V for all candidates 
        # but only the top-K alphas will have significant magnitude (or we hard mask).
        # However, storing V for ALL candidates would be O(N^2) (dense).
        # To be memory efficient, we can only store V for the current active diagonals?
        # But DynaDiag is "Dynamic Sparse Training", so offsets change.
        # The paper implies we learn alpha to select diagonals.
        # If we want to be truly sparse-to-sparse, we should only allocate V for active diagonals.
        # BUT, the paper mentions "TopK Based Diagonal Selection" in a differentiable way.
        # "We introduce a learnable vector of importance weights alpha... We use a TopK function..."
        # Section 3.1 Eq 3: "Sum_{j=1}^K P_j diag(V_j)".
        # If we strictly follow the formula with Softmax-TopK, all alphas are non-zero (just small).
        # So technically we need V for all diagonals?
        # That would be dense storage.
        # Section 3.3 says "DynaDiag enforces a diagonal sparsity pattern throughout training...".
        # This implies we only store/compute active diagonals.
        # In this implementation, to be safe and simple for now (and since we simulated DynaDiag),
        # we will use a dense V buffer if memory allows (it's same size as weights),
        # or we implement the dynamic selection strictly.
        # Given "DynaDiag is a ... sparse-to-sparse DST method", we should probably use sparse storage.
        # However, for the *differentiable* TopK to work (gradients flowing to alpha), 
        # we usually need to evaluate the "soft" selection.
        # Let's check Eq 5: min(k * exp(a/T) / sum(...), 1).
        # This is Softmax-TopK.
        
        # Optimization: We'll store `values` as a dense tensor (out, in) but conceptually it represents diagonal values.
        # Actually, if we store 'weight' directly, we can just mask it.
        # The paper formulation separates P (position) and V (values).
        # Let's stick to the paper's modular diagonal definition:
        # W[i, j] is non-zero if (j - i) % in_features in {selected_offsets}
        
        # For efficient implementation in PyTorch without custom CUDA kernels:
        # We can maintain a mask M based on TopK(alpha).
        # W = V * M.
        # This is "masking" approach.
        self.values = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.values, a=math.sqrt(5))
        
        # Store indices for all possible offsets: 0 to in_features-1
        self.all_offsets = torch.arange(self.num_candidates)
        
        self.register_buffer('active_offsets', torch.zeros(self.K, dtype=torch.long))
        
    def get_mask(self):
        # 1. Softmax TopK on alpha to get importance scores
        # Eq 5: alpha_tilde = min(k * softmax(alpha/T), 1)
        # Note: This gives soft weights. For strict sparsity (inference), we need hard TopK.
        
        if self.training:
            # Differentiable soft approximation? 
            # The paper mentions "DynaDiag enforces a diagonal sparsity pattern throughout training".
            # This suggests they might use Straight-Through Estimator or just Softmax for weighting?
            # "We use the following softmax-based TopK... alpha_tilde = min(...)"
            # And then W = Sum(alpha_tilde * P * V).
            # This W would be dense if alpha_tilde is not hard-sparse.
            # But they claim "sparse-to-sparse".
            # Usually this implies: Forward pass uses Hard TopK. Backward pass updates alpha.
            # Let's use Straight-Through Estimator (STE) for TopK.
            
            scores = self.alpha
            # Hard TopK
            _, topk_indices = torch.topk(scores, self.K)
            
            # Create binary mask for topk offsets
            mask = torch.zeros_like(self.values)
            
            # Vectorized mask creation for modular diagonals
            # Indices (i, j) are active if (j - i) % in_features in topk_indices
            
            # This step can be slow if done naively in python loop.
            # We can construct it via tensor broadcasting.
            rows = torch.arange(self.out_features, device=self.values.device).unsqueeze(1) # (O, 1)
            cols = torch.arange(self.in_features, device=self.values.device).unsqueeze(0)  # (1, I)
            
            # diff = (col - row) % in_features
            diffs = (cols - rows) % self.in_features
            
            # Check membership efficiently
            # We create a boolean vector of valid offsets
            valid_offsets = torch.zeros(self.in_features, device=self.values.device, dtype=torch.bool)
            valid_offsets[topk_indices] = True
            
            mask = valid_offsets[diffs].float()
            
            # STE: 
            # forward: use hard mask
            # backward: gradients flow to alpha?
            # The paper's formulation in Eq 4 uses alpha_tilde as a multiplication factor.
            # W = Sum(alpha_tilde * ...).
            # If alpha_tilde is soft, W is dense.
            # If alpha_tilde is hard (via STE), W is sparse.
            
            # Let's implement the soft weighting during training but masked to only TopK?
            # Or maybe they accept dense updates during training (like Soft-Movement Pruning)?
            # "DynaDiag ... preserves sparse computation in forward and backward passes."
            # This strongly implies W is physically sparse.
            # Thus, we must use Hard TopK.
            # To make alpha learnable, we likely need to backprop through the selection?
            # Typically done via: mask = hard_mask + (soft_mask - soft_mask.detach())
            
            soft_alpha = torch.clamp(
                self.K * F.softmax(self.alpha / self.temperature, dim=0), 
                max=1.0
            )
            
            # Extract soft values for the selected offsets
            soft_alpha_expanded = torch.zeros_like(mask)
            # Map alpha to mask positions
            # soft_alpha_expanded[i, j] = soft_alpha[(j-i)%N]
            soft_alpha_expanded = soft_alpha[diffs]
            
            # STE
            mask = mask.detach() - soft_alpha_expanded.detach() + soft_alpha_expanded
            
            return mask
        else:
            # Inference: strict Hard TopK
            _, topk_indices = torch.topk(self.alpha, self.K)
            rows = torch.arange(self.out_features, device=self.values.device).unsqueeze(1)
            cols = torch.arange(self.in_features, device=self.values.device).unsqueeze(0)
            diffs = (cols - rows) % self.in_features
            valid_offsets = torch.zeros(self.in_features, device=self.values.device, dtype=torch.bool)
            valid_offsets[topk_indices] = True
            mask = valid_offsets[diffs].float()
            return mask

    def forward(self, x):
        # This is a base param class, actual forward is in sub-classes
        pass

class DynaDiagLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, sparsity=0.9, temperature=1.0):
        super().__init__(in_features, out_features, bias)
        self.sparsity = sparsity
        self.param_diag = DiagonalParam(out_features, in_features, sparsity, temperature)
        
        # We override self.weight with a property or just ignore it and use param_diag?
        # nn.Linear defines self.weight. We can just overwrite it in forward.
        # But we need to keep 'weight' parameter for optimizers if we want standard behavior?
        # Actually DynaDiag learns values V and alphas.
        # We should disable standard weight parameter or link it.
        del self.weight
        # Re-register V as parameters is done in DiagonalParam.
        # But we need to make sure optimizer sees them.
        # self.param_diag is a submodule, so its params are seen.
        
    def forward(self, input):
        mask = self.param_diag.get_mask()
        masked_weight = self.param_diag.values * mask
        return F.linear(input, masked_weight, self.bias)

class DynaDiagConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros', 
                 sparsity=0.9, temperature=1.0):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, 
                         dilation, groups, bias, padding_mode)
        
        # For Conv2d, diagonals are usually defined on the (Out, In) matrix formed by 
        # flattening the kernel? Or just channel-wise?
        # Paper says: "DynaDiag faces scalability challenges with CNNs due to the overhead of searching for distinct diagonal patterns across each channel."
        # And "We evaluate... MLP-Mixer and ViT... CNNs... in Appendix F.1".
        # Appendix F.1 uses ResNet.
        # The standard way to sparsify Conv2d is to flatten (C_out, C_in * K * K) or (C_out * K * K, C_in)?
        # SRigL treats it as (Out, In * Kh * Kw).
        # Let's assume we treat the kernel tensor as a matrix [Out, In * Kh * Kw].
        
        self.flatten_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.param_diag = DiagonalParam(out_channels, self.flatten_in, sparsity, temperature)
        del self.weight
        
    def forward(self, input):
        mask = self.param_diag.get_mask()
        # Reshape mask to 4D
        mask_4d = mask.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        weight_4d = self.param_diag.values.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        
        masked_weight = weight_4d * mask_4d
        
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            masked_weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, masked_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DynaDiagScheduler:
    """
    Manages the temperature annealing for DynaDiag.
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
        # Cosine annealing for temperature
        progress = min(1.0, self.current_step / self.total_steps)
        # Temp goes from High to Low? 
        # Paper: "adjust T from a high starting value...".
        # Eq: T decay schedule.
        # Let's use simple cosine decay.
        temp = self.temp_final + 0.5 * (self.temp_init - self.temp_final) * (1 + math.cos(math.pi * progress))
        
        for m in self.diag_modules:
            m.param_diag.temperature = temp

def convert_to_dynadiag(model, sparsity=0.9, exclude_first_layer=True):
    """
    Replaces Linear and Conv2d layers with DynaDiag equivalents.
    """
    import copy
    
    # We need to traverse and replace.
    # Simple recursive replacement
    
    def replace_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check exclusions (e.g. first layer conv)
            if exclude_first_layer and "input" in full_name.lower(): # Simple heuristic
                continue
                
            if isinstance(child, nn.Linear):
                new_layer = DynaDiagLinear(child.in_features, child.out_features, 
                                           child.bias is not None, sparsity=sparsity)
                # Initialize values with original weights
                new_layer.param_diag.values.data = child.weight.data.clone()
                if child.bias is not None:
                    new_layer.bias.data = child.bias.data.clone()
                setattr(module, name, new_layer)
                
            elif isinstance(child, nn.Conv2d):
                # Skip 1x1 or groups>1 if needed (DynaDiag mainly targets large layers)
                if child.groups > 1: 
                    continue
                if child.kernel_size == (1, 1): # Optional: skip 1x1
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
