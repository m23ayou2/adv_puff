from pdb import set_trace as T
import numpy as np

import torch
import torch.nn as nn

import pufferlib.emulation
import pufferlib.pytorch
import pufferlib.spaces




class Player2(nn.Module):
    def __init__(self, obs_dim=17, input_size=128, hidden_size=128):
        """
        LSTM-based predictor for drone positions.
        Takes a sequence of observations and predicts the position in the next timestep.

        Args:
            obs_dim:        dimension of each observation vector (features from index 9 onwards)
            input_size:     size of LSTM input embedding
            hidden_size:    hidden dimension of LSTM
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Encoder: map observations → LSTM input size
        self.in_proj = nn.Linear(obs_dim, input_size)

        # LSTM for sequential encoding
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

        # LSTMCell for autoregressive prediction (1 step at a time)
        self.cell = nn.LSTMCell(input_size, hidden_size)

        # Share weights between LSTM and LSTMCell (optional, keeps them consistent)
        self.cell.weight_ih = self.lstm.weight_ih_l0
        self.cell.weight_hh = self.lstm.weight_hh_l0
        self.cell.bias_ih = self.lstm.bias_ih_l0
        self.cell.bias_hh = self.lstm.bias_hh_l0

        # Decoder: map hidden → predicted observation
        self.out_proj = nn.Linear(hidden_size, obs_dim)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization for stable LSTM training."""
        for name, param in self.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and param.ndim >= 2:
                nn.init.orthogonal_(param)

    def forward(self, obs, state=None):
        """
        Standard forward for training.
        obs:   (B, T, obs_dim)
        state: optional dict {'lstm_h', 'lstm_c'} each of shape (1, B, hidden_size)
        Returns:
            preds: (B, T, obs_dim)
            new_state: updated LSTM states
        """
        B, T, D = obs.shape
        if state is None:
            # Ensure dtype and device match input tensor
            h0 = torch.zeros(1, B, self.hidden_size, device=obs.device, dtype=obs.dtype)
            c0 = torch.zeros(1, B, self.hidden_size, device=obs.device, dtype=obs.dtype)
        else:
            h0, c0 = state['lstm_h'], state['lstm_c']



        x = self.in_proj(obs)  # (B, T, input_size)
        out, (h_n, c_n) = self.lstm(x, (h0, c0))
        preds = self.out_proj(out)  # (B, T, obs_dim)

        new_state = {'lstm_h': h_n.detach(), 'lstm_c': c_n.detach()}
        return preds, new_state


class AdversaryPolicy(nn.Module):
    def __init__(self, obs_dim=17, input_size=128, hidden_size=128, temperature=1.0):
        """
        Adversarial observation masking policy.
        Learns to select which observations to pass to Player2 to maximize prediction error.
        Uses gumbel-softmax for differentiable binary mask generation.

        Args:
            obs_dim:        dimension of each observation vector
            input_size:     size of encoder output
            hidden_size:    hidden dimension for the mask generator
            temperature:    temperature for gumbel-softmax (lower = harder masks)
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.temperature = temperature

        # Encoder: map observations → hidden representation
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(obs_dim, input_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(input_size, hidden_size)),
            nn.ReLU(),
        )

        # Mask generator: output logits for each observation dimension
        self.mask_generator = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, obs_dim), std=0.01),
        )

    def forward(self, obs, hard=False):
        """
        Generate observation masks using gumbel-softmax.
        
        Args:
            obs:   (B, T, obs_dim) observation sequence
            hard:  if True, use hard discrete masks; if False, use soft continuous masks
            
        Returns:
            masked_obs: (B, T, obs_dim) masked observations
            mask:       (B, T, obs_dim) soft or hard mask values
        """
        B, T, D = obs.shape
        
        # Flatten time dimension for processing
        obs_flat = obs.reshape(B * T, D)
        
        # Generate mask logits
        hidden = self.encoder(obs_flat)
        mask_logits = self.mask_generator(hidden)
        
        # Apply gumbel-softmax to get differentiable mask
        # Use sigmoid instead of softmax since we want per-element masking
        mask_soft = torch.sigmoid(mask_logits / self.temperature)
        
        if hard:
            # Straight-through estimator: hard mask for forward, soft for backward
            mask = (mask_soft > 0.5).float()
            mask = mask + (mask_soft - mask_soft.detach())
        else:
            mask = mask_soft
        
        # Reshape back to (B, T, obs_dim)
        mask = mask.reshape(B, T, D)
        
        # Apply mask to observations
        masked_obs = obs * mask
        
        return masked_obs, mask

