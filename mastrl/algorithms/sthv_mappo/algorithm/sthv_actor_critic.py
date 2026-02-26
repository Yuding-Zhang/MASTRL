import torch
import torch.nn as nn

from mastrl.algorithms.utils.util import init, check
from mastrl.algorithms.utils.cnn import CNNBase
from mastrl.algorithms.utils.mlp import MLPBase
from mastrl.algorithms.utils.act import ACTLayer
from mastrl.algorithms.utils.popart import PopArt
from mastrl.utils.util import get_shape_from_obs_space

from mastrl.algorithms.sthv_mappo.algorithm.st_encoder import STEncoder


class STHV_Actor(nn.Module):
    """Actor with an extra ST encoder and a credit head.

    NOTE: We keep action distribution head (ACTLayer) unchanged.
    We only change the feature pathway and the forward return signature.
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        # --- NEW: ST encoder and credit head ---
        self.st_encoder = STEncoder(
            d_model=self.hidden_size,
            n_heads_s=getattr(args, "st_n_heads_s", 4),
            n_heads_t=getattr(args, "st_n_heads_t", 4),
            dropout=getattr(args, "st_dropout", 0.0)
        )

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)
        self.to(device)

    def forward(self, obs, masks, available_actions=None, deterministic=True, need_aux: bool = True):
        obs = check(obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        feat = self.base(obs) # [T,B,N,降维]

        # Full ST encoder + credit head (used in training / eval)
        z, credit_logits = self.st_encoder(feat, masks)  # [T,B,N,D] -> [T,B,N,D], [T,B,N,1]
        # Extract the last time step (T-1) and reshape to [B, N, D]
        z_last = z[-1]  # Shape: [B, N, D]
        credit_logits_last = credit_logits[-1]  # Shape: [B, N, 1]

        actions, action_log_probs = self.act(z_last, available_actions, deterministic) # [B,N,D] -> [B,N,1]
        return actions, action_log_probs, z_last, credit_logits_last

    def evaluate_actions(self, obs, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)
            

        feat = self.base(obs)

        z, credit_logits = self.st_encoder(feat, masks)
        # 将 z 从 [T, B, N, D] 转换为 [T*B*N, D]
        T, B, N, D = z.shape
        z = z.view(T * B * N, D)
        credit_logits = credit_logits.view(T * B * N, -1)
        action = action.reshape(T * B * N, -1)
        if available_actions is not None:
            available_actions = available_actions.reshape(T * B * N, -1)
        if active_masks is not None:
            active_masks = active_masks.reshape(T * B * N, -1)
        
        action_log_probs, dist_entropy = self.act.evaluate_actions(
            z, action, available_actions,
            active_masks=active_masks if self._use_policy_active_masks else None
        )
        return action_log_probs, dist_entropy, z, credit_logits

class STHV_Critic(nn.Module):
    """Keep the original MAPPO critic as a baseline V(s) critic for returns/GAE (unchanged semantics)."""
    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super().__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]
        
        # --- NEW: ST encoder and credit head ---
        self.st_encoder = STEncoder(
            d_model=self.hidden_size,
            n_heads_s=getattr(args, "st_n_heads_s", 4),
            n_heads_t=getattr(args, "st_n_heads_t", 4),
            dropout=getattr(args, "st_dropout", 0.0)
        )

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)


        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        feat = self.base(cent_obs)
        z, _ = self.st_encoder(feat, masks)
        z_last = z[-1]  
        values = self.v_out(z_last)
        return values
    
    def evaluate_values(self, cent_obs, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        feat = self.base(cent_obs)
        z, _ = self.st_encoder(feat, masks)
        values = self.v_out(z)
        return values
    
