import torch
from mastrl.algorithms.sthv_mappo.algorithm.sthv_actor_critic import STHV_Actor, Baseline_R_Critic
from mastrl.algorithms.sthv_mappo.algorithm.hgvd_critic import HypergraphCritic
from mastrl.utils.util import update_linear_schedule


def _infer_act_dim(act_space):
    # Discrete: act_space.n; MultiDiscrete: sum(nvec); Box: shape[-1]
    if hasattr(act_space, 'n'):
        return int(act_space.n)
    if hasattr(act_space, 'nvec'):
        return int(sum(act_space.nvec))
    if hasattr(act_space, 'shape'):
        return int(act_space.shape[-1])
    raise ValueError("Unsupported action space type for act_dim inference.")


class STHV_MAPPOPolicy:
    """Policy wrapper with:
      - Actor: STHV_Actor (returns credit logits and embeddings)
      - Baseline critic: V(s) for returns/GAE/value loss
      - hgvd critic: HypergraphCritic for Q_tot/Q_i (used in actor advantage shaping + additional critic loss)
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.hgvd_critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.num_agents = args.num_agents

        self.actor = STHV_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Baseline_R_Critic(args, self.share_obs_space, self.device)

        act_dim = _infer_act_dim(act_space)
        self.hgvd_critic = HypergraphCritic(embed_dim=args.hidden_size, act_dim=act_dim, hidden_dim=getattr(args, "hgvd_hidden_dim", 128)).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        self.hgvd_critic_optimizer = torch.optim.Adam(self.hgvd_critic.parameters(), lr=self.hgvd_critic_lr, eps=self.opti_eps, weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.hgvd_critic_optimizer, episode, episodes, self.hgvd_critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, deterministic=False):
        # actor: (actions, logp, rnn, z, credit_logits)
        actions, action_log_probs, rnn_states_actor, z, credit_logits = self.actor(
            obs, rnn_states_actor, masks, available_actions, deterministic
        )
        # baseline critic
        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, z, credit_logits

    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, available_actions=None, active_masks=None):
        # actor evaluate -> logp, entropy, z, credit_logits
        action_log_probs, dist_entropy, z, credit_logits = self.actor.evaluate_actions(obs, rnn_states_actor, action, masks, available_actions, active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values, action_log_probs, dist_entropy, z, credit_logits

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor, _, _ = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
