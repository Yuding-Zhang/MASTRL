import numpy as np
import torch
import torch.nn as nn
from mastrl.utils.util import get_gard_norm, huber_loss, mse_loss
from mastrl.utils.valuenorm import ValueNorm
from mastrl.algorithms.utils.util import check

class STHV_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    Trainer class for MAPPO to update policies.
    Extended with:
      - HGVD (Hypergraph Value Decomposition) in critic (handled inside critic)
      - STCA (Spatio-Temporal Credit Assignment): redistribute advantages using HGVD group assignment.
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # PPO hyper-params
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        # switches
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # STCA switches
        self._use_stca = args.use_stca
        self._stca_detach = args.stca_detach_credit
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / (active_masks_batch.sum() + 1e-6)
        else:
            value_loss = value_loss.mean()

        return value_loss


    @torch.no_grad()
    def _compute_stca_advantages(self, buffer, advantages_np: np.ndarray) -> np.ndarray:
        """
        STCA: redistribute agent advantages using HGVD soft group assignment from the current critic.

        advantages_np: [T, n_threads, n_agents, 1]
        returns: same shape, will return redistributed advantage of same shape.
        """
        # If critic is not HGVD-enabled, fallback to original advantages.
        critic = self.policy.critic
        if not getattr(critic, "_use_hgvd", False):
            return advantages_np

        T, n_threads, n_agents, _ = advantages_np.shape
        # Flatten batch as in training: B_env = T * n_threads, BN = B_env * n_agents
        share_obs = buffer.share_obs[:-1].reshape(T * n_threads * n_agents, -1)
        rnn_states_critic = buffer.rnn_states_critic[:-1].reshape(T * n_threads * n_agents,
                                                                  buffer.rnn_states_critic.shape[-2],
                                                                  buffer.rnn_states_critic.shape[-1])
        masks = buffer.masks[:-1].reshape(T * n_threads * n_agents, -1)

        share_obs_t = check(share_obs).to(**self.tpdv)
        rnn_states_critic_t = check(rnn_states_critic).to(**self.tpdv)
        masks_t = check(masks).to(**self.tpdv)

        # Get group assignment S: [B_env, N, G] and denom: [B_env, 1, G]
        _, _, group_info = critic(share_obs_t, rnn_states_critic_t, masks_t, return_group_info=True)
        S = group_info["S"]  # [B_env, N, G]
        denom = group_info["denom"]  # [B_env,1,G]
        if self._stca_detach:
            S = S.detach()
            denom = denom.detach()

        # advantages: [B_env, N, 1]
        A = torch.as_tensor(advantages_np.reshape(T * n_threads, n_agents, 1), device=self.device, dtype=torch.float32)

        # group advantage: A_g = sum_i S_i,g * A_i  -> [B_env, G, 1]
        A_g = torch.einsum("bng,bn1->bg1", S, A)

        # redistribute to agents: A'_i = sum_g S_i,g * A_g / sum_j S_j,g
        denom_g = denom.transpose(1, 2).clamp_min(1e-6)  # [B_env,G,1]
        A_redist = torch.einsum("bng,bg1->bn1", S, A_g / denom_g)

        return A_redist.cpu().numpy().reshape(T, n_threads, n_agents, 1)

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        # compute raw advantages
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        # STCA redistribution (algorithmic credit assignment)
        if self._use_stca:
            advantages = self._compute_stca_advantages(buffer, advantages)

        # normalize advantages (mask dead agents)
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {
            "value_loss": 0.0,
            "policy_loss": 0.0,
            "dist_entropy": 0.0,
            "actor_grad_norm": 0.0,
            "critic_grad_norm": 0.0,
            "ratio": 0.0,
        }

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights =                     self.ppo_update(sample, update_actor)

                train_info["value_loss"] += float(value_loss.item())
                train_info["policy_loss"] += float(policy_loss.item())
                train_info["dist_entropy"] += float(dist_entropy.item())
                train_info["actor_grad_norm"] += float(actor_grad_norm)
                train_info["critic_grad_norm"] += float(critic_grad_norm)
                train_info["ratio"] += float(imp_weights.mean().item())

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= max(num_updates, 1)

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
