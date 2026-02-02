import numpy as np
import torch
import torch.nn as nn

from mastrl.utils.util import get_gard_norm, huber_loss, mse_loss
from mastrl.utils.valuenorm import ValueNorm
from mastrl.algorithms.utils.util import check

class STHV_MAPPO():
    """Trainer: MAPPO + (STCA advantage shaping) + (HVD critic loss).

    Key design choice for *minimal integration*:
      - We DO NOT change runner/buffer fields.
      - We keep baseline critic and returns/GAE pipeline identical to R_MAPPO.
      - We compute STCA advantages inside ppo_update using actor-produced credit_logits and
        HVD-computed Q_i (built from embeddings + actions + discovered hyperedges).
    """
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        # PPO hyperparams (same as R_MAPPO)
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # STCA/HVD hyperparams
        self.use_stca = getattr(args, "use_stca", True)
        self.credit_temperature = float(getattr(args, "credit_temperature", 1.0))
        self.credit_detach = bool(getattr(args, "credit_detach", True))
        self.use_hvd = getattr(args, "use_hvd", True)
        self.hyperedge_k = int(getattr(args, "hyperedge_k", 3))
        self.max_group_size = int(getattr(args, "max_group_size", 6))
        self.hvd_loss_coef = float(getattr(args, "hvd_loss_coef", 1.0))
                
        assert (self._use_popart and self._use_valuenorm) == False, ("popart and valuenorm cannot both be True")
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
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
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
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()
        return value_loss

    def _actions_to_onehot(self, actions_batch):
        # actions_batch can be discrete indices shaped [B,1] or [B] for each agent
        act_space = self.policy.act_space
        if hasattr(act_space, "n"):
            n = int(act_space.n)
            a = actions_batch.long().view(-1)
            oh = torch.zeros(a.size(0), n, device=a.device)
            oh.scatter_(1, a.unsqueeze(1), 1.0)
            return oh
        # continuous or already vector
        return actions_batch.float()
    

    def _compute_hvd_q(self, z_batch, actions_batch, n_agents):
        """Compute Q_tot and Q_i per sample.

        z_batch: [B*N, D] flattened over agents
        actions_batch: [B*N, ...] flattened over agents
        Returns:
            Q_tot: [B,1]
            Q_i:   [B,N]
        """
        B = z_batch.size(0) // n_agents
        D = z_batch.size(1)
        z = z_batch.view(B, n_agents, D)
        # actions -> one-hot or vector
        a_flat = self._actions_to_onehot(actions_batch)
        A = a_flat.size(-1)
        a = a_flat.view(B, n_agents, A)

        Q_tot_list = []
        Q_i_list = []
        for b in range(B):
            hyperedges = discover_hyperedges_knn(z[b], k=self.hyperedge_k, max_group_size=self.max_group_size)
            q_tot, q_i = self.policy.hvd_critic(z[b], a[b], hyperedges)
            Q_tot_list.append(q_tot)
            Q_i_list.append(q_i.unsqueeze(0))
        Q_tot = torch.stack(Q_tot_list, dim=0)  # [B,1]
        Q_i = torch.cat(Q_i_list, dim=0)        # [B,N]
        return Q_tot, Q_i
    
    def ppo_update(self, sample, update_actor=True):
        # sample tuple shape follows original buffer generators
        if len(sample) == 12:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch = sample
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # evaluate actions + get z and credit logits
        values, action_log_probs, dist_entropy, z, credit_logits = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
            masks_batch, available_actions_batch, active_masks_batch
        )

        # --- STCA advantage shaping (replace adv_targ) ---
        if self.use_stca and self.use_hvd:
            # infer number of agents from obs batch shape; in this codebase obs_batch is flattened [B*N, obs_dim]
            # We infer n_agents from available_actions_batch if present or from active_masks (has shape [B*N,1])
            # A robust way is to take args.num_agents, but args isn't accessible here; we infer from buffer's shapes.
            # Heuristic: share_obs_batch and obs_batch are flattened by feed_forward_generator; it also returns rnn_states shaped [B*N, ...]
            # We therefore require args.num_agents provided as policy attribute; if absent, fallback to 1.
            n_agents = int(getattr(self.policy, "num_agents", 1))
            if n_agents <= 1:
                # can't do inter-agent hypergraph if unknown; fall back to original adv_targ
                shaped_adv = adv_targ
                hvd_loss = torch.zeros([], device=values.device)
            else:
                Q_tot, Q_i = self._compute_hvd_q(z, actions_batch, n_agents)
                # baseline V_i: values is [B*N,1] -> [B,N]
                V_i = values.view(-1, n_agents, 1).squeeze(-1)
                # credit weights: credit_logits [B*N,1] -> [B,N]
                c = credit_logits.view(-1, n_agents, 1).squeeze(-1) / max(self.credit_temperature, 1e-6)
                w = torch.softmax(c, dim=-1)
                if self.credit_detach:
                    w = w.detach()
                # advantage per agent
                A_i = w * (Q_i.detach() - V_i.detach())
                shaped_adv = A_i.reshape(-1, 1)  # [B*N,1]
                # HVD critic TD target: use return_batch (already returns-to-go for each agent) as a proxy for global
                # Minimal: match Q_tot to mean return across agents (keeps scale stable).
                ret = return_batch.view(-1, n_agents, 1).mean(dim=1)  # [B,1]
                hvd_loss = (Q_tot - ret).pow(2).mean()
        else:
            shaped_adv = adv_targ
            hvd_loss = torch.zeros([], device=values.device)

        # PPO actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * shaped_adv
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * shaped_adv

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
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

        # baseline critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()

        # HVD critic update
        self.policy.hvd_critic_optimizer.zero_grad()
        (hvd_loss * self.hvd_loss_coef).backward()
        self.policy.hvd_critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, hvd_loss

    def train(self, buffer, update_actor=True):
        # compute standard advantages (baseline) to preserve normalization pipeline
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_adv = np.nanmean(advantages_copy)
        std_adv = np.nanstd(advantages_copy)
        advantages = (advantages - mean_adv) / (std_adv + 1e-5)

        train_info = {k: 0 for k in ['value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','ratio','hvd_loss']}

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_gn, policy_loss, dist_entropy, actor_gn, imp_w, hvd_loss = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += float(actor_gn)
                train_info['critic_grad_norm'] += float(critic_gn)
                train_info['ratio'] += imp_w.mean().item()
                train_info['hvd_loss'] += hvd_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.hvd_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.hvd_critic.eval()
