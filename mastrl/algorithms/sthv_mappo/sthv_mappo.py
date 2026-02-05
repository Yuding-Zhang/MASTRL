import copy
import numpy as np
import torch
import torch.nn as nn

from mastrl.utils.util import get_gard_norm, huber_loss, mse_loss
from mastrl.utils.valuenorm import ValueNorm
from mastrl.algorithms.utils.util import check

from mastrl.algorithms.sthv_mappo.algorithm.hyperedge import discover_hyperedges_knn

class STHV_MAPPO():
    """Trainer: MAPPO + (STCA advantage shaping) + (hgvd critic loss).

    Key design choice for *minimal integration*:
      - We DO NOT change runner/buffer fields.
      - We keep baseline critic and returns/GAE pipeline identical to R_MAPPO.
      - We compute STCA advantages inside ppo_update using actor-produced credit_logits and
        hgvd-computed Q_i (built from embeddings + actions + discovered hyperedges).
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
        self.gamma = args.gamma

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        # STCA/HGVD hyperparams
        self.use_stca = args.use_stca
        self.credit_temperature = args.credit_temperature
        self.credit_detach = args.credit_detach
        self.use_hgvd = args.use_hgvd
        self.hyperedge_k = args.hyperedge_k
        self.max_group_size = args.max_group_size
        self.hgvd_loss_coef = args.hgvd_loss_coef
        self.hgvd_target_tau = args.hgvd_target_tau
                
        assert (self._use_popart and self._use_valuenorm) == False, ("popart and valuenorm cannot both be True")
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

        self.target_actor = copy.deepcopy(self.policy.actor)
        self.target_hgvd_critic = copy.deepcopy(self.policy.hgvd_critic)
        self._freeze_target(self.target_actor)
        self._freeze_target(self.target_hgvd_critic)

    def _freeze_target(self, module):
        module.eval()
        for p in module.parameters():
            p.requires_grad = False

    def _soft_update(self, target, source, tau):
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

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
        actions_batch = check(actions_batch).to(**self.tpdv)
        act_space = self.policy.act_space
        space_name = act_space.__class__.__name__
        if space_name == "MultiDiscrete":
            actions = actions_batch.long()
            if actions.dim() == 1:
                actions = actions.view(-1, 1)
            nvec = (act_space.high - act_space.low + 1).astype(np.int64)
            onehots = []
            for i, n_i in enumerate(nvec):
                a_i = actions[:, i] - int(act_space.low[i])
                oh_i = torch.zeros(actions.size(0), int(n_i), device=actions.device)
                oh_i.scatter_(1, a_i.unsqueeze(1), 1.0)
                onehots.append(oh_i)
            return torch.cat(onehots, dim=-1)
        if space_name == "Discrete" or hasattr(act_space, "n"):
            n = int(act_space.n)
            a = actions_batch.long().view(-1)
            oh = torch.zeros(a.size(0), n, device=a.device)
            oh.scatter_(1, a.unsqueeze(1), 1.0)
            return oh
        # continuous or already vector
        return actions_batch.float()
    

    def _compute_hgvd_q(self, z_batch, actions_batch, n_agents):
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
            q_tot, q_i = self.policy.hgvd_critic(z[b], a[b], hyperedges)
            Q_tot_list.append(q_tot)
            Q_i_list.append(q_i.unsqueeze(0))
        Q_tot = torch.stack(Q_tot_list, dim=0)  # [B,1]
        Q_i = torch.cat(Q_i_list, dim=0)        # [B,N]
        return Q_tot, Q_i
    
    def ppo_update(self, sample, update_actor=True):
        """One PPO minibatch update (V3).

        Expected sample tuple (feed-forward generator):
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
            value_preds_batch, return_batch, rewards_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,
            adv_targ, available_actions_batch,
            old_credit_logits_batch, z_batch, next_obs_batch, next_rnn_states_batch, next_masks_batch

        Older tuples are supported for baseline compatibility.
        """
        if len(sample) >= 18:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, rewards_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch, old_credit_logits_batch, z_batch, next_obs_batch, next_rnn_states_batch, next_masks_batch = sample[:18]
        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, available_actions_batch = sample[:12]
            rewards_batch = None
            old_credit_logits_batch = None
            z_batch = None
            next_obs_batch = None
            next_rnn_states_batch = None
            next_masks_batch = None

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        values, action_log_probs, dist_entropy, z_new, credit_logits_new = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,
            masks_batch, available_actions_batch, active_masks_batch
        )

        n_agents = int(getattr(self.policy, "num_agents", 1))
        shaped_adv = adv_targ
        hgvd_loss = torch.zeros([], device=values.device)

        # ---------------------------
        # STCA (standalone): advantage shaping from stored (old) credit logits
        # shaped_adv = adv_targ * w, where w = softmax(credit_logits_old / T)
        # ---------------------------
        w = None
        if self.use_stca and (old_credit_logits_batch is not None) and n_agents > 1:
            c_old = check(old_credit_logits_batch).to(**self.tpdv).view(-1, n_agents, 1).squeeze(-1)
            c_old = c_old / max(self.credit_temperature, 1e-6)
            w = torch.softmax(c_old, dim=-1)
            if getattr(self, "credit_detach", False):
                w = w.detach()

            adv_i = adv_targ.view(-1, n_agents, 1).squeeze(-1)  # [B,N]
            shaped_adv = (adv_i * w).reshape(-1, 1)

        # ---------------------------
        # HGVD (standalone): TD loss for hypergraph critic using stored z_t and bootstrapped Q_tot
        # y = r_tot + gamma * mask_{t+1} * Q_tot_target(s_{t+1}, a_{t+1})
        # ---------------------------
        if self.use_hgvd and (z_batch is not None) and (rewards_batch is not None) and (next_obs_batch is not None) and (next_rnn_states_batch is not None) and (next_masks_batch is not None) and n_agents > 1:
            z_old = check(z_batch).to(**self.tpdv)
            Q_tot, _ = self._compute_hgvd_q(z_old, actions_batch, n_agents)

            rewards_t = check(rewards_batch).to(**self.tpdv)  # [B*N,1]
            r_tot = rewards_t.view(-1, n_agents, 1).mean(dim=1)  # [B,1]

            next_obs_t = check(next_obs_batch).to(**self.tpdv)
            next_rnn_t = check(next_rnn_states_batch).to(**self.tpdv)
            next_masks_t = check(next_masks_batch).to(**self.tpdv)

            with torch.no_grad():
                actions_next, _, _, z_next, _ = self.target_actor(next_obs_t, next_rnn_t, next_masks_t, None, deterministic=False)
                Q_tot_next, _ = self._compute_hgvd_q(z_next, actions_next, n_agents)
                mask_next = next_masks_t.view(-1, n_agents, 1)[:, 0]  # [B,1]
                y = r_tot + self.gamma * mask_next * Q_tot_next

            hgvd_loss = (Q_tot - y).pow(2).mean()


        # PPO update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * shaped_adv
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * shaped_adv

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        self.policy.actor_optimizer.zero_grad()
        if update_actor:
            (policy_action_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()

        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        # HGVD update is optional. If HGVD is disabled or hgvd_loss is a constant (no grad), skip.
        if self.use_hgvd and isinstance(hgvd_loss, torch.Tensor) and hgvd_loss.requires_grad:
            self.policy.hgvd_critic_optimizer.zero_grad()
            (hgvd_loss * self.hgvd_loss_coef).backward()
            self.policy.hgvd_critic_optimizer.step()

            self._soft_update(self.target_hgvd_critic, self.policy.hgvd_critic, self.hgvd_target_tau)
            self._soft_update(self.target_actor, self.policy.actor, self.hgvd_target_tau)
        else:
            hgvd_loss = torch.zeros([], device=values.device)

        return value_loss, critic_grad_norm, policy_action_loss, dist_entropy, actor_grad_norm, imp_weights, hgvd_loss

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

        train_info = {k: 0 for k in ['value_loss','policy_loss','dist_entropy','actor_grad_norm','critic_grad_norm','ratio','hgvd_loss']}

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_gn, policy_loss, dist_entropy, actor_gn, imp_w, hgvd_loss = self.ppo_update(sample, update_actor)
                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += float(actor_gn)
                train_info['critic_grad_norm'] += float(critic_gn)
                train_info['ratio'] += imp_w.mean().item()
                train_info['hgvd_loss'] += hgvd_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch
        for k in train_info.keys():
            train_info[k] /= num_updates
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.hgvd_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.hgvd_critic.eval()
