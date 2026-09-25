import torch
from skrl.agents.torch.ppo import PPO
from .rnd import RandomNetworkDistillation

class RND_PPO(PPO):
    def __init__(self, *, models, memory=None, observation_space=None, state_space=None, action_space=None, device=None, cfg={}):
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=cfg,
        )
        self.rnd = RandomNetworkDistillation(
            input_dim=10,
            feature_dim=128,
            learning_rate=1e-4,
            device=device,
        )
        self.rnd_beta = 0.01
        self.rnd_obs_dim = 10
        self.checkpoint_modules["rnd"] = self.rnd
        self.checkpoint_modules["rnd_optimizer"] = self.rnd.optimizer

    def init(self, *, trainer_cfg=None):
        super().init(trainer_cfg=trainer_cfg)
        self.memory.create_tensor(
            name="rnd_observations",
            size=self.rnd_obs_dim,
            dtype=torch.float32,
        )
        self._rnd_cumulative_rewards = None

    def record_transition(
        self,
        *,
        observations,
        states,
        actions,
        rewards,
        next_observations,
        next_states,
        terminated,
        truncated,
        infos,
        timestep,
        timesteps,
    ):
        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states
            rnd_observations = next_observations[:, :self.rnd_obs_dim]
            with torch.no_grad():
                intrinsic_reward = self.rnd.get_intrinsic_reward(rnd_observations)
            done = terminated | truncated
            start_pos = self.env_origins[:, :2]
            current_pos = self.robot.data.root_pos_w[:, :2]
            dist_from_start = torch.norm(current_pos - start_pos, dim=-1)
            range_mask = dist_from_start < 5.0
            intrinsic_reward = intrinsic_reward * range_mask.unsqueeze(-1).float()
            intrinsic_reward = intrinsic_reward * (~done).float()
            scaled_intrinsic_reward = self.rnd_beta * intrinsic_reward
            rewards = rewards + scaled_intrinsic_reward
            if self._rnd_cumulative_rewards is None:
                self._rnd_cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self._rnd_cumulative_rewards += rewards
            if done.any():
                episode_rewards = self._rnd_cumulative_rewards[done]
                self.track_data("RND / Episode reward", episode_rewards.mean().item())
                self._rnd_cumulative_rewards[done] = 0
            self.track_data("RND / Intrinsic reward", intrinsic_reward.mean().item())
            self.track_data("RND / Scaled intrinsic reward", scaled_intrinsic_reward.mean().item())
            self.track_data("RND / Total reward (Max)", rewards.max().item())
            self.track_data("RND / Total reward (Mean)", rewards.mean().item())
            self.track_data("RND / Total reward (Min)", rewards.min().item())
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)
            if self.cfg.time_limit_bootstrap and truncated.any():
                with torch.no_grad():
                    inputs = {
                        "observations": self._observation_preprocessor(next_observations),
                        "states": self._state_preprocessor(next_states),
                    }
                    next_values, _ = self.value.act(inputs, role="value")
                    next_values = self._value_preprocessor(next_values, inverse=True)
                rewards += self.cfg.discount_factor * next_values * truncated
            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                log_prob=self._current_log_prob,
                values=self._current_values,
                rnd_observations=rnd_observations,
            )

    def update(self, *, timestep, timesteps):
        rnd_observations = self.memory.get_tensor_by_name("rnd_observations")
        rnd_observations = rnd_observations.reshape(-1, self.rnd_obs_dim)
        rnd_loss = self.rnd.update(rnd_observations)
        self.track_data("RND / Loss", rnd_loss.item())
        super().update(timestep=timestep, timesteps=timesteps)