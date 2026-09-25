import dataclasses
from skrl.utils.runner.torch import Runner
from .rnd_ppo import RND_PPO

class RNDRunner(Runner):
    def _component(self, name):
        if name.lower() == "rnd_ppo":
            return RND_PPO
        return super()._component(name)

    def _generate_agent(self, env, cfg, models):
        multi_agent = False
        device = env.device
        num_envs = env.num_envs
        possible_agents = ["agent"]
        observation_spaces = {"agent": env.observation_space}
        state_spaces = {"agent": env.state_space}
        action_spaces = {"agent": env.action_space}
        agent_class = cfg["agent"]["class"].lower()
        memory_class = self._component(cfg["memory"]["class"])
        if cfg["memory"]["memory_size"] < 0:
            cfg["memory"]["memory_size"] = cfg["agent"]["rollouts"]
        memories = {
            agent_id: memory_class(num_envs=num_envs, device=device, **self._process_cfg(cfg["memory"]))
            for agent_id in possible_agents
        }
        if agent_class == "rnd_ppo":
            agent_id = possible_agents[0]
            agent_cfg = dataclasses.asdict(
                self._component("ppo_cfg")(**self._process_cfg(cfg["agent"]))
            )
            agent_cfg.get("observation_preprocessor_kwargs", {}).update(
                {"size": observation_spaces[agent_id], "device": device}
            )
            agent_cfg.get("state_preprocessor_kwargs", {}).update(
                {"size": state_spaces[agent_id], "device": device}
            )
            agent_cfg.get("value_preprocessor_kwargs", {}).update(
                {"size": 1, "device": device}
            )
            agent_cfg.get("exploration_noise_kwargs", {}).update(
                {"device": device}
            )
            agent_cfg.get("smooth_regularization_noise_kwargs", {}).update(
                {"device": device}
            )
            agent_kwargs = {
                "models": models[agent_id],
                "memory": memories[agent_id],
                "observation_space": observation_spaces[agent_id],
                "state_space": state_spaces[agent_id],
                "action_space": action_spaces[agent_id],
            }
            return self._component("rnd_ppo")(
                cfg=agent_cfg,
                device=device,
                **agent_kwargs,
            )
        return super()._generate_agent(env, cfg, models)