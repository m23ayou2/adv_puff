import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.drone import binding

class Drone(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=16,
        num_drones=64,
        max_rings=5,
        render_mode=None,
        report_interval=1024,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(26,),
            dtype=np.float32,
        )

        self.single_action_space = gymnasium.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )

        self.num_agents = num_envs*num_drones
        self.render_mode = render_mode
        self.report_interval = report_interval
        self.tick = 0
        self._num_envs = num_envs
        self._num_drones = num_drones
        self._predicted_to_target = None

        super().__init__(buf)
        self.actions = self.actions.astype(np.float32)

        c_envs = []
        for i in range(num_envs):
            c_envs.append(binding.env_init(
                self.observations[i*num_drones:(i+1)*num_drones],
                self.actions[i*num_drones:(i+1)*num_drones],
                self.rewards[i*num_drones:(i+1)*num_drones],
                self.terminals[i*num_drones:(i+1)*num_drones],
                self.truncations[i*num_drones:(i+1)*num_drones],
                i,
                num_agents=num_drones,
                max_rings=max_rings,
            ))

        self._c_env_handles = c_envs
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:
                info.append(log_data)

        return (self.observations, self.rewards, self.terminals, self.truncations, info)

    def render(self, predicted_to_target=None):
        if predicted_to_target is not None:
            self.set_predicted_to_target(predicted_to_target)
        binding.vec_render(self.c_envs, 0)

    def set_predicted_to_target(self, predicted_to_target):
        if predicted_to_target is None:
            for env_handle in self._c_env_handles:
                binding.env_put(env_handle, predicted_to_target=None)
            return

        predicted_to_target = np.asarray(predicted_to_target, dtype=np.float32)

        if predicted_to_target.shape == (self.num_agents, 3):
            predicted_to_target = predicted_to_target.reshape(
                self._num_envs, self._num_drones, 3
            )
        elif predicted_to_target.shape == (self._num_envs, self._num_drones, 3):
            pass
        elif (
            predicted_to_target.ndim == 2
            and predicted_to_target.shape[1] == 3
            and predicted_to_target.shape[0] % self.num_agents == 0
        ):
            predicted_to_target = predicted_to_target[: self.num_agents].reshape(
                self._num_envs, self._num_drones, 3
            )
        elif (
            predicted_to_target.ndim == 4
            and predicted_to_target.shape[1:] == (self._num_envs, self._num_drones, 3)
        ):
            predicted_to_target = predicted_to_target[0]
        else:
            raise ValueError(
                "predicted_to_target must have shape (num_agents, 3), "
                "(num_envs, num_drones, 3), or be a multiple of num_agents"
            )

        # Keep a reference to prevent garbage collection
        self._predicted_to_target_buffer = predicted_to_target

        for i, env_handle in enumerate(self._c_env_handles):
            binding.env_put(
                env_handle,
                predicted_to_target=self._predicted_to_target_buffer[i],
            )

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = Drone(num_envs=1000)
    env.reset()
    tick = 0

    actions = [env.action_space.sample() for _ in range(atn_cache)]

    import time
    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    print(f"SPS: {env.num_agents * tick / (time.time() - start)}")

if __name__ == "__main__":
    test_performance()
