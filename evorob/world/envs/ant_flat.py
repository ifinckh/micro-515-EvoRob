from pathlib import Path

import numpy as np

from gymnasium.spaces import Box
from gymnasium.envs.mujoco import MujocoEnv


class AntFlatEnvironment(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self, render_mode=None, robot_path: str = "ant_flat_terrain.xml", **kwargs
    ):
        # Load MuJoCo environment in Gymnasium
        # Get path to XML file relative to this module
        xml_file_path = str(Path(__file__).parent / "assets" / robot_path)

        MujocoEnv.__init__(
            self,
            model_path=xml_file_path,
            frame_skip=5,
            observation_space=None,  # needs to be defined after
            render_mode=render_mode,
            default_camera_config={
                "distance": 4.0,
            },
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        
        self.terrain_type = "ice" if "ice" in robot_path else "flat" # terrain type for different rewards


        self._reset_noise_scale: float = 0.1

        # Define observation space.
        # Action space is automatically defined by MuJoCo.
        # Exclude x,y position (first 2 elements of qpos) to match _get_obs()
        obs_size = (self.data.qpos.size - 2) + self.data.qvel.size
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

    def reset_model(self):
        noise_low = -0.1
        noise_high = 0.1

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()

        return observation

    def step(self, action):
        torso_body_id = 1
        xy_position_before = self.data.body(torso_body_id).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(torso_body_id).xpos[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        observation = self._get_obs()
        reward, reward_info = self._get_rew(x_velocity, action)
        terminated = self._get_termination()
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, terminated, False, info

    def _get_obs(self):
        # TODO: Return observation as concatenation of:
        # - position EXCLUDING x,y: self.data.qpos[2:].flatten() (13 values)
        # - velocity: self.data.qvel.flatten() (14 values)
        # This gives 27 total dimensions, making the task translation-invariant
        # Hint: Use np.concatenate() to combine both arrays
        pos = self.data.qpos[2:].flatten() # 13 values
        vel = self.data.qvel.flatten() # 14 values
        obs = np.concatenate((pos,vel))
        return obs
    
        raise NotImplementedError("TODO: Implement observation function")

    def _get_rew(self, x_velocity: float, action):
        # TODO: Implement reward function with three components:
        # 1. forward_reward = ...
        # 2. healthy_reward = ...
        # 3. ctrl_cost = ...
        # Final reward is the sum of these three components.
        # 1. forward_reward = ...
        # 2. healthy_reward = ...
        # 3. ctrl_cost = ...
        # Final reward is the sum of these three components.
        # Return: (reward, reward_info_dict)
        
        # reward not terminating
        healthy_reward = not self._get_termination()
        # reward forward velocity
        forward_reward = x_velocity * healthy_reward
        # reward low motor toques (penalize high control inputs)
        ctrl_cost = np.sum(np.square(action))
        
        # might want to try body height as a guassian
        # might want to start with rewarding the robot for any distance (e.g. eucledian) and then reward it going in that same direction (e.g. dot product with forward direction) to encourage it to move in a straight line rather than just spinning in circles, which could be a local minima for forward velocity reward
        # mujoco_gym website for innovative reward
        
        
        
        # recommended weights
        if self.terrain_type == "ice":
            weights = np.array([1, 1, -1.5])   # harsher ctrl penalty on ice
        else:
            weights = np.array([1, 1, -0.5])   # standard flat
            
        reward_array = np.array([forward_reward, healthy_reward, ctrl_cost])
        # print("Reward components: ", reward_array**weights)
        
        reward = np.sum(reward_array * weights)
        
        # have access to the individual losses just in case
        reward_info_dict = {
            "reward_forward" : forward_reward,
            "reward_survive" : healthy_reward, 
            "reward_ctrl" : ctrl_cost}
        
        # print(reward_info_dict)
        
        return (reward, reward_info_dict)
    
        raise NotImplementedError("TODO: Implement reward function")

    def _get_termination(self):
        # TODO: Robot should terminate when:
        # - Torso height is below 0.26 or above 1.0
        # Return True if NOT healthy (i.e., should terminate)
        # Hint: Use self.state_vector() to get current state.
        
        torso_height = self.state_vector()[2]
        
        # termination condition
        if (torso_height < 0.26) or (torso_height > 1.0):
            return True
        else:
            return False
        
        
        raise NotImplementedError("TODO: Implement termination function")
