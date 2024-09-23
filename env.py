import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from visualization import GridRenderer  # Импорт классов для визуализации
import functools


def _get_info():
    # Возвращаем дополнительную информацию, если необходимо
    return {}


class WorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 4,
        "name": "v0",
    }

    def __init__(self, size_x, size_y, target_location, walls_positions, doors_positions, render_mode=None):
        self.render_mode = render_mode
        self.agents = {}
        self.size_x = size_x
        self.size_y = size_y
        self.target_location = target_location
        self.walls_positions = walls_positions
        self.doors_positions = doors_positions
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }

    def _get_obs(self):
        state = {"target": self.target_location}
        for agent_id, agent_instance in self.agents.items():
            state[agent_id] = agent_instance.location
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        for agent_instance in self.agents.values():
            agent_instance.location = random.choice(agent_instance.start_zone)
        observation = self._get_obs()
        info = _get_info()
        return observation, info

    def step(self, action):
        terminated = False
        new_positions = set()
        for agent_id, agent_instance in self.agents.items():
            direction = self._action_to_direction[action[agent_id]]
            agent_instance.location = self.decision_process(agent_instance, direction, new_positions)
            if not terminated and agent_id.startswith("patron"):
                if np.array_equal(agent_instance.location, self.target_location):
                    terminated = True
                    break
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = _get_info()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human":
            # Отрисовываем только при вызове этого метода
            self.renderer.render(self.agents.values(), self.target_location, self.walls_positions, self.doors_positions)

    def close(self):
        if self.render_mode == "human":
            self.renderer.close()

    def decision_process(self, agent_instance, direction, new_positions):
        new_position = self.decision_grid_edges(agent_instance, direction)
        if (self.decision_walls_positions(new_position)
                and self.decision_doors_positions(agent_instance, new_position)
                and self.decision_other_agents(new_position, new_positions)):
            new_positions.add(tuple(new_position))
            return tuple(new_position)
        return agent_instance.location

    def decision_grid_edges(self, agent_instance, direction):
        new_position = np.clip(
            agent_instance.location + direction, [0, 0], [self.size_x - 1, self.size_y - 1]
        )
        return new_position

    def decision_walls_positions(self, new_position):
        if self._is_immutable_block(new_position):
            return False
        return True

    def decision_doors_positions(self, agent_instance, new_position):
        if not self._is_door(new_position):
            return True
        return False

    @staticmethod
    def decision_other_agents(new_position, new_positions):
        if tuple(new_position) in new_positions:
            return False
        return True

    def _is_immutable_block(self, position):
        return tuple(position) in self.walls_positions

    def _is_door(self, position):
        return self.doors_positions.get(tuple(position), False)
    
    @property
    def render_mode(self):
        return self._render_mode

    @render_mode.setter
    def render_mode(self, render_mode):
        if render_mode is not None and render_mode not in self.metadata.get("render_modes", []):
            raise ValueError(f"Invalid render_mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}")

        self._render_mode = render_mode

        if render_mode == "human":
            self.renderer = GridRenderer(
                grid_width=self.size_x,
                grid_height=self.size_y
            )


    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return spaces.MultiDiscrete([self.size_x, self.size_y])

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return spaces.Discrete(5)
