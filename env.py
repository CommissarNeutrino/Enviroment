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

    def __init__(self, render_mode=None, size_x=20, size_y=4, key_position=np.array([1, 1])):
        self.size_x = size_x  # Размер сетки по X
        self.size_y = size_y  # Размер сетки по Y

        self.render_mode = render_mode
        self.agents = {}

        # Инициализируем визуализатор только если render_mode задан
        if self.render_mode is not None:
            self.renderer = GridRenderer(
                grid_width=self.size_x,
                grid_height=self.size_y
            )

        self.create_obstacles()

        if render_mode is not None and render_mode not in self.metadata.get("render_modes", []):
            raise ValueError(f"Invalid render_mode '{render_mode}'. Supported modes: {self.metadata['render_modes']}")
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }
        self.key_position = key_position

    def _get_obs(self):
        state = {"target": self._target_location}
        for agent_id, agent_instance in self.agents.items():
            state[agent_id] = agent_instance.location
        return state

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        agent_start_area_x = 5
        agent_start_area_y = self.size_y
        available_locations = {(x, y) for x in range(agent_start_area_x) for y in range(agent_start_area_y)}
        for agent_id, agent_instance in self.agents.items():
            agent_instance.location = random.choice(list(available_locations))
            available_locations.remove(agent_instance.location)
        target_area_x = (7, 13)
        target_area_y = self.size_y
        self._target_location = tuple(self.np_random.integers(
            low=[target_area_x[0], 0], high=[target_area_x[1], target_area_y], size=2, dtype=int
        ))
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
                if np.array_equal(agent_instance.location, self._target_location):
                    terminated = True
                    break
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = _get_info()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "human" or self.render_mode == "rgb_array":
            # Отрисовываем только при вызове этого метода
            # print(self.agents)
            self.renderer.render(self.agents.values(), self._target_location, self._immutable_blocks, self._doors, self._buttons)

    def close(self):
        if self.render_mode is not None:
            self.renderer.close()

    # Остальные методы остаются без изменений
    def decision_process(self, agent_instance, direction, new_positions):
        new_position = self.decision_grid_edges(agent_instance, direction)
        if (self.decision_immutable_blocks(new_position)
                and self.decision_doors(agent_instance, new_position)
                and self.decision_other_agents(new_position, new_positions)):
            self.decision_key(agent_instance, new_position)
            new_positions.add(tuple(new_position))
            return tuple(new_position)
        return agent_instance.location

    def decision_grid_edges(self, agent_instance, direction):
        new_position = np.clip(
            agent_instance.location + direction, [0, 0], [self.size_x - 1, self.size_y - 1]
        )
        return new_position

    def decision_immutable_blocks(self, new_position):
        if self._is_immutable_block(new_position):
            return False
        return True

    def decision_doors(self, agent_instance, new_position):
        if not self._is_door(new_position) or agent_instance.key:
            return True
        return False

    @staticmethod
    def decision_other_agents(new_position, new_positions):
        if tuple(new_position) in new_positions:
            return False
        return True

    def decision_key(self, agent_instance, new_position):
        if np.array_equal(new_position, self.key_position):
            agent_instance.key = True
            self.key_position = np.array([-1, -1])

    def _is_immutable_block(self, position):
        return tuple(position) in self._immutable_blocks

    def _is_door(self, position):
        return self._doors.get(tuple(position), False)

    def create_obstacles(self):
        obstacles = set()
        doors = {}
        for x in range(6, self.size_x, 6):
            doors[x, 3] = True
            obstacles.add((x, 2))
            doors[x, 1] = True
            obstacles.add((x, 0))
        self._immutable_blocks = obstacles
        self._doors = doors
        self._buttons = {(0,2), (3,1)}

    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return spaces.MultiDiscrete([self.size_x, self.size_y])

    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return spaces.Discrete(5)