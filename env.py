import gymnasium as gym
from gymnasium import spaces
import pygame
# from pettingzoo import ParallelEnv
import functools
import random
import numpy as np


"""
TODO:

Custom-form env (tunnel 4 blocks wide, 2 blocks with doors in center each 8 column, 100 blocks in length)
Doors mechanics
Fix reset func

"""


class WorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 4,
        "name": "v0",
        }
    agents = {}

    def __init__(self, render_mode=None, size_x=20, size_y=4, key_position = np.array([1, 1])):
        self.size_x = size_x  # The size of the square grid
        self.size_y = size_y  # The size of the square grid
        self.window_size_x = 1024  # The size of the PyGame window
        self.window_size_y = 512  # The size of the PyGame window
        # Observations are dictionaries with the agents's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.create_obstacles()
        # Проверка того, что в render_mode не лежит ничего странного.
        assert render_mode is None or render_mode in self.metadata.get("render_modes", [])
        # Возможные действия для агентов
        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([0, 0]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, -1]),
        }
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.key_position = key_position

    def _get_obs(self):
        state = {"target": self._target_location}
        for agent_id, agent_instance in self.agents.items():
            state[agent_id] = agent_instance.location
        return state

    def _get_info(self):
        return {
            # "distance": np.linalg.norm(
            #     self._agent_patron_location - self._target_location, ord=1
            # )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        # Определим стартовую площадку для агентов (размер 5x4)
        agent_start_area_x = 5
        agent_start_area_y = self.size_y
        available_locations = {(x, y) for x in range(agent_start_area_x) for y in range(agent_start_area_y)}
        for agent_id, agent_instance in self.agents.items():
            agent_instance.location = random.choice(list(available_locations))
            available_locations.remove(agent_instance.location)
        # Определим область для появления цели (x между 7 и 12 включительно)
        target_area_x = (7, 13)  # 13, потому что верхняя граница в randint не включается
        target_area_y = self.size_y
        self._target_location = tuple(self.np_random.integers(
            low=[target_area_x[0], 0], high=[target_area_x[1], target_area_y], size=2, dtype=int
        ))
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, info


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        terminated= False
        new_positions = set()
        for agent_id, agent_instance in self.agents.items():
            direction = self._action_to_direction[action[agent_id]]
            agent_instance.location = self.decision_process(agent_instance, direction, new_positions)
            if not terminated and agent_id.startswith("patron"):
                if np.array_equal(agent_instance.location, self._target_location):
                    terminated = True
                    break
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def decision_process(self, agent_instance, direction, new_positions):
        new_position = self.decision_grid_edges(agent_instance, direction)
        if self.decision_immutable_blocks(new_position) and self.decision_doors(agent_instance, new_position) and self.decision_other_agents(new_position, new_positions):
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
    
    def decision_other_agents(self, new_position, new_positions):
        if tuple(new_position) in new_positions:
            return False
        return True

    def decision_key(self, agent_instance, new_position):
        if np.array_equal(new_position, self.key_position):
            agent_instance.key = True
            # print(f"{agent_instance.agent_type} got the key")
            self.key_position = np.array([-1, -1])

    def _is_immutable_block(self, position):
        # Проверка, является ли позиция неизменяемым блоком
        return tuple(position) in self._immutable_blocks

    def _is_door(self, position):
        # Проверка, является ли позиция дверью
        # return tuple(position) in self._doors
        return self._doors.get(tuple(position), False)

    def create_obstacles(self):
        """
        Автоматически создает препятствия и двери, начиная с 6-го столбца и далее через каждые 6 столбцов.
        Порядок: дверь, препятствие, дверь, препятствие.
        """
        obstacles = set()
        doors = {}
        for x in range(6, self.size_x, 6):
            doors[x, 3] = True  # дверь
            obstacles.add((x, 2))  # препятствие
            doors[x, 1] = True  # дверь
            obstacles.add((x, 0))  # препятствие

        self._immutable_blocks = obstacles
        self._doors = doors

    def render(self):
        # print(f"Current State: {self._get_obs()}")

        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size_x, self.window_size_y))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size_x, self.window_size_y))
        canvas.fill((255, 255, 255))

        # Размеры клеток по X и Y в пикселях
        pix_square_size_x = self.window_size_x / self.size_x
        pix_square_size_y = self.window_size_y / self.size_y

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # Цвет цели
            pygame.Rect(
                (self._target_location[0] * pix_square_size_x, self._target_location[1] * pix_square_size_y),
                (pix_square_size_x, pix_square_size_y),
            ),
        )
        # Отрисовка стен
        for wall in self._immutable_blocks:
            pygame.draw.rect(
                canvas,
                (128, 128, 128),  # Цвет стен (серый)
                pygame.Rect(
                    (wall[0] * pix_square_size_x, wall[1] * pix_square_size_y),  # Позиция блока
                    (pix_square_size_x, pix_square_size_y),  # Размер блока
                ),
            )

        # Отрисовка дверей
        for door in self._doors:
            pygame.draw.rect(
                canvas,
                (0, 0, 255),  # Цвет дверей (синий)
                pygame.Rect(
                    (door[0] * pix_square_size_x, door[1] * pix_square_size_y),  # Позиция двери
                    (pix_square_size_x, pix_square_size_y),  # Размер двери
                ),
            )
            pygame.draw.rect(
                canvas,
                (0, 255, 0),  # Цвет открывающих блоков (зеленый)
                pygame.Rect(
                    ((door[0] - 1) * pix_square_size_x, (door[1] - 1) * pix_square_size_y),  # Позиция открывающего блока
                    (pix_square_size_x, pix_square_size_y),  # Размер блока
                ),
            )

        pygame.draw.rect(
            canvas,
            (255, 0, 0),  # Цвет цели
            pygame.Rect(
                (self._target_location[0] * pix_square_size_x, self._target_location[1] * pix_square_size_y),
                (pix_square_size_x, pix_square_size_y),
            ),
        )

        for agent_id, agent_instance in self.agents.items():
            if agent_id.startswith("patron"):
                color = (0, 0, 255)
            elif agent_id.startswith("altruist"):
                color = (0, 255, 0)
            pygame.draw.circle(
                canvas,
                color,  # Цвет агента-патрона - синий
                (
                    (agent_instance.location[0] + 0.5) * pix_square_size_x,
                    (agent_instance.location[1] + 0.5) * pix_square_size_y,
                ),
                min(pix_square_size_x, pix_square_size_y) / 3,  # Радиус зависит от наименьшего размера клетки
            )

        # Now we draw the agent-altruist
        pygame.draw.circle(
            canvas,
            color,  # Цвет агента-альтруиста - зеленый
            (
                (agent_instance.location[0] + 0.5) * pix_square_size_x,
                (agent_instance.location[1] + 0.5) * pix_square_size_y,
            ),
            min(pix_square_size_x, pix_square_size_y) / 3,  # Радиус зависит от наименьшего размера клетки
        )

        # Finally, add some gridlines
        for x in range(self.size_x + 1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size_x * x, 0),
                (pix_square_size_x * x, self.window_size_y),
                width=3,
            )
        for y in range(self.size_y + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size_y * y),
                (self.window_size_x, pix_square_size_y * y),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    # Пространство наблюдений для каждого агента
    @functools.lru_cache(maxsize=None)
    def observation_space(self):
        return spaces.MultiDiscrete([self.size_x, self.size_y])

    # Пространство действий для каждого агента
    @functools.lru_cache(maxsize=None)
    def action_space(self):
        return spaces.Discrete(5)