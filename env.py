import gymnasium as gym
from gymnasium import spaces
import pygame
from pettingzoo import ParallelEnv
import functools
import random
import numpy as np


"""
TODO:

Custom-form env (tunnel 4 blocks wide, 2 blocks with doors in center each 8 column, 100 blocks in length)
Doors mechanics
Fix reset func

"""


class ShSV_WorldEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"], 
        "render_fps": 4,
        "name": "shsv_env_v0",
        }

    def __init__(self, render_mode=None, size_x=20, size_y=4):
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

    def _get_obs(self):
        return {"agent_patron": self._agent_patron_location, "agent_altruist": self._agent_altruist_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_patron_location - self._target_location, ord=1
            )
        }

    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Определим стартовую площадку для агентов (размер 5x4)
        agent_start_area_x = 5
        agent_start_area_y = self.size_y

        # Выбираем случайное положение для первого агента
        self._agent_patron_location = self.np_random.integers(
            low=[0, 0], high=[agent_start_area_x, agent_start_area_y], size=2, dtype=int
        )

        # Для второго агента необходимо убедиться, что он не на той же клетке
        # Вот здесь может быть уход в долгий цикл, пофиксить.
        while True:
            self._agent_altruist_location = self.np_random.integers(
                low=[0, 0], high=[agent_start_area_x, agent_start_area_y], size=2, dtype=int
            )
            if not np.array_equal(self._agent_altruist_location, self._agent_patron_location):
                break

        # Определим область для появления цели (x между 7 и 12 включительно)
        target_area_x = (7, 13)  # 13, потому что верхняя граница в randint не включается
        target_area_y = self.size_y

        self._target_location = self.np_random.integers(
            low=[target_area_x[0], 0], high=[target_area_x[1], target_area_y], size=2, dtype=int
        )

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info


    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction_altruist = self._action_to_direction[action["patron"]]
        direction_patron = self._action_to_direction[action["altruist"]]
        # ALTRIUST
        # Проверяем, не является ли новая позиция неизменяемым блоком
        new_position_altruist = self._agent_altruist_location + direction_altruist
        if self._is_immutable_block(new_position_altruist):
            new_position_altruist = self._agent_altruist_location  # Оставляем позицию без изменений, если блок неизменяемый
        else:
            # We use `np.clip` to make sure we don't leave the grid
            new_position_altruist = np.clip(
                self._agent_altruist_location + direction_altruist, [0, 0], [self.size_x - 1, self.size_y - 1]
            )
        # PATRON
        new_position_patron = self._agent_patron_location + direction_patron
        if self._is_immutable_block(new_position_patron):
            new_position_patron = self._agent_patron_location  # Оставляем позицию без изменений, если блок неизменяемый
        elif self._is_door(new_position_patron):
            door_unlock_position = new_position_altruist - [1, 1]
            if not np.array_equal(new_position_altruist, door_unlock_position):
                new_position_patron = self._agent_patron_location
        else:
            # We use `np.clip` to make sure we don't leave the grid
            new_position_patron = np.clip(
                self._agent_patron_location + direction_patron, [0, 0], [self.size_x - 1, self.size_y - 1]
            )
        self._agent_patron_location = new_position_patron
        self._agent_altruist_location = new_position_altruist
        # An episode is done if the agent has reached the target
        terminated = np.array_equal(self._agent_patron_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def _is_immutable_block(self, position):
        # Проверка, является ли позиция неизменяемым блоком
        return tuple(position) in self._immutable_blocks

    def _is_door(self, position):
        # Проверка, является ли позиция дверью
        return tuple(position) in self._doors

    def create_obstacles(self):
        """
        Автоматически создает препятствия и двери, начиная с 6-го столбца и далее через каждые 6 столбцов.
        Порядок: дверь, препятствие, дверь, препятствие.
        """
        obstacles = []
        doors = []
        for x in range(6, self.size_x, 6):
            doors.append((x, 3))  # дверь
            obstacles.append((x, 2))  # препятствие
            doors.append((x, 1))  # дверь
            obstacles.append((x, 0))  # препятствие

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

        # Now we draw the agent-patron
        pygame.draw.circle(
            canvas,
            (0, 0, 255),  # Цвет агента-патрона - синий
            (
                (self._agent_patron_location[0] + 0.5) * pix_square_size_x,
                (self._agent_patron_location[1] + 0.5) * pix_square_size_y,
            ),
            min(pix_square_size_x, pix_square_size_y) / 3,  # Радиус зависит от наименьшего размера клетки
        )

        # Now we draw the agent-altruist
        pygame.draw.circle(
            canvas,
            (0, 255, 0),  # Цвет агента-альтруиста - зеленый
            (
                (self._agent_altruist_location[0] + 0.5) * pix_square_size_x,
                (self._agent_altruist_location[1] + 0.5) * pix_square_size_y,
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
        return spaces.Discrete(4)