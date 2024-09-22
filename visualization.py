import pygame

# Глобальные переменные для путей к изображениям
GOAL_IMAGE_PATH = 'img/goal.png'
BACKGROUND_IMAGE_PATH = 'img/background.png'
PATRON_IMAGE_PATH = 'img/patron.png'
ALTRUIST_IMAGE_PATH = 'img/altruist.png'
NO_IMAGE = 'img/no_image.png'
DOOR_IMAGE_PATH = 'img/door.png'
BUTTON_IMAGE_PATH = 'img/button.png'
OBSTACLE_IMAGE_PATH = 'img/obstacle.png'

CELL_SIZE = 50
FPS = 60


def scale_image(image, target_size=(CELL_SIZE, CELL_SIZE), keep_aspect_ratio=True):
    target_width = target_size[0]
    target_height = target_size[1]

    if keep_aspect_ratio:
        # Масштабирует изображение с сохранением пропорций.
        original_width, original_height = image.get_size()
        aspect_ratio = original_width / original_height

        # Вычисляем новый размер с сохранением пропорций
        if aspect_ratio > 1:
            target_height = int(target_width / aspect_ratio)
        else:
            target_width = int(target_height * aspect_ratio)

    return pygame.transform.scale(image, (target_width, target_height))


class GridRenderer:
    """ Класс визуализации, включает управление размерами окна и FPS """
    def __init__(self, grid_width, grid_height):
        self.grid_width = grid_width
        self.grid_height = grid_height

        self.window_size_x = grid_width * CELL_SIZE
        self.window_size_y = grid_height * CELL_SIZE

        self.cell_size = min(self.window_size_x // self.grid_width, self.window_size_y // self.grid_height)

        # Инициализация PyGame и экрана
        screen_size = (self.window_size_x, self.window_size_y)
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        # Загружаем и сохраняем оригинальные изображения
        self.original_background_image = scale_image(
            pygame.image.load(BACKGROUND_IMAGE_PATH),
            screen_size, False
        )
        self.background_image = self.original_background_image  # Изначально равен оригиналу

        self.original_goal_image = scale_image(
            pygame.image.load(GOAL_IMAGE_PATH)
        )
        self.goal_image = self.original_goal_image

        # Сохраняем оригиналы изображений агентов
        self.original_agent_images = {
            "Patron": scale_image(pygame.image.load(PATRON_IMAGE_PATH), keep_aspect_ratio=True),
            "Altruist": scale_image(pygame.image.load(ALTRUIST_IMAGE_PATH), keep_aspect_ratio=True),
            "Default": scale_image(pygame.image.load(NO_IMAGE), keep_aspect_ratio=True)
        }
        self.agent_images = self.original_agent_images

        self.original_object_images = {
            "Door": scale_image(pygame.image.load(DOOR_IMAGE_PATH), keep_aspect_ratio=True),
            "Button": scale_image(pygame.image.load(BUTTON_IMAGE_PATH), keep_aspect_ratio=True),
            "Obstacle": scale_image(pygame.image.load(OBSTACLE_IMAGE_PATH), keep_aspect_ratio=True),
        }

        self.object_images = self.original_object_images

        # Масштабируем изображения под размер клетки

        self.initiate_scaling()

        # Создаем предварительно отрисованную сетку
        self.grid_surface = self.create_grid_surface()
        self.fps = FPS

    def create_grid_surface(self):
        """ Создаем поверхность с сеткой """
        grid_surface = pygame.Surface(
            (self.grid_width * self.cell_size, self.grid_height * self.cell_size),
            pygame.SRCALPHA
        )
        for x in range(0, self.grid_width * self.cell_size, self.cell_size):
            for y in range(0, self.grid_height * self.cell_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(grid_surface, (0, 0, 0, 50), rect, 1)  # Прозрачная черная сетка
        return grid_surface

    def initiate_scaling(self):
        """ Масштабируем изображения агентов и цели в зависимости от текущего размера клетки """

        self.background_image = scale_image(
            self.original_background_image,
            (self.window_size_x, self.window_size_y),
            True
        )

        # Масштабируем изображение цели
        self.goal_image = scale_image(self.original_goal_image, (self.cell_size, self.cell_size))

        # Масштабируем изображения агентов
        self.agent_images = {
            agent_type: scale_image(original_image, (self.cell_size, self.cell_size))
            for agent_type, original_image in self.original_agent_images.items()
        }

        self.object_images = {
            object_type: scale_image(original_image, (self.cell_size, self.cell_size))
            for object_type, original_image in self.original_object_images.items()
        }

    def handle_resize_event(self):
        """ Обрабатываем изменение размеров окна """
        for event in pygame.event.get():
            if event.type == pygame.VIDEORESIZE:
                self.window_size_x, self.window_size_y = event.w, event.h
                self.cell_size = min(self.window_size_x // self.grid_width, self.window_size_y // self.grid_height)

                # Обновляем размеры окна и масштабируем изображения
                screen_size = (self.window_size_x, self.window_size_y)
                self.screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)

                # Масштабируем изображения агентов и цели
                self.initiate_scaling()

                # Пересоздаем сетку с новым размером клеток
                self.grid_surface = self.create_grid_surface()

    def render(self, agents, goal_location, immutable_blocks, doors, buttons):
        """ Отрисовка сетки, агентов, цели и объектов на экране """
        self.handle_resize_event()

        # Отрисовка фонового изображения
        self.screen.blit(self.background_image, (0, 0))

        # Используем предварительно созданную сетку
        self.screen.blit(self.grid_surface, (0, 0))

        # Отрисовка цели
        goal_rect = pygame.Rect(
            goal_location[0] * self.cell_size,
            goal_location[1] * self.cell_size,
            self.cell_size, self.cell_size
        )
        self.screen.blit(self.goal_image, goal_rect)

        # Отрисовка препятствий (immutable_blocks)
        for block in immutable_blocks:
            block_rect = pygame.Rect(
                block[0] * self.cell_size,
                block[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            obstacle_image = self.object_images.get("Obstacle")
            self.screen.blit(obstacle_image, block_rect)

        # Отрисовка дверей
        for door in doors:
            door_rect = pygame.Rect(
                door[0] * self.cell_size,
                door[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            door_image = self.object_images.get("Door")
            self.screen.blit(door_image, door_rect)

        # Отрисовка кнопок
        for button in buttons:
            button_rect = pygame.Rect(
                button[0] * self.cell_size,
                button[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            button_image = self.object_images.get("Button")
            self.screen.blit(button_image, button_rect)

        # Рендеринг агентов
        for agent in agents:
            agent_rect = pygame.Rect(
                agent.location[0] * self.cell_size,
                agent.location[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            agent.type = agent.__class__.__name__
            agent_image = self.agent_images.get(agent.type, self.agent_images["Default"])
            self.screen.blit(agent_image, agent_rect)

        # Обновляем экран
        pygame.display.flip()

        # Задержка для поддержания заданного FPS
        self.clock.tick(self.fps)

    # def render(self, agents, goal_location):
    #     """ Отрисовка сетки, агентов и цели на экране """
    #     self.handle_resize_event()
    #
    #     # Отрисовка фонового изображения
    #     self.screen.blit(self.background_image, (0, 0))
    #
    #     # Используем предварительно созданную сетку
    #     self.screen.blit(self.grid_surface, (0, 0))
    #
    #     # Отрисовка цели
    #     goal_rect = pygame.Rect(
    #         goal_location[0] * self.cell_size,
    #         goal_location[1] * self.cell_size,
    #         self.cell_size, self.cell_size
    #     )
    #     self.screen.blit(self.goal_image, goal_rect)
    #
    #     # Рендеринг агентов
    #     for agent in agents:
    #         agent_rect = pygame.Rect(
    #             agent.location[0] * self.cell_size,
    #             agent.location[1] * self.cell_size,
    #             self.cell_size, self.cell_size
    #         )
    #         agent.type = agent.__class__.__name__
    #         agent_image = self.agent_images.get(agent.type, self.agent_images["Default"])
    #         self.screen.blit(agent_image, agent_rect)
    #
    #     # Обновляем экран
    #     pygame.display.flip()
    #
    #     # Задержка для поддержания заданного FPS
    #     self.clock.tick(self.fps)

    @staticmethod
    def close():
        """ Закрываем окно и выходим из PyGame """
        pygame.quit()
