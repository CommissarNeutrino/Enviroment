import pygame
import os
import imageio.v2 as imageio

# Глобальные переменные для путей к изображениям
GOAL_IMAGE_PATH = 'img/goal.png'
BACKGROUND_IMAGE_PATH = 'img/background.png'
PATRON_IMAGE_PATH = 'img/patron.png'
ALTRUIST_IMAGE_PATH = 'img/altruist.png'
NO_IMAGE = 'img/no_image.png'
DOOR_IMAGE_PATH = 'img/door.png'
BUTTON_IMAGE_PATH = 'img/button.png'
OBSTACLE_IMAGE_PATH = 'img/obstacle.png'

CELL_SIZE = 100
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
    def __init__(self, grid_width, grid_height, save_frames=False):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.save_frames = save_frames  # Флаг для сохранения кадров
        self.frame_count = 0  # Счётчик кадров
        self.frames_dir = "frames"  # Папка для сохранения кадров

        self.window_size_x = grid_width * CELL_SIZE
        self.window_size_y = grid_height * CELL_SIZE
        screen_size = (self.window_size_x, self.window_size_y)

        # Убедимся, что папка для кадров существует
        if self.save_frames and not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)

        self.cell_size = min(self.window_size_x // self.grid_width, self.window_size_y // self.grid_height)
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size_x, self.window_size_y), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.is_running = True

        self.grid_surface = self.create_grid_surface()
        self.fps = FPS

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
            "Patron": scale_image(pygame.image.load(PATRON_IMAGE_PATH), keep_aspect_ratio=False),
            "Altruist": scale_image(pygame.image.load(ALTRUIST_IMAGE_PATH), keep_aspect_ratio=True),
            "Default": scale_image(pygame.image.load(NO_IMAGE), keep_aspect_ratio=True)
        }
        self.agent_images = self.original_agent_images

        self.original_object_images = {
            "Door": scale_image(pygame.image.load(DOOR_IMAGE_PATH), keep_aspect_ratio=True),
            "Button": scale_image(pygame.image.load(BUTTON_IMAGE_PATH), keep_aspect_ratio=True),
            "Obstacle": scale_image(pygame.image.load(OBSTACLE_IMAGE_PATH), keep_aspect_ratio=False),
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

    def handle_events(self):
        """ Обрабатываем изменение размеров окна """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False  # Останавливаем работу приложения
            elif event.type == pygame.VIDEORESIZE:
                self.window_size_x, self.window_size_y = event.w, event.h
                self.cell_size = min(self.window_size_x // self.grid_width, self.window_size_y // self.grid_height)

                # Обновляем размеры окна и масштабируем изображения
                screen_size = (self.window_size_x, self.window_size_y)
                self.screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)

                # Масштабируем изображения агентов и цели
                self.initiate_scaling()

                # Пересоздаем сетку с новым размером клеток
                self.grid_surface = self.create_grid_surface()

    def render(self, agents, goal_location, immutable_blocks, doors):
        self.handle_events()

        if not self.is_running:
            return

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
        for door, button in doors.items():
            door_rect = pygame.Rect(
                door[0] * self.cell_size,
                door[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            button_rect = pygame.Rect(
                button[0] * self.cell_size,
                button[1] * self.cell_size,
                self.cell_size, self.cell_size
            )
            door_image = self.object_images.get("Door")
            button_image = self.object_images.get("Button")
            self.screen.blit(door_image, door_rect)
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

        # Сохраняем текущий кадр, если включена запись
        if self.save_frames:
            self.save_frame()

        self.clock.tick(self.fps)

    def save_frame(self):
        """ Сохраняет текущий кадр в папку с изображениями """
        frame_filename = os.path.join(self.frames_dir, f"frame_{self.frame_count:04d}.png")
        pygame.image.save(self.screen, frame_filename)
        self.frame_count += 1

    @staticmethod
    def create_video_from_frames(output_path='output_video.mp4', fps=60):
        """ Собирает видео из сохранённых кадров """
        frames_dir = "frames"
        frames = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')])
        with imageio.get_writer(output_path, fps=fps) as writer:
            for frame_path in frames:
                image = imageio.imread(frame_path)
                writer.append_data(image)

    @staticmethod
    def close():
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
