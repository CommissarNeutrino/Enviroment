import pygame
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image

# Глобальные переменные для путей к изображениям
GOAL_IMAGE_PATH = 'img/goal.png'
BACKGROUND_IMAGE_PATH = 'img/background.png'
PATRON_IMAGE_PATH = 'img/patron.png'
ALTRUIST_IMAGE_PATH = 'img/altruist.png'
NO_IMAGE = 'img/no_image.png'
DOOR_OPENED_IMAGE_PATH = 'img/door_opened.png'
DOOR_CLOSED_IMAGE_PATH = 'img/door_closed.png'
BUTTON_IMAGE_PATH = 'img/button.png'
OBSTACLE_IMAGE_PATH = 'img/obstacle.png'
FRAMES_DIR = 'video/frames'
SAVE_FRAMES = False
CREATE_VIDEO = False
CELL_SIZE = 112
FPS = 60
DELAY = 200

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

def create_video_from_frames(size_x, size_y, video_dir, fps=int(FPS / 10)):
    """ Собирает видео из сохранённых кадров, добавляя эпизодные заставки по названиям файлов """
    output_dir = video_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    
    v_dir = os.path.join(output_dir, "output_video.mp4")
    frames = sorted([os.path.join(output_dir, f) for f in os.listdir(output_dir) if f.endswith('.png')])
    with imageio.get_writer(v_dir, fps=fps) as writer:
        current_episode = None

        for frame_path in frames:
            # Извлечение номера эпизода из первых двух символов имени файла
            frame_name = os.path.basename(frame_path)
            episode_number = frame_name[:2]  # Первые два символа — номер эпизода

            # Если номер эпизода изменился, вставляем титульный кадр
            if episode_number != current_episode:
                current_episode = episode_number
                episode_frame = create_episode_frame(int(current_episode), size_x, size_y)

                # Добавляем титульный кадр на 1 секунду
                for _ in range(fps):
                    writer.append_data(episode_frame)

            # Добавляем текущий кадр
            image = imageio.imread(frame_path)
            
            # Ensure the image has 3 channels (convert RGBA to RGB if necessary)
            if image.shape[-1] == 4:  # If the image has an alpha channel (RGBA)
                image = image[:, :, :3]  # Drop the alpha channel, convert to RGB
            # Resize image to match video size if necessary
            if image.shape[:2] != (size_y, size_x):  # Check if image dimensions match the required video size
                image = np.array(Image.fromarray(image).resize((size_x, size_y)))  # Resize using PIL
            
            writer.append_data(image)
    print("video successfully created")
    return




def create_episode_frame(episode_number, size_x, size_y):
    """ Создает изображение с текстом 'Episode #X' """
    font = pygame.font.Font(None, 74)
    text_surface = font.render(f'Round #{episode_number}', True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(size_x/2, size_y/2))

    # Создаем изображение с черным фоном и текстом
    episode_image = pygame.Surface((size_x, size_y))
    episode_image.fill((0, 0, 0))
    episode_image.blit(text_surface, text_rect)

    # Преобразуем изображение в массив данных
    return pygame.surfarray.array3d(episode_image).swapaxes(0, 1)  # swapaxes нужно для корректного порядка



class GridRenderer:
    def __init__(self, grid_width, grid_height, save_frames=True, save_video=True, scenary_type="1a"):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.save_frames = save_frames  # Флаг для сохранения кадров
        self.save_video = save_video
        self.frame_count = 0  # Счётчик кадров
        self.scenario = scenary_type
        self.create_frames_dir()
        self.delay = DELAY
        self.window_size_x = grid_width * CELL_SIZE
        self.window_size_y = grid_height * CELL_SIZE
        self.colors_per_door = {
            (1, 2): (89, 114, 154),
            (4, 2): (239, 239, 239),
        }
        self.footer_size = 48
        self.pushed_buttons = set()
        self.screen_size_with_footer = (self.window_size_x, self.window_size_y + self.footer_size)

        # Убедимся, что папка для кадров существует
        if self.save_frames and not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        self.cell_size = min(self.window_size_x // self.grid_width, self.window_size_y // self.grid_height)
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size_x, self.window_size_y + self.footer_size), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.is_running = True

        self.grid_surface = self.create_grid_surface()
        self.fps = FPS

        # Загружаем и сохраняем оригинальные изображения
        self.original_background_image = scale_image(
            pygame.image.load(BACKGROUND_IMAGE_PATH),
            self.screen_size_with_footer, False
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
            "Door_Opened": scale_image(pygame.image.load(DOOR_OPENED_IMAGE_PATH), keep_aspect_ratio=True),
            "Door_Closed": scale_image(pygame.image.load(DOOR_CLOSED_IMAGE_PATH), keep_aspect_ratio=True),
            "Button": scale_image(pygame.image.load(BUTTON_IMAGE_PATH), keep_aspect_ratio=True),
            "Obstacle": scale_image(pygame.image.load(OBSTACLE_IMAGE_PATH), keep_aspect_ratio=False),
        }

        self.object_images = self.original_object_images

        # Масштабируем изображения под размер клетки

        self.initiate_scaling()

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

    def render(self, agents, goal_location, immutable_blocks, doors, step_number, episod_number):
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
            color = self.colors_per_door[door]
            # Получаем изображения двери и кнопки
            if button in self.pushed_buttons:
                door_image = self.object_images.get("Door_Opened")
            else:
                door_image = self.object_images.get("Door_Closed")
            button_image = self.object_images.get("Button")
            # Отображаем их на экране
            self.screen.blit(door_image, door_rect)
            self.screen.blit(button_image, button_rect)
            # Рисуем сетку зеленым цветом вокруг двери и кнопки
            # x_start_door = door_rect.left  # левая граница клетки
            # y_start_door = door_rect.bottom  # нижняя граница клетки
            # x_end_door = door_rect.right  # правая граница клетки
            # y_end_door = door_rect.bottom  # линия идет вдоль нижней границы

            # # Рисуем линию
            # pygame.draw.line(self.screen, color, (x_start_door, y_start_door), (x_end_door, y_end_door), 4)

            # x_start_button = button_rect.left  # левая граница клетки
            # y_start_button = button_rect.bottom  # нижняя граница клетки
            # x_end_button = button_rect.right  # правая граница клетки
            # y_end_button = button_rect.bottom  # линия идет вдоль нижней границы

            # # Рисуем линию
            # pygame.draw.line(self.screen, color, (x_start_button, y_start_button), (x_end_button, y_end_button), 4)
            pygame.draw.rect(self.screen, color, door_rect, 4)
            pygame.draw.rect(self.screen, color, button_rect, 4)

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
        # Отображение номера шага
        self.draw_info(step_number, episod_number)

        # Обновляем экран
        pygame.display.flip()
        pygame.time.wait(self.delay)

        # Сохраняем текущий кадр, если включена запись
        if self.save_frames:
            self.save_frame(episod_number, step_number)

        self.clock.tick(self.fps)

    def create_frames_dir(self):
        cache_dir = os.path.join("cache", self.scenario)
        try_dir_base = "progon_"
        existing_folders = [f for f in os.listdir(cache_dir) if
                            f.startswith(try_dir_base) and os.path.isdir(os.path.join(cache_dir, f))]
        if existing_folders:
            max_i = max([int(f.split('_')[1]) for f in existing_folders])
            progon_number = max_i
        else:
            raise ValueError("Нет сохранённых прогонов для загрузки.")
        progon_folder = os.path.join(cache_dir, f"{try_dir_base}{progon_number}")
        self.frames_dir = os.path.join(progon_folder, "video")
        print(self.frames_dir)
    
    def save_frame(self, episode_number, step_number):
        """ Сохраняет текущий кадр в папку с изображениями """
        frame_episode_dir = os.path.join(self.frames_dir, f"{episode_number + 1:02d}{step_number:03d}.png")
        pygame.image.save(self.screen, frame_episode_dir)
        self.frame_count += 1

    def draw_info(self, step_number, episode_number):
        # """ Отображает номер шага на экране """
        # # Отображаем текст на экране
        # self.screen.blit(text_surface, text_rect)
        """ Отображает номер шага и эпизода на экране с переводом строки """
        font = pygame.font.Font(None, 36)  # Шрифт
        color = (0, 0, 0)  # Белый цвет текста

        # Создаем строки для шага и эпизода
        episode_text = f'Round: {episode_number+1}'
        step_text = f'Step: {step_number}'

        white_background = (255, 255, 255)  # Цвет фона (белый)
        step_surface = font.render(step_text, True, color, white_background)  # Белый фон
        episode_surface = font.render(episode_text, True, color, white_background)  # Белый фон

        step_rect = step_surface.get_rect(midbottom=(self.screen_size_with_footer[0] - CELL_SIZE * 1, self.screen_size_with_footer[1] - self.footer_size/4))  # Справа
        episode_rect = episode_surface.get_rect(midbottom=(0 + CELL_SIZE * 1, self.screen_size_with_footer[1] - self.footer_size/4))  # Слева

        # Отображаем текст на экране
        self.screen.blit(step_surface, step_rect)
        self.screen.blit(episode_surface, episode_rect)

    def close(self):
        if self.save_video:
            create_video_from_frames(self.window_size_x, self.window_size_y + self.footer_size, self.frames_dir)
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
        pygame.quit()
        return
