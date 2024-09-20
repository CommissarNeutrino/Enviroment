import pygame


def scale_image_with_aspect_ratio(image, target_size):
    """
    Масштабирует изображение с сохранением пропорций.

    :param image: Исходное изображение.
    :param target_size: Целевой размер (ширина, высота).
    :return: Масштабированное изображение с сохранением пропорций.
    """
    original_width, original_height = image.get_size()
    target_width, target_height = target_size

    aspect_ratio = original_width / original_height

    # Вычисляем новый размер с сохранением пропорций
    if aspect_ratio > 1:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    return pygame.transform.scale(image, (new_width, new_height))


class ImageLoader:
    """ Класс для загрузки и масштабирования изображений """
    def __init__(self, default_size):
        self.default_size = default_size  # Размер по умолчанию, например, размер клетки

    def load_and_scale_image(self, image_path, target_size=None, keep_aspect_ratio=True):
        """
        Универсальный метод для загрузки и масштабирования изображения.

        :param image_path: Путь к изображению.
        :param target_size: Целевой размер изображения (ширина, высота). Если не указан, используется default_size.
        :param keep_aspect_ratio: Флаг для сохранения пропорций изображения.
        :return: Масштабированное изображение.
        """
        image = pygame.image.load(image_path)

        if target_size is None:
            target_size = (self.default_size, self.default_size)

        if keep_aspect_ratio:
            image = scale_image_with_aspect_ratio(image, target_size)
        else:
            image = pygame.transform.scale(image, target_size)

        return image


class GridRenderer:
    """ Класс про визуализацию всего (славься Pygame).В нем есть:"""
    def __init__(self, grid_width, grid_height, goal_image_path, agents_info, cell_size=100, background_image_path='background.png'):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.cell_size = cell_size

        # Инициализация ImageLoader с размером клетки по умолчанию
        image_loader = ImageLoader(self.cell_size)

        # Загрузка изображения цели
        self.goal_image = image_loader.load_and_scale_image(goal_image_path)

        # Загрузка изображений агентов с сохранением пропорций
        self.agent_images = {
            agent_type: image_loader.load_and_scale_image(image_path)
            for agent_type, image_path in agents_info.items()
        }

        # Инициализация экрана
        screen_size = (self.grid_width * self.cell_size, self.grid_height * self.cell_size)
        pygame.init()
        self.screen = pygame.display.set_mode(screen_size)
        self.clock = pygame.time.Clock()

        # Загрузка фонового изображения без сохранения пропорций (чтобы точно подогнать под экран)
        self.background_image = image_loader.load_and_scale_image(
            background_image_path, target_size=screen_size, keep_aspect_ratio=False
        )

        # Предварительно отрисованная сетка
        self.grid_surface = self.create_grid_surface()

    def create_grid_surface(self):
        """ Создаём поверхность с отрисованной сеткой для последующего использования """
        grid_surface = pygame.Surface((self.grid_width * self.cell_size, self.grid_height * self.cell_size), pygame.SRCALPHA)
        for x in range(0, self.grid_width * self.cell_size, self.cell_size):
            for y in range(0, self.grid_height * self.cell_size, self.cell_size):
                rect = pygame.Rect(x, y, self.cell_size, self.cell_size)
                pygame.draw.rect(grid_surface, (0, 0, 0, 50), rect, 1)  # Черная прозрачная сетка
        return grid_surface

    def render(self, agents, goal_location, delay):
        """ Отрисовка сетки, агентов и цели на экране """
        # Отрисовка фонового изображения
        self.screen.blit(self.background_image, (0, 0))

        # Используем предварительно созданную сетку
        self.screen.blit(self.grid_surface, (0, 0))

        # Отрисовка цели
        goal_rect = pygame.Rect(
            goal_location[1] * self.cell_size,
            goal_location[0] * self.cell_size,
            self.cell_size, self.cell_size
        )
        self.screen.blit(self.goal_image, goal_rect)

        # Рендеринг агентов с их изображениями
        for agent in agents:
            agent_rect = pygame.Rect(
                agent.location[1] * self.cell_size,
                agent.location[0] * self.cell_size,
                self.cell_size, self.cell_size
            )
            agent_image = self.agent_images["patron"]
            self.screen.blit(agent_image, agent_rect)

        pygame.display.flip()  # Обновляем экран
        pygame.time.wait(delay)  # Добавляем задержку для замедления движения

    @staticmethod
    def close():
        pygame.quit()
