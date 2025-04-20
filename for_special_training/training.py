# import json  # # для Q table нужно
# import os  # для Q table нужно
# import matplotlib.pyplot as plt # ЗАЧЕМ? ПРОКЛЯТО? Проверить
# import numpy as np
# from typing import Optional, List, Tuple, Dict  # раньше тут вызывался еще и List
# from agents import Patron, Altruist  # берем классы агентов из соседнего файла
# from env import WorldEnv  # берем класс среды из соседнего файла
# from config import num_episodes, patron_num, altruist_num
# from config import show_graph, cache_results

class Training_Manager:
    """
    Что делает этот класс:
        Управляет вызовом всех необходимых методов (run_simulation)
        Инициализирует среду
        Добавляет заданное количество агентов
        Запускает процесс обучения агентов
        Выполняет один эпизод взаимодействия агентов со средой
        Тестирует обученных агентов без дальнейшего обучения
        Сохраняет текущие Q-таблицы агентов на диск
        Преобразует ключи Q-таблицы в строки для возможности сохранения в JSON
    """
    def special_training_function(self, num_episodes = 1000):
        rewards = []
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step()
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()
        return rewards