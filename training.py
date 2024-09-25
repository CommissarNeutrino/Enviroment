import json  # # для Q table нужно
import os  # для Q table нужно
import matplotlib.pyplot as plt # ЗАЧЕМ? ПРОКЛЯТО? Проверить
from typing import Optional, List  # раньше тут вызывался еще и List
from agents import Patron, Altruist  # берем классы агентов из соседнего файла
from env import WorldEnv  # берем класс среды из соседнего файла
from config import num_episodes, patron_num, altruist_num
from config import show_graph, cache_results

#from collections import deque

class BaseTrainer:
    def __init__(self, agent, env):
        pass 

    def train(self):
        pass
    
    def evaluate(self):
        pass


class AltruistTraining(BaseTrainer):
    def __init__(self,
            action_space,
            q_table, 
            time_horizon, 
            decay_coefficient,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.95,
            min_epsilon=0.01,
            **kwargs):

        super().__init__(**kwargs)
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # todo
        self.table_to_belong_q = q_table.copy()
        self.states_of_env = {}    # { int: { "patron_position": tuple, "altruist_position": tuple, "doors_positions": tuple } }
        self.current_time = 0
        self.time_horizon = time_horizon
        self.decay_coefficient = decay_coefficient
        self.action_space = action_space
        # todo
        self._action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0]),
        }

    def select_action(self):
        pass

    def update_table_to_belong_q(self, t):
        score = 0
        exp = 1
        score_time = t
        
        if len(self.states_of_env) - t < time_horizon:
            print("aaaaaaaaaaa")

        reachable_tiles = set(self.states_of_env[t]["patron_position"])
        
        while(score_time - t < self.time_horizon):
            next_tiles = set()
            for tile in reachable_tiles:
                for offset in self._action_to_direction.items():
                    #todo
                    if _allowed_step(tile, offset):
                        next_tiles.add(tile + offset)
                 
            score = score + exp * len(next_tiles)
            score_time += 1
            exp = exp * self.decay_coefficient
            reachable_tiles = next_tiles

        
        




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

    def train_agents(self, render_mode: str = "rgb_array") -> None:
        rewards = []
        self.init_environment_and_agents(render_mode)
        # Запуск цикла обучения
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step()
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        if cache_results:
            self.cache_tables()
        if show_graph:
            self.build_plot(rewards)
        self.env.close()

    def init_environment_and_agents(self, render_mode) -> None:
        self.env = WorldEnv(render_mode=render_mode)
        self.add_agents(patron_num, altruist_num)

    def add_agents(self) -> None:
        for counter in range(patron_num):
            self.env.agents[f"patron_{counter}"] = Patron(self.env.action_space())
        for counter in range(altruist_num):
            self.env.agents[f"altruist_{counter}"] = Altruist(self.env.action_space())

    def run_simulation_step(
            self,
            steps: int = 0,
            total_reward: int = 0,
            action: dict = {},
            possible_actions: int = 300,
            done=False
        ):
        state, _ = self.env.reset()
        state_tupled = tuple(state.values())
        while steps < possible_actions and not done:
            steps += 1
            for agent_id, agent_instance in self.env.agents.items():
                action[agent_id] = agent_instance.select_action(state_tupled)
            next_state, reward, done, _, _ = self.env.step(action)
            for agent_id, agent_instance in self.env.agents.items():
                agent_instance.update_q(state[agent_id], action[agent_id], reward, next_state[agent_id])
            state = next_state
            total_reward += reward
            self.env.render()
        return total_reward, steps

    def cache_tables(self, cache_dir: str = "cache", try_dir_base: str = "progon_"):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        existing_folders = [f for f in os.listdir(cache_dir) if f.startswith(try_dir_base) and os.path.isdir(os.path.join(cache_dir, f))]
        if existing_folders:
            max_i = max([int(f.split('_')[1]) for f in existing_folders])
        else:
            max_i = 0
        new_folder = os.path.join(cache_dir, f"{try_dir_base}{max_i + 1}")
        os.makedirs(new_folder)
        for agent_id, agent_instance in self.env.agents.items():
            table_agent_path = os.path.join(new_folder, f"table_{agent_id}.json")
            table = self.serialize_keys(agent_instance.q_table)
            print(agent_id, table)
            with open(table_agent_path, 'w') as f1:
                json.dump(table, f1)
        print(f"Q-таблицы сохранены в {new_folder}")
        
    def serialize_keys(self, table):
        new_table = {}
        for key, value in table.items():
            str_key = str(key)
            new_table[str_key] = value
        return new_table

    def build_plot(self, rewards: List):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps')
        plt.title('Learning Progress')
        plt.show()


if __name__ == "__main__":
    Training_Manager().train_agents()

