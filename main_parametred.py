from agents import Patron, Altruist  # берем классы агентов из соседнего файла
from env import WorldEnv  # берем класс среды из соседнего файла
import matplotlib.pyplot as plt
from typing import Optional, List
import os  # для Q table нужно
import json  # # для Q table нужно
from map_creation import Map_Creation
from for_special_training.training import Training_Manager

class SimulationManager:
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
        

    def run_simulation(
            self,
            map_type: str = "11",
            learning_flag: bool = True,
            testing_flag: bool = True,
    ):
        self.map_type = map_type
        self.init_environment_and_agents()
        if learning_flag:
            self.learning()
        else:
            self.load_tables()
        if testing_flag:
            self.show_trained_behavior()

    def init_environment_and_agents(self):
        env_size_x, env_size_y, env_agent_patron_start_zone, env_agent_altruist_start_zone, env_target_location, env_walls_positions, env_doors_positions, agent_patron_status, agent_altruist_status = Map_Creation().select_scenary(self.map_type)
        self.env = WorldEnv(env_size_x,
                            env_size_y,
                            env_target_location,
                            env_walls_positions,
                            env_doors_positions,
                            render_mode=None
                        )
        self.add_agents(env_agent_patron_start_zone, env_agent_altruist_start_zone, agent_patron_status, agent_altruist_status)

    def add_agents(self, env_agent_patron_start_zone, env_agent_altruist_start_zone, agent_patron_status, agent_altruist_status, patron_num: int = 1, altruist_num: int = 1):
        for counter in range(patron_num):
            agent_id = f"patron_{counter}"
            self.env.agents[agent_id] = Patron(self.env.action_space())
            self.env.agents[agent_id].start_zone = env_agent_patron_start_zone
            match agent_patron_status:
                case "training":
                    pass
                case "trained":
                    self.load_tables([agent_id])
        for counter in range(altruist_num):
            agent_id = f"altruist_{counter}"
            match agent_altruist_status:
                case "training":
                    self.env.agents[agent_id] = Altruist(self.env.action_space())
                    self.env.agents[agent_id].start_zone = env_agent_altruist_start_zone
                case "not_there":
                    pass
                case "random":
                    self.env.agents[agent_id] = Altruist(self.env.action_space())
                    self.env.agents[agent_id].start_zone = env_agent_altruist_start_zone

    def learning(
        self,
        render_mode: str = "rgb_array",
    ):
        self.env.render_mode = render_mode
        rewards = self.special_training_function()
        self.cache_tables()
        self.build_plot(rewards)
        print("Episode finished!")

    def special_training_function(self, num_episodes = 100):
        rewards = []
        total_reward = 0
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(total_reward, learning_flag=True)
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()
        return rewards

    def show_trained_behavior(
        self,
        render_mode: str = "human",
        num_episodes: int = 10
    ):
        self.env.render_mode = render_mode
        total_reward = 0
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(total_reward, learning_flag=False)
            print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()

    def run_simulation_step(
            self,
            total_reward: int,
            learning_flag: bool = False,
            possible_actions: int = 800,
            ):
        state, _ = self.env.reset()
        state_tupled = tuple(state.values())
        steps = 0
        action = {}
        done=False
        while possible_actions > 0 and not done:
            steps += 1
            for agent_id, agent_instance in self.env.agents.items():
                action[agent_id] = agent_instance.select_action(state_tupled)
            next_state, reward, done, _, _ = self.env.step(action)
            if learning_flag:
                for agent_id, agent_instance in self.env.agents.items():
                    agent_instance.update_q(state[agent_id], action[agent_id], reward, next_state[agent_id])
            state = next_state
            total_reward += reward
            possible_actions -= 1
            self.env.render()
        if learning_flag:
            for agent_instance in self.env.agents.values():
                agent_instance.decay_epsilon()
        return total_reward, steps

    def cache_tables(self, cache_dir: str = "cache", try_dir_base: str = "progon_"):
        directory_path = f"{cache_dir}/{self.map_type}"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        existing_folders = [f for f in os.listdir(directory_path) if f.startswith(try_dir_base) and os.path.isdir(os.path.join(directory_path, f))]
        if existing_folders:
            max_i = max([int(f.split('_')[1]) for f in existing_folders])
        else:
            max_i = 0
        new_folder = os.path.join(directory_path, f"{try_dir_base}{max_i + 1}")
        os.makedirs(new_folder)
        for agent_id, agent_instance in self.env.agents.items():
            table_agent_path = os.path.join(new_folder, f"table_{agent_id}.json")
            table = self.serialize_keys(agent_instance.q_table)
            with open(table_agent_path, 'w') as f1:
                json.dump(table, f1)
        print(f"Q-таблицы сохранены в {new_folder}")
        
    def serialize_keys(self, table):
        new_table = {}
        for key, value in table.items():
            str_key = str(key)
            new_table[str_key] = value
        return new_table

    def load_tables(self, agents_to_load: List, progon_number: int = None, cache_dir: str = "cache", try_dir_base: str = "progon_"):
        if progon_number is None:
            existing_folders = [f for f in os.listdir(cache_dir) if
                                f.startswith(try_dir_base) and os.path.isdir(os.path.join(cache_dir, f))]
            if existing_folders:
                max_i = max([int(f.split('_')[1]) for f in existing_folders])
                progon_number = max_i
            else:
                raise ValueError("Нет сохранённых прогонов для загрузки.")
        progon_folder = os.path.join(cache_dir, f"{try_dir_base}{progon_number}")
        if not os.path.exists(progon_folder):
            raise ValueError(f"Попытка {progon_number} не существует.")
        for agent_id in agents_to_load:
            file_path = os.path.join(progon_folder, f"table_{agent_id}.json")
            if os.path.exists(file_path):
                with open(file_path, 'r') as f1:
                    self.env.agents[agent_id].q_table = json.load(f1)
            else:
                print(f"Файл с agent_id {agent_id} не найден.")
        print(f"Таблицы успешно загружены из {progon_folder}")

    def build_plot(self, rewards: List):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps')
        plt.title('Learning Progress')
        plt.show()




# Блок в котором определяется поведение скрипта
# (по умолчанию при вызове напрямую агенты и обучаются и тестируют, агрументами вызова это можно поменять)
if __name__ == "__main__":
    import sys  # Это для обработки аргументов командной строки

    # по умолчанию запускается на обучение агента + демонстрацию (как и было)
    learning_needed = True
    testing_needed = True

    # можно передать из командной строки агрументы no_learn или no_test и тогда будет что-то одно
    # ПЕРЕПИСАТЬ НА ЛОГИКУ КОНЕЧНОГО ПОЛЬЗОВАТЕЛЯ ***
    map_type = "11"
    if "no_learn" in sys.argv:
        learning_needed = False
    if "no_test" in sys.argv:
        testing_needed = False
    for arg in sys.argv:
        if arg.startswith("map_type"):
            map_type=arg.split("=")[1]
    SimulationManager().run_simulation(map_type=map_type, learning_flag=learning_needed, testing_flag=testing_needed)

