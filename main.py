from agents import Patron, Altruist  # берем классы агентов из соседнего файла
from env import WorldEnv  # берем класс среды из соседнего файла
#  import matplotlib.pyplot as plt # ЗАЧЕМ? ПРОКЛЯТО? Проверить
from typing import Optional  # раньше тут вызывался еще и List
import os  # для Q table нужно
import json  # # для Q table нужно
from map_creation import Map_Creation


# Технический долг: посмотреть как реализуется action space в гимназиум по-нормальному?
class SimulationManager:

    def __init__(self, map_type):
        self.map_type = map_type

    def run_simulation(
            self,
            progon_number: Optional[int] = None,
            learning_flag: bool = True,
            testing_flag: bool = True,
            
    ):
        if learning_flag:
            self.learning()
        else:
            self.load_tables(progon_number)
        if testing_flag:
            self.show_trained_behavior()

    def learning(
            self,
            patron_num: int = 1,
            altruist_num: int = 1,
            render_mode: str = "rgb_array",
            num_episodes: int = 1000
    ):
        # Этот метод запускает процесс обучения агентов
        # Я ВООБЩЕ НЕ ПОНЯЛА ЭТУ ЛОГИКУ. ОБСУДИТЬ ***

        rewards = []

        # Инициализация среды и агентов
        self.init_environment_and_agents(render_mode)

        # Запуск цикла обучения
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(learning_flag=True)
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.cache_tables()
        # self.build_plot(rewards)

        print("Episode finished!")

        self.env.close()

    def init_environment_and_agents(self, render_mode):
        env_size_x, env_size_y, env_agent_patron_start_zone, env_agent_altruist_start_zone, env_target_location, env_walls_positions, env_doors_positions = Map_Creation().select_scenary(self.map_type)
        self.env = WorldEnv(env_size_x,
                            env_size_y,
                            env_target_location,
                            env_walls_positions,
                            env_doors_positions,
                            render_mode=render_mode
                        )
        self.add_agents(env_agent_patron_start_zone, env_agent_altruist_start_zone)

    def add_agents(self, env_agent_patron_start_zone, env_agent_altruist_start_zone, patron_num=1, altruist_num=1):
        # переписать без циклового вызова методов add_agents и env
        for counter in range(patron_num):
            self.env.agents[f"patron_{counter}"] = Patron(self.env.action_space())
            self.env.agents[f"patron_{counter}"].start_zone = env_agent_patron_start_zone

        for counter in range(altruist_num):
            self.env.agents[f"altruist_{counter}"] = Altruist(self.env.action_space())
            self.env.agents[f"altruist_{counter}"].start_zone = env_agent_altruist_start_zone

    def show_trained_behavior(
            self,
            patron_num: int = 1,
            altruist_num: int = 1,
            render_mode: str = "human",
            num_episodes: int = 10
    ):
        # Устанавливаем epsilon на минимальное значение и переводим в режим наблюдения
        self.init_environment_and_agents(render_mode)
        for agent_id, agent_instance in self.env.agents.items():
            agent_instance.epsilon = 0.01
        # Запускаем агента для тестирования его поведения
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(learning_flag=False)
            print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()

    def run_simulation_step(
            self,
            learning_flag: bool,
            steps: int = 0,
            total_reward: int = 0,
            action: dict = {},
            possible_actions: int = 800,
            done=False):
        state, _ = self.env.reset()
        state_tupled = tuple(state.values())
        while possible_actions > 0 and not done:
            steps += 1
            for agent_id, agent_instance in self.env.agents.items():
                old_agent_location = agent_instance.location
                action[agent_id] = agent_instance.select_action(state_tupled)
            next_state, reward, done, _, _ = self.env.step(action)
            if learning_flag:
                for agent_id, agent_instance in self.env.agents.items():
                    agent_instance.update_q(state[agent_id], action[agent_id], reward, next_state[agent_id])
                    agent_instance.decay_epsilon()
            state = next_state
            total_reward += reward
            possible_actions -= 1
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

    def load_tables(self, progon_number: int = None, cache_dir: str = "cache", try_dir_base: str = "progon_"):
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

    # def load_tables(self, progon_number: int, cache_dir: str = "cache", try_dir_base: str = "progon_"):
    #     progon_folder = os.path.join(cache_dir, f"{try_dir_base}{progon_number}")
    #     if not os.path.exists(progon_folder):
    #         raise ValueError(f"Попытка {progon_number} не существует.")
    #     table_patron_path = os.path.join(progon_folder, "table_patron.json")
    #     table_altruist_path = os.path.join(progon_folder, "table_altruist.json")
    #     if not os.path.exists(table_patron_path) or not os.path.exists(table_altruist_path):
    #         raise ValueError(f"Файлы table_1.json и/или table_2.json не найдены в папке {progon_folder}.")
    #     # Загружаем table_1
    #     with open(table_patron_path, 'r') as f1:
    #         self.agent_patron.q_table = json.load(f1)
    #     # Загружаем table_2
    #     with open(table_altruist_path, 'r') as f2:
    #         self.agent_altruist.q_table = json.load(f2)
    #     print(f"Таблицы успешно загружены из {progon_folder}")

    # def build_plot(self, rewards: List):
    #     plt.plot(rewards)
    #     plt.xlabel('Episode')
    #     plt.ylabel('Total Steps')
    #     plt.title('Learning Progress')
    #     plt.show()

#
#
#
# Блок в котором определяется поведение скрипта
# (по умолчанию при вызове напрямую агенты и обучаются и тестируют, агрументами вызова это можно поменять)
if __name__ == "__main__":
    import sys  # Это для обработки аргументов командной строки

    # по умолчанию запускается на обучение агента + демонстрацию (как и было)
    learning_needed = True
    testing_needed = True

    # можно передать из командной строки агрументы no_learn или no_test и тогда будет что-то одно
    # ПЕРЕПИСАТЬ НА ЛОГИКУ КОНЕЧНОГО ПОЛЬЗОВАТЕЛЯ ***
    if "no_learn" in sys.argv:
        learning_needed = False
    if "no_test" in sys.argv:
        testing_needed = False
    SimulationManager(map_type=3).run_simulation(learning_flag=learning_needed, testing_flag=testing_needed)

