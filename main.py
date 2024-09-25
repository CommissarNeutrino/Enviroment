from agents import Patron, Altruist  # берем классы агентов из соседнего файла
from env import WorldEnv  # берем класс среды из соседнего файла
import matplotlib.pyplot as plt
from typing import Optional, List
import os  # для Q table нужно
import json  # # для Q table нужно
#from map_creation import Map_Creation
from training import Training_Manager
import os
import pickle

# Технический долг: посмотреть как реализуется action space в гимназиум по-нормальному?

class SimulationManager:

    def Scenary_1a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            target_location=(4, 0),
                            walls_positions=set(),
                            doors_positions={},
                            render_mode=None
                        )
        agent_id = f"patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0)]
        self.env.agents[agent_id].status = "training"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/1a")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/1a")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_1b(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            target_location=(4, 0),
                            walls_positions=set(),
                            doors_positions={},
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "training"
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "random"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/1b")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/1b")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_2a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            target_location=(4, 0),
                            walls_positions=set([(1, 0), (1, 1), (4, 1)]),
                            doors_positions={},
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "training"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/2a")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/2a")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(3):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_2b(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1)])
        doors_positions={}
        length_of_grid = 5
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            target_location=(4, 0),
                            walls_positions=walls_positions,
                            doors_positions=doors_positions,
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "training"
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        #self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "random"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            #print("qaaaaaaa self.env.agents", self.env.agents)
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/2b")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/2b")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(3):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()
    
    def Scenary_2c(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1)])
        doors_positions={}
        length_of_grid = 5
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            target_location=(4, 0),
                            walls_positions=walls_positions,
                            doors_positions=doors_positions,
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "trained"
        self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
        self.load_tables(agents_to_load = ["patron_0"], cache_dir="cache/2b")
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "training"

        self.env.agents[agent_id].states_of_env["walls_positions"] = walls_positions
        self.env.agents[agent_id].states_of_env["doors_positions"] = doors_positions
        self.env.agents[agent_id].states_of_env["length_of_grid"] = length_of_grid
        self.env.agents[agent_id].states_of_env["height_of_grid"] = height_of_grid

        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/2c")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0", "altruist_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/2c")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon

            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(3):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_3a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            target_location=(4, 0),
                            walls_positions=set([(1, 0), (1, 1), (4, 1)]),
                            doors_positions={(1, 2): (3, 1)},
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "training"
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "random"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/3a")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/3a")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_3b(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1)])
        doors_positions={(1, 2): (3, 1)}
        length_of_grid = 5
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            target_location=(4, 0),
                            walls_positions=walls_positions,
                            doors_positions=doors_positions,
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "trained"
        self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
        self.load_tables(agents_to_load = ["patron_0"], cache_dir="cache/3a")
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "training"

        self.env.agents[agent_id].states_of_env["walls_positions"] = walls_positions
        self.env.agents[agent_id].states_of_env["doors_positions"] = doors_positions
        self.env.agents[agent_id].states_of_env["length_of_grid"] = length_of_grid
        self.env.agents[agent_id].states_of_env["height_of_grid"] = height_of_grid

        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/3b")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0", "altruist_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/3b")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_4a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=7,
                            size_y=3,
                            target_location=(4, 0),
                            walls_positions=set([(1, 0), (1, 1), (4, 1), (5, 0)]),
                            doors_positions={(1, 2): (3, 1), (4, 2): (3, 0)},
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "training"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/4a")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/4a")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_4b(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=7,
                            size_y=3,
                            target_location=(4, 0),
                            walls_positions=set([(1, 0), (1, 1), (4, 1), (5, 0)]),
                            doors_positions={(1, 2): (3, 1), (4, 2): (3, 0)},
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "training"
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "random"
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/4b")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/4b")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenary_4c(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1), (5, 0)])
        doors_positions={(1, 2): (3, 1), (4, 2): (3, 0)}
        length_of_grid = 7
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            target_location=(4, 0),
                            walls_positions=walls_positions,
                            doors_positions=doors_positions,
                            render_mode=None
                        )
        agent_id = "patron_0"
        self.env.agents[agent_id] = Patron(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(0, 0), (0, 1), (0, 2)]
        self.env.agents[agent_id].status = "trained"
        self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
        self.load_tables(agents_to_load = ["patron_0"], cache_dir="cache/4b")
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "training"

        self.env.agents[agent_id].states_of_env["walls_positions"] = walls_positions
        self.env.agents[agent_id].states_of_env["doors_positions"] = doors_positions
        self.env.agents[agent_id].states_of_env["length_of_grid"] = length_of_grid
        self.env.agents[agent_id].states_of_env["height_of_grid"] = height_of_grid

        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/4c")
            self.build_plot(rewards)
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0", "altruist_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/4c")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(10):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def special_training_function(self, num_episodes = 300000):
        rewards = []
        #print("self.env.agents", self.env.agents)
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(episode, total_reward=0, learning_flag=True)
            # print(self.env.agents["patron_0"].q_table)
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()
        return rewards

    def run_simulation_step(
            self,
            episode_number,
            total_reward: int,
            learning_flag: bool = False,
            possible_actions: int = 100,
            ):
        state, _ = self.env.reset()

        #print("self.env.agents", self.env.agents)
        if "altruist_0" in self.env.agents.keys():
            #print("has")
            if self.env.agents["altruist_0"].status == "training":
                altruist_instance = self.env.agents["altruist_0"]
                altruist_instance.time = 0
                altruist_instance.states_of_env[altruist_instance.time] = {}
                #print("state", state["patron_0"], state["altruist_0"])
                altruist_instance.states_of_env[altruist_instance.time]["patron_position"] = state["patron_0"]
                altruist_instance.states_of_env[altruist_instance.time]["altruist_position"] = state["altruist_0"]

        steps = 0
        action = {}
        done = False
        while possible_actions > 0 and not done:
            steps += 1
            for agent_id, agent_instance in self.env.agents.items():
                action[agent_id] = agent_instance.select_action(state[agent_id])
            next_state, reward, done, _, _ = self.env.step(action)
            if learning_flag:
                for agent_id, agent_instance in self.env.agents.items():
                    # print(state[agent_id], action[agent_id], reward, next_state[agent_id])

                    # change this shit!!!!!
                    if agent_id[:-1] == "altruist_":
                        agent_instance.states_of_env[agent_instance.time] = {}
                        agent_instance.states_of_env[agent_instance.time]["patron_position"] = next_state["patron_0"]
                        agent_instance.states_of_env[agent_instance.time]["altruist_position"] = next_state["altruist_0"]
                    if agent_instance.status == "training":
                        agent_instance.update_q(state[agent_id], action[agent_id], reward, next_state[agent_id])

            state = next_state
            total_reward += reward
            possible_actions -= 1
            self.env.render(steps, episode_number)
        if learning_flag:
            for agent_instance in self.env.agents.values():
                agent_instance.decay_epsilon()
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
            table_agent_path = os.path.join(new_folder, f"table_{agent_id}.pkl")
            with open(table_agent_path, 'wb') as f:
                pickle.dump(agent_instance.q_table, f)  # Сохраняем Q-таблицы с помощью pickle
        print(f"Q-таблицы сохранены в {new_folder}")

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
            file_path = os.path.join(progon_folder, f"table_{agent_id}.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.env.agents[agent_id].q_table = pickle.load(f)  # Загружаем Q-таблицы с помощью pickle
            else:
                print(f"Файл с agent_id {agent_id} не найден.")
        print(f"Таблицы успешно загружены из {progon_folder}")


    def build_plot(self, rewards: List):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps')
        plt.title('Learning Progress')
        plt.show()

if __name__ == "__main__":
    import sys  # Это для обработки аргументов командной строки

    # по умолчанию запускается на обучение агента + демонстрацию (как и было)
    learning_needed = True
    testing_needed = True

    # можно передать из командной строки агрументы no_learn или no_test и тогда будет что-то одно
    # ПЕРЕПИСАТЬ НА ЛОГИКУ КОНЕЧНОГО ПОЛЬЗОВАТЕЛЯ ***
    map_type = "2c"
    if "no_learn" in sys.argv:
        learning_needed = False
    if "no_test" in sys.argv:
        testing_needed = False
    for arg in sys.argv:
        if arg.startswith("map_type"):
            map_type=arg.split("=")[1]
    match map_type:
        case "1a":
            SimulationManager().Scenary_1a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "1b":
            SimulationManager().Scenary_1b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "2a":
            SimulationManager().Scenary_2a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "2b":
            SimulationManager().Scenary_2b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "2c":
            SimulationManager().Scenary_2c(learning_flag=learning_needed, testing_flag=testing_needed)
        case "3a":
            SimulationManager().Scenary_3a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "3b":
            SimulationManager().Scenary_3b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4a":
            SimulationManager().Scenary_4a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4b":
            SimulationManager().Scenary_4b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4c":
            SimulationManager().Scenary_4c(learning_flag=learning_needed, testing_flag=testing_needed)