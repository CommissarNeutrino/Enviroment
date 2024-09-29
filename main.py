import os  # для Q table нужно
import numpy as np
import pickle
import matplotlib.pyplot as plt
from typing import List
from agents import Patron, Altruist  # берем классы агентов из соседнего файла
from env import WorldEnv  # берем к4ласс среды из соседнего файла

# Технический долг: посмотреть как реализуется action space в гимназиум по-нормальному?

class SimulationManager:

    def Scenario_1a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            scenary_type="1a",
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_1b(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            scenary_type="1b",
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_2a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            scenary_type="2a",
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_2b(
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
                            scenary_type="2b",
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()
    
    def Scenario_2c(
            self,
            time_horizon: int,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1)])
        doors_positions={}
        length_of_grid = 5
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            scenary_type="2c",
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
        self.env.agents[agent_id].time_horizon = time_horizon

        self.env.agents[agent_id].states_of_env["walls_positions"] = walls_positions
        self.env.agents[agent_id].states_of_env["doors_positions"] = doors_positions
        self.env.agents[agent_id].states_of_env["length_of_grid"] = length_of_grid
        self.env.agents[agent_id].states_of_env["height_of_grid"] = height_of_grid

        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/2c")
            self.save_plot(rewards, cache_dir="cache/2c")
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0", "altruist_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/2c")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_3a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=5,
                            size_y=3,
                            scenary_type = "3a",
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_3b(
            self,
            time_horizon: int,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1)])
        doors_positions={(1, 2): (3, 1)}
        length_of_grid = 5
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            scenary_type="3b",
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
        self.env.agents[agent_id].time_horizon = time_horizon

        self.env.agents[agent_id].states_of_env["walls_positions"] = walls_positions
        self.env.agents[agent_id].states_of_env["doors_positions"] = doors_positions
        self.env.agents[agent_id].states_of_env["length_of_grid"] = length_of_grid
        self.env.agents[agent_id].states_of_env["height_of_grid"] = height_of_grid

        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/3b")
            self.save_plot(rewards, cache_dir="cache/3b")
            print("Episode finished!")
        if testing_flag:
            agents_to_test = ["patron_0", "altruist_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/3b")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_4a(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=7,
                            size_y=3,
                            scenary_type="4a",
                            target_location=(4, 0),
                            walls_positions=set([(1, 0), (1, 1), (4, 1), (5, 0)]),
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_4b(
            self,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        self.env = WorldEnv(size_x=7,
                            size_y=3,
                            scenary_type="4b",
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
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def Scenario_4c(
            self,
            time_horizon: int,
            learning_flag: bool = True,
            testing_flag: bool = True
    ):
        walls_positions=set([(1, 0), (1, 1), (4, 1), (5, 0)])
        doors_positions={(1, 2): (3, 1), (4, 2): (3, 0)}
        length_of_grid = 7
        height_of_grid = 3
        self.env = WorldEnv(size_x=length_of_grid,
                            size_y=height_of_grid,
                            scenary_type = "4c",
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
        self.patron_q_table = self.env.agents[agent_id].q_table
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "training"
        self.env.agents[agent_id].time_horizon = time_horizon
        self.env.agents[agent_id].states_of_env = {
            "walls_positions": walls_positions,
            "doors_positions": doors_positions,
            "length_of_grid": length_of_grid,
            "height_of_grid": height_of_grid,
        }
        if learning_flag:
            self.env.render_mode = "rgb_array"
            rewards = self.special_training_function()
            self.cache_tables(cache_dir="cache/4c")
            self.save_plot(rewards, cache_dir="cache/4c")
            print("Episode finished!")
            self.env.close()
        if testing_flag:
            agents_to_test = ["patron_0", "altruist_0"]
            self.load_tables(agents_to_load = agents_to_test, cache_dir="cache/4c")
            for agent_id in agents_to_test:
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()
    
    def Scenario_4c_testing_params(
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
        self.patron_q_table = self.env.agents[agent_id].q_table
        agent_id = "altruist_0"
        self.env.agents[agent_id] = Altruist(self.env.action_space())
        self.env.agents[agent_id].start_zone = [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.env.agents[agent_id].status = "training"
        self.env.agents[agent_id].states_of_env = {
            "walls_positions": walls_positions,
            "doors_positions": doors_positions,
            "length_of_grid": length_of_grid,
            "height_of_grid": height_of_grid,
        }
        if learning_flag:
            learning_params = {
                "alpha_min": 0.001,
                "alpha_steps": 10,
                "alpha_max": 0.1,
                "negative_reward_min": 0.05,
                "negative_reward_steps": 10,
                "negative_reward_max": 0.15,
            }
            total_steps = learning_params["alpha_steps"]*learning_params["negative_reward_steps"]
            percent_per_step = round(100 / total_steps, 2)
            current_step = 0
            self.env.render_mode = "rgb_array"
            for alpha_changing in np.linspace(learning_params["alpha_max"], learning_params["alpha_min"], learning_params["alpha_steps"]):
                for negative_reward_changing in np.linspace(learning_params["negative_reward_min"], learning_params["negative_reward_max"], learning_params["negative_reward_steps"]):
                    current_step += 1
                    self.nulling_agent(walls_positions, doors_positions, length_of_grid, height_of_grid)
                    self.env.agents["altruist_0"].alpha_changing = alpha_changing
                    self.env.agents["altruist_0"].negative_reward = negative_reward_changing
                    steps_list = self.special_training_function_testing_params()
                    self.cache_tables_testing_params(agents_to_load = ["altruist_0", "patron_0"], plot_data=steps_list, cache_dir=f"cache/4c/testing_params/{alpha_changing}_{negative_reward_changing}")
                    print(f"{current_step*percent_per_step}% done. Alpha: {alpha_changing}, Negative Reward: {negative_reward_changing}")
            self.env.close()
        if testing_flag:
            self.load_tables(agents_to_load = ["patron_0"], cache_dir="cache/4c")
            self.load_tables_testing_params(agents_to_load = ["altruist_0"], alpha_altruist=0.1, negative_reward_altruist=0.061111111111111116,cache_dir="cache/4c/testing_params")
            for agent_id in self.env.agents.keys():
                self.env.agents[agent_id].epsilon = self.env.agents[agent_id].min_epsilon
            self.env.render_mode = "human"
            total_reward = 0
            for episode in range(2):
                total_reward, steps = self.run_simulation_step(episode, total_reward, learning_flag=False)
                print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
            self.env.close()

    def nulling_agent(self, walls_positions, doors_positions, length_of_grid, height_of_grid):
        patron_instance = self.env.agents["patron_0"]
        patron_instance.q_table = self.patron_q_table
        patron_instance.epsilon = patron_instance.min_epsilon
        altruist_instance = self.env.agents["altruist_0"]
        altruist_instance.time = 0
        altruist_instance.q_table = {}
        altruist_instance.epsilon = 1.0
        altruist_instance.decay_epsilon_counter = 0
        altruist_instance.score_time = 0
        altruist_instance.states_of_env = {
            "walls_positions": walls_positions,
            "doors_positions": doors_positions,
            "length_of_grid": length_of_grid,
            "height_of_grid": height_of_grid,
        }

    def special_training_function_testing_params(self, num_episodes = 30000):
        steps_list = []
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(episode, total_reward=0, learning_flag=True)
            steps_list.append(steps)
        return steps_list
    
    def special_training_function(self, num_episodes = 30000):
        steps_list = []
        #print("self.env.agents", self.env.agents)
        for episode in range(num_episodes):
            total_reward, steps = self.run_simulation_step(episode, total_reward=0, learning_flag=True)
            # print(self.env.agents["patron_0"].q_table)
            steps_list.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        return steps_list

    def run_simulation_step(
            self,
            episode_number,
            total_reward: int,
            learning_flag: bool = False,
            possible_actions: int = 30,
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
        self.env.render(steps, episode_number)
        while possible_actions > 0 and not done:
            steps += 1
            for agent_id, agent_instance in self.env.agents.items():
                action[agent_id] = agent_instance.select_action(state[agent_id])
            next_state, reward, done, _, _ = self.env.step(action)
            if learning_flag:
                for agent_id, agent_instance in self.env.agents.items():
                    if agent_id[:-1] == "altruist_" and self.env.agents["altruist_0"].status == "training":
                        agent_instance.states_of_env[agent_instance.time] = {}
                        agent_instance.states_of_env[agent_instance.time]["patron_position"] = next_state["patron_0"]
                        agent_instance.states_of_env[agent_instance.time]["altruist_position"] = next_state["altruist_0"]
                        agent_instance.states_of_env[agent_instance.time - 1]["patron_action"] = action["patron_0"]
                        agent_instance.states_of_env[agent_instance.time - 1]["altruist_action"] = action["altruist_0"]
                    if agent_instance.status == "training":
                        agent_instance.update_q(state[agent_id], action[agent_id], reward, next_state[agent_id])

            state = next_state
            total_reward += reward
            possible_actions -= 1
            self.env.render(steps, episode_number)
        if "altruist_0" in self.env.agents and self.env.agents["altruist_0"].status == "training":
            additional_steps = self.env.agents["altruist_0"].time_horizon - 1
            for step in range(0, additional_steps):
                for agent_id, agent_instance in self.env.agents.items():
                    action[agent_id] = agent_instance.select_action(state[agent_id])
                next_state, reward, done, _, _ = self.env.step(action)
                if learning_flag:
                    for agent_id, agent_instance in self.env.agents.items():
                        if agent_id[:-1] == "altruist_" and self.env.agents["altruist_0"].status == "training":
                            agent_instance.states_of_env[agent_instance.time] = {}
                            agent_instance.states_of_env[agent_instance.time]["patron_position"] = next_state["patron_0"]
                            agent_instance.states_of_env[agent_instance.time]["altruist_position"] = next_state["altruist_0"]
                            agent_instance.states_of_env[agent_instance.time - 1]["patron_action"] = action["patron_0"]
                            agent_instance.states_of_env[agent_instance.time - 1]["altruist_action"] = action["altruist_0"]
                        if agent_instance.status == "training":
                            agent_instance.update_q(state[agent_id], action[agent_id], reward, next_state[agent_id])
                state = next_state
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

    def cache_tables_testing_params(self, agents_to_load, plot_data, cache_dir: str = "cache"):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        for agent_id in agents_to_load:
            table_agent_path = os.path.join(cache_dir, f"table_{agent_id}.pkl")
            with open(table_agent_path, 'wb') as f:
                pickle.dump(self.env.agents[agent_id].q_table, f)
        plot_data_path = os.path.join(cache_dir, f"table_data.pkl")
        with open(plot_data_path, 'wb') as f:
            pickle.dump(plot_data, f)

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

    def load_tables_testing_params(self, agents_to_load: List, alpha_altruist: float, negative_reward_altruist: float, cache_dir: str = "cache"):
        folder_path = os.path.join(cache_dir, f"{alpha_altruist}_{negative_reward_altruist}")
        if not os.path.exists(folder_path):
            raise ValueError(f"Попытка {alpha_altruist}_{negative_reward_altruist} не существует.")
        for agent_id in agents_to_load:
            file_path = os.path.join(folder_path, f"table_{agent_id}.pkl")
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    self.env.agents[agent_id].q_table = pickle.load(f)  # Загружаем Q-таблицы с помощью pickle
            else:
                print(f"Файл с agent_id {agent_id} не найден.")
        print(f"Таблицы успешно загружены из {folder_path}")

    def build_plot(self, rewards: List):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps')
        plt.title('Learning Progress')
        plt.show()

    def save_plot(self, rewards: List, cache_dir: str, try_dir_base: str = "progon_"):
        existing_folders = [f for f in os.listdir(cache_dir) if
                            f.startswith(try_dir_base) and os.path.isdir(os.path.join(cache_dir, f))]
        if existing_folders:
            max_i = max([int(f.split('_')[1]) for f in existing_folders])
            progon_number = max_i
        else:
            raise ValueError("Нет сохранённых прогонов для загрузки.")
        progon_folder = os.path.join(cache_dir, f"{try_dir_base}{progon_number}")
        filename = os.path.join(progon_folder, "plot.png")
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps')
        plt.title('Learning Progress')
        plt.savefig(filename)


def scenario_chooser():
    import sys  # Это для обработки аргументов командной строки

    # по умолчанию запускается на обучение агента + демонстрацию (как и было)
    learning_needed = True
    testing_needed = True

    # можно передать из командной строки агрументы no_learn или no_test и тогда будет что-то одно
    # ПЕРЕПИСАТЬ НА ЛОГИКУ КОНЕЧНОГО ПОЛЬЗОВАТЕЛЯ ***
    map_type = "4b"
    if "no_learn" in sys.argv:
        learning_needed = False
    if "no_test" in sys.argv:
        testing_needed = False
    for arg in sys.argv:
        if arg.startswith("map_type"):
            map_type=arg.split("=")[1]
    match map_type:
        case "1a":
            SimulationManager().Scenario_1a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "1b":
            SimulationManager().Scenario_1b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "2a":
            SimulationManager().Scenario_2a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "2b":
            SimulationManager().Scenario_2b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "2c":
            SimulationManager().Scenario_2c(learning_flag=learning_needed, testing_flag=testing_needed)
        case "3a":
            SimulationManager().Scenario_3a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "3b":
            SimulationManager().Scenario_3b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4a":
            SimulationManager().Scenario_4a(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4b":
            SimulationManager().Scenario_4b(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4c":
            SimulationManager().Scenario_4c(learning_flag=learning_needed, testing_flag=testing_needed)
        case "4c_testing_params":
            SimulationManager().Scenario_4c_testing_params(learning_flag=learning_needed, testing_flag=testing_needed)
        case _:
            print("Error: Choose scenario to play out with attribute map_type={choose type}")

def altruist_horizon_iterator():
    time_horizon_max_by_scenario = {
        "2c": 13,
        "3b": 15,
        "4c": 21
    }
    for scenario in time_horizon_max_by_scenario:
        for time_horizon in range(1, time_horizon_max_by_scenario[scenario]):
            match scenario:
                case "2c":
                    SimulationManager().Scenario_2c(time_horizon=time_horizon, learning_flag=True, testing_flag=False)
                case "3b":
                    SimulationManager().Scenario_3b(time_horizon=time_horizon, learning_flag=True, testing_flag=False)
                case "4c":
                    SimulationManager().Scenario_4c(time_horizon=time_horizon, learning_flag=True, testing_flag=False)


if __name__ == "__main__":
    altruist_horizon_iterator()
    # scenario_chooser()