from agents import QLearningAgent, Altruistic_agent
from env import WorldEnv
import matplotlib.pyplot as plt
from typing import List, Optional
import os
import json

class Main_Frame():

    def venv_init(self, patron_num, altruist_num, render_mode):
        self.env = WorldEnv(render_mode=render_mode)
        self.add_agents(patron_num, altruist_num)

    def add_agents(self, patron_num, altruist_num):
        for counter in range(patron_num):
            self.env.agents[f"patron_{counter}"] = QLearningAgent(self.env.action_space())
        for counter in range(altruist_num):
            self.env.agents[f"altruist_{counter}"] = Altruistic_agent(self.env.action_space())

    def learning(self, patron_num: int = 1, altruist_num: int = 1, render_mode: str = "rgb_array", num_episodes: int = 1000):
        self.venv_init(patron_num, altruist_num, render_mode)
        rewards = []
        for episode in range(num_episodes):
            total_reward, steps = self.progon(learning_flag=True)
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.cache_tables()
        self.build_plot(rewards)
        print("Episode finished!")
        self.env.close()

    def checking_learning(self, patron_num: int = 1, altruist_num: int = 1, render_mode: str = "human", num_episodes: int = 10):
        # Устанавливаем epsilon на минимальное значение и переводим в режим наблюдения
        self.venv_init(patron_num, altruist_num, render_mode)
        for agent_id, agent_instance in self.env.agents.items():
            agent_instance.epsilon = 0.01
        # Запускаем агента для тестирования его поведения
        for episode in range(num_episodes):
            total_reward, steps = self.progon(learning_flag=False)
            print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()

    def progon(self, learning_flag: bool, steps: int = 0, total_reward: int = 0, action: dict = {}, possible_actions: int = 300, done = False):
        state, _ = self.env.reset()
        state_tupled = tuple(state.values())
        while possible_actions>0 and not done:
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

    def load_tables(self, progon_number: int, cache_dir: str = "cache", try_dir_base: str = "progon_"):
        progon_folder = os.path.join(cache_dir, f"{try_dir_base}{progon_number}")
        if not os.path.exists(progon_folder):
            raise ValueError(f"Попытка {progon_number} не существует.")
        table_patron_path = os.path.join(progon_folder, "table_patron.json")
        table_altruist_path = os.path.join(progon_folder, "table_altruist.json")
        if not os.path.exists(table_patron_path) or not os.path.exists(table_altruist_path):
            raise ValueError(f"Файлы table_1.json и/или table_2.json не найдены в папке {progon_folder}.")
        # Загружаем table_1
        with open(table_patron_path, 'r') as f1:
            self.agent_patron.q_table = json.load(f1)
        # Загружаем table_2
        with open(table_altruist_path, 'r') as f2:
            self.agent_altruist.q_table = json.load(f2)
        print(f"Таблицы успешно загружены из {progon_folder}")

    def build_plot(self, rewards: List):
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Steps')
        plt.title('Learning Progress')
        plt.show()

    def main_frame(self, progon_number: Optional[int] = None, learning_needed: bool = True, testing_needed: bool = True):
        if learning_needed:
            self.learning()
        else:
            self.load_tables(progon_number)
        if testing_needed:
            self.checking_learning()


if __name__ == "__main__":
    Main_Frame().main_frame()
