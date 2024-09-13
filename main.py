from agents import QLearningAgent, Altruistic_agent
from ShSV_project.env import ShSV_WorldEnv
import matplotlib.pyplot as plt
from typing import List, Optional
import os
import json

class Main_Frame():

    def __init__(self):
        self.env = ShSV_WorldEnv(render_mode="rgb_array")
        self.agent_patron = QLearningAgent(self.env.action_space())
        self.agent_altruist = Altruistic_agent(self.env.action_space()) 

    def learning(self, num_episodes: int = 1000):
        rewards = []
        for episode in range(num_episodes):
            total_reward, steps = self.progon(learning_flag=True)
            self.agent_patron.decay_epsilon()
            self.agent_altruist.decay_epsilon()
            rewards.append(steps)
            print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.cache_tables()
        self.build_plot(rewards)
        print("Episode finished!")
        self.env.close()

    def checking_learning(self, num_episodes: int = 10):
        # Устанавливаем epsilon на минимальное значение и переводим в режим наблюдения
        self.agent_patron.epsilon = 0.01
        # Reinit env
        self.env = ShSV_WorldEnv(render_mode="human")
        # Запускаем агента для тестирования его поведения
        for episode in range(num_episodes):
            total_reward, steps = self.progon(state, learning_flag=False)
            print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
        self.env.close()

    def progon(self, learning_flag: bool, steps: int = 0, total_reward: int = 0, action: dict = {}, possible_actions: int = 300, done = False):
        state, _ = self.env.reset()
        state = tuple(state['agent_patron']) + tuple(state['agent_altruist']) + tuple(state['target'])
        while possible_actions>0 and not done:
            steps += 1
            action["patron"] = self.agent_patron.select_action(state)
            action["altruist"] = self.agent_altruist.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = tuple(next_state['agent_patron']) + tuple(next_state['agent_altruist']) + tuple(next_state['target'])
            if learning_flag:
                self.agent_patron.update_q(state, action["patron"], reward, next_state)
                self.agent_altruist.update(state, action["altruist"], reward, next_state)
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
        table_patron_path = os.path.join(new_folder, "table_patron.json")
        table_altruist_path = os.path.join(new_folder, "table_altruist.json")
        table_patron = self.agent_patron.q_table
        table_altruist = self.agent_altruist.q_table
        with open(table_patron_path, 'w') as f1:
            json.dump(table_patron, f1)
        with open(table_altruist_path, 'w') as f2:
            json.dump(table_altruist, f2)
        print(f"Q-таблицы сохранены в {new_folder}")

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
    SHSV_Frame().main_frame()