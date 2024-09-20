import numpy as np
from typing import Tuple


class BaseAgent:
    def __init__(self, agent_name: str = "", agent_type: str = "", location: Tuple = (), key: bool = False):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.location = location
        self.key = key


class Patron(BaseAgent):
    def __init__(
            self,
            action_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q(self, state, action, reward, next_state):
        all_actions = range(self.action_space.n)
        best_next_action = max(all_actions, key=lambda a: self.get_q(next_state, a))
        td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
        td_error = td_target - self.get_q(state, action)
        new_q = self.get_q(state, action) + self.alpha * td_error
        self.q_table[(state, action)] = new_q

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            all_actions = range(self.action_space.n)
            return max(all_actions, key=lambda a: self.get_q(state, a))  # Exploitation

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


class Altruist(BaseAgent):
    def __init__(
            self,
            action_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.01,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.key = False

    def update_q(self, state, action, reward, next_state):
        pass

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update(self, state, action, reward, next_state):
        pass

    def select_action(self, state):
        return self.action_space.sample()  # Exploration

    def decay_epsilon(self):
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)


# Создаем окружение
# env = gym.make('gym_examples/GridWorld-v0', render_mode='rgb_array')
# agent = QLearningAgent(env.action_space)

# num_episodes = 1000
# rewards = []
# for episode in range(num_episodes):
#     state, _ = env.reset()
#     state = tuple(state['agent']) + tuple(state['target'])
#     total_reward = 0
#     steps = 0
#     done = False
    
#     while not done:
#         steps += 1
#         action = agent.select_action(state)
#         next_state, reward, done, _, _ = env.step(action)
#         next_state = tuple(next_state['agent']) + tuple(next_state['target'])
#         agent.update_q(state, action, reward, next_state)
#         state = next_state
#         total_reward += reward
#         env.render()

#     agent.decay_epsilon()
#     print(f"Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")
#     rewards.append(steps)

# plt.plot(rewards)
# plt.xlabel('Episode')
# plt.ylabel('Total Steps')
# plt.title('Learning Progress')
# plt.show()
# env.close()

# # Устанавливаем epsilon на минимальное значение и переводим в режим наблюдения
# agent.epsilon = 0.01
# env = gym.make('gym_examples/GridWorld-v0', render_mode='human')

# # Запускаем агента для тестирования его поведения
# num_test_episodes = 10
# for episode in range(num_test_episodes):
#     state, _ = env.reset()
#     state = tuple(state['agent']) + tuple(state['target'])
#     total_reward = 0
#     steps = 0
#     done = False
    
#     while not done:
#         steps += 1
#         action = agent.select_action(state)
#         next_state, reward, done, _, _ = env.step(action)
#         next_state = tuple(next_state['agent']) + tuple(next_state['target'])
#         state = next_state
#         total_reward += reward
#         env.render()

#     print(f"Test Episode {episode + 1}: Total Reward = {total_reward}, Steps - {steps}")

# env.close()
