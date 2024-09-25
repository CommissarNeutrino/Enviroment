import numpy as np
from typing import Tuple


class BaseAgent:
    def __init__(self, agent_name: str = "", agent_type: str = "", location: Tuple = (), key: bool = False):
        self.agent_name = agent_name
        self.agent_type = agent_type
        self.location = location
        self.start_zone = []
        self.status = ""

    def get_q(self, state, action):
        pass

    def update_q(self, state, action, reward, next_state):
        pass

    def select_action(self, state):
        pass

    def decay_epsilon(self):
        pass


class Patron(BaseAgent):
    def __init__(
            self,
            action_space,
            alpha=0.1,
            gamma=0.99,
            epsilon=1.0,
            epsilon_decay=0.995,
            min_epsilon=0.0,
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

    def select_action(self, agent_location):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            all_actions = range(self.action_space.n)
            # print("all_actions", all_actions)
            # print("agent_location", agent_location)
            # print("action_list", [self.get_q(agent_location, a) for a in all_actions])
            # print("max", max(all_actions, key=lambda a: self.get_q(agent_location, a)))
            return max(all_actions, key=lambda a: self.get_q(agent_location, a))  # Exploitation

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
            decay_coefficient = 0.95,
            time_horizon = 3,
            **kwargs
    ):
        print("action_space", action_space)
        super().__init__(**kwargs)
        self.q_table = {}
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.key = False

        self.time = 0
        self.time_horizon = time_horizon
        self._action_to_direction = {
            0: np.array([0, -1]),
            1: np.array([1, 0]),
            2: np.array([0, 1]),
            3: np.array([-1, 0]),
            4: np.array([0, 0]),
        }
        self.decay_coefficient = decay_coefficient
        self.states_of_env = {}
        self.score_time = 0
        self.decay_epsilon_counter = 0
        


    def get_q(self, state, action):
        #print("in get_q", state, action)
        return self.q_table.get((state, tuple(action)), 0.0)

    def update_q(self, state, action, reward, next_state):
        all_actions = range(self.action_space.n)
        # best_next_action = max(all_actions, key=lambda a: self.get_q(next_state, a))
        # td_target = reward + self.gamma * self.get_q(next_state, best_next_action)
        # td_error = td_target - self.get_q(state, action)
        # new_q = self.get_q(state, action) + self.alpha * td_error
        # self.q_table[(state, action)] = new_q
        score = 0
        exp = 1
        self.score_time = self.time - self.time_horizon
        print("126", self.score_time, self.time,  self.time_horizon)
        if self.score_time < 0:
            return
        #print("tiles", self.states_of_env[self.score_time]["patron_position"])
        reachable_tiles = [self.states_of_env[self.score_time]["patron_position"]]
        print("131", self.states_of_env[self.score_time]["patron_position"])

        while(self.score_time < self.time):
            print("134", self.score_time)
            next_tiles = set()
            for tile in reachable_tiles:
                #print("tile", tile)
                for direction in self._action_to_direction.values(): #??
                    #todo
                    if self._allowed_step(tile, direction):
                        next_tiles.add(tuple(direction + tile))
                    
            score = score + exp * len(next_tiles)
            self.score_time += 1
            exp = exp * self.decay_coefficient
            reachable_tiles = list(next_tiles)

        # change state
        state = self.states_of_env[self.time - self.time_horizon]["altruist_position"]
        action = np.array(self.states_of_env[self.time - self.time_horizon + 1]["altruist_position"]) - np.array(state)
        #print("change state in update_q", state, action)
        new_q = self.get_q(state, tuple(action)) + self.alpha * score
        print("score", score)
        self.q_table[(state, tuple(action))] = new_q

    def select_action(self, agent_location):
        self.time += 1
        match self.status:
            case "random":
                #return self.action_space.sample()
                return 0
            case "training":
                if np.random.rand() < self.epsilon:
                    return self.action_space.sample()  # Exploration
                else:
                    all_actions = range(self.action_space.n)
                    return max(all_actions, key=lambda a: self.get_q(agent_location, self._action_to_direction[a]))  # Exploitation
            case _:
                return self.action_space.sample()



    def _allowed_step(self, agent_location, direction):

        new_position = self.decision_grid_edges(agent_location, direction)

        #print(new_position, agent_location, direction, agent_location + direction)
        #print("check", np.array_equal(new_position, agent_location + direction))

        if not np.array_equal(new_position, agent_location + direction):
            return False

        if (self.decision_walls_positions(new_position)
                and self.decision_doors_positions(new_position)
                and self.decision_other_agents(new_position)):
            return True
        return False

    
    def decision_grid_edges(self, agent_location, direction):
        new_position = np.clip(
            agent_location + direction, [0, 0], [self.states_of_env["length_of_grid"] - 1, self.states_of_env["height_of_grid"] - 1] #!!!!!!!!!
        )
        return new_position

    def decision_walls_positions(self, new_position):
        if tuple(new_position) in self.states_of_env["walls_positions"]:
            return False
        return True

    def decision_doors_positions(self, new_position):
        button_coords = self.states_of_env["doors_positions"].get(tuple(new_position), False)
        if not button_coords:
            return True
        if button_coords == self.states_of_env[self.score_time]["altruist_position"]:
            return True
        return False

    def decision_other_agents(self, new_position):
        if tuple(new_position) == self.states_of_env[self.score_time]["altruist_position"]:
            return False
        return True

    def decay_epsilon(self):
        self.decay_epsilon_counter += 1
        if self.decay_epsilon_counter > 300:
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
