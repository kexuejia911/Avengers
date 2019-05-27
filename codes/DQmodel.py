import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import random
import collections


action_list = ['forward', 'back', 'left', 'right']
map_list = [0, 0, 2, 0, 0,
       0, -1, 0, 0, 0,
       0, -1, 0, 0, 0,
       0, 0, 0, -1, 0,
       1, 0, 0, 0, -1]


class AgentAi:
    def __init__(self, map_size, action_size, gamma, epsilon):
        self.state_size = map_size**2
        self.action_size = action_size
        self.memory = collections.deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer='Adam')

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        min_batch = self.memory
        if batch_size < len(self.memory):
            min_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in min_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)


def check_movable(pos, action):
    if pos%5 == 0 and action == 2:
        return False
    if (pos-4)%5 == 0 and action == 3:
        return False
    if action == 0:
        pos -= 5
    if action == 1:
        pos += 5
    if pos<0 or pos >= 25:
        return False
    return True


# this is wrong way to do it, just for test
def take_action(map, action, reward):
    agent = map.index(1)
    if action == action_list.index('forward'):
        if check_movable(agent, action):
            map[agent] = 0
            agent -= 5
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] > 0:
                reward += 100
            return map, reward, True
        else:
            reward -= 1
            return map, reward, False
    if action == action_list.index('back'):
        if check_movable(agent, action):
            map[agent] = 0
            agent += 5
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] > 0:
                reward += 100
            return map, reward, True
        else:
            reward -= 1
            return map, reward, False
    if action == action_list.index('left'):
        if check_movable(agent, action):
            map[agent] = 0
            agent -= 1
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] > 0:
                reward += 100
            return map, reward, True
        else:
            reward -= 1
            return map, reward, False
    if action == action_list.index('right'):
        if check_movable(agent, action):
            map[agent] = 0
            agent += 1
            if map[agent] < 0:
                reward = -100
                return map, reward, True
            if map[agent] == 0:
                reward -= 1
                map[agent] = 1
                return map, reward, False
            if map[agent] > 0:
                reward += 100
            return map, reward, True
        else:
            reward -= 1
            return map, reward, False


agent = AgentAi(5, 4, 0.9, 0.15)
episode = 100000
for e in range(episode):

    state = [0, 0, 2, 0, 0,
       0, -1, 0, 0, 0,
       0, -1, 0, 0, 0,
       0, 0, 0, -1, 0,
       1, 0, 0, 0, -1]
    reward = 0

    for step in range(100):
        ndstate = np.reshape(state, [1, 25])
        action = agent.act(ndstate)
        next_state, reward, done = take_action(state,action,reward)
        ndnext_state = np.reshape(next_state, [1, 25])
        agent.remember(ndstate, action, reward, ndnext_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}".format(e, episode, reward))
            break
    agent.replay(32)
