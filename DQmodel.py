import numpy as np


class AgentAi:
    def __init__(self, map_size, action_size, gamma, epsilon):
        self.state_size = map_size**2
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = Sequential()
        self.model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        self.model.add(Dense(24, activation='relu'))
        self.model.add(Dense(self.action_size, activation='linear'))
        self.model.compile(loss='mse', optimizer=Adam)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        min_batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in min_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
agent = AgentAi(5,5,0.9,0.2)
            
for epis in range(episodes):

    state = get_state()
    state = np.reshape(state, [1, 4])

    for step in range(100):
        action = agent.act(state)
        next_state, reward, done = take_action(action)
        next_state = np.reshape(next_state, [1, 4])
        if done:
            reward = -100
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}".format(e, episodes, time_t))
            break
    agent.replay(32)
