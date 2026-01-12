import numpy as np
import tensorflow as tf
from collections import deque
import random

class TicTacToeEnvironment:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = 0
        return self.get_state()
    
    def get_state(self):
        return np.reshape(self.board, (3, 3, 1))
    
    def get_valid_moves(self):
        return [i for i, val in enumerate(self.board) if val == 0]
    
    def did_block_opponent(self, action):
        opponent = -self.current_player
        win_combinations = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
        for combo in win_combinations:
            if action in combo:
                others = [self.board[i] for i in combo if i != action]
                if others.count(opponent) == 2:
                    return True
        return False

    def step(self, action):
        if self.board[action] != 0:
            return self.get_state(), -50, True 

        blocked = self.did_block_opponent(action)
        self.board[action] = self.current_player
        
        if self.check_winner(self.current_player):
            reward = 10 
            self.game_over = True
            self.winner = self.current_player
        elif 0 not in self.board:
            reward = 5
            self.game_over = True
        else:
            reward = 2 if blocked else -0.05
            self.current_player = -self.current_player
        
        return self.get_state(), reward, self.game_over

    def check_winner(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]              # Diagonals
        ]
        for combo in win_combinations:
            if all(self.board[i] == player for i in combo):
                return True
        return False

    def heuristic_move(self):
        valid = self.get_valid_moves()
        for m in valid:
            self.board[m] = self.current_player
            if self.check_winner(self.current_player):
                self.board[m] = 0
                return m
            self.board[m] = 0
        opponent = -self.current_player
        for m in valid:
            self.board[m] = opponent
            if self.check_winner(opponent):
                self.board[m] = 0
                return m
            self.board[m] = 0
        if 4 in valid: return 4
        return np.random.choice(valid)
        
class DQNAgent:
    def __init__(self):
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.05 
        self.epsilon_decay = 0.9995 
        self.batch_size = 64
        self.update_target_every = 20
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(3, 3, 1)),  
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(9, activation='linear') 
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_moves)
        state = np.expand_dims(state, axis=0)
        act_values = self.model.predict(state, verbose=0)
        act_values = act_values[0]
        valid_actions = np.argsort(act_values)[::-1]
        for action in valid_actions:
            if action in valid_moves:
                return action
        return np.random.choice(valid_moves)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        minibatch = random.sample(self.memory, self.batch_size)
        states = np.array([x[0] for x in minibatch])
        targets = self.model.predict(states)
        
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                target = reward
            else:
                next_action = np.argmax(self.model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0])
                next_q = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)[0][next_action]
                target = reward + self.gamma * next_q
            
            targets[i][action] = target
        
        self.model.fit(states, targets, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, episodes=2000):
        env = TicTacToeEnvironment()
        for episode in range(episodes):
            state = env.reset()
            done = False
            smart_opponent = random.random() < 0.5 

            while not done:
                valid_moves = env.get_valid_moves()
                action = self.act(state, valid_moves)
                next_state, reward, done = env.step(action)
                
                if done and env.winner == -1: reward = -20

                self.remember(state, action, reward, next_state, done)
                state = next_state
                
                if not done:
                    if smart_opponent:
                        opp_action = env.heuristic_move()
                    else:
                        opp_action = self.act(state, env.get_valid_moves())
                    
                    state, _, done = env.step(opp_action)
                
                self.replay()
            
            if (episode+1) % 100 == 0:
                print(f"Episode: {episode+1}, Epsilon: {self.epsilon:.2f}")

        self.model.save('tic_tac_toe_smart.h5')

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train(episodes=1000)