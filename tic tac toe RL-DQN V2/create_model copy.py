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
            reward = 50 
            self.game_over = True
            self.winner = self.current_player
        elif 0 not in self.board:
            reward = 7
            self.game_over = True
        else:
            reward = 2 if blocked else -0.05
            self.current_player = -self.current_player
        
        return self.get_state(), reward, self.game_over

    def check_winner(self, player):
        win_combinations = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
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
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_every = 20
        
    def build_model(self):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(3, 3, 1)),
            tf.keras.layers.Flatten(),  
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(9, activation='linear') 
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done, player_perspective):
        self.memory.append((state, action, reward, next_state, done, player_perspective))
    
    def act(self, state, valid_moves, player_perspective=1):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(valid_moves)
        current_state = state * player_perspective
        current_state = np.expand_dims(current_state, axis=0)
        act_values = self.model.predict(current_state, verbose=0)[0]
        sorted_actions = np.argsort(act_values)[::-1]
        for action in sorted_actions:
            if action in valid_moves:
                return action
        return np.random.choice(valid_moves)
        
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = np.array([x[0] * x[5] for x in minibatch])
        next_states = np.array([x[3] * x[5] for x in minibatch]) 

        targets = self.model.predict(states, verbose=0)
        next_q_values_model = self.model.predict(next_states, verbose=0) 
        next_q_values_target = self.target_model.predict(next_states, verbose=0) 

        for i, (state, action, reward, next_state, done, perspective) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                next_action = np.argmax(next_q_values_model[i])
                targets[i][action] = reward + self.gamma * next_q_values_target[i][next_action]

        self.model.fit(states, targets, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def evaluate_performance(self, games=200):
        env = TicTacToeEnvironment()
        old_epsilon = self.epsilon
        self.epsilon = 0.0

        wins = losses = draws = 0

        for _ in range(games):
            state = env.reset()
            done = False

            while not done:
                action = self.act(state, env.get_valid_moves(), 1)
                state, _, done = env.step(action)
                if done:
                    break
                opp_action = env.heuristic_move()
                state, _, done = env.step(opp_action)

            if env.winner == 1:
                wins += 1
            elif env.winner == -1:
                losses += 1
            else:
                draws += 1

        self.epsilon = old_epsilon

        win_rate = wins / games
        loss_rate = losses / games
        draw_rate = draws / games

        return win_rate, loss_rate, draw_rate


    def train(self, episodes=2000, eval_every=100):
        env = TicTacToeEnvironment()
        episode_rewards = []
        
        win_count = loss_count = draw_count = 0

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            smart_opponent = random.random() < 0.6

            while not done:
                valid_moves = env.get_valid_moves()
                action = self.act(state, valid_moves, player_perspective=1)
                next_state, reward, done = env.step(action)

                if done and env.winner == -1:
                    reward = -30

                self.remember(state, action, reward, next_state, done, player_perspective=1)
                state = next_state
                total_reward += reward

                if not done:
                    if smart_opponent:
                        opp_action = env.heuristic_move()
                    else:
                        opp_action = self.act(
                            state, 
                            env.get_valid_moves(), 
                            player_perspective=-1
                        )

                    state, _, done = env.step(opp_action)

                    if done and env.winner == -1 and self.memory:
                        s, a, r, ns, d, p = self.memory.pop()
                        self.memory.append((s, a, -30, ns, True, p))

            if len(self.memory) > self.batch_size:
                for _ in range(10):
                    self.replay()

            if episode % self.update_target_every == 0:
                self.update_target_model()

            episode_rewards.append(total_reward)
            if env.winner == 1:
                win_count += 1
            elif env.winner == -1:
                loss_count += 1
            else:
                draw_count += 1

            if (episode + 1) % eval_every == 0:
                avg_reward = np.mean(episode_rewards[-eval_every:])
                win_rate, loss_rate, draw_rate = self.evaluate_performance(games=100)

                print(f"\n" + "="*40)
                print(f"Episode: {episode + 1} | Epsilon: {self.epsilon:.3f}")
                print(f"Avg Reward: {avg_reward:.2f}")
                print(f"Training (Last {eval_every}): Wins: {win_count}, Losses: {loss_count}, Draws: {draw_count}")
                print(f"Eval (No Epsilon): Win: {win_rate:.1%}, Loss: {loss_rate:.1%}, Draw: {draw_rate:.1%}")
                print("="*40)

                win_count = loss_count = draw_count = 0

        self.model.save("tic_tac_toe.h5")
        print("\nTraining Complete. Model saved as 'tic_tac_toe.h5'")

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train(episodes=1000)
