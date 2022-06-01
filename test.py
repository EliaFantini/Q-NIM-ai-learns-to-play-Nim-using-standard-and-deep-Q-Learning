import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import copy
from itertools import count
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
import random
from nim_env import NimEnv, OptimalPlayer
import numpy as np


MAX_STICKS=7
NUM_HEAPS=3
METRICS_FREQUENCY=250

def heaps_to_binary(heaps):
    """
    Converts heaps state into three binary numbers to feed them into the Q-Network for DQN
    Args:
        heaps: list of integers
            list of heap sizes.

    Returns:
        bin_seq: torch.tensor
            3-bit binary representation of all 3 heaps, concatenated
    """
    bin_seq = []
    for i in range(NUM_HEAPS):
        # convert the number of sticks in each heap in a 3-bit binary representation
        binary_string = format(heaps[i], "b").zfill(3)
        for i in binary_string:
            bin_seq.append(int(i))
    return torch.tensor(bin_seq, dtype=torch.float)


class Qnetwork(nn.Module):
    def __init__(self):
        super(Qnetwork, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU())
        self.linear = nn.Linear(128, 21)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.linear(x)



class DQNAgent:
    def __init__(self, epsilon = 0.4, gamma = 0.99, batch_size = 64, learning_rate = 5e-4, c = 500, criterion = nn.HuberLoss(delta=1.0)): #learning_rate = 0.001
        # if a GPU is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.player = None
        self.isLearning = True
        self.buffer_states = deque(maxlen=10000)
        self.buffer_actions = deque(maxlen=10000)
        self.buffer_rewards = deque(maxlen=10000)
        self.buffer_next_states = deque(maxlen=10000)
        self.buffer_target_q_values = deque(maxlen=10000)
        """TODO: change fifo queue with a list"""

        self.state = None
        self.action = None

        #self.q_value_theta = None
        self.gamma = gamma                        # discount factor

        # for epsilon-greedy policy and decreasing exploration
        self.epsilon = epsilon
        self.epsilon_min = epsilon
        self.epsilon_max = epsilon
        self.t = 0  # round counter
        self.c = c
        self.n = 0  # game counter
        self.n_star = 50

        self.batch_size = min(batch_size, 10000)
        self.learning_rate = learning_rate
        # as required, the criterion is the Huber loss with delta = 1
        self.criterion = criterion.to(self.device)

        self.model = Qnetwork().to(self.device)
        self.target = copy.deepcopy(self.model).to(self.device)
        self.target.load_state_dict(self.model.state_dict())

        # as required, we use the Adam optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr = learning_rate)

    def decrease_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_max * (1 - self.n / self.n_star))

    def memorize(self, state, action , reward, next_state, target_q_value):
        """
        Adds information at a given time step to the replay buffer for the DQN algorithm
        """
        self.buffer_states.append(state)
        self.buffer_actions.append(action)
        self.buffer_rewards.append(reward)
        self.buffer_next_states.append(next_state)
        self.buffer_target_q_values.append(target_q_value)

    def randomMove(self, heaps):
        """
        same function as the one from OptimalPlayer
        """
        # the indexes of the heaps available are given by
        heaps_avail = [i for i in range(len(heaps)) if heaps[i] > 0]
        chosen_heap = random.choice(heaps_avail)
        n_obj = random.choice(range(1, heaps[chosen_heap] + 1))
        move = [chosen_heap + 1, n_obj]
        return move

    def act_dqn(self, heaps):
        """
        action function, epsilon-greedy
        """
        self.state = heaps_to_binary(heaps).to(self.device)
        if np.random.rand() < self.epsilon:
            """REMEMBER TO CHANGE THIS IF IT DOESNT WORK"""
            #action = int(torch.randint(size=[1],low=0, high=MAX_STICKS*NUM_HEAPS).item())
            move = self.randomMove(heaps)
            action = (move[0] - 1) * MAX_STICKS + move[1] - 1
        else:
            with torch.no_grad():
                action = int(torch.argmax(self.model(self.state)).item())
                move = [(action // MAX_STICKS) + 1, (action % MAX_STICKS) + 1]
             # action with best Q-value : maximum value of output of Q-net

        # transform index of best action to a representation that the environment understands (from heap a, take b sticks)
        self.action = action

        return move

    def learn(self, state, reward, end = False):
        if self.isLearning:
            #if not end:
                # transcribe game state
            next_state = heaps_to_binary(state).to(self.device)
            if reward == 0:
                with torch.no_grad():
                    target_q_value = torch.max(self.target(next_state))
                self.memorize(self.state, self.action, torch.tensor(reward,dtype=torch.float, requires_grad=True), next_state, target_q_value)
            else:
                self.memorize(self.state, self.action, torch.tensor(reward,dtype=torch.float, requires_grad=True), next_state, torch.tensor(0,dtype=torch.float, requires_grad=True, device=self.device))
            if end:
            #else:
                # if the game is finished, there is no next state : that is still added in the replay buffer
                """0 is added so that it removes the second term and it's only reward in the loss calculation"""
                # reset state and action attributes
                #self.q_value_theta = None
                self.action = None
                self.state = None
                # increase game counter
                self.n += 1
                """Careful for self learning to not do this twice"""
                # decrease exploration
                self.decrease_epsilon()

            # we then have to manage the first few rounds, when the replay buffer is smaller than the batch size (default case 64)
            if len(self.buffer_states) < self.batch_size :
                #states = torch.zeros((len(self.buffer_states),9), dtype=torch.float, device= self.device )
                states = torch.cat([x.reshape([-1,9]) for x in self.buffer_states]).to(self.device)
                q_values = self.model(states)
                actions = list(self.buffer_actions)
                rewards = torch.tensor(list(self.buffer_rewards), requires_grad= True, device= self.device)
                q_values_target = torch.tensor(list(self.buffer_target_q_values), requires_grad= True, device= self.device)
                q_values_theta = q_values[np.arange(len(actions)), actions]

            else:

                rand_idxs= np.random.permutation(np.arange(len(self.buffer_states)))[:self.batch_size]
                rand_buffer_states = torch.cat([self.buffer_states[idx].reshape([-1,9]) for idx in rand_idxs]).to(self.device)
                actions = np.array(self.buffer_actions)[rand_idxs]
                rewards = torch.cat([self.buffer_rewards[idx].reshape([-1,1]) for idx in rand_idxs]).to(self.device)

                states = torch.cat([x.reshape([-1, 9]) for x in self.buffer_states]).to(self.device)
                q_values_target = torch.cat([self.buffer_target_q_values[idx].reshape([-1,1]) for idx in rand_idxs]).to(self.device)
                #states = torch.tensor((len(rand_buffer_states),9), dtype=torch.float, device= self.device )
                #torch.cat(rand_buffer_states, out=states)
                q_values = self.model(rand_buffer_states)
                #rewards = torch.tensor(rand_buffer_rewards, requires_grad= True, device= self.device)
                #q_values_target = torch.tensor(rand_buffer_target_q_values, requires_grad= True, device= self.device)
                q_values_theta = q_values[np.arange(len(actions)), actions].reshape((len(actions), 1))


            self.optimizer.zero_grad()
            loss = self.criterion(q_values_theta, rewards + self.gamma * q_values_target)
            # prediction for this round
            #theta = self.model.forward(states)
            #next_states_mask = torch.tensor([r[3] is not None for r in batch], dtype=torch.bool).to(self.device)

            #if next_states_mask.any(): # if not empty
           #     next_states = torch.cat(torch.tensor([r[3] for r in batch if r[3] is not None])).to(self.device)
                # select maximum values for network update
           #     max_target[next_states_mask] = self.target.forward(next_states).max(dim=1).values.detach()
                # computing Huber loss between current weights and observed rewards (current and previous with discount factor)
            #    loss = self.criterion(theta[torch.arange(len(actions)), actions], rewards + self.gamma * max_target)
            #else: # if empty
            #    loss = self.criterion(theta[torch.arange(len(actions)), actions], rewards)

            # backpropagation
            loss.backward(retain_graph=True)
            # update
            self.optimizer.step()
            # increment round counter
            """
            self.t += 1
            if self.t == self.c:
                self.t = 0
                # update network every c steps
            """

        elif end:
            # reset state and action attributes
            self.state, self.action = None, None

    def update_target(self):
        self.target.load_state_dict(self.model.state_dict())
        next_states = torch.cat([x.reshape([-1, 9]) for x in self.buffer_next_states]).to(self.device)
        result = self.target(next_states).max(dim=1).values
        for i in range(len(self.buffer_next_states)):
            self.buffer_target_q_values[i]=result[i]


    def load(self, name):
        return torch.load(name)

    def save(self, name):
        torch.save(self.model.state_dict(), name)




def deep_q_learning(maxGames=20000, env=NimEnv(), c=500):
    rewards = []
    rewards_mean = []
    player_dq = DQNAgent()

    pBar = trange(maxGames)
    Turn = 0
    for nbGames in range(maxGames):
        player_adv=OptimalPlayer(player = Turn, epsilon=0.5)
        player_dq.player = 1-Turn
        env.reset()
        heaps, _, __ = env.observe()

        roundGame = 0
        while not env.end:
            if env.current_player == player_adv.player:
                # if it's the turn of the adversary
                if isinstance(player_adv, DQNAgent):
                    # player_adv.state = heaps
                    player_adv.learn(heaps, 0)
                    move = player_adv.act_dqn(heaps)
                else:
                    move = player_adv.act(heaps)
            else:
                # if it's the agent's turn
                assert env.current_player == player_dq.player, "It has to be the agent's turn"
                if isinstance(player_dq, DQNAgent):
                    # player_dq.state = heaps
                    if roundGame>1: # if it's round 0 or 1, it's still first move for dqnAgent, there's no state and action to be memorized yet (they are the default, None)
                        player_dq.learn(heaps, 0)
                    move = player_dq.act_dqn(heaps)

            try:
                heaps, end, winner = env.step(move)
            except AssertionError:
                # if move is unavailable, the player that made the move loses and the game ends
                env.end = True
                end = True
                env.num_step += 1
                """winner = env.current_player"""
                winner = 1 - env.current_player # The opposite of the one who did the last, incorrect move.

            """TODO: in case the player won because the adv did an unavailable move, don't store it as a win for the other player (if dqn agent of course) (?)"""
            if end:
                if winner == player_dq.player:
                    rewards.append(1)
                    if isinstance(player_adv, DQNAgent):
                        player_adv.learn(heaps, -1, end = True)
                    if isinstance(player_dq, DQNAgent):
                        player_dq.learn(heaps, 1, end = True)
                elif winner == player_adv.player:
                    rewards.append(-1)
                    if isinstance(player_adv, DQNAgent):
                        player_adv.learn(heaps, 1, end = True)
                    if isinstance(player_dq, DQNAgent):
                        player_dq.learn(heaps, -1, end = True)
                else:
                    if isinstance(player_adv, DQNAgent):
                        player_adv.learn(heaps, 0, end = True)
                    if isinstance(player_dq, DQNAgent):
                        player_dq.learn(heaps, 0, end = True)
                break
            roundGame += 1

        pBar.update(1)
        Turn = 1 - Turn
        # every 250 games, compute average reward for plotting
        if nbGames%METRICS_FREQUENCY == METRICS_FREQUENCY -1 and nbGames!=0:
            print(np.array(rewards).mean())
            rewards_mean.append(np.array(rewards).mean())
            rewards = []
        if (nbGames + 1)%c ==0:
            if nbGames >900:
                a=1
            player_dq.update_target()
    pBar.close()
    env.reset()
    metrics = []
    metrics.append(rewards_mean)
    return metrics

rewards_mean = deep_q_learning()
#single_image_subplots("Training", rewards_mean)