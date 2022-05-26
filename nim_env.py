import random
import numpy as np


class NimEnv:
    def __init__(self, seed=None):
        self.n_heap = 3
        self.n_agents = 2
        self.current_player = 0
        self.winner = None
        self.end = False
        self.num_step = 0

        if seed is not None:
            random.seed(seed)
        self.heaps = random.sample(range(1, 8), 3)
        self.heap_avail = [True, True, True]
        self.heap_keys = ["1", "2", "3"]
        self.winner = None

    def check_valid(self, action):
        h, n = map(int, action)
        if not self.heap_avail[h - 1]:
            return False
        if n < 1:
            return False
        if n > self.heaps[h - 1]:
            return False
        return True

    def step(self, action):
        """
        step method takin an action as input

        Parameters
        ----------
        action : list(int)
            action[0] = 1, 2, 3 is the selected heap to take from
            action[1] is the number of objects to take from the heap

        Returns
        -------
        getObservation()
            State space (printable).
        reward : tuple
            (0,0) when not in final state, +1 for winner and -1 for loser
            otherwise.
        done : bool
            is the game finished.
        dict
            dunno.

        """

        # extracting integer values h: heap id, n: nb objects to take
        h, n = map(int, action)

        assert self.heap_avail[h - 1], "The selected heap is already empty"
        assert n >= 1, "You must take at least 1 object from the heap"
        assert (
            n <= self.heaps[h - 1]
        ), "You cannot take more objects than there are in the heap"

        self.heaps[h - 1] -= n  # core of the action

        if self.heaps[h - 1] == 0:
            self.heap_avail[h - 1] = False

        reward = (0, 0)
        done = False
        if self.heap_avail.count(True) == 0:
            done = True
            self.winner = self.current_player

        self.end = done
        # update
        self.num_step += 1
        self.current_player = 0 if self.num_step % 2 == 0 else 1
        next_heaps = self.heaps[:]
        return self.heaps, self.end, self.winner

    def observe(self):
        return self.heaps, self.end, self.winner

    def reward(self, player=0):
        if self.end:
            if self.winner is None:
                return 0
            else:
                return 1 if player == self.winner else -1
        else:
            return 0

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        self.heaps = random.sample(range(1, 8), 3)
        self.heap_avail = [True, True, True]
        self.current_player = 0
        self.winner = None
        self.end = False
        self.num_step = 0
        return self.heaps.copy()

    def render(self, simple=False):
        if simple:
            print(self.heaps)
        else:
            print(u"\u2500" * 35)
            for i in range(len(self.heaps)):
                print(
                    "Heap {}: {:15s} \t ({})".format(
                        self.heap_keys[i], "|" * self.heaps[i], self.heaps[i]
                    )
                )
                print(u"\u2500" * 35)


class OptimalPlayer:
    '''
    Description:
        A class to implement an epsilon-greedy optimal player in Nim.

    About optimial policy:
        Optimal policy relying on nim sum (binary XOR) taken from
        https://en.wikipedia.org/wiki/Nim#Example_implementation
        We play normal (i.e. not misere) game: the player taking the last object wins

    Parameters:
        epsilon: float, in [0, 1]. This is a value between 0-1 that indicates the
            probability of making a random action instead of the optimal action
            at any given time.

    '''

    def __init__(self, player=0, epsilon=0.2):
        self.epsilon = epsilon
        self.player = player  # 0 or 1

    def set_player(self, player=0, j=-1):
        self.player = player
        if j != -1:
            self.player = 0 if j % 2 == 0 else 1

    def randomMove(self, heaps):
        """
        Random policy (then optimal when obvious):
            - Select an available heap
            - Select a random integer between 1 and the number of objects in this heap.

        Parameters
        ----------
        heaps : list of integers
                list of heap sizes.

        Returns
        -------
        move : list
            move[0] is the heap to take from (starts at 1)
            move[1] is the number of obj to take from heap #move[0]
        """
        # the indexes of the heaps available are given by
        heaps_avail = [i for i in range(len(heaps)) if heaps[i] > 0]
        chosen_heap = random.choice(heaps_avail)
        n_obj = random.choice(range(1, heaps[chosen_heap] + 1))
        move = [chosen_heap + 1, n_obj]

        return move

    def compute_nim_sum(self, heaps):
        """
        The nim sum is defined as the bitwise XOR operation,
        this is implemented in python with the native caret (^) operator.

        The bitwise XOR operation is such that:
            if we have heaps = [3, 4, 5],
            it can be written in bits as heaps = [011, 100, 101],
            and the bitwise XOR problem gives 010 = 2 (the nim sum is 2)

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.

        Returns
        -------
        nim : int
            nim sum of all heap sizes.

        """
        nim = 0
        for i in heaps:
            nim = nim ^ i
        return nim

    def act(self, heaps, **kwargs):
        """
        Optimal policy relying on nim sum (binary XOR) taken from
        https://en.wikipedia.org/wiki/Nim#Example_implementation

        We play normal (i.e. not misere) game: the player taking the last object wins

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.

        Returns
        -------
        move : list
            move[0] is the heap to take from (starts at 1)
            move[1] is the number of obj to take from heap #move[0]

        """
        if random.random() < self.epsilon:
            return self.randomMove(heaps)

        else:
            nim_sum = self.compute_nim_sum(heaps)
            if nim_sum == 0:
                # You will lose :(
                count_non_0 = sum(x > 0 for x in heaps)
                if count_non_0 == 0:
                    # Game is already finished, return a dumb move
                    move = [-1, -1]
                else:
                    # Take any possible move
                    move = [heaps.index(max(heaps)) + 1, 1]
                return move

            # Calc which move to make
            for index, heap in enumerate(heaps):
                target_size = heap ^ nim_sum
                if target_size < heap:
                    amount_to_remove = heap - target_size
                    move = [index + 1, amount_to_remove]
                    return move

    def act_q(self, heaps, q_table_row, greedy=False):
        """
        Policy relying on the Q-values learned applying q-learning algorithm.
        The player taking the last object wins. If greedy is True, the policy
        will select the action with the highest q-value. Otherwise it might choose
        a random action with probability env.epsilon

        Parameters
        ----------
        heaps : list of integers
            list of heap sizes.
        q_table_row: numpy.ndarray
            row of the Q-table containing the q-values of
            all the possible actions in the heaps' state
        greedy: bool
            If greedy is True, the policy
            will select the action with the highest q-value. Otherwise it might choose
            a random action with probability env.epsilon. Default is False

        Returns
        -------
        move : list
            move[0] is the heap to take from (starts at 1)
            move[1] is the number of obj to take from heap #move[0]
        q_value: float
            q_value of the chosen move
        action: int
            column of the chosen action in the row of the Q-table.

        """
        if not greedy and random.random() < self.epsilon:
            move = self.randomMove(heaps)
            # converting move (list of 2 int) in the corresponding action (q_table_row's cell)
            if move[0] == 1:
                q_value = q_table_row[move[1]-1]
                action = move[1]-1
            if move[0] == 2:
                q_value = q_table_row[heaps[0] + move[1]-1]
                action = heaps[0] + move[1]-1
            if move[0] == 3:
                q_value = q_table_row[heaps[0] + heaps[1] + move[1]-1]
                action = heaps[0] + heaps[1] + move[1]-1

        else:
            # taking the highest q-value and saving the corresponding cell's index in action, then
            # converting the action to the move format (list of 2 int)
            q_value = np.max(q_table_row)
            action = np.argmax(q_table_row)
            if action <= (heaps[0]-1):
                move = [1, action + 1]
            if action > (heaps[0]-1) and action <= (heaps[0] + heaps[1] - 1):
                move = [2, action - heaps[0] + 1]
            if action > (heaps[0] + heaps[1] - 1):
                move = [3, action - heaps[1] - heaps[0] + 1]
        return move, q_value, action
