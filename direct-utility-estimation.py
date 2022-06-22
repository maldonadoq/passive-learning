import numpy as np
import random

# Actions
UP, DOWN, LEFT, RIGHT = range(4)

P_F = 0.8
P_L = 0.1
P_R = 0.1

GAMMA = 0.9


# Gets the perpendicular actions of the given action
def getPerpActions(action):
    if action == UP or action == DOWN:
        return [LEFT, RIGHT]
    return [UP, DOWN]


def printPolicy(policy):
    for i in range(policy.shape[0]):
        for j in range(policy.shape[1]):
            character = ''
            if policy[i, j] == UP:
                character = 'U'
            elif policy[i, j] == DOWN:
                character = 'D'
            elif policy[i, j] == LEFT:
                character = 'L'
            elif policy[i, j] == RIGHT:
                character = 'R'
            else:
                character = '*'
            print(character, end=" ")
        print()


class Agent:
    def __init__(self, height, width, walls, dest, R):
        self.height = height
        self.width = width
        self.walls = walls
        self.dest = dest
        self.R = R
        self.P = np.zeros((height, width))
        self.P[:] = -100
        self.U = np.zeros((height, width))

    def validActions(self, i, j):
        actions = [UP, DOWN, LEFT, RIGHT]

        # Check borders
        if i == self.height - 1:
            actions.remove(DOWN)
        if i == 0:
            actions.remove(UP)
        if j == self.width - 1:
            actions.remove(RIGHT)
        if j == 0:
            actions.remove(LEFT)

        # Check walls
        for wallPos in self.walls:
            if i == wallPos[0] and j == wallPos[1] - 1:
                actions.remove(RIGHT)
            if i == wallPos[0] and j == wallPos[1] + 1:
                actions.remove(LEFT)
            if i == wallPos[0] - 1 and j == wallPos[1]:
                actions.remove(DOWN)
            if i == wallPos[0] + 1 and j == wallPos[1]:
                actions.remove(UP)
        return actions

    # Returns the next state given the current state and action
    def applyAction(self, i, j, action):
        n_i = 0
        n_j = 0

        if action == UP:
            n_i = i - 1
            n_j = j
        elif action == DOWN:
            n_i = i + 1
            n_j = j
        elif action == LEFT:
            n_i = i
            n_j = j - 1

        elif action == RIGHT:
            n_i = i
            n_j = j + 1

        # Check borders
        if n_i < 0 or n_i >= self.height:
            n_i = i
        if n_j < 0 or n_j >= self.width:
            n_j = j

        return n_i, n_j

    def valueIteration(self, epsilon=0.01):
        while True:
            delta = 0
            for i in range(self.height):
                for j in range(self.width):
                    a = self.U[i, j]
                    self.updateState(i, j)

                    delta = max(delta, abs(a - self.U[i, j]))
            if delta <= epsilon * (1 - GAMMA) / GAMMA:
                break

    def updateState(self, i, j):
        #  Get the valid actions at the current state
        validActions = self.validActions(i, j)

        # Can't update walls states, you shouldn't get here
        if (i, j) in self.walls:
            return None

        # This list saves the values from the right side of the bellman equation
        # It will be used to calculate the max value
        sumVals = []

        for act in [0, 1, 2, 3]:
            # Your current state s is the tuple in (i, j)

            # The state you would arrive at after applying the action is s'
            # There are three possibilities:
            # Forward state
            Sn_f = self.applyAction(i, j, act)

            # You failed and go to the perpendicular blocks
            perAct = getPerpActions(act)

            # Left state
            Sn_1 = self.applyAction(i, j, perAct[0])

            # Right state
            Sn_2 = self.applyAction(i, j, perAct[1])

            # However going left and right might fail, so we need to check if they are valid
            if not (perAct[0] in validActions):
                Sn_1 = [i, j]  # You don't move
            if not (perAct[1] in validActions):
                Sn_2 = [i, j]  # You don't move

            E = P_F * self.U[Sn_f[0], Sn_f[1]] + P_L * \
                self.U[Sn_1[0], Sn_1[1]] + P_R * self.U[Sn_2[0], Sn_2[1]]

            sumVals.append(E)

        bestAction = [0, 1, 2, 3][np.argmax(sumVals)]  # Get the best action
        bestE = max(sumVals)  # Get the best value from the summatory

        self.P[i, j] = bestAction
        # Here the complete bellman equation
        self.U[i, j] = self.R[i, j] + GAMMA * bestE

    def randomAction(self, action):
        prob = random.uniform(0, 1)
        perpActions = getPerpActions(action)
        if prob < P_F:
            return action
        elif prob < (P_F + P_L):
            return perpActions[0]
        elif prob < (P_F + P_L + P_R):
            return perpActions[1]

    def passiveDUEAgent(self, policy, origin=(0, 0)):
        state_history = []
        reward_history = []

        # calculate the path along which it travels according to the policy

        current_state = origin
        while True:
            i, j = current_state
            current_reward = self.R[i, j]

            state_history.append(current_state)
            reward_history.append(current_reward)

            if current_state in self.dest or current_state in self.walls:
                break
            else:
                action = self.randomAction(policy[i, j])
                current_state = self.applyAction(i, j, action)

        # calculate probability accumulative based on state and reward history
        U = dict()
        for i in range(len(state_history)):
            state = state_history[i]
            accum = sum(reward_history[i:])

            if state in U:
                U[state] += [accum]
            else:
                U[state] = [accum]

        # update utility based on states visited and his probabilities
        U = {k: sum(v) / len(v) for k, v in U.items()}
        for state in U:
            i, j = state
            if self.U[i, j] == 0.0:
                self.U[i, j] = U[state]
            else:
                self.U[i, j] = (self.U[i, j] + U[state]) / 2


# walls
W = [(1, 1)]

# rewards
R = np.zeros((3, 4))
R[:] = -0.04
R[2, 3] = 1
R[1, 3] = -1

# destins
D = [(2, 3), (1, 3)]

# get optimal policy
agent1 = Agent(3, 4, W, D, R)
agent1.valueIteration()

# optimal policy
policy1 = agent1.P

# random policy
policy2 = np.array([
    [1, 2, 2, 1],
    [1, -100, 2, 1],
    [3, 3, 3, 1],
])

# random policy
policy3 = np.array([
    [3, 3, 1, 1],
    [1, -100, 1, 1],
    [3, 2, 3, 1],
])

# all policies
policies = [policy1, policy2, policy3]

for i in range(len(policies)):
    printPolicy(policies[i])

    agent2 = Agent(3, 4, W, D, R)
    for _ in range(250):
        agent2.passiveDUEAgent(policies[i])
    print(agent2.U, '\n')
