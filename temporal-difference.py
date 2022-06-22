from email import policy
import numpy as np
import random


# Gets the perpendicular actions of the given action
def getPerpActions(action):
    if action == UP or action == DOWN:
        return [LEFT, RIGHT]
    return [UP, DOWN]


# Actions
UP, DOWN, LEFT, RIGHT = range(4)

P_F = 0.8
P_L = 0.1
P_R = 0.1

GAMMA = 0.9


class Agent:
    def __init__(self, height, width, walls, dest, R):
        self.height = height
        self.width = width
        self.walls = walls
        self.dest = dest
        self.R = R
        self.P = np.zeros((height, width))
        self.P[:] = -100  # -100 value for missing null policy
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
        for wallPos in walls:
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

    def valueIteration(self):
        for i in range(self.height):
            for j in range(self.width):
                self.update_state(i, j)

    def update_state(self, i, j):
        #  Get the actions available at the current state
        # actions = self.actionsAt(i, j)

        #  Get the valid actions at the current state
        validActions = self.validActions(i, j)

        # Can't update destination state because it is the end of the path
        # if i == dest[0] and j ==dest[1]:
        #     return None

        # Can't update walls states, you shouldn't get here
        if i == walls[0][0] and j == walls[0][1]:
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

    def printPolicy(self):
        for i in range(self.height):
            for j in range(self.width):
                character = ''
                if self.P[i, j] == UP:
                    character = 'U'
                elif self.P[i, j] == DOWN:
                    character = 'D'
                elif self.P[i, j] == LEFT:
                    character = 'L'
                elif self.P[i, j] == RIGHT:
                    character = 'R'
                else:
                    character = '*'
                print(character, end=" ")
            print()

    def randomAction(self, action):
        prob = random.uniform(0, 1)
        perpActions = getPerpActions(action)
        if prob < P_F:
            return action
        elif prob < (P_F + P_L):
            return perpActions[0]
        elif prob < (P_F + P_L + P_R):
            return perpActions[1]

    def emulate(self, origin):
        T_R = 0
        curr = origin
        while not(curr[0] == self.dest[0] and curr[1] == self.dest[1]):
            policyAct = self.P[curr[0], curr[1]]
            act = self.randomAction(policyAct)
            validActions = self.validActions(curr[0], curr[1])
            if act in validActions:
                act = int(act)
                if act == 0:
                    curr = [curr[0] - 1, curr[1]]
                elif act == 1:
                    curr = [curr[0] + 1, curr[1]]
                elif act == 2:
                    curr = [curr[0], curr[1] - 1]
                elif act == 3:
                    curr = [curr[0], curr[1] + 1]

            T_R = T_R + self.R[curr[0], curr[1]]
        return T_R

    def matrix_convergence(self, U, NU, conv_error=0.01):

        if np.count_nonzero(np.isnan(U)) == (U.shape[0] * U.shape[1]) or np.count_nonzero(np.isnan(NU)) == (U.shape[0] * U.shape[1]):
            return False

        U_nan = np.argwhere(np.isnan(U))
        U_nan = np.sort(U_nan)
        NU_nan = np.argwhere(np.isnan(U))
        NU_nan = np.sort(U_nan)

        equals_nan = (U_nan == NU_nan).all()

        U_c = np.nan_to_num(U, nan=0)
        NU_c = np.nan_to_num(NU, nan=0)

        equals_vals = abs(np.sum(U_c - NU_c)) < conv_error

        print(equals_vals)
        print(equals_nan)
        return equals_nan and equals_vals

    # The main function for temporal difference
    # Assumes you already have the policy P

    def temporalDifference(self, policy, sn, iterations):
        self.nonenum = np.nan
        self.U = np.zeros((self.height, self.width))
        self.U[:] = self.nonenum
        self.Ns = np.zeros((self.height, self.width))
        self.alpha = 0.1
        self.P = policy

        self.s = None
        self.a = None
        self.r = None

        for i in range(iterations):
            sn = self.passiveTDAgent(sn)

    def passiveTDAgent(self, sn):
        s = self.s
        rn = self.R[sn]

        if np.isnan(self.U[sn]):
            self.U[sn] = rn

        if s is not None:
            self.Ns[s] = self.Ns[s] + 1

            alpha_func = 60/(59 + self.Ns[s])  # From the book

            self.U[s] = self.U[s] + self.alpha * \
                (alpha_func) * (self.r + GAMMA *
                                self.U[sn] - self.U[s])  # Update utilities

        if sn[0] == self.dest[0] and sn[1] == self.dest[1]:
            self.a = None
            self.s = None
            self.r = None
            return (0, 0)
        else:
            self.s = sn
            self.a = self.P[self.s]
            self.r = rn

        act = self.randomAction(self.a)
        curr = sn
        validActions = self.validActions(sn[0], sn[1])
        if act in validActions:
            act = int(act)
            if act == 0:
                curr = [curr[0] - 1, curr[1]]
            elif act == 1:
                curr = [curr[0] + 1, curr[1]]
            elif act == 2:
                curr = [curr[0], curr[1] - 1]
            elif act == 3:
                curr = [curr[0], curr[1] + 1]

        return (curr[0], curr[1])


walls = [[1, 1]]


# Rewards
R = np.zeros((3, 4))

# Set rewards values
R[:] = -0.04
R[2, 3] = 1
R[1, 3] = -1

dest = [2, 3]

agent = Agent(3, 4, walls, dest, R)


for i in range(250):
    agent.valueIteration()

# optimal policy
policy1 = np.copy(agent.P)

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
    agent.temporalDifference(policies[i], (0, 0), 1000)
    agent.printPolicy()
    print(agent.U, '\n')
