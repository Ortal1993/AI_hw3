from copy import deepcopy
import random
import numpy as np


def value_iteration(mdp, U_init, epsilon=10 ** (-3)):
    # TODO:
    # Given the mdp, the initial utility of each state - U_init,
    #   and the upper limit - epsilon.
    # run the value iteration algorithm and
    # return: the U obtained at the end of the algorithms' run.
    #

    # ====== YOUR CODE: ======
    discount_gamma = mdp.gamma
    num_rows = mdp.num_row
    num_cols = mdp.num_col
    actions = mdp.actions
    rewards = mdp.board

    U_tag = np.zeros((num_rows, num_cols)) # TODO: check what to do with U_init
    for row in range(num_rows):
        for col in range(num_cols):
            if (row, col) in mdp.terminal_states and rewards[row][col] != 'WALL':
                    U_tag[row][col] = float(rewards[row][col])

    while True:
        #print(U_tag)
        U = np.copy(U_tag)
        delta = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if rewards[row][col] != 'WALL' and (row, col) not in mdp.terminal_states:
                    temp = [average_utility(mdp, (row, col), action, U) for action in actions.keys()]
                    U_tag[row][col] = (float(rewards[row][col]) + (discount_gamma * max(temp)))
                    diff = abs(U_tag[row][col] - U[row][col])
                    if diff > delta:
                        delta = diff
        if delta < (epsilon * ((1.0 - discount_gamma) / discount_gamma)):
            return U
    # ========================


def average_utility(mdp, state, action_to_evaluate, U):
    if mdp.step(state, action_to_evaluate) != state:
        probabilities = mdp.transition_function[action_to_evaluate]
        sum_pu = 0.0
        for (action, probability) in zip(mdp.actions.keys(), probabilities):
            row, col = mdp.step(state, action)
            sum_pu += probability * U[row][col]
        return sum_pu
    return 0.0


def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======3
    P = [['UP' for col in range(mdp.num_col)] for row in range(mdp.num_row)]
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            temp = [average_utility(mdp, (row, col), action, U) for action in mdp.actions.keys()]
            max = 0
            for action, u in zip(mdp.actions.keys(), temp):
                row_new, col_new = mdp.step((row, col), action)
                if u > max:
                    P[row][col] = action
                    max = u
    return P
    # ========================


def q_learning(mdp, init_state, total_episodes=10000, max_steps=999, learning_rate=0.7, epsilon=1.0,
               max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.8):
    # TODO:
    # Given the mdp and the Qlearning parameters:
    # total_episodes - number of episodes to run for the learning algorithm
    # max_steps - for each episode, the limit for the number of steps
    # learning_rate - the "learning rate" (alpha) for updating the table values
    # epsilon - the starting value of epsilon for the exploration-exploitation choosing of an action
    # max_epsilon - max value of the epsilon parameter
    # min_epsilon - min value of the epsilon parameter
    # decay_rate - exponential decay rate for exploration prob
    # init_state - the initial state to start each episode from
    # return: the Qtable learned by the algorithm
    #

    # ====== YOUR CODE: ======
    actions_string = list(mdp.actions.keys())
    qtable = np.zeros((mdp.num_row, mdp.num_col, len(mdp.actions.keys())))
    for final in mdp.terminal_states:
        for i in range(len(actions_string)):
            qtable[final[0]][final[1]][i] = float(mdp.board[final[0]][final[1]])
    for episode in range(total_episodes):
        state = init_state
        for step_i in range(max_steps):
            explore_exploit = random.uniform(0, 1)
            if explore_exploit > epsilon:  # exploit
                action = np.argmax(qtable[state[0]][state[1]])
                actionstring = actions_string[action]
            else:  # explore
                actionstring = random.sample(actions_string, 1)
                actionstring = actionstring[0]
                action = actions_string.index(actionstring)
            new_state, reward = take_step(mdp, state, actionstring)
            qtable[state[0]][state[1]][action] = ((1 - learning_rate) * qtable[state[0]][state[1]][action]) + \
                                                 learning_rate * (reward + (
                        mdp.gamma * np.max(qtable[new_state[0]][new_state[1]])))
            state = new_state
            if new_state in mdp.terminal_states:
                break
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp((-decay_rate) * episode)
    return qtable
    # ========================


def take_step(mdp, state, action):
    # supposed to be fine
    action = random.choices(population=list(mdp.actions.keys()), weights=mdp.transition_function[action])
    action = action[0]
    row, col = mdp.step(state, action)
    return (row, col), float(mdp.board[state[0]][state[1]])


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    # supposed to be fine
    P = [['UP' for col in range(mdp.num_col)] for row in range(mdp.num_row)]
    actions_string = list(mdp.actions.keys())
    for row in range(mdp.num_row):
        for col in range(mdp.num_col):
            index_action = np.argmax(qtable[row][col])
            P[row][col] = actions_string[index_action]
    return P
    # ========================


# BONUS

def policy_evaluation(mdp, policy):
    # TODO:
    # Given the mdp, and a policy
    # return: the utility U(s) of each state s
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================


def policy_iteration(mdp, policy_init):
    # TODO:
    # Given the mdp, and the initial policy - policy_init
    # run the policy iteration algorithm
    # return: the optimal policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
    # ========================
