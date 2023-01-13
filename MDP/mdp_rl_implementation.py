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
    U_tag = np.zeros((num_rows, num_cols))

    while True:
        U = np.copy(U_tag)
        delta = 0
        for row in range(num_rows):
            for col in range(num_cols):
                if rewards[row][col] != 'WALL':
                    temp = [average_utility(mdp, (row, col), action, U) for action in actions.keys()]
                    U_tag[row][col] = float(rewards[row][col]) + discount_gamma * max(temp)
                    diff = abs(U_tag[row][col] - U[row][col])
                    if diff > delta:
                        delta = diff
        if delta < epsilon * ((1 - discount_gamma) / discount_gamma):
            return U
    # ========================

def average_utility(mdp, state, action_to_evaluate, U):
    probabilities = mdp.transition_function[action_to_evaluate]
    sum = 0
    for (action, probability) in zip(mdp.actions.keys(), probabilities):
        new_state = mdp.step(state, action)
        sum += probability * U[new_state[0]][new_state[1]]
    return sum

def get_policy(mdp, U):
    # TODO:
    # Given the mdp and the utility of each state - U (which satisfies the Belman equation)
    # return: the policy
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
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
    raise NotImplementedError
    # ========================


def q_table_policy_extraction(mdp, qtable):
    # TODO:
    # Given the mdp and the Qtable:
    # return: the policy corresponding to the Qtable
    #

    # ====== YOUR CODE: ======
    raise NotImplementedError
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

