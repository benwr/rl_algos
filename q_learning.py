#!/usr/bin/env python

"""Q Learning Implementation"""
from __future__ import print_function

import random
from collections import defaultdict

import numpy as np
import gym

def softmax(values):
    """Softmax takes a 1-d array of scalars, and assigns them weights between 0 and 1"""
    greatest_x = np.max(values)
    ex = np.exp(values - greatest_x)
    return ex / ex.sum()

def random_policy(state, action_space, q_values):
    """Take a random action from the action space"""
    # pylint: disable=unused-argument
    return action_space.sample()

def epsilon_greedy_policy(state, action_space, q_values, epsilon=0.7):
    """Take a uniformly random action with probability epsilon, otherwise take the greedy action"""
    if random.uniform(0.0, 1.0) > epsilon:
        return greedy_policy(state, action_space, q_values)
    else:
        return random.randrange(action_space.n)

def softmax_policy(state, action_space, q_values, inverse_temperature=1.0):
    """Choose from actions, with probabilities softmax(q_values)"""
    sample = random.uniform(0.0, 1.0)
    action_probs = softmax(
        [q_values[state, action] * inverse_temperature for action in range(action_space.n)])
    for i, probability in enumerate(action_probs):
        sample -= probability
        if sample <= 0.0:
            return i


def greedy_policy(state, action_space, q_values):
    """Return the action with the highest q value"""
    best_action = None
    best_action_q = float("-inf")
    for action in range(action_space.n):
        q = q_values[state, action]
        if q > best_action_q:
            best_action = action
            best_action_q = q
    return best_action

def q_learning(env, exploration_policy, n_episodes, alpha=0.1, gamma=0.999):
    """Q Learning from the given exploration policy"""
    q_values = defaultdict(lambda: 0.0)
    total_returns = 0.0
    parameter = 0.01
    parameter_step = (30.0 - parameter) / n_episodes
    for episode in range(n_episodes):
        if episode % 100 == 0:
            print("Episode " + str(episode))
            print("Average reward for this set:", total_returns / 100.0)
            total_returns = 0.0
        state = env.reset()
        done = False
        while not done:
            # env.render()
            # print(state)
            action = exploration_policy(
                state, env.action_space, q_values, inverse_temperature=parameter)
            # print(action)
            next_state, reward, done, _ = env.step(action)
            total_returns += reward
            # print "received reward ", reward, " by taking action ", action
            greedy_next_action = greedy_policy(next_state, env.action_space, q_values)
            # update q:
            #   q(s, a) := q(s, a) + alpha * [(reward + gamma * max_{a'} (q(s', a'))) - q(s, a)]
            initial_q = q_values[state, action]
            q_values[state, action] = (
                initial_q
                + alpha * (reward + gamma * q_values[next_state, greedy_next_action] - initial_q))
            state = next_state

        parameter += parameter_step

        # print()
    return q_values

def execute_policy(env, policy, n_episodes, q_values):
    """Perform n_episodes rollouts of the given policy"""
    total_reward = 0.0
    for episode in range(n_episodes):
        if episode % 100 == 0:
            print ("Episode " + str(episode))
        state = env.reset()
        done = False
        while not done:
            action = policy(state, env.action_space, q_values)
            # print "taking action ", action, " in state ", state
            state, reward, done, _ = env.step(action)
            total_reward += reward
    return total_reward

def q_to_string(q_values):
    """Create printable version of defaultdict q_values"""
    q_list = sorted(list(q_values.iteritems()))
    last_state = 0
    result_strings = []
    for (state, action), q_value in q_list:
        if state != last_state:
            result_strings.append("\n")
        result_strings.append("{0}:{1:.5e}, ".format((state, action), q_value))
        last_state = state
    return "".join(result_strings)


def main():
    """Run the dang thing"""
    env = gym.make('FrozenLake-v0')
    q = q_learning(env, softmax_policy, 10000)
    print(q_to_string(q))
    final_reward = execute_policy(env, greedy_policy, 1000, q)
    print("observed reward out of 1000 trials: ", final_reward)

if __name__ == "__main__":
    main()
