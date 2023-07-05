import gym
import time
import numpy as np
import matplotlib.pyplot as plt
import copy
import random
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches


############################### env method : you don't need to know them


def modify_rewards(next_state, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    state_type = custom_map_flaten[next_state]

    if state_type == "H":
        return hole_reward  # Decrease the reward for falling into a hole
    elif state_type == "G":
        return goal_reward  # Increase the reward for reaching the goal
    else:
        return move_reward  # Decrease the reward for moving


def modify_rewards_P(envP, custom_map, hole_reward, goal_reward, move_reward):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    # old_envP = copy.deepcopy(envP)
    old_envP = copy.copy(envP)

    new_envP = {}
    for state, v1 in old_envP.items():
        inner_dict = {}
        for action, v2 in v1.items():
            inner_list = []
            for tpl in v2:
                (prob_of_transition, s_prime, old_reward, terminated) = tpl
                if custom_map_flaten[s_prime] == "H":
                    new_reward = (
                        hole_reward  # Decrease the reward for falling into a hole
                    )
                elif custom_map_flaten[s_prime] == "G":
                    new_reward = (
                        goal_reward  # Increase the reward for reaching the goal
                    )
                else:
                    new_reward = move_reward  # Decrease the reward for movin
                inner_list.append((prob_of_transition, s_prime, new_reward, terminated))
            inner_dict[action] = inner_list
        new_envP[state] = inner_dict

    return new_envP


class ModifyRewards(gym.Wrapper):
    def __init__(
            self, env, custom_map, hole_reward=-10, goal_reward=10, move_reward=-0.1
    ):
        super().__init__(env)
        self.hole_reward = hole_reward
        self.goal_reward = goal_reward
        self.move_reward = move_reward
        self.custom_map = custom_map
        self.P = modify_rewards_P(
            env.P, custom_map, hole_reward, goal_reward, move_reward
        )

    def step(self, action):
        next_state, reward, done, truncated, info = self.env.step(action)
        modified_reward = modify_rewards(
            next_state,
            self.custom_map,
            self.hole_reward,
            self.goal_reward,
            self.move_reward,
        )
        return next_state, modified_reward, done, truncated, info


############################### plot methods : you can use them to plot 
# your policy and state value


#  plot policy with arrows in four direction to understand policy better
def plot_policy_arrows(policy, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    n = len(custom_map)
    m = len(custom_map[0])
    fig, ax = plt.subplots(n, m, figsize=(8, 8))
    for i in range(n):
        for j in range(m):
            ax[i, j].set_xlim([0, 3])
            ax[i, j].set_ylim([0, 3])
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
    for state, subdict in policy.items():
        row = state // m
        col = state % m
        if custom_map_flaten[state] == "S":
            square_fill = plt.Rectangle(
                (0.5, 0.5), 2, 2, linewidth=0, edgecolor=None, facecolor="y", alpha=0.5
            )
            ax[row, col].add_patch(square_fill)
        for direction, value in subdict.items():
            dx, dy = 0, 0
            if direction == 0:
                dx = -value
            elif direction == 1:
                dy = -value
            elif direction == 2:
                dx = value
            else:
                dy = value
            if value != 0:
                ax[row, col].arrow(1.5, 1.5, dx, dy, head_width=0.35, head_length=0.25)
        if subdict[0] == 0 and subdict[1] == 0 and subdict[2] == 0 and subdict[3] == 0:
            if custom_map_flaten[state] == "G":
                color = "g"
            else:
                color = "r"
            square_fill = plt.Rectangle(
                (0.5, 0.5),
                2,
                2,
                linewidth=0,
                edgecolor=None,
                facecolor=color,
                alpha=0.5,
            )
            ax[row, col].add_patch(square_fill)
    plt.show()


# plot policy in terminal using best action for each state
def plot_policy_terminal(policy, custom_map):
    arr = np.empty((len(custom_map), len(custom_map[0])), dtype=object)
    state = 0
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            subdict = policy[state]
            best_action = max(subdict, key=subdict.get)
            if best_action == 0:
                arr[i, j] = "Lt"  # 0: LEFT
            elif best_action == 1:
                arr[i, j] = "Dn"  # 1: DOWN
            elif best_action == 2:
                arr[i, j] = "Rt"  # 2: RIGHT
            elif best_action == 3:
                arr[i, j] = "UP"  # 3: UP
            else:
                arr[i, j] = "##"
            state += 1
    print(arr)


# plot state value
def plot_state_value(state_value, custom_map):
    custom_map_flaten = get_flaten_custom_map(custom_map)
    rows = len(custom_map)
    cols = len(custom_map[0])
    table = state_value.reshape(rows, cols)
    # Define custom colors
    green = mcolors.to_rgba("green", alpha=0.5)
    blue = mcolors.to_rgba("blue", alpha=0.5)
    fig, ax = plt.subplots()
    im = ax.imshow(table, cmap="Reds")
    state = 0
    for i in range(rows):
        for j in range(cols):
            if custom_map_flaten[state] == "H":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0.1,
                        facecolor=blue,
                    )
                )
            elif custom_map_flaten[state] == "G":
                ax.add_patch(
                    mpatches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        linewidth=0,
                        facecolor=green,
                    )
                )

            ax.text(
                j,
                i,
                str(i * cols + j) + "\n" + custom_map_flaten[state],
                ha="center",
                va="center",
            )
            state += 1
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels([""] * cols)
    ax.set_yticklabels([""] * rows)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")
    ax.set_title("state value")
    plt.show()


############################### handler methods : you don't need to know them,
# they have been used in other methods


def act_wrt_prob(probability):
    if random.random() < probability:
        return 1
    else:
        return 0


def get_action_wrt_policy(state, policy):
    action = -1
    while action == -1:
        if act_wrt_prob(policy[state][0]) == 1:
            action = 0
        elif act_wrt_prob(policy[state][1]) == 1:
            action = 1
        elif act_wrt_prob(policy[state][2]) == 1:
            action = 2
        elif act_wrt_prob(policy[state][3]) == 1:
            action = 3
    return action


def get_flaten_custom_map(custom_map):
    custom_map_flaten = []
    for row in custom_map:
        for char in row:
            custom_map_flaten.append(char)
    return custom_map_flaten


############################### helper methods : you can use them in your code to create
# random policy and check your policy


# it gives a randome walk policy w.r.t costum 
def get_init_policy(custom_map):
    policy = {}
    random_sub_dict = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":

                policy[state] = terminal_sub_dict
            else:
                policy[state] = random_sub_dict

    return policy


# it gives walk policy according to direction w.r.t costum
def get_policy_direction(direction, custom_map):  # direction :"left", "down", "right"
    policy = {}
    left_sub_dict = {0: 1.0, 1: 0.0, 2: 0.0, 3: 0.0}
    down_sub_dict = {0: 0.0, 1: 1.0, 2: 0.0, 3: 0.0}
    right_sub_dict = {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}
    terminal_sub_dict = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    for i in range(len(custom_map)):
        for j in range(len(custom_map[i])):
            state = (i * len(custom_map[i])) + j
            if custom_map[i][j] == "H" or custom_map[i][j] == "G":

                policy[state] = terminal_sub_dict
            else:
                if direction == "left":
                    policy[state] = left_sub_dict
                elif direction == "down":
                    policy[state] = down_sub_dict
                elif direction == "right":
                    policy[state] = right_sub_dict

    return policy


# it run game according to given policy
def do_policy(env, policy, episdoes=5):
    # episdoes = 10
    for ep in range(episdoes):
        n_state = env.reset()[0]
        done = False
        rewards = 0
        moves = 0
        while done is False:
            action = get_action_wrt_policy(n_state, policy)
            n_state, reward, done, truncated, info = env.step(action)
            rewards += reward
            moves += 1
        print("rewards:", rewards, " - moves:", moves, " - final state:", n_state)
    env.render()


############################### algorithm methods : you have to implement these algorithms


# Implement policy iteration using the Policy Evaluation and Policy Improvement steps. In the Policy Evaluation step, you compute
# the state values for each state in the environment using the Bellman equation and the current policy. In the Policy Improvement
# step, you improve the policy by choosing the action that maximizes the value function for each state.


# *****************************************************************************
# def policy_iteration(env, custom_map, max_ittr=30, theta=0.01, discount_factor=0.9):
#     policy = get_init_policy(custom_map)   # it gives a random-walk policy
#     V = np.zeros(env.observation_space.n)  # you can change it with any init value
#     P = env.P
#     # This attribute stores the transition probabilities
#                                            # and rewards for each possible action in each possible
#                                            # state of the environment
#
#     # loop till policy_stable becomes True or itter >= max_ittr
#     ittr = 0
#     policy_stable = False
#     while not policy_stable and ittr < max_ittr:
#         # policy evaluation
#         flag = True
#         while flag:
#             delta = 0
#             for s in range(env.observation_space.n):
#                 v = 0
#                 for a, action_prob in enumerate(policy[s]):
#                     # ns, r, d, ti, ij
#                     for next_state_prob, r, d, ti, ij in P[s][a]:
#                         v += action_prob * next_state_prob * (r + discount_factor * V[next_state])
#
#                 delta = max(delta, abs(v - V[s]))
#                 V[s] = v
#
#             if delta < theta:
#                 flag = False
#
#
#         # policy improvement
#         for s in range(env.observation_space.n):
#             q_values = np.zeros(env.observation_space.n)
#
#             for a in range(env.observation_space.n):
#                 for next_state_prob, r in env.P[s][a]:
#                     q_values[a] += next_state_prob * (r + discount_factor * V[next_state])
#
#             best_a = np.argmax(q_values)
#             policy[s, best_a] = 1
#
#
#         ittr += 1
#     return V, policy
# *****************************************************************************

def format_policy(policy):
    formatted_policy = {}
    for state, action_probs in policy.items():
        formatted_action_probs = {}
        for action, prob in enumerate(action_probs):
            formatted_action_probs[action] = prob
        formatted_policy[state] = formatted_action_probs
    return formatted_policy


def policy_iteration(env, custom_map, max_ittr=30, theta=0.01, discount_factor=0.9):
    policy = get_init_policy(custom_map)
    V = np.zeros(env.observation_space.n)
    P = env.P

    ittr = 0
    policy_stable = False
    while not policy_stable and ittr < max_ittr:
        # evalution
        while True:
            delta = 0
            for s in range(env.observation_space.n):
                v_old = V[s]
                V[s] = sum(policy[s][a] * sum(p * (r + discount_factor * V[s_]) for p, s_, r, done in P[s][a]) for a in
                           range(env.action_space.n))
                # print(V[s])
                delta = max(delta, abs(v_old - V[s]))
                # print(delta)
            if delta < theta:
                break
        # improvment
        policy_stable = True
        for s in range(env.observation_space.n):
            old_action = np.argmax(policy[s])
            # print(old_action)
            new_action_values = [sum(p * (r + discount_factor * V[s_]) for p, s_, r, done in P[s][a]) for a in
                                 range(env.action_space.n)]
            new_action = np.argmax(new_action_values)
            # print(new_action)
            if old_action != new_action:
                policy_stable = False
            policy[s] = np.eye(env.action_space.n)[new_action]

        ittr += 1
    # print("original policy is :")
    # print(policy)
    # p = format_policy(policy)
    # return V, p
    p=convert_policy(env,policy)
    return V,p


#
def convert_policy(env, policy):
    p = {}
    for s in range(env.observation_space.n):
        arr_list = policy[s]
        arr_dict = {i: arr_list[i] for i in range(len(arr_list))}
        p[s] = arr_dict
        return p


# This algorithm allows you to estimate the state values of a given policy by sampling episodes and
# calculating the average returns(in first visit of a state in each episode)
# def first_visit_mc_prediction(env, policy, num_episodes, gamma):
# initilize
# V = np.zeros(env.observation_space.n)
# N = np.zeros(env.observation_space.n)

# loop in range num_episodes(for each episode)
# for i_episode in range(num_episodes):

# generate episode w.r.t policy

# loop for each step of episode , t= T-1, T-2, ..., 0

# return V


def first_visit_mc_prediction(env, policy, num_episodes, gamma):
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)

    for i_episode in range(num_episodes):
        episode = []
        state = env.reset()
        done = False

        # print(type(state))
        # print(state[0])
        # print(policy)

        while not done:
            # print(policy[0])
            # print(len(ps[0]))
            # print(policy[state[0]])
            # print(np.arange(len(policy[state])))
            # action = np.random.choice(np.arange(len(policy[state[0]])), p=policy[state[0]])

            # action = policy[state[0]][random.randint(0, len(policy[state[0]]) - 1)]
            # if state[0] in policy.keys():

            if (type(state) == tuple):
                keys = list(policy[state[0]].keys())
                values = list(policy[state[0]].values())
            else:
                keys = list(policy[state].keys())
                values = list(policy[state].values())

            # print(keys)
            # print(values)
            action = random.choices(keys, weights=values)[0]
            # print(env.step(action))
            # print("--------------------------------")
            next_state, reward, done, truncated, info = env.step(action)
            # print(jj)
            # print(info)
            # print(done)

            if reward == env.goal_reward:
                print("goal")
                break
                # return V
            episode.append((state, action, reward))
            state = next_state

        visited_states = set()
        for t in range(len(episode)):
            state, _, _ = episode[t]
            if type(state) == tuple:
                if state[0] not in visited_states:
                    # print("******")
                    visited_states.add(state[0])
                    G = 0
                    for k in range(t, len(episode)):
                        _, _, reward = episode[k]
                        G += gamma ** (k - t) * reward
                    N[state[0]] += 1
                    V[state[0]] += (1 / N[state[0]]) * (G - V[state[0]])
            else:
                if state not in visited_states:
                    # print("----")
                    visited_states.add(state)
                    G = 0
                    for k in range(t, len(episode)):
                        _, _, reward = episode[k]
                        G += gamma ** (k - t) * reward
                    N[state] += 1
                    V[state] += (1 / N[state]) * (G - V[state])
    return V


# def first_visit_mc_prediction(env, policy, num_episodes, gamma):
#     # initialize
#     V = np.zeros(env.observation_space.n)
#     N = np.zeros(env.observation_space.n)
#
#     for i_episode in range(num_episodes):
#         # generate episode w.r.t policy
#         episode = []
#         state = env.reset()
#         d = False
#         while not d:
#             print(type(policy))
#             a = policy[state]
#             state = 0  # replace with the actual current state
#             action_probs = policy[state]
#             a = random.choices(list(action_probs.keys()), weights=list(action_probs.values()))[0]
#             ns, r, d, ti, ij = env.step(a)
#             # ns, r, d =
#             episode.append((state, a, r))
#             state = ns
#
#         # loop for each step of episode, t = T-1, T-2, ..., 0
#         g = 0
#         visited_states = set()
#
#         for ti in range(len(episode) - 1, -1, -1):
#             state, act, r = episode[ti]
#             g = gamma * g + r
#
#             if state not in visited_states:
#                 visited_states.add(state)
#                 N[state] += 1
#                 V[state] += (g - V[state]) / N[state]
#
#
#         if d:
#             print("rewards")
#             break
#
#     return V


# This algorithm allows you to estimate the state values of a given policy by sampling episodes and
# calculating the average returns(in every visit of a state)


def every_visit_mc_prediction(env, policy, num_episodes, gamma):
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)

    for i_episode in range(num_episodes):
        episode = []
        state = env.reset()
        d = False
        while not d:
            if (type(state) == tuple):
                keys = list(policy[state[0]].keys())
                values = list(policy[state[0]].values())
            else:
                keys = list(policy[state].keys())
                values = list(policy[state].values())

            action = random.choices(keys, weights=values)[0]
            ns, r, d, ti, ij = env.step(action)  # next_state, modified_reward, done, truncated, info
            episode.append((state, action, r))
            state = ns
            # print("_________")
            # print(f"done is {d}")
            # print(f"reward is {r}")
            # print("_________")
            if r == env.goal_reward:
                print("goal")
                # return V
                break

        g = 0
        visited_states = set()

        for ti in range(len(episode) - 1, -1, -1):
            state, act, r = episode[ti]
            g = gamma * g + r
            if type(state) == tuple:
                if state[0] not in visited_states:
                    visited_states.add(state[0])
                    N[state[0]] += 1
                    alpha = 1 / N[state[0]]
                    V[state[0]] += alpha * (g - V[state[0]])
            else:
                if state not in visited_states:
                    visited_states.add(state)
                    N[state] += 1
                    alpha = 1 / N[state]
                    V[state] += alpha * (g - V[state])

    return V


############################### custom maps : you have to use them according to the problem

custom_map_1 = ["HFSFFFFG"]

custom_map_2 = ["SFFFF",
                "HHHFF",
                "FFFFH",
                "FFFFF",
                "FFFFG"]

custom_map_3 = ["SFFFF",
                "HFFFF",
                "HFFFF",
                "HFFFF",
                "GFFFF"]

custom_map_4 = ["FFFSFFF",
                "FHHHHFF",
                "FFFFFFF",
                "HFFFFFF",
                "FGFFFFF"]

custom_map_5 = ["HFSFFFFG"]

custom_map_6 = ["HFSFFFFG",
                "HFFFFFFF",
                "HFFFFFFF"]

custom_map_7 = ["SFFFF",
                "FFFFH",
                "HHFFF",
                "HFFFH",
                "FFFFG"]

custom_map_8 = ["HFFSFFH",
                "FFFFFFF",
                "FFFFFFF",
                "GFFHFFG"]
#############################
if __name__ == "__main__":
    map = custom_map_7
    env = gym.make("FrozenLake-v1", render_mode="human", desc=map, is_slippery=True)
    # env = gym.make("FrozenLake-v1", render_mode="human", desc=map, is_slippery=False)
    # env = gym.make("FrozenLake-v1", desc=map, is_slippery=True)
    env = ModifyRewards(
        # env, custom_map=map, hole_reward=0, goal_reward=1, move_reward=-0.2
        # env, custom_map=map,hole_reward=-0.1, goal_reward=1, move_reward=-0.1
        # env, custom_map=map, hole_reward=-4, goal_reward=10, move_reward=-0.9
        # env, custom_map=map, hole_reward=-5, goal_reward=5, move_reward=-0.5
        env, custom_map=map,hole_reward=-3, goal_reward=7, move_reward=-2
        # env, custom_map=map, hole_reward=-2, goal_reward=50, move_reward=-1
    )
    env.reset()
    env.render()
    ###
    policy = get_init_policy(map)

    # plot_policy_arrows(policy, map)
    # do_policy(env, policy)

    # print(env.P[0])
    # print(len(env.P[0]))
    # print(env.P)
    # print(len(env.P))
    # print(policy[0])
    # print(len(policy[0]))
    # print(policy)
    # print(len(policy))
    # print(env.action_space)
    #
    #
    # print("******************************************************")
    # for s in range(env.observation_space.n):
    #     print(s)
    #     for action in range(env.action_space.n):
    #     # for action in env.P[s]:
    #
    #         for m in env.P[s][action]:
    #             print("(((((((((((((((((((((((((((")
    #             print(m)
    #             print(m[0])
    #             print(m[1])
    #             print(m[2])
    #     print("-------------------------------")

    # rewards = 0
    # for t in range(100):
    #     action = env.action_space.sample()
    #     next_state, reward, done, truncated, info = env.step(action)
    #     rewards += reward

    # action = 2
    # next_state, reward, done, truncated, info = env.step(action)
    # rewards += reward
    # action = 1
    # next_state, reward, done, truncated, info = env.step(action)
    # rewards += reward
    # action = 2
    # next_state, reward, done, truncated, info = env.step(action)
    # rewards += reward
    # if done:
    #     print(rewards)
    #     break

    # V, policy = policy_iteration(env, map, theta=0.0001, discount_factor=0.9)
    # print(policy)
    # V, policy = policy_iteration(env, map, theta=0.0001, discount_factor=0.9)
    # plot_state_value(V, map)
    # plot_policy_arrows(policy, map)
    # plot_policy_ter/minal(policy, map)
    # do_policy(env, policy, episdoes=5)

    num_episodes = 10
    gamma = 0.9
    # print(get_policy_direction("left",map))
    #
    V_MC = first_visit_mc_prediction(env, get_policy_direction("right", map), num_episodes, gamma)
    # V_MC = first_visit_mc_prediction(env, policy, num_episodes, gamma)
    # V_MC = every_visit_mc_prediction(env, policy, num_episodes, gamma)
    # V_MC = every_visit_mc_prediction(env, get_policy_direction("down", map), num_episodes, gamma)
    plot_state_value(V_MC, map)
    # end=False
    # while not end:
    #     newpol=
    time.sleep(2)
