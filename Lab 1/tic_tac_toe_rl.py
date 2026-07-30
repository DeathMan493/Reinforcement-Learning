import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


EMPTY = 0
X = 1      # Agent
O = -1     # Environment / opponent

CELL_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


# Hyperparameters
TOTAL_EPISODES = 20000
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.0003


def empty_board():
    return (EMPTY,) * 9


def available_actions(state):
    return [index for index, value in enumerate(state) if value == EMPTY]


def make_move(state, action, player):
    board = list(state)
    board[action] = player
    return tuple(board)


def get_winner(state):
    for a, b, c in WIN_LINES:
        line_sum = state[a] + state[b] + state[c]
        if line_sum == 3:
            return X
        if line_sum == -3:
            return O
    return None


def is_draw(state):
    return get_winner(state) is None and not available_actions(state)


def is_terminal(state):
    return get_winner(state) is not None or is_draw(state)


def reward_for(state):
    winner = get_winner(state)
    if winner == X:
        return 1
    if winner == O:
        return -1
    return 0


def choose_action(q_table, state, epsilon):
    actions = available_actions(state)
    if random.random() < epsilon:
        return random.choice(actions)

    action_values = q_table[state]
    best_value = max(action_values[action] for action in actions)
    best_actions = [action for action in actions if action_values[action] == best_value]
    return random.choice(best_actions)


def opponent_action(state):
    return random.choice(available_actions(state))


def update_q_value(q_table, state, action, reward, next_state):
    old_value = q_table[state][action]

    if is_terminal(next_state):
        future_value = 0
    else:
        next_actions = available_actions(next_state)
        future_value = max(q_table[next_state][next_action] for next_action in next_actions)

    q_table[state][action] = old_value + LEARNING_RATE * (
        reward + DISCOUNT_FACTOR * future_value - old_value
    )


def train_agent():
    q_table = defaultdict(lambda: np.zeros(9))
    episode_rewards = []
    epsilon = MAX_EPSILON

    print("Training Tic-Tac-Toe Q-learning agent...")
    for episode in range(TOTAL_EPISODES):
        state = empty_board()
        total_reward = 0
        done = False

        while not done:
            action = choose_action(q_table, state, epsilon)
            after_agent_state = make_move(state, action, X)
            reward = reward_for(after_agent_state)

            if is_terminal(after_agent_state):
                update_q_value(q_table, state, action, reward, after_agent_state)
                total_reward += reward
                break

            after_opponent_state = make_move(after_agent_state, opponent_action(after_agent_state), O)
            reward = reward_for(after_opponent_state)
            update_q_value(q_table, state, action, reward, after_opponent_state)

            total_reward += reward
            state = after_opponent_state
            done = is_terminal(state)

        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)
        episode_rewards.append(total_reward)

    print("Training finished!\n")
    return q_table, episode_rewards


def evaluate_agent(q_table, games=1000):
    results = {"X wins": 0, "O wins": 0, "Draws": 0}

    for _ in range(games):
        state = empty_board()

        while not is_terminal(state):
            action = choose_action(q_table, state, epsilon=0.0)
            state = make_move(state, action, X)

            if is_terminal(state):
                break

            state = make_move(state, opponent_action(state), O)

        winner = get_winner(state)
        if winner == X:
            results["X wins"] += 1
        elif winner == O:
            results["O wins"] += 1
        else:
            results["Draws"] += 1

    return results


def print_board(state):
    symbols = {X: "X", O: "O", EMPTY: "-"}
    cells = [symbols[value] if value != EMPTY else CELL_LABELS[index] for index, value in enumerate(state)]
    print(f"{cells[0]} | {cells[1]} | {cells[2]}")
    print("--+---+--")
    print(f"{cells[3]} | {cells[4]} | {cells[5]}")
    print("--+---+--")
    print(f"{cells[6]} | {cells[7]} | {cells[8]}")


def play_demo_game(q_table):
    state = empty_board()
    print("Demo game: trained X agent vs random O opponent")
    print_board(state)

    while not is_terminal(state):
        action = choose_action(q_table, state, epsilon=0.0)
        state = make_move(state, action, X)
        print(f"\nAgent X chooses cell {action + 1}")
        print_board(state)

        if is_terminal(state):
            break

        action = opponent_action(state)
        state = make_move(state, action, O)
        print(f"\nOpponent O chooses cell {action + 1}")
        print_board(state)

    reward = reward_for(state)
    if reward == 1:
        print("\nResult: X wins (+1 reward)")
    elif reward == -1:
        print("\nResult: O wins (-1 reward)")
    else:
        print("\nResult: draw (0 reward)")


def plot_learning_curve(episode_rewards):
    window = 200
    moving_average = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Reward per Episode", color="steelblue", alpha=0.25)
    plt.plot(moving_average, label=f"{window}-Episode Moving Average", color="crimson")
    plt.title("Tic-Tac-Toe Q-Learning Training Progress")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    q_table, episode_rewards = train_agent()

    results = evaluate_agent(q_table)
    print("Evaluation against random opponent:")
    for label, count in results.items():
        print(f"{label}: {count}")

    print()
    play_demo_game(q_table)
    plot_learning_curve(episode_rewards)


if __name__ == "__main__":
    main()
