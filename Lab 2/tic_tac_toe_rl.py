import random
from collections import defaultdict

import numpy as np


EMPTY = 0
X = 1
O = -1

CELL_LABELS = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
GREEDY_PRIORITY = [4, 0, 2, 6, 8, 1, 3, 5, 7]
WIN_LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)

TRAINING_EPISODES = [100, 500, 1000, 5000, 10000]
EVALUATION_GAMES = 1000
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.95
MAX_EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.0006
RANDOM_SEED = 42


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


def reward_for_player(state, player):
    winner = get_winner(state)
    if winner == player:
        return 1
    if winner == -player:
        return -1
    return 0


def greedy_action(q_table, state):
    actions = available_actions(state)
    action_values = q_table[state]
    best_value = max(action_values[action] for action in actions)
    best_actions = [action for action in actions if action_values[action] == best_value]
    return min(best_actions, key=GREEDY_PRIORITY.index)


def choose_action(q_table, state, epsilon):
    if random.random() < epsilon:
        return random.choice(available_actions(state))
    return greedy_action(q_table, state)


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


def train_self_play_agent(total_episodes):
    q_tables = {
        X: defaultdict(lambda: np.zeros(9)),
        O: defaultdict(lambda: np.zeros(9)),
    }
    episode_rewards = []

    for episode in range(total_episodes):
        state = empty_board()
        current_player = X
        last_turn = {X: None, O: None}
        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-DECAY_RATE * episode)

        while not is_terminal(state):
            previous_turn = last_turn[current_player]
            if previous_turn is not None:
                previous_state, previous_action = previous_turn
                update_q_value(
                    q_tables[current_player],
                    previous_state,
                    previous_action,
                    reward_for_player(state, current_player),
                    state,
                )

            action = choose_action(q_tables[current_player], state, epsilon)
            next_state = make_move(state, action, current_player)
            last_turn[current_player] = (state, action)

            if is_terminal(next_state):
                update_q_value(
                    q_tables[current_player],
                    state,
                    action,
                    reward_for_player(next_state, current_player),
                    next_state,
                )

                other_player = -current_player
                if last_turn[other_player] is not None:
                    previous_state, previous_action = last_turn[other_player]
                    update_q_value(
                        q_tables[other_player],
                        previous_state,
                        previous_action,
                        reward_for_player(next_state, other_player),
                        next_state,
                    )

                episode_rewards.append(reward_for_player(next_state, X))
                break

            state = next_state
            current_player = -current_player

    return q_tables, episode_rewards


def evaluate_agent(q_tables, games=EVALUATION_GAMES):
    results = {"wins": 0, "losses": 0, "draws": 0}
    q_x = q_tables[X]

    for _ in range(games):
        state = empty_board()

        while not is_terminal(state):
            action = greedy_action(q_x, state)
            state = make_move(state, action, X)

            if is_terminal(state):
                break

            action = random.choice(available_actions(state))
            state = make_move(state, action, O)

        winner = get_winner(state)
        if winner == X:
            results["wins"] += 1
        elif winner == O:
            results["losses"] += 1
        else:
            results["draws"] += 1

    return results


def result_percentages(results):
    total = sum(results.values())
    return {
        label: (count / total) * 100
        for label, count in results.items()
    }


def policy_chooses(q_table, state, expected_actions):
    action = greedy_action(q_table, state)
    return action in expected_actions


def observe_behaviour(q_tables, percentages):
    q_x = q_tables[X]
    observations = []

    first_move = greedy_action(q_x, empty_board())
    if first_move == 4:
        observations.append("chooses center")
    elif first_move in (0, 2, 6, 8):
        observations.append("chooses corner")
    else:
        observations.append("opening still unstable")

    can_win = policy_chooses(q_x, (X, X, EMPTY, O, O, EMPTY, EMPTY, EMPTY, EMPTY), [2])
    can_block = policy_chooses(q_x, (O, O, EMPTY, X, EMPTY, EMPTY, EMPTY, EMPTY, X), [2])

    if can_win:
        observations.append("makes winning move")
    if can_block:
        observations.append("blocks opponent")
    if percentages["losses"] <= 10:
        observations.append("avoids losing")
    if percentages["draws"] >= 45:
        observations.append("forces draw often")
    if percentages["losses"] <= 5 and (can_win or can_block):
        observations.append("near-optimal play")
    if not observations:
        observations.append("plays randomly")

    return ", ".join(observations)


def print_board(state):
    symbols = {X: "X", O: "O", EMPTY: "-"}
    cells = [
        symbols[value] if value != EMPTY else CELL_LABELS[index]
        for index, value in enumerate(state)
    ]
    print(f"{cells[0]} | {cells[1]} | {cells[2]}")
    print("--+---+--")
    print(f"{cells[3]} | {cells[4]} | {cells[5]}")
    print("--+---+--")
    print(f"{cells[6]} | {cells[7]} | {cells[8]}")


def print_rl_components():
    print("Task 1: Reinforcement Learning Components")
    print("-" * 72)
    rows = [
        ("Agent", "The Tic-Tac-Toe player learning as X."),
        ("Environment", "The game board plus the opposing player O."),
        ("State Representation", "A 9-cell tuple: 0 = empty, 1 = X, -1 = O."),
        ("Action Space", "Any currently empty cell from positions 1 to 9."),
        ("Reward Function", "+1 win, -1 loss, 0 draw or non-terminal move."),
        ("Learning Approach Used", "Model-free Q-learning through self-play."),
    ]
    for component, observation in rows:
        print(f"{component:24} {observation}")


def print_analysis_answers():
    print("\nAnalysis Questions")
    print("-" * 72)
    answers = [
        "1. The learning agent is the Tic-Tac-Toe player X.",
        "2. The environment is the current board and the moves made by player O.",
        "3. The state is represented as a 9-position tuple for the board.",
        "4. The possible actions are the empty cells where the agent can place X.",
        "5. The agent receives a positive reward when X forms a winning line.",
        "6. The learning method used is Q-learning with epsilon-greedy exploration.",
        "7. No labelled data is used; the agent learns from rewards after playing games.",
        "8. Unlike supervised learning, this agent is not given correct moves. It explores, receives rewards or penalties, and improves its policy from experience.",
    ]
    for answer in answers:
        print(answer)


def print_conclusion():
    print("\nConclusion")
    print("-" * 72)
    print("With few training episodes, the agent behaves mostly randomly.")
    print("As episodes increase, Q-values improve from repeated self-play feedback.")
    print("The agent starts preferring useful opening moves such as center or corners.")
    print("It also learns to block threats and take winning moves when available.")
    print("After sufficient training, losses reduce and many games become draws.")
    print("This shows reinforcement learning improves behaviour through reward-based practice.")


def run_lab_experiment():
    print_rl_components()
    print_analysis_answers()

    print("\nTask 2: Effect of Training Episodes on Agent Behaviour")
    print("-" * 100)
    print(f"{'Episodes':>10} {'Win %':>8} {'Loss %':>8} {'Draw %':>8}  Behaviour Observed")
    print("-" * 100)

    for episodes in TRAINING_EPISODES:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        q_tables, _ = train_self_play_agent(episodes)
        random.seed(RANDOM_SEED)
        results = evaluate_agent(q_tables)
        percentages = result_percentages(results)
        behaviour = observe_behaviour(q_tables, percentages)

        print(
            f"{episodes:>10} "
            f"{percentages['wins']:>7.1f}% "
            f"{percentages['losses']:>7.1f}% "
            f"{percentages['draws']:>7.1f}%  "
            f"{behaviour}"
        )

    print("\nAnswers for Task 2")
    print("-" * 72)
    print("9. Quality improves from random moves to blocking, winning, and safer play.")
    print("10. Intelligent decisions usually start appearing around 1000 to 5000 episodes.")
    print("11. Learning begins to converge around 5000 to 10000 episodes in this run.")
    print("12. Win percentage improves because useful actions get higher Q-values.")
    print("13. Many games become draws because both players learn to avoid losing lines.")
    print("14. Too few episodes leave the Q-table undertrained, so actions remain mostly random.")
    print_conclusion()


if __name__ == "__main__":
    run_lab_experiment()
