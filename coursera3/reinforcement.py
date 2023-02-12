import random
from abc import ABC, abstractmethod


class Action(ABC):
    @abstractmethod
    def hash(self) -> str:
        pass

    @staticmethod
    @abstractmethod
    def from_hash(hash: str) -> "Action":
        pass


class State(ABC):
    @abstractmethod
    def get_reward(self) -> float:
        pass

    @abstractmethod
    def is_terminal(self) -> bool:
        pass

    @abstractmethod
    def get_next_state(self, action: Action) -> "State":
        pass

    @abstractmethod
    def get_possible_actions(self) -> list[Action]:
        pass

    @abstractmethod
    def hash(self) -> str:
        pass


class Environment(ABC):
    @abstractmethod
    def get_start_states(self) -> list[State]:
        pass


class TickTackToeAction(Action):
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __str__(self):
        return f"({self.row}, {self.col})"

    def hash(self):
        return f"{self.row}{self.col}"

    @staticmethod
    def from_hash(hash: str):
        return TickTackToeAction(int(hash[0]), int(hash[1]))


TickTackToeBoard = list[list[int]]


def reverse_board(board: TickTackToeBoard) -> TickTackToeBoard:
    return [
        [
            (-1 * cell) for cell in row
        ]
        for row in board
    ]


def empty_board() -> TickTackToeBoard:
    return [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]


class TickTackToeState(State):
    player = 1
    opponent = -1
    empty = 0

    booster = 1000

    def __init__(self, board: TickTackToeBoard):
        if len(board) != 3 or len(board[0]) != 3:
            raise ValueError("Invalid board size")

        self.board = board

    def __str__(self):
        return "\n".join(
            [
                " ".join([str(cell).rjust(2) for cell in row])
                for row in self.board
            ]
        ) + '\n'

    def hash(self) -> str:
        return str(self.board)

    def get_reward(self) -> float:
        winner = self._get_winner()
        if winner == self.player:
            return 1.0 * self.booster
        elif winner == self.opponent:
            return -1.0 * self.booster
        else:
            return 0.0

    def is_terminal(self) -> bool:
        return self._get_winner() != self.empty or self._is_full()

    def get_next_state(self, action: Action) -> "TickTackToeState":
        if not isinstance(action, TickTackToeAction):
            raise ValueError("Invalid action type")

        if self.board[action.row][action.col] != self.empty:
            raise ValueError("Invalid action")

        new_board = [row[:] for row in self.board]
        new_board[action.row][action.col] = self.player
        return TickTackToeState(new_board)

    def get_possible_actions(self) -> list[Action]:
        actions = []
        for row in range(3):
            for col in range(3):
                if self.board[row][col] == self.empty:
                    actions.append(TickTackToeAction(row, col))
        return actions

    def get_reverse(self) -> 'TickTackToeState':
        return TickTackToeState(reverse_board(self.board))

    def _get_winner(self) -> int:
        # rows
        for row in self.board:
            if row[0] == row[1] == row[2] != self.empty:
                return row[0]
        # cols
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != self.empty:
                return self.board[0][col]
        # diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != self.empty:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != self.empty:
            return self.board[0][2]

        return self.empty

    def _is_full(self) -> bool:
        for row in self.board:
            for col in row:
                if col == self.empty:
                    return False
        return True


class TickTackToeEnvironment(Environment):
    def get_start_states(self) -> list[TickTackToeState]:
        empty_state = TickTackToeState(empty_board())
        return [empty_state] + [empty_state.get_next_state(a) for a in empty_state.get_possible_actions()]


class Agent(ABC):
    @abstractmethod
    def get_action(self, state: State, training=False) -> Action:
        pass

    @abstractmethod
    def update(self, state: State, action: Action, next_state: State, reward: float):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def summary(self) -> str:
        pass


class RandomAgent(Agent):
    def get_action(self, state: State, training=False) -> Action:
        return random.choice(state.get_possible_actions())

    def update(self, state: State, action: Action, next_state: State, reward: float):
        pass

    def reset(self):
        pass

    def summary(self) -> str:
        return f'random agent'


class HumanAgent(Agent):
    def get_action(self, state: State, training=False) -> Action:
        print(state, flush=True)
        print("Available actions:")
        for i, action in enumerate(state.get_possible_actions()):
            print(f"{i}: {action}")
        while True:
            action_index = int(input("Enter action index: "))
            if 0 <= action_index < len(state.get_possible_actions()):
                return state.get_possible_actions()[action_index]

    def update(self, state: State, action: Action, next_state: State, reward: float):
        pass

    def reset(self):
        pass

    def summary(self) -> str:
        return f'human agent'


class DeterministicLearningAgent(Agent):
    def __init__(self, gamma: float, q: dict[str, dict[str, float]] = {}):
        self.gamma = gamma
        self.q: dict[str, dict[str, float]] = q

    def get_action(self, state: TickTackToeState, training=False) -> Action:
        state = state.get_reverse()
        if state.hash() not in self.q:
            raise ValueError(f"q: state not available: {state}")
        rewards = self.q[state.hash()]
        if len(rewards.values()) != len(state.get_possible_actions()):
            raise ValueError(f"q: state action count mismatch: {state}")
        return TickTackToeAction.from_hash(max(self.q[state.hash()].items(), key=lambda x: x[1])[0])

    def update(self, state: State, action: Action, next_state: State, reward: float):
        pass

    def reset(self):
        pass

    def summary(self) -> str:
        return f'deterministic learning agent'

    def train(self, env: TickTackToeEnvironment):
        for state in env.get_start_states():
            self.q[state.hash()] = {}
            for action in state.get_possible_actions():
                self.q[state.hash()][action.hash()] = self._train(state, action)
                # print(f"Trained {state} {action}")
            print(f"Trained {state.hash()}")
        print(len(self.q))

    def _train(self, last_state: TickTackToeState, last_action: Action, depth: int = 0) -> float:
        state = last_state.get_next_state(last_action)
        if state.is_terminal():
            return state.get_reward() * self.gamma ** depth
        else:
            if state.hash() in self.q:
                return max(self.q[state.hash()].values()) * self.gamma ** depth
            self.q[state.hash()] = {}
            for action in state.get_possible_actions():
                self.q[state.hash()][action.hash()] = self._train(
                    state.get_reverse(), action, depth + 1)
            return max(self.q[state.hash()].values()) * self.gamma ** depth


class TicTacToeGame:
    def __init__(self, agent1: Agent, agent2: Agent):
        self.agent1 = agent1
        self.agent2 = agent2

    def train_play(self, env: TickTackToeEnvironment, num_episodes: int, render: bool = False):
        for episode in range(num_episodes):
            self.agent1.reset()
            self.agent2.reset()

            state = random.choice(env.get_start_states())
            playerI = 0
            while not state.is_terminal():
                agent = [self.agent1, self.agent2][playerI]
                state = self._turn(state.get_reverse(), agent, False, True)
                playerI = (playerI + 1) % 2
            if playerI == 1:
                state = state.get_reverse()

            if render:
                print(state)
                print(
                    f"Episode {episode + 1} finished with reward {state.get_reward()}")

            # if episode % 100 == 0:
            #     print(
            #         f"Episode {episode} finished with reward {state.get_reward()}")
            #     print(f"Agent 1: {self.agent1.summary()}")
            #     print(f"Agent 2: {self.agent2.summary()}")

    def _turn(self, state: TickTackToeState, agent: Agent, render: bool = False, training=False) -> TickTackToeState:
        if render:
            print(state)
            print(f'playing: {agent.summary()}')
        action = agent.get_action(state, training)
        next_state = state.get_next_state(action)
        reward = next_state.get_reward()
        agent.update(state, action, next_state, reward)
        return next_state

    def contest_play(self, env: TickTackToeEnvironment):
        state = env.get_start_states()[0]
        playerI = 0
        while not state.is_terminal():
            agent = [self.agent1, self.agent2][playerI]
            state = self._turn(state.get_reverse(), agent, True, False)
            playerI = (playerI + 1) % 2
        if playerI == 0:
            state = state.get_reverse()
        print(state)
        print(f"Finished with reward {state.get_reward()}")
        print(f"Agent 1: {self.agent1.summary()}")
        print(f"Agent 2: {self.agent2.summary()}")


def main():
    env = TickTackToeEnvironment()
    # agent1 = RandomAgent()
    agent1 = DeterministicLearningAgent(0.999)
    agent1.train(env)
    # agent2 = RandomAgent()
    # game = TicTacToeGame(agent1, agent2)
    # game.train_play(env, 10, render=True)
    # print(agent1.summary())
    # print(agent1.q.keys())
    # print(agent1.q[
    #     TickTackToeState([
    #         [1,  1,  0],
    #         [0, -1,  0],
    #         [0,  0,  0]
    #     ]).hash()
    # ])
    human_agent = HumanAgent()
    game = TicTacToeGame(human_agent, agent1)
    game.contest_play(env)


if __name__ == "__main__":
    main()
