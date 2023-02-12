/**
 * coin king game  works like this:
 * each turn you can add 1 or 2 coins to the pile
 * the player who adds the last coin wins
 *
 * for simplicity, the game will be allays played by 2 players
 */

export module Reinforcement {
  // export type DecisionTable<Action> = {
  //   [state: string]: QDecision<Action>;
  // };
  export abstract class DecisionTable<Action> {
    public abstract get(hash: string): Promise<QDecision<Action> | undefined>;
    public abstract set(
      hash: string,
      decision: QDecision<Action>
    ): Promise<void>;
    public abstract has(hash: string): Promise<boolean>;
  }
  export class MemoryDecisionTable<Action> implements DecisionTable<Action> {
    constructor(
      public readonly table: { [state: string]: QDecision<Action> } = {}
    ) {}
    public async get(hash: string): Promise<QDecision<Action> | undefined> {
      return this.table[hash];
    }
    public async set(hash: string, decision: QDecision<Action>): Promise<void> {
      this.table[hash] = decision;
    }
    public async has(hash: string): Promise<boolean> {
      return hash in this.table;
    }
  }

  export class FileDecisionTable<Action> implements DecisionTable<Action> {
    constructor(public readonly dirPath: string) {}
    private hashString = (str: string): string => {
      let hash = 0;
      for (let i = 0, len = str.length; i < len; i++) {
        let chr = str.charCodeAt(i);
        hash = (hash << 5) - hash + chr;
        hash |= 0; // Convert to 32bit integer
      }
      return hash.toString(16);
    };
    public async get(hash: string): Promise<QDecision<Action> | undefined> {
      try {
        return JSON.parse(
          await Deno.readTextFile(
            `${this.dirPath}/${this.hashString(hash)}.json`
          )
        );
      } catch (e) {
        console.log(
          `error while reading and parsing ${this.dirPath}/${this.hashString(
            hash
          )}.json`
        );
      }
    }
    public async set(hash: string, decision: QDecision<Action>): Promise<void> {
      try {
        await Deno.writeTextFile(
          `${this.dirPath}/${this.hashString(hash)}.json`,
          JSON.stringify(decision),
          { create: true }
        );
      } catch (e) {
        console.log(
          `error while signifying and writing ${this.dirPath}/${this.hashString(
            hash
          )}.json`
        );
      }
    }
    public async has(hash: string): Promise<boolean> {
      try {
        return Deno.stat(`${this.dirPath}/${this.hashString(hash)}.json`)
          .then(() => true)
          .catch(() => false);
      } catch (e) {
        console.log(
          `error while checking status of ${this.dirPath}/${this.hashString(
            hash
          )}.json`
        );
        return false;
      }
    }
  }

  export type QDecision<Action> = {
    action: Action;
    reward: number;
  };

  export abstract class State<Action> {
    public abstract getPossibleActions(): Action[];
    public abstract isTerminal(): boolean;
    public abstract move(action: Action): void;
    public abstract moveCopy(action: Action): State<Action>;
    public abstract toHash(): string;
  }
  export abstract class Player<Action> {
    public abstract getMove(state: State<Action>): Action;
    public abstract getName(): string;
  }

  export abstract class RewardEnvironment<Action> {
    public abstract getInitialStates(): State<Action>[];
    public abstract getStateReward(state: State<Action>): number;
  }

  export abstract class RewardTwoPlayerEnvironment<
    Action
  > extends RewardEnvironment<Action> {
    public abstract getStateReward(state: State<Action>): number;
    public abstract reverseState(state: State<Action>): State<Action>;

    public readonly reverseReward = (reward: number): number => -reward;
  }

  export const multiPlayerTrain = async <Action>(
    env: RewardTwoPlayerEnvironment<Action>,
    gamma = 1,
    decisionTable: DecisionTable<Action> = new MemoryDecisionTable<Action>()
  ): Promise<DecisionTable<Action>> => {
    /***
     * @description getReward is a recursive function that calculates the reward of a state by calculating the reward of all possible next states
     * @param state the state to calculate the reward for
     * @returns the reward of the state, converted to be the reward of the previous player's perspective
     * ---
     * it uses the decision table as a cache \
     * it returns the reward of the state \
     * it fills the decision table with the best action for each state \
     * the returned reward is converted to be the reward of the previous player's perspective
     */
    const getReward = async (state: State<Action>): Promise<number> => {
      const stateHash = state.toHash();
      // const _l = Object.keys(decisionTable).length;
      // if (_l % 1000 < 5) console.log(_l);
      if (await decisionTable.has(stateHash))
        return env.reverseReward((await decisionTable.get(stateHash))!.reward);
      if (state.isTerminal())
        return env.reverseReward(env.getStateReward(state));
      const best = (
        await Promise.all(
          state.getPossibleActions().map(async (action) => ({
            action,
            reward: await getReward(env.reverseState(state.moveCopy(action))),
          }))
        )
      ).reduce((a, b) => (a.reward >= b.reward ? a : b));
      await decisionTable.set(stateHash, best);
      return env.reverseReward(gamma * best.reward);
    };
    for (const state of env.getInitialStates()) await getReward(state);

    return decisionTable;
  };
}

export module KingCoin {
  export enum Action {
    Add1 = "1",
    Add2 = "2",
  }
  export const pileMove = (action: Action): number =>
    action === Action.Add1 ? 1 : 2;

  export abstract class Player {
    public abstract getMove(state: State): Promise<Action>;
    public abstract getName(): string;
  }

  export class RandomPlayer extends Player {
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      return possibleActions[
        Math.floor(Math.random() * possibleActions.length)
      ];
    }
    public getName = (): string => "Random";
  }

  export class HumanPlayer extends Player {
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      const move = prompt("Enter your move: 1 or 2");
      if (parseInt(move || "0") === 1) return Action.Add1;
      else return Action.Add2;
    }
    public getName = (): string => "Human";
  }

  export class BotPlayer extends Player {
    constructor(
      public readonly decisionTable: Reinforcement.DecisionTable<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public async getMove(state: State): Promise<Action> {
      const decision = await this.decisionTable.get(state.toHash());
      if (decision === undefined) throw new Error("No decision found");
      return decision.action;
    }
    public getName = (): string => `Bot#${this.id}`;
  }

  export class Game {
    constructor(
      public readonly players: Player[],
      public readonly state = new State(5)
    ) {}

    public async play(render = false): Promise<void> {
      let playerTurn = 0;
      if (render) {
        console.log(
          `Game: ${this.players.map((p) => p.getName()).join(" vs ")}`
        );
        console.log(this.state.toString());
      }
      while (!this.state.isTerminal()) {
        const player = this.players[playerTurn];
        const move = await player.getMove(this.state);
        this.state.move(move);
        if (render)
          console.log(
            `${player.getName()}: +${move}, ${this.state.toString()}`
          );
        playerTurn = (playerTurn + 1) % 2;
      }
      const winner = this.players.at(playerTurn - 1);
      if (winner !== undefined && render)
        console.log(`Game: winner is ${winner.getName()}`);
    }
  }

  export class State implements Reinforcement.State<Action> {
    public current_pile: number = 0;
    constructor(public readonly pile_size: number) {}
    public getPossibleActions = () => [Action.Add1, Action.Add2];
    public isTerminal = (): boolean => this.current_pile >= this.pile_size;
    public move = (action: Action): void => {
      this.current_pile += pileMove(action);
    };
    public moveCopy = (action: Action): State => {
      const copy = new State(this.pile_size);
      copy.current_pile = this.current_pile + pileMove(action);
      return copy;
    };
    public toHash = (): string => this.current_pile.toString();
    public toString(): string {
      return `${this.current_pile}/${this.pile_size}`;
    }
  }
  export class GameEnvironment extends Reinforcement.RewardTwoPlayerEnvironment<Action> {
    public readonly playerCount: number = 2;
    constructor(public readonly pile_size: number) {
      super();
    }
    public getInitialStates = (): State[] => [new State(this.pile_size)];
    public getStateReward = (state: State): number =>
      state.isTerminal() ? -1 : 0;
    public reverseState = (state: State): State => state;
  }
}

export module KingCoin3 {
  export enum Action {
    Add1 = "1",
    Add2 = "2",
    Add3 = "3",
  }
  export const pileMove = (action: Action): number =>
    action === Action.Add1 ? 1 : action === Action.Add2 ? 2 : 3;

  export abstract class Player {
    public abstract getMove(state: State): Promise<Action>;
    public abstract getName(): string;
  }

  export class RandomPlayer extends Player {
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      return possibleActions[
        Math.floor(Math.random() * possibleActions.length)
      ];
    }
    public getName = (): string => "Random";
  }

  export class HumanPlayer extends Player {
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      const move = prompt("Enter your move: 1 or 2");
      return parseInt(move || "0").toString() as Action;
    }
    public getName = (): string => "Human";
  }

  export class BotPlayer extends Player {
    constructor(
      public readonly decisionTable: Reinforcement.DecisionTable<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public async getMove(state: State): Promise<Action> {
      const decision = await this.decisionTable.get(state.toHash());
      if (decision === undefined) throw new Error("No decision found");
      return decision.action;
    }
    public getName = (): string => `Bot#${this.id}`;
  }

  export class Game {
    constructor(
      public readonly players: Player[],
      public readonly state = new State(5)
    ) {}

    public async play(render = false): Promise<void> {
      let playerTurn = 0;
      if (render) {
        console.log(
          `Game: ${this.players.map((p) => p.getName()).join(" vs ")}`
        );
        console.log(this.state.toString());
      }
      while (!this.state.isTerminal()) {
        const player = this.players[playerTurn];
        const move = await player.getMove(this.state);
        this.state.move(move);
        if (render)
          console.log(
            `${player.getName()}: +${move}, ${this.state.toString()}`
          );
        playerTurn = (playerTurn + 1) % 2;
      }
      const winner = this.players.at(playerTurn - 1);
      if (winner !== undefined && render)
        console.log(`Game: winner is ${winner.getName()}`);
    }
  }

  export class State implements Reinforcement.State<Action> {
    public current_pile: number = 0;
    constructor(public readonly pile_size: number) {}
    public getPossibleActions = () => [Action.Add1, Action.Add2, Action.Add3];
    public isTerminal = (): boolean => this.current_pile >= this.pile_size;
    public move = (action: Action): void => {
      this.current_pile += pileMove(action);
    };
    public moveCopy = (action: Action): State => {
      const copy = new State(this.pile_size);
      copy.current_pile = this.current_pile + pileMove(action);
      return copy;
    };
    public toHash = (): string => this.current_pile.toString();
    public toString(): string {
      return `${this.current_pile}/${this.pile_size}`;
    }
  }
  export class GameEnvironment extends Reinforcement.RewardTwoPlayerEnvironment<Action> {
    public readonly playerCount: number = 2;
    constructor(public readonly pile_size: number) {
      super();
    }
    public getInitialStates = (): State[] => [new State(this.pile_size)];
    public getStateReward = (state: State): number =>
      state.isTerminal() ? -1 : 0;
    public reverseState = (state: State): State => state;
  }
}

export module TicTacToe {
  //  export type BoardIndex = 0 | 1 | 2;
  export type BoardIndex = number;
  export class Action {
    constructor(
      public readonly row: BoardIndex,
      public readonly col: BoardIndex
    ) {}

    public toString = () => `[${this.row}, ${this.col}]`;
  }
  export type Cell = -1 | 0 | 1;
  export type Board = [
    [Cell, Cell, Cell],
    [Cell, Cell, Cell],
    [Cell, Cell, Cell]
  ];
  const emptyBoard: Board = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const getWinner = (board: Board): Cell => {
    const flat = board.flat();
    for (let i = 0; i < 3; i++) {
      if (flat[i] === flat[i + 3] && flat[i] === flat[i + 6] && flat[i] !== 0)
        return flat[i];
      if (
        flat[i * 3] === flat[i * 3 + 1] &&
        flat[i * 3] === flat[i * 3 + 2] &&
        flat[i * 3] !== 0
      )
        return flat[i * 3];
    }
    if (flat[0] === flat[4] && flat[0] === flat[8] && flat[0] !== 0)
      return flat[0];
    if (flat[2] === flat[4] && flat[2] === flat[6] && flat[2] !== 0)
      return flat[2];
    return 0;
  };

  export class State implements Reinforcement.State<Action> {
    constructor(public readonly board: Board = emptyBoard) {}
    public getPossibleActions = () => {
      const actions: Action[] = [];
      this.board.forEach((_, row) =>
        _.forEach((_, col) => {
          if (this.board[row][col] === 0) actions.push(new Action(row, col));
        })
      );
      return actions;
    };

    public isTerminal = (): boolean =>
      !(getWinner(this.board) === 0 && this.board.flat().some((c) => c === 0));
    public move = (action: Action): void => {
      this.board[action.row][action.col] = 1;
    };
    public moveCopy = (action: Action): State => {
      const copy = new State(structuredClone(this.board));
      copy.move(action);
      return copy;
    };
    public toHash = (): string => this.board.toString();
    public toString(): string {
      return this.board
        .map((row) => row.map((c) => c.toString().padStart(2)).join(","))
        .join("\n");
    }
  }
  export class GameEnvironment extends Reinforcement.RewardTwoPlayerEnvironment<Action> {
    public readonly playerCount: number = 2;
    constructor() {
      super();
    }
    public getInitialStates = (): State[] => [
      new State(structuredClone(emptyBoard)),
    ];
    public getStateReward = (state: State): number => this.getWinner(state);

    public getWinner = (state: State): Cell => {
      return getWinner(state.board);
    };
    public reverseState = (state: State): State =>
      new State(state.board.map((row) => row.map((cell) => -cell)) as Board);
  }

  // Game:
  export abstract class Player {
    public abstract getMove(state: State): Promise<Action>;
    public abstract getName(): string;
  }

  export class RandomPlayer extends Player {
    constructor(private readonly id = Math.random().toString(32).slice(2, 4)) {
      super();
    }
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      return possibleActions[
        Math.floor(Math.random() * possibleActions.length)
      ];
    }
    public getName = (): string => `Random#${this.id}`;
  }

  export class HumanPlayer extends Player {
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      const move = prompt("Enter your move: (row, col)")?.split(",") as [
        string,
        string
      ];
      return new Action(parseInt(move[0]), parseInt(move[1]));
    }
    public getName = (): string => "Human";
  }

  export class BotPlayer extends Player {
    constructor(
      public readonly decisionTable: Reinforcement.DecisionTable<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public async getMove(state: State): Promise<Action> {
      const decision = await this.decisionTable.get(state.toHash());
      if (decision === undefined) throw new Error("No decision found");
      return decision.action;
    }
    public getName = (): string => `Bot#${this.id}`;
  }

  export class Game {
    constructor(
      public readonly env: GameEnvironment,
      public readonly players: Player[],
      public state = new State(emptyBoard)
    ) {}

    public async play(render = false): Promise<void> {
      let playerTurn = 0;
      if (render) {
        console.log(
          `Game: ${this.players.map((p) => p.getName()).join(" vs ")}`
        );
        console.log(this.state.toString());
      }
      while (!this.state.isTerminal()) {
        const player = this.players[playerTurn];
        const move = await player.getMove(this.state);
        this.state.move(move);
        if (render)
          console.log(
            `${player.getName()}: +${move}: \n${this.state.toString()}\n`
          );
        playerTurn = (playerTurn + 1) % 2;
        this.state = this.env.reverseState(this.state);
      }
      if (render) {
        switch (this.env.getWinner(this.state)) {
          case 1:
            console.log(
              `Game: winner is ${this.players[playerTurn].getName()}`
            );
            break;
          case -1:
            console.log(
              `Game: winner is ${this.players.at(playerTurn - 1)!.getName()}`
            );
            break;
          case 0:
            console.log(`Game: draw`);
            break;
        }
      }
    }
  }
}

export module FourInARow {
  //  export type BoardIndex = 0 | 1 | 2;
  export type BoardIndex = number;
  export class Action {
    constructor(public readonly col: BoardIndex) {}

    public toString = () => `[${this.col}]`;
  }
  export type Cell = -1 | 0 | 1;
  export type Board = [
    [Cell, Cell, Cell, Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell, Cell, Cell, Cell]
  ];
  const emptyBoard: Board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
  ];
  const getWinner = (board: Board): Cell => {
    // horizontal
    for (let row = 0; row < 6; row++) {
      for (let col = 0; col < 4; col++) {
        const cell = board[row][col];
        if (
          cell !== 0 &&
          board[row][col + 1] === cell &&
          board[row][col + 2] === cell &&
          board[row][col + 3] === cell
        )
          return cell;
      }
    }
    // vertical
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 7; col++) {
        const cell = board[row][col];
        if (
          cell !== 0 &&
          board[row + 1][col] === cell &&
          board[row + 2][col] === cell &&
          board[row + 3][col] === cell
        )
          return cell;
      }
    }
    for (let row = 0; row < 3; row++) {
      // diagonal
      for (let col = 0; col < 4; col++) {
        const cell = board[row][col];
        if (
          cell !== 0 &&
          board[row + 1][col + 1] === cell &&
          board[row + 2][col + 2] === cell &&
          board[row + 3][col + 3] === cell
        )
          return cell;
      }
      // anti-diagonal
      for (let col = 3; col < 7; col++) {
        const cell = board[row][col];
        if (
          cell !== 0 &&
          board[row + 1][col - 1] === cell &&
          board[row + 2][col - 2] === cell &&
          board[row + 3][col - 3] === cell
        )
          return cell;
      }
    }
    return 0;
  };

  export class State implements Reinforcement.State<Action> {
    constructor(public readonly board: Board = emptyBoard) {}
    public getPossibleActions = () => {
      const flat = this.board.flat();
      const actions: Action[] = [];
      for (let col = 0; col < 7; col++) {
        for (let row = 0; row < 6; row++) {
          if (flat[row * 7 + col] === 0) {
            actions.push(new Action(col));
            break;
          }
        }
      }
      return actions;
    };

    public isTerminal = (): boolean =>
      !(getWinner(this.board) === 0 && this.getPossibleActions().length > 0);
    public move = (action: Action): void => {
      let row = 5;
      while (this.board[row][action.col] !== 0) row--;
      if (row < 0) throw new Error("Invalid move: column is full");
      this.board[row][action.col] = 1;
    };
    public moveCopy = (action: Action): State => {
      const copy = new State(structuredClone(this.board));
      copy.move(action);
      return copy;
    };
    public toHash = (): string => this.board.toString();
    public toString(): string {
      return this.board
        .map((row) => row.map((c) => c.toString().padStart(2)).join(","))
        .join("\n");
    }
  }
  export class GameEnvironment extends Reinforcement.RewardTwoPlayerEnvironment<Action> {
    public readonly playerCount: number = 2;
    constructor() {
      super();
    }
    public getInitialStates = (): State[] => [
      new State(structuredClone(emptyBoard)),
    ];
    public getStateReward = (state: State): number => this.getWinner(state);

    public getWinner = (state: State): Cell => {
      return getWinner(state.board);
    };
    public reverseState = (state: State): State =>
      new State(state.board.map((row) => row.map((cell) => -cell)) as Board);
  }

  // Game:
  export abstract class Player {
    public abstract getMove(state: State): Promise<Action>;
    public abstract getName(): string;
  }

  export class RandomPlayer extends Player {
    constructor(private readonly id = Math.random().toString(32).slice(2, 4)) {
      super();
    }
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      return possibleActions[
        Math.floor(Math.random() * possibleActions.length)
      ];
    }
    public getName = (): string => `Random#${this.id}`;
  }

  export class HumanPlayer extends Player {
    public async getMove(state: State): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      const move = prompt("Enter your move: (col)") ?? "0";
      return new Action(parseInt(move));
    }
    public getName = (): string => "Human";
  }

  export class BotPlayer extends Player {
    constructor(
      public readonly decisionTable: Reinforcement.DecisionTable<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public async getMove(state: State): Promise<Action> {
      const decision = await this.decisionTable.get(state.toHash());
      if (decision === undefined) throw new Error("No decision found");
      return decision.action;
    }
    public getName = (): string => `Bot#${this.id}`;
  }

  export class Game {
    constructor(
      public readonly env: GameEnvironment,
      public readonly players: Player[],
      public state = new State(emptyBoard)
    ) {}

    public async play(render = false): Promise<void> {
      let playerTurn = 0;
      if (render) {
        console.log(
          `Game: ${this.players.map((p) => p.getName()).join(" vs ")}`
        );
        console.log(this.state.toString());
      }
      while (!this.state.isTerminal()) {
        const player = this.players[playerTurn];
        const move = await player.getMove(this.state);
        this.state.move(move);
        if (render)
          console.log(
            `${player.getName()}: +${move}: \n${this.state.toString()}\n`
          );
        playerTurn = (playerTurn + 1) % 2;
        this.state = this.env.reverseState(this.state);
      }
      if (render) {
        switch (this.env.getWinner(this.state)) {
          case 1:
            console.log(
              `Game: winner is ${this.players[playerTurn].getName()}`
            );
            break;
          case -1:
            console.log(
              `Game: winner is ${this.players.at(playerTurn - 1)!.getName()}`
            );
            break;
          case 0:
            console.log(`Game: draw`);
            break;
        }
      }
    }
  }
}

/*export module Checkers {
  //  export type BoardIndex = 0 | 1 | 2;
  export type BoardIndex = number;
  export class Action {

  }
  export type Cell = -1 | 0 | 1;
  export type Board = [
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell],
    [Cell, Cell, Cell, Cell]
  ];
  const blankBoard: Board = [
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [-1, -1, -1, -1],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
    [1, 1, 1, 1],
  ];
  const getWinner = (board: Board): Cell => {
    const flat = board.flat();
    if (!flat.some((c) => c === 1)) return -1;
    if (!flat.some((c) => c === -1)) return 1;
    return 0;
  };

  export class State implements Reinforcement.State<Action> {
    constructor(public readonly board: Board = blankBoard) {}
    public getPossibleActions = () => {
      const actions: Action[] = [];
      this.board.forEach((_, row) =>
        _.forEach((_, col) => {
          if (this.board[row][col] === 0) actions.push(new Action(row, col));
        })
      );
      return actions;
    };

    public isTerminal = (): boolean =>
      !(getWinner(this.board) === 0 && this.board.flat().some((c) => c === 0));
    public move = (action: Action): void => {
      this.board[action.row][action.col] = 1;
    };
    public moveCopy = (action: Action): State => {
      const copy = new State(structuredClone(this.board));
      copy.move(action);
      return copy;
    };
    public toHash = (): string => this.board.toString();
    public toString(): string {
      return this.board
        .map((row) => row.map((c) => c.toString().padStart(2)).join(","))
        .join("\n");
    }
  }
  export class GameEnvironment extends Reinforcement.RewardTwoPlayerEnvironment<Action> {
    public readonly playerCount: number = 2;
    constructor() {
      super();
    }
    public getInitialStates = (): State[] => [
      new State(structuredClone(blankBoard)),
    ];
    public getStateReward = (state: State): number => this.getWinner(state);

    public getWinner = (state: State): Cell => {
      return getWinner(state.board);
    };
    public reverseState = (state: State): State =>
      new State(state.board.map((row) => row.map((cell) => -cell)) as Board);
  }

  // Game:
  export abstract class Player {
    public abstract getMove(state: State): Action;
    public abstract getName(): string;
  }

  export class RandomPlayer extends Player {
    constructor(private readonly id = Math.random().toString(32).slice(2, 4)) {
      super();
    }
    public getMove(state: State): Action {
      const possibleActions = state.getPossibleActions();
      return possibleActions[
        Math.floor(Math.random() * possibleActions.length)
      ];
    }
    public getName = (): string => `Random#${this.id}`;
  }

  export class HumanPlayer extends Player {
    public getMove(state: State): Action {
      const possibleActions = state.getPossibleActions();
      const move = prompt("Enter your move: (row, col)")?.split(",") as [
        string,
        string
      ];
      return new Action(parseInt(move[0]), parseInt(move[1]));
    }
    public getName = (): string => "Human";
  }

  export class BotPlayer extends Player {
    constructor(
      public readonly decisionTable: Reinforcement.DecisionTable<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public getMove(state: State): Action {
      const decision = this.decisionTable[state.toHash()];
      if (decision === undefined) throw new Error("No decision found");
      return decision.action;
    }
    public getName = (): string => `Bot#${this.id}`;
  }

  export class Game {
    constructor(
      public readonly env: GameEnvironment,
      public readonly players: Player[],
      public state = new State(blankBoard)
    ) {}

    public play(render = false): void {
      let playerTurn = 0;
      if (render) {
        console.log(
          `Game: ${this.players.map((p) => p.getName()).join(" vs ")}`
        );
        console.log(this.state.toString());
      }
      while (!this.state.isTerminal()) {
        const player = this.players[playerTurn];
        const move = player.getMove(this.state);
        this.state.move(move);
        if (render)
          console.log(
            `${player.getName()}: +${move}: \n${this.state.toString()}\n`
          );
        playerTurn = (playerTurn + 1) % 2;
        this.state = this.env.reverseState(this.state);
      }
      if (render) {
        switch (this.env.getWinner(this.state)) {
          case 1:
            console.log(
              `Game: winner is ${this.players[playerTurn].getName()}`
            );
            break;
          case -1:
            console.log(
              `Game: winner is ${this.players.at(playerTurn - 1)!.getName()}`
            );
            break;
          case 0:
            console.log(`Game: draw`);
            break;
        }
      }
    }
  }
}*/

const TestKingCoin = async () => {
  // console.clear();
  const coins = 5;
  const env = new KingCoin.GameEnvironment(coins);
  const table = await Reinforcement.multiPlayerTrain(env, 0.5);
  console.log(table);
  const bot = new KingCoin.BotPlayer(table, "1");
  const bot2 = new KingCoin.BotPlayer(table, "2");
  const human = new KingCoin.HumanPlayer();
  const random = new KingCoin.RandomPlayer();
  const game = new KingCoin.Game([bot, bot2], new KingCoin.State(coins));
  game.play(true);
};

const TestKingCoin3 = async () => {
  // console.clear();
  const coins = 20;
  const env = new KingCoin3.GameEnvironment(coins);
  const table = await Reinforcement.multiPlayerTrain(env, 1);
  console.log(table);
  const bot = new KingCoin3.BotPlayer(table, "1");
  const bot2 = new KingCoin3.BotPlayer(table, "2");
  const human = new KingCoin3.HumanPlayer();
  const random = new KingCoin3.RandomPlayer();
  const game = new KingCoin3.Game([bot, bot2], new KingCoin3.State(coins));
  game.play(true);
};

const TestTicTacToe = async () => {
  // console.clear();
  const env = new TicTacToe.GameEnvironment();
  const table = await Reinforcement.multiPlayerTrain(
    env,
    0.999,
    new Reinforcement.FileDecisionTable<TicTacToe.Action>(
      "./coursera3/saved/TicTacToeFolder"
    )
  );
  // const table = JSON.parse(
  //   await Deno.readTextFile("./coursera3/saved/TicTacToe.json")
  // ) as Reinforcement.DecisionTable<TicTacToe.Action>;
  // await Deno.writeTextFile(
  //   "./coursera3/saved/TicTacToe.json",
  //   JSON.stringify(table)
  // );
  console.log(Object.keys(table).length);
  const bot = new TicTacToe.BotPlayer(table, "1");
  const bot2 = new TicTacToe.BotPlayer(table, "2");
  const human = new TicTacToe.HumanPlayer();
  const random = new TicTacToe.RandomPlayer();
  // const game = new TicTacToe.Game(env, [bot, bot2]);
  const game = new TicTacToe.Game(env, [bot, human]);
  game.play(true);
};

const TestFourInARow = async () => {
  // console.clear();
  const env = new FourInARow.GameEnvironment();
  const table = await Reinforcement.multiPlayerTrain(
    env,
    0.999,
    new Reinforcement.FileDecisionTable<TicTacToe.Action>(
      "./coursera3/saved/FourInARowFolder"
    )
  );
  // await Deno.writeTextFile(
  //   "./coursera3/saved/FourInARow.json",
  //   JSON.stringify(table)
  // );
  // const table = JSON.parse(
  //   await Deno.readTextFile("./coursera3/saved/FourInARow.json")
  // ) as Reinforcement.DecisionTable<FourInARow.Action>;
  console.log(Object.keys(table).length);
  // const bot = new FourInARow.BotPlayer(table, "1");
  // const bot2 = new FourInARow.BotPlayer(table, "2");
  const human = new FourInARow.HumanPlayer();
  const random = new FourInARow.RandomPlayer();
  // const game = new TickTackToe.Game(env, [bot, bot2]);
  // const game = new FourInARow.Game(env, [bot, human]);
  // const game = new FourInARow.Game(env, [random, random]);
  // game.play(true);
};

const main = TestFourInARow;

main();
