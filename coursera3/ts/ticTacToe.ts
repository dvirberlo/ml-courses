import * as Reinforcement from "./reinforcement.ts";

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
    return possibleActions[Math.floor(Math.random() * possibleActions.length)];
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
      console.log(`Game: ${this.players.map((p) => p.getName()).join(" vs ")}`);
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
          console.log(`Game: winner is ${this.players[playerTurn].getName()}`);
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

const TestTicTacToe = async () => {
  // console.clear();
  const env = new GameEnvironment();
  // const table = new Reinforcement.MemoryDecisionTable<Action>();
  const table = new Reinforcement.FileDecisionTable<Action>(
    "../saved/TicTacToe"
  );

  await Reinforcement.multiPlayerTrain(env, 0.999, table);
  // const table = JSON.parse(
  //   await Deno.readTextFile("./coursera3/saved/TicTacToe.json")
  // ) as Reinforcement.DecisionTable<Action>;
  // await Deno.writeTextFile(
  //   "./coursera3/saved/TicTacToe.json",
  //   JSON.stringify(table)
  // );
  // console.log(Object.keys(table.table).length);
  const bot = new BotPlayer(table, "1");
  const bot2 = new BotPlayer(table, "2");
  const human = new HumanPlayer();
  const random = new RandomPlayer();
  // const game = new Game(env, [bot, bot2]);
  const game = new Game(env, [bot, bot2]);
  game.play(true);
};

export const main = TestTicTacToe;
