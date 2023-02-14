import * as Reinforcement from "./reinforcement.ts";

const ROWS = 4;
const COLS = 5;
const WIN_LENGTH = 4;
//  export type BoardIndex = 0 | 1 | 2;
export type BoardIndex = number;
export class Action {
  constructor(public readonly col: BoardIndex) {}

  public toString = () => `[${this.col}]`;
  static toString = (obj: any) => `[${obj.col}]`;
}
export type Cell = -1 | 0 | 1;
export type Board = Cell[][];
const emptyBoard = Array.from({ length: ROWS }, () =>
  Array.from({ length: COLS }, () => 0)
) as Board;
function getWinner(board: Board): Cell {
  const checkSeries = (cells: Cell[]) =>
    cells[0] !== 0 && cells.every((cell) => cell === cells[0]);

  const getSeries = (
    startCol: number,
    startRow: number,
    dCol: number,
    dRow: number
  ) => {
    const series: Cell[] = [];
    for (let i = 0; i < WIN_LENGTH; i++) {
      const col = startCol + i * dCol;
      const row = startRow + i * dRow;
      if (row < 0 || row >= ROWS || col < 0 || col >= COLS) return null;
      series.push(board[row][col]);
    }
    return series;
  };

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const cell = board[row][col];
      if (cell === 0) continue;
      const rowSeries = getSeries(col, row, 1, 0);
      const colSeries = getSeries(col, row, 0, 1);
      const diagSeries1 = getSeries(col, row, 1, 1);
      const diagSeries2 = getSeries(col, row, 1, -1);
      if (
        checkSeries(rowSeries ?? [0]) ||
        checkSeries(colSeries ?? [0]) ||
        checkSeries(diagSeries1 ?? [0]) ||
        checkSeries(diagSeries2 ?? [0])
      ) {
        return cell;
      }
    }
  }

  return 0;
}
/*const getWinner2 = (board: Board): Cell => {
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
};*/

export class State implements Reinforcement.State<Action> {
  constructor(public readonly board: Board = emptyBoard) {}
  public getPossibleActions = () => {
    const flat = this.board.flat();
    const actions: Action[] = [];
    for (let col = 0; col < COLS; col++) {
      for (let row = 0; row < ROWS; row++) {
        if (flat[row * COLS + col] === 0) {
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
    let row = ROWS - 1;
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
      .map((row) =>
        row.map((c) => (c === 1 ? "X" : c === -1 ? "O" : " ")).join(" | ")
      )
      .join("\n");
    // .map((row) => row.map((c) => c.toString().padStart(2)).join(","))
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
// export abstract class Player {
//   public abstract getMove(state: State): Promise<Action>;
//   public abstract getName(): string;
// }

// export class RandomPlayer implements Player {
//   constructor(private readonly id = Math.random().toString(32).slice(2, 4)) {
//     super();
//   }
//   public async getMove(state: State): Promise<Action> {
//     const possibleActions = state.getPossibleActions();
//     return possibleActions[Math.floor(Math.random() * possibleActions.length)];
//   }
//   public getName = (): string => `Random#${this.id}`;
// }

export class HumanPlayer implements Reinforcement.Player<Action> {
  public async getMove(state: State): Promise<Action> {
    const possibleActions = state.getPossibleActions();
    const move = prompt("Enter your move: (col)") ?? "0";
    return new Action(parseInt(move));
  }
  public getName = (): string => "Human";
}

// export class BotPlayer implements Player {
//   constructor(
//     public readonly decisionTable: Reinforcement.DecisionTable<Action>,
//     private readonly id = Math.random().toString(32).slice(2, 4)
//   ) {
//     super();
//   }
//   public async getMove(state: State): Promise<Action> {
//     const decision = await this.decisionTable.get(state.toHash());
//     if (decision === undefined) throw new Error("No decision found");
//     return decision.action;
//   }
//   public getName = (): string => `Bot#${this.id}`;
// }

// export class Game {
//   constructor(
//     public readonly env: GameEnvironment,
//     public readonly players: Player[],
//     public state = new State(emptyBoard)
//   ) {}

//   public async play(render = false): Promise<void> {
//     let playerTurn = 0;
//     if (render) {
//       console.log(`Game: ${this.players.map((p) => p.getName()).join(" vs ")}`);
//       console.log(this.state.toString());
//     }
//     while (!this.state.isTerminal()) {
//       const player = this.players[playerTurn];
//       const move = await player.getMove(this.state);
//       this.state.move(move);
//       if (render)
//         console.log(
//           `${player.getName()}: +${Action.toString(
//             move
//           )}: \n${this.state.toString()}\n`
//         );
//       playerTurn = (playerTurn + 1) % 2;
//       this.state = this.env.reverseState(this.state);
//     }
//     if (render) {
//       switch (this.env.getWinner(this.state)) {
//         case 1:
//           console.log(`Game: winner is ${this.players[playerTurn].getName()}`);
//           break;
//         case -1:
//           console.log(
//             `Game: winner is ${this.players.at(playerTurn - 1)!.getName()}`
//           );
//           break;
//         case 0:
//           console.log(`Game: draw`);
//           break;
//       }
//     }
//   }
// }

const TestFourInARow = async () => {
  // console.clear();
  const env = new GameEnvironment();
  const table = await Reinforcement.multiPlayerTrain(
    env,
    0.999,
    new Reinforcement.FileDecisionTable<Action>(
      `../saved/${WIN_LENGTH}_InARowFolder_${ROWS}x${COLS}`
    )
  );
  // console.log(Object.keys(table.table).length);
  const bot = new Reinforcement.BotPlayer(table, "1");
  const bot2 = new Reinforcement.BotPlayer(table, "2");
  const human = new HumanPlayer();
  const random = new Reinforcement.RandomPlayer();
  const game = new Reinforcement.Game(env, [human, bot2]);
  // const game = new Game(env, [bot, human]);
  // const game = new Game(env, [human, bot]);
  game.play(true);
};

export const main = TestFourInARow;
