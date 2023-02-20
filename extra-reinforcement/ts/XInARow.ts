import * as Reinforcement from "./reinforcement.ts";

const ROWS = 3;
const COLS = 3;
const WIN_LENGTH = 3;
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

  public actionToString = (action: Action): string => `[${action.col}]`;
}

export class HumanPlayer implements Reinforcement.Game.Player<Action> {
  public async getMove(state: State): Promise<Action> {
    const possibleActions = state.getPossibleActions();
    const move = prompt("Enter your move: (col)") ?? "0";
    return new Action(parseInt(move));
  }
  public getName = (): string => "Human";
}

const TestFourInARow = async () => {
  // console.clear();
  const env = new GameEnvironment();
  // const dt = new Reinforcement.PreTrained.FileDecisionTable<Action>(
  //   `../saved/${WIN_LENGTH}_InARow_${ROWS}x${COLS}`
  // );
  const table = new Reinforcement.PreTrained.MemoryDecisionTable<Action>();
  await Reinforcement.PreTrained.multiPlayerTrain(env, 0.999, table);

  // seq.fit(
  //   tf.tensor2d(
  //     Object.values(table.table).map(({ action }, state) => state)
  //   ) as tf.Tensor2D,
  //   tf.tensor2d(
  //     Object.values(table.table).map(({ action }, state) => action.col)
  //   ) as tf.Tensor2D,
  //   {
  //     epochs: 100,
  //     batchSize: 32,
  //   }
  // );

  // console.log(Object.keys(table.table).length);
  const bot = new Reinforcement.Game.BotPlayer(table, "1");
  const bot2 = new Reinforcement.Game.BotPlayer(table, "2");
  const human = new HumanPlayer();
  const random = new Reinforcement.Game.RandomPlayer();
  const game = new Reinforcement.Game.Game(env, [bot, bot2]);
  // const game = new Reinforcement.Game.Game(env, [bot, human]);
  game.play(true);
};

export const main = TestFourInARow;
