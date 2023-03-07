import * as Reinforcement from "./reinforcement";
import { prompt } from "./index";
// import * as Reinforcement from "./reinforcement.ts";

// tensorflow is not supported in deno yet
import * as tf from "@tensorflow/tfjs-node";

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

  public toVector = (): number[] => this.board.flat();
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

  public actionToString = (action: Action): string =>
    `[${action.row}, ${action.col}]`;
}

// Game:

export class HumanPlayer extends Reinforcement.Game.Player<Action> {
  public async getMove(state: State): Promise<Action> {
    const possibleActions = state.getPossibleActions();
    const move = (await prompt("Enter your move: (row, col)"))?.split(",") as [
      string,
      string
    ];
    return new Action(parseInt(move[0]), parseInt(move[1]));
  }
  public getName = (): string => "Human";
}

const TestTicTacToe = async () => {
  // console.clear();
  const env = new GameEnvironment();

  const table = new Reinforcement.PreTrained.MemoryDecisionTable<Action>();
  // const table = new Reinforcement.PreTrained.FileDecisionTable<Action>(
  //   "../saved/TicTacToe"
  // );

  await Reinforcement.PreTrained.multiPlayerTrain(env, 0.999, table);
  // const table = JSON.parse(
  //   await Deno.readTextFile("./coursera3/saved/TicTacToe.json")
  // ) as Reinforcement.DecisionTable<Action>;
  // await Deno.writeTextFile(
  //   "./coursera3/saved/TicTacToe.json",
  //   JSON.stringify(table)
  // );
  const bot = new Reinforcement.Game.BotPlayer(table, "1");
  const bot2 = new Reinforcement.Game.BotPlayer(table, "2");
  const RTbot = new Reinforcement.Game.RealTimeBotPlayer<Action>(
    Reinforcement.RealTime.getMinimaxDecider(env, 1000),
    "3"
  );
  const human = new HumanPlayer();
  const random = new Reinforcement.Game.RandomPlayer();

  // tensorflow model that accepts a vector of length 9 (board) and outputs the estimated reward value of the state
  const model = tf.sequential();
  model.add(
    tf.layers.dense({
      units: 729,
      inputShape: [9],
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 243,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 81,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 27,
      activation: "relu",
    })
  );
  model.add(
    tf.layers.dense({
      units: 1,
      activation: "linear",
    })
  );
  model.compile({
    optimizer: "rmsprop",
    loss: "meanSquaredError",
    metrics: ["accuracy"],
  });
  await Reinforcement.PreTrained.modelTrain(
    env,
    (state, gamma) => {
      let result = Reinforcement.RealTime.minimaxDecision(
        env,
        state,
        1000,
        gamma
      )[0];
      if (result < 0) console.log(result);
      return result;
    },
    10000,
    model,
    0.999,
    10000
  );
  const ModelBot = new Reinforcement.Game.ModelBotPlayer<Action>(model, "4");

  const game = new Reinforcement.Game.Game(env, [RTbot, ModelBot]);
  // const game = new Reinforcement.Game.Game(env, [RTbot, human]);
  game.play(true);

  // while (true) {
  //   let game = new Reinforcement.Game.Game(env, [bot, ModelBot]);
  //   await game.play(true);
  //   game = new Reinforcement.Game.Game(env, [ModelBot, human]);
  //   await game.play(true);
  //   game = new Reinforcement.Game.Game(env, [human, ModelBot]);
  //   await game.play(true);
  // }
  // const state = env.reverseState(
  //   env.getInitialStates()[0].moveCopy(new Action(0, 0))
  // );
  // for (const action of state.getPossibleActions()) {
  //   console.log(
  //     action.toString(),
  //     model.predict(tf.tensor2d([state.moveCopy(action).toVector()])).toString()
  //   );
  // }
};

export const main = TestTicTacToe;
