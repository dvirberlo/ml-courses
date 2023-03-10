import * as Reinforcement from "./reinforcement";
// import * as Reinforcement from "./reinforcement.ts";

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
    return possibleActions[Math.floor(Math.random() * possibleActions.length)];
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
    public readonly decisionTable: Reinforcement.PreTrained.DecisionTable<Action>,
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
      console.log(`Game: ${this.players.map((p) => p.getName()).join(" vs ")}`);
      console.log(this.state.toString());
    }
    while (!this.state.isTerminal()) {
      const player = this.players[playerTurn];
      const move = await player.getMove(this.state);
      this.state.move(move);
      if (render)
        console.log(`${player.getName()}: +${move}, ${this.state.toString()}`);
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
  public getPossibleActions = () =>
    [Action.Add1, Action.Add2, Action.Add3].slice(
      0,
      this.pile_size - this.current_pile
    );
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
  public getStateReward = (state: State, stateDepth = 1): number =>
    state.isTerminal() ? (-1) ** stateDepth : 0;
  public reverseState = (state: State): State => state;

  public getWinner = (state: Reinforcement.State<Action>): number =>
    state.isTerminal() ? 1 : 0;

  public actionToString = (action: Action): string => action;
}

const TestKingCoin3 = async () => {
  // console.clear();
  const coins = 10;
  const env = new GameEnvironment(coins);

  const table = await Reinforcement.PreTrained.multiPlayerTrain(env, 1);
  // console.log(table);
  const bot = new Reinforcement.Game.BotPlayer(table, "1");
  const bot2 = new Reinforcement.Game.BotPlayer(table, "2");
  const RTbot = new Reinforcement.Game.RealTimeBotPlayer(
    Reinforcement.RealTime.getMinimaxDecider(env, 1000),
    "3"
  );
  const human = new HumanPlayer();
  const random = new Reinforcement.Game.RandomPlayer<Action>();
  // const game = new Game([bot, human], new State(coins));
  const game = new Game([RTbot, random], new State(coins));
  game.play(true);
};

export const main = TestKingCoin3;
