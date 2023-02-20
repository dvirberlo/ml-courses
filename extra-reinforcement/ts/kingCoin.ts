import * as Reinforcement from "./reinforcement.ts";

/**
 * coin king game  works like this:
 * each turn you can add 1 or 2 coins to the pile
 * the player who adds the last coin wins
 *
 * for simplicity, the game will be allays played by 2 players
 */

export enum Action {
  Add1 = "1",
  Add2 = "2",
}
export const pileMove = (action: Action): number =>
  action === Action.Add1 ? 1 : 2;

export class HumanPlayer extends Reinforcement.Game.Player<Action> {
  public async getMove(state: State): Promise<Action> {
    const possibleActions = state.getPossibleActions();
    const move = prompt("Enter your move: 1 or 2");
    if (parseInt(move || "0") === 1) return Action.Add1;
    else return Action.Add2;
  }
  public getName = (): string => "Human";
}

export class Game {
  constructor(
    public readonly players: Reinforcement.Game.Player<Action>[],
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
    [Action.Add1, Action.Add2].slice(0, this.pile_size - this.current_pile);
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
  // state.isTerminal() ? -1 : 0;
  public reverseState = (state: State): State => state;

  public getWinner = (state: Reinforcement.State<Action>): number =>
    state.isTerminal() ? 1 : 0;

  public actionToString = (action: Action): string => action;
}

const TestKingCoin = async () => {
  // console.clear();
  const coins = 5;
  const env = new GameEnvironment(coins);

  const table = await Reinforcement.PreTrained.multiPlayerTrain(env, 1);
  console.log(table);

  const bot = new Reinforcement.Game.BotPlayer(table, "1");
  const bot2 = new Reinforcement.Game.BotPlayer(table, "2");
  const human = new HumanPlayer();
  const random = new Reinforcement.Game.RandomPlayer();
  // const game = new Game([bot, human], new State(coins));
  const game = new Game([bot, bot2], new State(coins));
  game.play(true);
};

export const main = TestKingCoin;
