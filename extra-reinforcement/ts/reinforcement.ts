// export type DecisionTable<Action> = {
//   [state: string]: QDecision<Action>;
// };

export abstract class State<Action> {
  public abstract getPossibleActions(): Action[];
  public abstract isTerminal(): boolean;
  public abstract move(action: Action): void;
  public abstract moveCopy(action: Action): State<Action>;
  public abstract toHash(): string;
}

export abstract class RewardEnvironment<Action> {
  public abstract getInitialStates(): State<Action>[];
  public abstract getStateReward(state: State<Action>): number;
  public abstract actionToString(action: Action): string;
}

export abstract class RewardTwoPlayerEnvironment<
  Action
> extends RewardEnvironment<Action> {
  public abstract getStateReward(
    state: State<Action>,
    stateDepth?: number
  ): number;
  public abstract reverseState(state: State<Action>): State<Action>;

  public readonly reverseReward = (reward: number): number => -reward;
  public abstract getWinner(state: State<Action>): number;
}

export module PreTrained {
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
    private dirCreated;
    constructor(public readonly dirPath: string) {
      this.dirCreated = Deno.mkdir(dirPath, { recursive: true });
    }
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
      await this.dirCreated;
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
      await this.dirCreated;
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
      await this.dirCreated;
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
      // const best = (
      //   await Promise.all(
      //     state.getPossibleActions().map(async (action) => ({
      //       action,
      //       reward: await getReward(env.reverseState(state.moveCopy(action))),
      //     }))
      //   )
      // ).reduce((a, b) => (a.reward >= b.reward ? a : b));
      // with loop:
      const actions = state.getPossibleActions();
      let best: QDecision<Action>;
      if (actions.length === 1)
        best = {
          action: actions[0],
          reward: await getReward(env.reverseState(state.moveCopy(actions[0]))),
        };
      else {
        best = {
          action: actions[0],
          reward: -Infinity,
        };
        for (const action of actions) {
          const reward = await getReward(
            env.reverseState(state.moveCopy(action))
          );
          if (reward > best.reward) best = { action, reward };
        }

        decisionTable.set(stateHash, best);
      }
      return env.reverseReward(gamma * best.reward);
    };
    for (const state of env.getInitialStates()) await getReward(state);

    return decisionTable;
  };
}

/**
 * 0
 *  +1 (0)
 *    1
 *     +1 (0)
 *       2
 *        +1 (0)
 *          3 ***
 *     +2 (1)
 *       3 ***
 *  +2 (1)
 *    2
 *     +1 (0)
 *       3 ***
 */

//  -> 1 -> 1 -> 1
//       \> 2
//  \> 2 -> 1

export module Game {
  export abstract class Player<Action> {
    public abstract getMove(state: State<Action>): Promise<Action>;
    public abstract getName(): string;
  }
  export class RandomPlayer<Action> extends Player<Action> {
    constructor(private readonly id = Math.random().toString(32).slice(2, 4)) {
      super();
    }
    public async getMove(state: State<Action>): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      return possibleActions[
        Math.floor(Math.random() * possibleActions.length)
      ];
    }
    public getName = (): string => `Random#${this.id}`;
  }
  export class HumanPlayer<Action> extends Player<Action> {
    public async getMove(state: State<Action>): Promise<Action> {
      const possibleActions = state.getPossibleActions();
      console.log("Possible moves:", possibleActions);
      const input = await prompt("Your move: ");
      const action = possibleActions[parseInt(input ?? "0")];
      if (action === undefined) throw new Error("Invalid move");
      return action;
    }
    public getName = (): string => "Human";
  }
  export class BotPlayer<Action> extends Player<Action> {
    constructor(
      public readonly decisionTable: PreTrained.DecisionTable<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public async getMove(state: State<Action>): Promise<Action> {
      if (state.getPossibleActions().length === 1)
        return state.getPossibleActions()[0];
      const decision = await this.decisionTable.get(state.toHash());
      if (decision === undefined) throw new Error("No decision found");
      return decision.action;
    }
    public getName = (): string => `Bot#${this.id}`;
  }

  export class RealTimeBotPlayer<Action> extends Player<Action> {
    constructor(
      public readonly getAction: (state: State<Action>) => Promise<Action>,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public getMove = (state: State<Action>): Promise<Action> =>
      this.getAction(state);
    public getName = (): string => `Bot#${this.id}`;
  }

  export class Game<Action> {
    constructor(
      public readonly env: RewardTwoPlayerEnvironment<Action>,
      public readonly players: Player<Action>[],
      public state: State<Action> = env.getInitialStates()[
        Math.floor(Math.random() * env.getInitialStates().length)
      ]
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
            `${player.getName()}: +${this.env.actionToString(
              move
            )}: \n${this.state.toString()}\n`
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
