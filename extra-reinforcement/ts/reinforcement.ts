import fs from "fs";

// export module IO {
//   export const writePromise = (file: string, data: string) =>
//     Deno.writeTextFile(file, data, { create: true });

//   export const readPromise = (file: string) => Deno.readTextFile(file);

//   export const existsPromise = (file: string) =>
//     Deno.stat(file)
//       .then(() => true)
//       .catch(() => false);

//   export const mkdirPromise = (dir: string) =>
//     Deno.mkdir(dir, { recursive: true });
// }

export module IO {
  // TODO: create?
  export const writePromise = (file: string, data: string) =>
    new Promise<void>((res, rej) => {
      fs.writeFile(file, data, (err) => {
        if (err) return rej(err);
        else return res();
      });
    });

  export const readPromise = (file: string) =>
    new Promise<string>((res, rej) => {
      fs.readFile(file, "utf-8", (err, data) => {
        if (err) return rej(err);
        return res(data);
      });
    });

  export const existsPromise = (file: string) =>
    fs.promises
      .access(file, fs.constants.F_OK)
      .then(() => true)
      .catch(() => false);

  export const mkdirPromise = (dir: string) =>
    fs.promises.mkdir(dir, { recursive: true });
}
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
      this.dirCreated = IO.mkdirPromise(dirPath);
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
          await IO.readPromise(`${this.dirPath}/${this.hashString(hash)}.json`)
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
        await IO.writePromise(
          `${this.dirPath}/${this.hashString(hash)}.json`,
          JSON.stringify(decision)
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
        return await IO.existsPromise(
          `${this.dirPath}/${this.hashString(hash)}.json`
        );
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
      public readonly getAction: (
        state: State<Action>
      ) => Promise<Action> | Action,
      private readonly id = Math.random().toString(32).slice(2, 4)
    ) {
      super();
    }
    public getMove = async (state: State<Action>): Promise<Action> =>
      await this.getAction(state);
    public getName = (): string => `RTbot#${this.id}`;
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

export module RealTime {
  export enum Minimax {
    Max = 1,
    Min = -1,
  }
  export const minimaxWithAlphaBetaPruning = <Action>(
    env: RewardTwoPlayerEnvironment<Action>,
    state: State<Action>,
    maxDepth: number,
    gamma = 0.999,
    method = Minimax.Max
  ) =>
    _minimaxWithAlphaBetaPruning(
      env,
      state,
      maxDepth,
      gamma,
      -Infinity,
      Infinity,
      method
    );
  const _minimaxWithAlphaBetaPruning = <Action>(
    env: RewardTwoPlayerEnvironment<Action>,
    state: State<Action>,
    maxDepth: number,
    gamma = 0.999,
    alpha = -Infinity,
    beta = Infinity,
    method = Minimax.Max
  ): [number, Action?] => {
    if (state.isTerminal() || maxDepth === 0)
      return [env.getStateReward(state) * method];
    const possibleActions = state.getPossibleActions();
    let bestValue = method === Minimax.Max ? -Infinity : Infinity;
    let bestAction: Action | undefined;
    for (const action of possibleActions) {
      const nextState = env.reverseState(state.moveCopy(action));
      // note: the reverseState also reverses the reward
      const [value] = _minimaxWithAlphaBetaPruning(
        env,
        nextState,
        maxDepth - 1,
        gamma,
        alpha,
        beta,
        -method
      );
      if (method === Minimax.Max) {
        if (value > bestValue) {
          bestValue = value;
          bestAction = action;
        }
        alpha = Math.max(alpha, bestValue);
      }
      if (method === Minimax.Min) {
        if (value < bestValue) {
          bestValue = value;
          bestAction = action;
        }
        beta = Math.min(beta, bestValue);
      }
      if (beta <= alpha) break;
    }
    return [bestValue * gamma, bestAction];
  };

  export const minimaxDecision = <Action>(
    env: RewardTwoPlayerEnvironment<Action>,
    state: State<Action>,
    depth: number,
    gamma?: number
  ): Action => {
    const [value, action] = minimaxWithAlphaBetaPruning(
      env,
      state,
      depth,
      gamma,
      Minimax.Max
    );
    return action ?? state.getPossibleActions()[0];
  };

  export const getMinimaxDecider = <Action>(
    env: RewardTwoPlayerEnvironment<Action>,
    depth: number,
    gamma?: number
  ): ((state: State<Action>) => Action) => {
    return (state) => minimaxDecision(env, state, depth, gamma);
  };

  // export abstract class StateNode<Action> {
  //   constructor(public totalReward: number = 0, public visits: number = 0) {}
  //   public abstract getState(): State<Action> | Promise<State<Action>>;
  //   public abstract getParent():
  //     | Promise<StateNode<Action> | undefined>
  //     | (StateNode<Action> | undefined);
  //   public abstract getChildren():
  //     | Promise<[Action, StateNode<Action>][] | undefined>
  //     | ([Action, StateNode<Action>][] | undefined);
  //   public abstract createChildren():
  //     | Promise<[Action, StateNode<Action>][] | undefined>
  //     | ([Action, StateNode<Action>][] | undefined);
  //   public getAverageReward = (): number => this.totalReward / this.visits;
  //   public USB1 = (logN: number): number =>
  //     this.visits === 0
  //       ? Infinity
  //       : this.getAverageReward() + 2 * Math.sqrt(logN / this.visits);
  //   public propagate = (value: number): void | Promise<void> => {
  //     this.totalReward += value;
  //     this.visits++;
  //   };
  //   public simulate = async (
  //     env: RewardTwoPlayerEnvironment<Action>
  //   ): Promise<number> => {
  //     let stateDepth = 0;
  //     let rewardState = await this.getState();
  //     while (!rewardState.isTerminal()) {
  //       const actions = rewardState.getPossibleActions();
  //       rewardState = env.reverseState(
  //         rewardState.moveCopy(
  //           actions[Math.floor(Math.random() * actions.length)]
  //         )
  //       );
  //       stateDepth++;
  //     }
  //     return env.getStateReward(rewardState, stateDepth);
  //   };
  // }
  // export class StateTree<Action> {
  //   constructor(public root: StateNode<Action>) {}
  //   public async print(): Promise<void> {
  //     const printNode = async (
  //       node: StateNode<Action>,
  //       depth: number = 1
  //     ): Promise<void> => {
  //       console.log(
  //         `${depth === 1 ? "*" : ""}${" ".repeat(depth)}${node.totalReward
  //           .toString()
  //           .padStart(2)}/${node.visits}`
  //       );
  //       const children = await node.getChildren();
  //       if (children) {
  //         for (const [action, child] of children) {
  //           // console.log(`${" ".repeat(depth + 1)}${action}`);
  //           await printNode(child, depth + 2);
  //         }
  //       }
  //     };
  //     await printNode(this.root);
  //   }
  // }
  // export class MemoryStateNode<Action> extends StateNode<Action> {
  //   private children: [Action, StateNode<Action>][] | undefined;
  //   constructor(
  //     public readonly env: RewardTwoPlayerEnvironment<Action>,
  //     private state: State<Action>,
  //     public totalReward: number = 0,
  //     public visits: number = 0,
  //     private parent?: StateNode<Action>
  //   ) {
  //     super(totalReward, visits);
  //   }
  //   public getState = (): State<Action> => this.state;
  //   public getParent = (): StateNode<Action> | undefined => this.parent;
  //   public getChildren = (): [Action, StateNode<Action>][] | undefined =>
  //     this.children;
  //   public createChildren = (): [Action, StateNode<Action>][] | undefined => {
  //     if (this.state.isTerminal()) return this.children;
  //     this.children ??= this.state
  //       .getPossibleActions()
  //       .map((action) => [
  //         action,
  //         new MemoryStateNode<Action>(
  //           this.env,
  //           this.env.reverseState(this.state.moveCopy(action)),
  //           0,
  //           0,
  //           this
  //         ),
  //       ]);
  //     return this.children;
  //   };
  // }
  // export class MemoryStateTree<Action> extends StateTree<Action> {
  //   constructor(env: RewardTwoPlayerEnvironment<Action>, state: State<Action>) {
  //     super(new MemoryStateNode<Action>(env, state));
  //   }
  // }
  // export const monteCarlo = async <Action>(
  //   env: RewardTwoPlayerEnvironment<Action>,
  //   state: State<Action>,
  //   gamma = 1,
  //   iterations = 1000
  // ): Promise<MemoryStateTree<Action>> => {
  //   // TODO: for now, assume that there is only one root state:
  //   const tree = new MemoryStateTree<Action>(env, state);
  //   while (iterations--) {
  //     // tree.print();
  //     let node = tree.root;
  //     let children = await node.getChildren();
  //     let depth = 0;
  //     // Selection:
  //     while (node.visits !== 0) {
  //       if (children === undefined) {
  //         children = await node.createChildren();
  //         if (children === undefined) break;
  //         node = children[0][1];
  //         break;
  //       }
  //       const logN = Math.log(node.visits);
  //       const best =
  //         depth % 2 === 0
  //           ? children.reduce((a, b) =>
  //               a[1].USB1(logN) >= b[1].USB1(logN) ? a : b
  //             )
  //           : children.reduce((a, b) =>
  //               a[1].USB1(logN) < b[1].USB1(logN) ? a : b
  //             );
  //       node = best[1];
  //       children = await node.getChildren();
  //       depth++;
  //     }
  //     // Simulation:
  //     let reward = await node.simulate(env);
  //     // Back Propagation:
  //     let rollNode: StateNode<Action> | undefined = node;
  //     while (true) {
  //       rollNode.propagate(reward);
  //       rollNode = await rollNode.getParent();
  //       reward *= gamma;
  //       if (rollNode === undefined) break;
  //     }
  //   }
  //   return tree;
  // };
  // export const getAction = async <Action>(
  //   env: RewardTwoPlayerEnvironment<Action>,
  //   state: State<Action>,
  //   gamma = 1,
  //   iterations = 1000
  // ): Promise<Action> => {
  //   const tree = await monteCarlo(env, state, gamma, iterations);
  //   const children = await tree.root.getChildren();
  //   if (children === undefined)
  //     throw new Error("Monte Carlo did not produced children");
  //   // await tree.print();
  //   if (true) {
  //     const mostVisited = children.reduce((a, b) =>
  //       a[1].visits >= b[1].visits ? a : b
  //     )[1];
  //     for (const child of children) {
  //       while (child[1].visits < mostVisited.visits) {
  //         let reward = await child[1].simulate(env);
  //         child[1].propagate(reward);
  //       }
  //     }
  //   }
  //   // await tree.print();
  //   return children.reduce((a, b) =>
  //     a[1].getAverageReward() >= b[1].getAverageReward() ? a : b
  //   )[0];
  // };
}
