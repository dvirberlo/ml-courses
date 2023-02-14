/// Path: coursera3\dart\reinforcement.dart

import 'dart:convert';
import 'dart:io';
import 'dart:math';

class QDecision<Action> {
  final Action action;
  final double reward;
  const QDecision(this.action, this.reward);
  QDecision.fromJson(Map<String, dynamic> json)
      : action = json['action'],
        reward = json['reward'];
  Map<String, dynamic> toJson() {
    return {
      'action': action,
      'reward': reward,
    };
  }

  @override
  String toString() {
    return 'QDecision{action: $action, reward: $reward}';
  }
}

abstract class DecisionTable<Action> {
  Future<QDecision<Action>?> get(String hash);
  Future<void> set(String hash, QDecision<Action> decision);
  Future<bool> has(String hash);
}

class MemoryDecisionTable<Action> implements DecisionTable<Action> {
  final Map<String, QDecision<Action>> table = {};
  Future<QDecision<Action>?> get(String hash) async {
    return table[hash];
  }

  Future<void> set(String hash, QDecision<Action> decision) async {
    table[hash] = decision;
  }

  Future<bool> has(String hash) async {
    return table.containsKey(hash);
  }

  @override
  String toString() {
    return table.toString();
  }

  void toPrint() {
    for (var key in table.keys.toList().reversed) {
      print('$key: ${table[key]}');
    }
  }
}

class FileDecisionTable<Action> implements DecisionTable<Action> {
  final String dirPath;
  const FileDecisionTable(this.dirPath);
  String hashString(String str) {
    int hash = 0;
    for (int i = 0, len = str.length; i < len; i++) {
      int chr = str.codeUnitAt(i);
      hash = ((hash << 5) - hash + chr) | 0;
    }
    return hash.toRadixString(16);
  }

  Future<QDecision<Action>?> get(String hash) async {
    try {
      return QDecision<Action>.fromJson(jsonDecode(
          await File('${dirPath}/${hashString(hash)}.json').readAsString()));
    } catch (e) {
      print(
          'error while reading and parsing ${dirPath}/${hashString(hash)}.json');
    }
  }

  Future<void> set(String hash, QDecision<Action> decision) async {
    try {
      await File('${dirPath}/${hashString(hash)}.json')
          .writeAsString(jsonEncode(decision.toJson()));
    } catch (e) {
      print(
          'error while signifying and writing ${dirPath}/${hashString(hash)}.json');
    }
  }

  Future<bool> has(String hash) async {
    try {
      return File('${dirPath}/${hashString(hash)}.json').existsSync();
    } catch (e) {
      print(
          'error while checking status of ${dirPath}/${hashString(hash)}.json');
      return false;
    }
  }
}

abstract class State<Action> {
  const State();
  List<Action> getPossibleActions();
  bool isTerminal();
  State<Action> moveCopy(Action action);
  String toHash();
}

abstract class RewardEnvironment<Action> {
  List<State<Action>> getInitialStates();
  int getStateReward(State<Action> state);
}

abstract class RewardTwoPlayerEnvironment<Action>
    extends RewardEnvironment<Action> {
  int getStateReward(State<Action> state);
  State<Action> reverseState(State<Action> state);

  double reverseReward(double reward) => -reward;
}

Future<DecisionTable<Action>> multiPlayerTrain<Action>(
  RewardTwoPlayerEnvironment<Action> env, {
  double gamma = 1,
  required DecisionTable<Action> decisionTable,
}) async {
  /// Description: getReward is a recursive function that calculates the reward of a state by calculating the reward of all possible next states
  /// Param: state the state to calculate the reward for
  /// Returns the reward of the state, converted to be the reward of the previous player's perspective
  /// ---
  /// it uses the decision table as a cache \
  /// it returns the reward of the state \
  /// it fills the decision table with the best action for each state \
  /// the returned reward is converted to be the reward of the previous player's perspective
  Future<double> getReward(State<Action> state) async {
    final stateHash = state.toHash();
    // final _l = Object.keys(decisionTable).length;
    // if (_l % 1000 < 5) print(_l);
    if (await decisionTable.has(stateHash))
      return env.reverseReward((await decisionTable.get(stateHash))!.reward);
    if (state.isTerminal())
      return env.reverseReward(env.getStateReward(state).toDouble());
    final best = (await Future.wait(
      state.getPossibleActions().map((action) async => QDecision(
          action, await getReward(env.reverseState(state.moveCopy(action))))),
    ))
        .reduce((a, b) => a.reward >= b.reward ? a : b);
    await decisionTable.set(stateHash, best);
    return env.reverseReward(gamma * best.reward);
  }

  for (final state in env.getInitialStates()) await getReward(state);

  return decisionTable;
}

abstract class Player<Action> {
  Future<Action> getMove(State<Action> state);
  String getName();
}

class RandomPlayer<Action> implements Player<Action> {
  final String id;
  RandomPlayer({String? id}) : id = id ?? Random().nextInt(100000).toString();
  Future<Action> getMove(State<Action> state) async {
    final possibleActions = state.getPossibleActions();
    return possibleActions[Random().nextInt(possibleActions.length)];
  }

  String getName() => "Random#$id";
}

class HumanPlayer<Action> implements Player<Action> {
  Future<Action> getMove(State<Action> state) async {
    final possibleActions = state.getPossibleActions();
    print("Possible actions: $possibleActions");
    return possibleActions[int.parse(stdin.readLineSync()!)];
  }

  String getName() => "Human";
}

class BotPlayer<Action> implements Player<Action> {
  final DecisionTable<Action> decisionTable;
  final String id;
  BotPlayer(this.decisionTable, {String? id})
      : id = id ?? Random().nextInt(100000).toString();
  Future<Action> getMove(State<Action> state) async {
    final decision = await decisionTable.get(state.toHash());
    if (decision == null) throw Exception("No decision found");
    return decision.action;
  }

  String getName() => "Bot#$id";
}

abstract class Game {
  final List<Player> players;
  final RewardEnvironment env;
  Game(this.players, this.env);
  Future<void> play({bool render = false}) async {
    final inits = env.getInitialStates();
    State state = inits[Random().nextInt(inits.length)];
    int playerTurn = 0;
    if (render) {
      print(
          'Game: ${players.map((p) => p.getName()).join(" vs ")}\n${state.toString()}');
    }
    while (!state.isTerminal()) {
      final player = players[playerTurn];
      final move = await player.getMove(state);
      state = state.moveCopy(move);
      if (render) print('${player.getName()}: +$move, ${state.toString()}');
      playerTurn = (playerTurn + 1) % 2;
    }
    final winner = players[playerTurn - 1];
    if (render) print('Game: winner is ${winner.getName()}');
  }
}

class TwoPlayerGame implements Game {
  final List<Player> players;
  final RewardTwoPlayerEnvironment env;
  TwoPlayerGame(this.players, this.env);
  Future<void> play({bool render = false}) async {
    final inits = env.getInitialStates();
    State state = inits[Random().nextInt(inits.length)];
    int playerTurn = 0;

    if (render) {
      print('Game: ${players.map((p) => p.getName()).join(" vs ")}');
      print(state.toString());
    }

    while (!state.isTerminal()) {
      final player = players[playerTurn];
      final move = await player.getMove(state);
      state = state.moveCopy(move);
      if (render)
        print('${player.getName()}: ${move}: \n${state.toString()}\n');
      playerTurn = (playerTurn + 1) % 2;
      state = env.reverseState(state);
    }

    if (render) {
      switch (env.getStateReward(state)) {
        case 1:
          print('Game: winner is ${players[playerTurn].getName()}');
          break;
        case -1:
          print('Game: winner is ${players[(playerTurn + 1) % 2].getName()}');
          break;
        case 0:
          print('Game: draw');
          break;
      }
    }
  }
}
