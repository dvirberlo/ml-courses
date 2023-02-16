import 'reinforcement.dart';

enum _Action {
  Add1,
  Add2,
  Add3;

  @override
  String toString() => "+${this.name.substring(3)}";

  int get value => int.parse(this.name.substring(3));
}

class _State implements State<_Action> {
  final int currentPile;
  final int pileSize;
  const _State(this.pileSize, {this.currentPile = 0});
  List<_Action> getPossibleActions() => _Action.values;
  bool isTerminal() => currentPile >= pileSize;
  _State moveCopy(_Action action) {
    return _State(pileSize, currentPile: currentPile + action.value);
  }

  String toHash() => currentPile.toString();
  String toString() => '$currentPile/$pileSize';
}

class _GameEnvironment extends RewardTwoPlayerEnvironment<_Action> {
  final int pile_size;
  _GameEnvironment(this.pile_size);
  List<_State> getInitialStates() => [_State(pile_size)];
  int getStateReward(State<_Action> state) => state.isTerminal() ? -1 : 0;
  State<_Action> reverseState(State<_Action> state) => state;
}

void main(List<String> args) {
  _main(args);
}

void _main(List<String> args) async {
  final env = _GameEnvironment(20);
  final table = MemoryDecisionTable<_Action>();
  await multiPlayerTrain(env, decisionTable: table, gamma: 1);
  table.toPrint();
  final random = RandomPlayer<_Action>(id: '1');
  final random2 = RandomPlayer<_Action>(id: '2');
  final human = HumanPlayer<_Action>();
  final bot = BotPlayer<_Action>(table);
  // final players = [bot, random];
  // final game = TwoPlayerGame(players, env);
  // game.play(render: true);
}
