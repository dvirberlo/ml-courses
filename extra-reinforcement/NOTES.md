# Extra - Reinforcement

## Intuitive Game Learning

Most games have one problem: the reward (or punishment) is only at the end. \
This fact make it difficult to develop good strategies or continuous reward system from that. \
In fact, the intuitive way to solve this problem is to go all the down to the last state of the game, obtain the reward, and slowly learn up from that until the initial state is reached - and we have a perfect strategy.

### Perfect Strategy

Of course, this process requires every single state to be classified in advanced, but once it is done- we have a model that plays **Perfectly**. \
I am not sure I can prove it is this model will always take the best action. \ <!-- TODO -->
But I have tested it in some games I know for certain it achieve the best result, so...

But, once we get to a complex games that have deep state tree, the model can be so large that it won't fit in an average computer's disk. \
Now, of course we can say that this is hardware, and not concept limitation. \
But is there a way to do better?

The whimsical thing to say is that it is the Perfectionist-Realistic tradeoff, again. \
And we can talk later about some "good enough" approaches, "given the current existing hardware". \
But why should we throw the dream away so fast?

### Human's Perspective

So, as humans we know that we ourselves have some ideas about how to learn, don't we? \
We begin our life as a very unexperienced babies with small to none reasoning or logics. \
But we grow up, and we learn. It takes years, but eventually we do (most of us, at least 😉).

But if you happen to read a smart person's writings, he might suggest that our brain is actually pre trained for many years, since the prehistoric ages. \
You may call it evolution, that forced the human brain to develop and progress. \
So, the pre-trained newborn brain is actually somewhat similar to a Supervised Learning model, where the supervisor is nature and survival, say something like a NN, just a large one, with ENORMOUS data set, of all the ancestors.

But wait, when we actually do end up learning something new, say a new game, we do learn for scratch, and we don't think about the X number of possible states all the way down to the last reward state. \
We try to recognize patterns, develop simple and complex strategies, memorize some key states and moves, and more.

Maybe we have some existing NN and we Transfer them to the new subject or game. \
All this process, is usually iterative. The more games and experience - the decisions are smarter and better.

We have a limited memory. How much moves can the best Chess player think ahead? 16? 17? \
We are not capable of holding in our brain all the states and possibilities. \
One might say it is a relief. If we could see too much ahead - we would be overwhelmed. We could not bare it, mentally.

## Monte Carlo Tree Search

$$
\begin{gather*}
UCB1 (S_i) = \bar{v_i} + 2\sqrt{\frac{\ln N}{n_i}} \\
\bar{v_i} = \frac{t_i}{n_i} \\
N = n_{i-1}
\end{gather*}
$$

`t_i` is the total sum of reward values, that rolled back to the node (i). \
`n_i` the number of times the node (i) was visited. \
`v_i` is the average reward value, at the node (i). \
`N` is the number of times the parent node was visited.

<!-- TODO: I think `N` is equal to `n_{i-1}`  -->

## The iteration:

1. From top node, choose the child \[state] with the largest `UCB1` score.
2. \
   If the node has already being visited once, add to the tree all its child nodes
   (then choose randomly a child node). \
   If not, choose the current node.
3. From the current node, take a random path, until a terminal node is reached, and get its reward.\
   Then, "rollback" and add the reward value to all parent nodes, till the root (and update the visited counters).

## The Result

After done training, the model will choose the action that will lead to the node with the largest `\bar{v_i}` (average reward value `t/n`). \
(Obviously, for real-world shipping the tree can be reduced to state:action table).

## The Idea
