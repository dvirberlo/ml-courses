<div style="text-align:center">

# **Extra - Reinforcement**

</div>

# Intuitive Game Learning

Most games have one problem: the reward (or punishment) is only at the end. \
This fact make it difficult to develop good strategies or continuous reward system from that. \
In fact, the intuitive way to solve this problem is to go all the down to the last state of the game, obtain the reward, and slowly learn up from that until the initial state is reached - and we have a perfect strategy.

## Perfect Strategy

Of course, this process requires every single state to be classified in advanced, but once it is done- we have a model that plays **Perfectly**. \
I am not sure I can prove it is this model will always take the best action. \ <!-- TODO -->
But I have tested it in some games I know for certain it achieve the best result, so...

But, once we get to a complex games that have deep state tree, the model can be so large that it won't fit in an average computer's disk. \
Now, of course we can say that this is hardware, and not concept limitation. \
But is there a way to do better?

The whimsical thing to say is that it is the Perfectionist-Realistic tradeoff, again. \
And we can talk later about some "good enough" approaches, "given the current existing hardware". \
But why should we throw the dream away so fast?

## Human's Perspective

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

### Real Time decisions

Another way we could approach this, is to give up on a well pre trained and enormous model that has everything sorted out. \
Instead, of disk-space based model, we can opt for a CPU (realtime and maybe on-client) based model. \
This way, we can every turn calculate the minimal part of the tree that is needed to make the decision.

Although my initial intuitive code was actually very similar to the non optimized minimax (instead of calling it "min" I used the same max, with negative reward), we could use "alpha-beta pruning" to reduce the number of states we need to calculate. \
This way, we do not give up the dream of a perfect decision, but we do it in a more realistic way. \
Or is it?

# Monte Carlo Tree Search

$$
\begin{gather*}
UCB1 (S_i) = \bar{v}_i + C \cdot\sqrt{\frac{\ln N}{n_i}} \\
\bar{v}_i = \frac{t_i}{n_i} \\
N = n_{i-1} \\
C = 2
\end{gather*}
$$

`t_i` is the total sum of reward values, that rolled back to the node (i). \
`n_i` the number of times the node (i) was visited. \
`v_i` is the average reward value, at the node (i). \
`N` is the number of times the parent node (of node `i`) was visited.
`C` is a constant, that is used to balance the exploration and exploitation.

## The iteration

1. ("Selection:") From the root node, choose the child \[state] with the largest `UCB1` score, and do the same for the current node, until a leaf node is reached.
2. - ("Expansion:") If the node has already being visited once, add to the tree all its child nodes \
     (then choose randomly a child node).
   - If not, choose the current node.
3. ("Simulation:") From the current node, take a random path, until a terminal node is reached, and get its reward.
4. ("Back-Propagation:") Then, "back-propagate" and add the reward value to all parent nodes, till the root (and update the visited counters).

## The Result

After done training, the model will choose the action that will lead to the node with the largest `\bar{v_i}` (average reward value `t/n`).

In some cases, since the model is attracted to the more promising states, the fraction `t/n` may be greater for the less promising state. \
Therefore, in some cases to make a final decision, some simulations are performed to the less visited nodes, to make a more accurate decision, with an equal visits count for all the nodes.

## The Idea

The hope, is that when the iteration count is equal to the minimax iteration count, the model will be as good as the minimax model. \
But, when it is lower, it will still to well.

It is implemented so that it will explore and exploit, in a statistical way, the entire state tree. \
But, the key limitation is that it will have any information or strategy against a weak opponent, since it tends to explore the promising parts of the tree.
But, as I noticed during implementation phases, the algorithm is optimizing the action only for the selected state, and do not generate any strategy for the next states, or a general model/ strategy for the game. \
So, it is real-time maybe on-client decision making, but not a model that tries to make some generalized idea about the game.
