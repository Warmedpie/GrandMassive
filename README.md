# GrandMassive

GrandMassive is a UCI a body building chess engine with a neural net for a brain and a search that bench-presses the game tree.

## How it works

Eval: an NNUE (efficiently updatable neural network), trained on the engine's own self-play games. The original hand-written evaluation can still be built as an option.

Search: alpha-beta with principal variation search (PVS) and iterative deepening, a transposition table, the usual pruning and reduction techniques, and multithreading (Lazy SMP).

