# Deep Solitire

A minimal demo runner for Deep Solitire.

Tokenized Solitire states are embedded and fed to a RoPE-enhanced Transformer.

It supports two decision policies:

- **One-step ε-greedy** (default)
- **Monte Carlo Tree Search (MCTS)**

## Usage
```bash
python demo.py --help

usage: demo.py [-h] [--mcts | --greedy]

Deep Solitire demo runner

options:
  -h, --help  show this help message and exit
  --mcts      Use Monte Carlo Tree Search instead of one-step greedy.
  --greedy    Force one-step epsilon-greedy (default).
```

## Examples
```bash
# Run with the default one-step ε-greedy policy
python demo.py --greedy

# Run with Monte Carlo Tree Search
python demo.py --mcts
```

## Sample Results

Completion rate (i.e., “all cards opened”), with a maximum of 200 moves:

-- One-step ε-greedy (ε = 0.1): 28.8%

-- Monte Carlo Tree Search (ε = 0.1, iterations = 1000): 37%

## Card image source
By Byron Knoll [OpenGameArt.org](https://opengameart.org/content/playing-cards-vector-png#:~:text=Playing%20Cards%20)

By Aidan_Walker [OpenGameArt.org](https://opengameart.org/content/playing-cards-5#:~:text=File)

