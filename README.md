# Deep Solitire

A minimal demo runner for Deep Solitire.

Tokenized Solitire states are embedded and fed to a RoPE-enhanced Transformer.

It supports two decision policies:

- **One-step ε-greedy** (default)
- **Monte Carlo Tree Search (MCTS)**

## Usage
```bash
python setup.py build_ext --inplace

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
python setup.py build_ext --inplace

# Run with the default one-step ε-greedy policy
python demo.py --greedy

# Run with Monte Carlo Tree Search
python demo.py --mcts
```

## Sample Results

Completion rate (i.e., “all cards opened”), with a maximum of 200 moves:
| Group | Complete/Total   | Win rate     | 95%CI (Wilson) |
|---|-----------:|---------:|---------------:|
| One-step ε-greedy (ε = 0.1) | 18250/63293 | **28.83%** | 28.48–29.19%   |
| Monte Carlo Tree Search (ε = 0.1, iterations = 1000) |  5793/15131 | **38.29%** | 37.51–39.06%   |

## Card image source
By Byron Knoll [OpenGameArt.org](https://opengameart.org/content/playing-cards-vector-png#:~:text=Playing%20Cards%20)

By Aidan_Walker [OpenGameArt.org](https://opengameart.org/content/playing-cards-5#:~:text=File)

