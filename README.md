# 🐍 AI Snake Game

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python" alt="Python Version" />
  <img src="https://img.shields.io/badge/Framework-Tkinter%20%7C%20Pygame-informational?style=for-the-badge" alt="GUI Frameworks" />
  <img src="https://img.shields.io/badge/AI-Hamiltonian%20%7C%20Greedy%20%7C%20DQN-orange?style=for-the-badge" alt="AI Solvers" />
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-TensorFlow-red?style=for-the-badge&logo=tensorflow" alt="TensorFlow RL" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License" />
</p>

An intelligent, multi-algorithmic implementation of the classic Snake Game in Python. This project bridges classical graph theory, heuristic search algorithms, and deep reinforcement learning by featuring automated AI agents capable of achieving perfect map saturation alongside an interactive manual arcade mode.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [AI Solvers & Algorithms](#-ai-solvers--algorithms)
  - [1. Hamiltonian Cycle Solver (Optimal & Safe)](#1-hamiltonian-cycle-solver-optimal--safe)
  - [2. Greedy Pathfinding Solver (Lookahead & Tail Chase)](#2-greedy-pathfinding-solver-lookahead--tail-chase)
  - [3. Deep Q-Network Solver (Dueling DQN + PER)](#3-deep-q-network-solver-dueling-dqn--per)
- [Project Architecture](#-project-architecture)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [Running the AI Simulation](#running-the-ai-simulation)
  - [Configuring Solvers & Modes](#configuring-solvers--modes)
  - [Headless Benchmarking](#headless-benchmarking)
  - [Manual Arcade Play](#manual-arcade-play)
- [Controls & Keybindings](#-controls--keybindings)
- [Configuration Reference](#-configuration-reference)
- [Algorithm Comparison](#-algorithm-comparison)
- [Logging & Telemetry](#-logging--telemetry)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

The Snake problem is fundamentally a constrained pathfinding problem on a dynamic discrete grid. As the snake eats food and grows, previously navigable space becomes impassable, turning short-sighted greedy approaches into self-trapping dead ends.

This repository implements three distinct paradigms to conquer the game:
1. **Hamiltonian Cycle Traversal**: Mathematical guarantee of never colliding with the snake body or walls, enhanced by greedy shortcuts to cut traversal time.
2. **Heuristic Lookahead Pathfinding**: Breadth-First Search (BFS) combined with virtual simulation and safe tail-chasing escape routes.
3. **Deep Reinforcement Learning (DQN)**: Convolutional neural network trained with Dueling DQN architectures and Prioritized Experience Replay (PER).

---

## ✨ Key Features

- **Guaranteed Win Strategy**: Includes a Hamiltonian cycle generator that guarantees 100% board fill without self-collision.
- **Dynamic Shortcut Optimization**: Smart Hamiltonian shortcutting that calculates relative cycle distances, dramatically speeding up food acquisition while preserving cycle topology.
- **Dual Visual Modes**:
  - **Tkinter HUD Interface**: Real-time HUD displaying episode index, step count, current length, map capacity, and snake state.
  - **Pygame Arcade Interface**: Snappy, arcade-style manual game for human testing and play.
- **Headless Benchmarking**: Run batch simulations across arbitrary episode counts to measure average score, step efficiency, and survival rates.
- **Comprehensive Logging**: Detailed ASCII grid snapshots saved per step into `logs/snake.log` for debugging and analysis.

---

## 🧠 AI Solvers & Algorithms

### 1. Hamiltonian Cycle Solver (Optimal & Safe)
*Implemented in `snake/solver/hamilton.py`*

A Hamiltonian cycle is a closed loop visiting every vertex on the grid exactly once. 

- **Cycle Construction**: By computing the longest path between head and tail on an even-dimensioned grid, the solver maps every grid coordinate $(x, y)$ to an indexed cycle position.
- **Dynamic Shortcutting**: Following a static cycle is 100% safe but slow ($O(N^2)$ steps). When the snake's length is under $50\%$ of the board's capacity, the solver evaluates the shortest path to food and takes shortcuts across the loop **only if** the relative ordering of the next step remains strictly between the head and food relative to the tail:
  $$\text{dist}_{\text{rel}}(\text{tail}, \text{next}) \le \text{dist}_{\text{rel}}(\text{tail}, \text{food})$$
- **Result**: Retains a $100\%$ win rate while cutting total execution steps by up to $80\%$.

### 2. Greedy Pathfinding Solver (Lookahead & Tail Chase)
*Implemented in `snake/solver/greedy.py` & `snake/solver/path.py`*

A 5-step heuristic search designed to aggressively target food while avoiding traps:

1. **Shortest Path Probe**: Computes BFS shortest path from head to food.
2. **Virtual Simulation**: Clones the game state and simulates the snake following this path to eat the food.
3. **Tail Route Validation**: In the simulated future state, verifies whether a valid path exists from the new head to the new tail. If safe, commits to the path.
4. **Tail-Chasing Fallback**: If the food path traps the snake, the agent aborts and instead follows the **longest path to its own tail**, stalling safely while waiting for empty spaces to open up.
5. **Wandering Maneuver**: If no tail path exists, moves to the safest neighbor maximizing Manhattan distance from food to avoid boxing itself in.

### 3. Deep Q-Network Solver (Dueling DQN + PER)
*Implemented in `snake/solver/dqn/`*

An end-to-end reinforcement learning solver utilizing value-based deep learning:

- **State Representation**: Multi-channel spatial tensor encoding empty spaces, food, snake head, and snake body.
- **Dueling Architecture**: Separates the Q-value estimation into state-value $V(s)$ and advantage $A(s, a)$ streams:
  $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left(A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|}\sum_{a'} A(s, a'; \theta, \alpha)\right)$$
- **Prioritized Experience Replay (PER)**: Uses a binary `SumTree` data structure (`snake/util/sumtree.py`) to sample transitions according to their Temporal Difference (TD) error magnitude $|\delta|^\alpha$.
- **Action Space**: Relative steering (Turn Left, Forward, Turn Right) or Absolute cardinal directions.

---

## 📂 Project Architecture

```text
AI_Snake_Game/
├── AI_SnakeGame.py          # Entry point for AI game execution & configuration
├── Manual_SnakeGame.py      # Standalone Pygame arcade mode for manual play
├── requirements.txt         # Project dependencies
├── logs/                    # Output directory for logs, checkpoints & models
│   └── snake.log            # Real-time ASCII board step logs
└── snake/                   # Core game engine package
    ├── base/                # Core domain primitives
    │   ├── direc.py         # Direction enum (UP, DOWN, LEFT, RIGHT, NONE)
    │   ├── map.py           # Discrete grid map & point matrix
    │   ├── point.py         # Point types (HEAD, BODY, FOOD, WALL, EMPTY)
    │   ├── pos.py           # 2D coordinates & Manhattan distance utilities
    │   └── snake.py         # Snake state machine & movement mechanics
    ├── solver/              # AI solver implementations
    │   ├── base.py          # Abstract BaseSolver interface
    │   ├── greedy.py        # Lookahead Greedy heuristic solver
    │   ├── hamilton.py      # Hamiltonian cycle with shortcutting solver
    │   ├── path.py          # BFS shortest path and longest path algorithms
    │   └── dqn/             # Deep Q-Network RL package
    │       ├── history.py   # Training metric tracker & matplotlib plotting
    │       ├── memory.py    # Prioritized Experience Replay memory buffer
    │       └── snakeaction.py # Discrete action representations
    ├── util/                # Algorithmic utility structures
    │   └── sumtree.py       # Binary sum tree for PER sampling
    ├── game.py              # Game controller, configuration & game loops
    └── gui.py               # Tkinter GUI window & HUD rendering
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/AhryRaj/AI_Snake_Game.git
cd AI_Snake_Game
```

### 2. Set Up a Virtual Environment (Recommended)
```bash
# On macOS / Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note**: For Tkinter on Linux (Ubuntu/Debian), ensure `python3-tk` is installed:
> ```bash
> sudo apt-get install python3-tk
> ```

---

## 🎮 Usage Guide

### Running the AI Simulation
Launch the default AI game runner:
```bash
python3 AI_SnakeGame.py
```

### Configuring Solvers & Modes
Open `AI_SnakeGame.py` to customize the solver and game mode:

```python
from snake.game import Game, GameConf, GameMode

conf = GameConf()

# Select Solver: "HamiltonSolver" or "GreedySolver"
conf.solver_name = "HamiltonSolver"

# Select Game Mode: GameMode.NORMAL or GameMode.BENCHMARK
conf.mode = GameMode.NORMAL

# Adjust map dimensions (even numbers required for Hamiltonian cycle)
conf.map_rows = 10
conf.map_cols = 10

# Adjust animation speed (milliseconds per tick)
conf.interval_draw = 40

Game(conf).run()
```

### Headless Benchmarking
To assess algorithm performance over statistical trials without GUI overhead, set `conf.mode = GameMode.BENCHMARK` in `AI_SnakeGame.py`:
```bash
python3 AI_SnakeGame.py
```
```text
Solver: GreedySolver 
Mode: GameMode.BENCHMARK
Please input the number of episodes: 50

Map size: 8x8
Solver: greedy

Episode 1 - FULL (len: 64 | steps: 312)
Episode 2 - FULL (len: 64 | steps: 298)
...
[Summary]
Average Length: 63.80
Average Steps: 320.14
```

### Manual Arcade Play
Test your own snake reflexes using the standalone Pygame arcade version:
```bash
python3 Manual_SnakeGame.py
```

---

## ⌨️ Controls & Keybindings

### AI Simulation (Tkinter Window)
| Key | Action |
| :--- | :--- |
| <kbd>W</kbd> / <kbd>A</kbd> / <kbd>S</kbd> / <kbd>D</kbd> | Manually steer snake direction (override AI) |
| <kbd>Space</kbd> | Pause / Resume simulation |
| <kbd>R</kbd> | Restart episode |
| <kbd>Esc</kbd> | Exit application |

### Manual Mode (Pygame Window)
| Key | Action |
| :--- | :--- |
| <kbd>↑</kbd> <kbd>↓</kbd> <kbd>←</kbd> <kbd>→</kbd> | Change direction (Up, Down, Left, Right) |
| Window Close | Quit game |

---

## ⚙️ Configuration Reference

All settings can be customized via `GameConf` in `snake/game.py`:

| Parameter | Default | Type | Description |
| :--- | :--- | :--- | :--- |
| `mode` | `GameMode.NORMAL` | `Enum` | `NORMAL` (GUI), `BENCHMARK` (headless), `TRAIN_DQN` |
| `solver_name` | `'HamiltonSolver'` | `str` | Class name of solver (`'HamiltonSolver'`, `'GreedySolver'`) |
| `map_rows` | `8` | `int` | Number of playable rows (excluding border walls) |
| `map_cols` | `8` | `int` | Number of playable columns (must be even for Hamiltonian) |
| `map_width` | `160` | `int` | Width of game board canvas in pixels |
| `interval_draw` | `50` | `int` | Step render interval in milliseconds (lower = faster) |
| `show_grid_line` | `False` | `bool` | Toggle visible grid separation lines |
| `show_info_panel` | `True` | `bool` | Display the real-time statistics HUD side panel |
| `color_bg` | `'#000000'` | `hex` | Canvas background color |
| `color_food` | `'#FFF59D'` | `hex` | Food rendering color |
| `color_head` | `'#F5F5F5'` | `hex` | Snake head color |
| `color_body` | `'#F5F5F5'` | `hex` | Snake body color |

---

## 📊 Algorithm Comparison

| Metric | Hamiltonian Solver | Greedy Lookahead | DQN Agent |
| :--- | :---: | :---: | :---: |
| **Win Rate (100% Board Fill)** | **100%** | ~75% - 85% | Variable (Learned) |
| **Step Efficiency** | High (with shortcuts) | Very High | Medium |
| **Safety Guarantee** | Absolute | Heuristic | Probabilistic |
| **Grid Constraint** | Even dimensions ($M \times N$) | Any grid size | Fixed dimension |
| **Computation Overhead** | $O(1)$ lookup + $O(V)$ shortcuts | $O(V + E)$ BFS per step | GPU / Neural inference |

---

## 📜 Logging & Telemetry

Each move during an episode writes state snapshots to `logs/snake.log`. A sample snapshot shows the exact grid status:

```text
[ Episode 1 / Step 42 ]
# # # # # # # # # # 
#                 # 
#     H B B       # 
#         B       # 
#         B       # 
#         T       # 
#       F         # 
#                 # 
# # # # # # # # # # 
[ last/next direc: RIGHT/DOWN ]
```
- `#`: Outer boundary wall
- `H`: Snake head
- `B`: Snake body segment
- `T`: Snake tail
- `F`: Food item

---

## 🤝 Contributing

Contributions, feature suggestions, and performance improvements are always welcome!
1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/AmazingSolver`).
3. Commit your Changes (`git commit -m 'Add new A* heuristic solver'`).
4. Push to the Branch (`git push origin feature/AmazingSolver`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
