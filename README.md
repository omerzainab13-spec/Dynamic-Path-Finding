# Dynamic Pathfinding Agent

A Pygame-based interactive grid pathfinding application implementing informed search algorithms with real-time dynamic obstacle re-planning.

## Requirements

```bash
pip install pygame
```

Python 3.8+ required.

## How to Run

```bash
python dynamic_pathfinding_agent.py
```

## Features

### Algorithms
| Algorithm | Description |
|-----------|-------------|
| **A\*** | `f(n) = g(n) + h(n)` — optimal, uses both path cost and heuristic |
| **GBFS** | `f(n) = h(n)` — greedy, faster but not always optimal |

### Heuristics
- **Manhattan Distance**: `|x1-x2| + |y1-y2|` — ideal for 4-directional grids
- **Euclidean Distance**: `sqrt((x1-x2)² + (y1-y2)²)` — straight-line distance

### Grid Controls
- **Left-click drag** on grid → place/remove walls
- **Edit Mode buttons** → switch between placing Walls / moving Start / moving Goal
- **Apply Grid Size** → resize grid (enter Rows & Cols in input boxes)
- **Random Map** → generate maze with user-defined obstacle density (0–1)
- **Reset Grid** → clear all walls

### Search & Visualization
- **Run Search** → runs selected algorithm and animates frontier/visited/path
- **Clear Path** → removes visualization overlay
- Color coding:
  - 🟢 Green = Start | 🔴 Red = Goal | 🟠 Orange = Agent
  - 🩵 Cyan = Final Path | 🟣 Purple = Visited | 🟡 Yellow = Frontier
  - ⬛ Dark = Static Wall | 🔴 Bright Red = Dynamic Obstacle

### Dynamic Mode
1. Toggle **Dynamic Mode: ON**
2. Set **Spawn Probability** (e.g. 0.03 = 3% chance per step)
3. Set **Agent Speed** in ms/step (lower = faster)
4. Click **▶ Start Traversal**

While the agent moves, new obstacles spawn randomly. If one blocks the current path, the agent **immediately re-plans** from its current position using the selected algorithm. The re-plan counter in Metrics tracks how many times this happened.

## Metrics Dashboard (live)
- **Nodes Visited** — total expanded nodes
- **Path Cost** — length of final path
- **Exec Time** — computation time in milliseconds
- **Re-plans** — number of dynamic re-plans triggered

## File Structure
```
dynamic_pathfinding_agent.py   ← single-file complete solution
README.md
```

## Algorithm Notes

### A* (Optimal)
- Guarantees shortest path when heuristic is admissible
- Manhattan is admissible on a 4-directional grid
- Higher memory usage than GBFS

### GBFS (Greedy)
- Much faster to reach goal in open spaces
- Not guaranteed optimal — can be misled by local minima
- Lower memory footprint

### Re-planning Efficiency
- On obstacle spawn, the system checks if the new obstacle falls on the **remaining path**
- If **not** on path → no re-plan needed (efficient)
- If **on path** → re-plan from current agent position only
