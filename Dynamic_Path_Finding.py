"""
Dynamic Pathfinding Agent
=========================
A Pygame-based interactive grid pathfinding application implementing:
  - Greedy Best-First Search (GBFS)
  - A* Search
  - Manhattan & Euclidean heuristics
  - Dynamic obstacles with real-time re-planning
  - Interactive map editor (click to place/remove walls)
  - Random map generation with user-defined obstacle density
  - Real-time metrics dashboard

Requirements: pip install pygame
Run:          python dynamic_pathfinding_agent.py
"""

import pygame
import sys
import heapq
import math
import random
import time
from typing import Optional

# ─────────────────────────── Constants ───────────────────────────
SCREEN_W, SCREEN_H = 1100, 700
SIDEBAR_W = 280
GRID_AREA_W = SCREEN_W - SIDEBAR_W
PANEL_PAD = 12

# Colours
WHITE       = (255, 255, 255)
BLACK       = (0,   0,   0  )
GRAY        = (180, 180, 180)
DARK_GRAY   = (60,  60,  60 )
LIGHT_GRAY  = (230, 230, 230)
WALL_COL    = (40,  40,  40 )
START_COL   = (34,  197, 94 )   # green
GOAL_COL    = (239, 68,  68 )   # red
PATH_COL    = (34,  211, 238)   # cyan
VISITED_COL = (167, 139, 250)   # purple
FRONTIER_COL= (251, 191, 36 )   # yellow
AGENT_COL   = (249, 115, 22 )   # orange
BG_COL      = (15,  23,  42 )   # dark navy
SIDEBAR_COL = (30,  41,  59 )
BTN_COL     = (51,  65,  85 )
BTN_HOV     = (71,  85,  105)
BTN_ACT     = (99,  102, 241)
TEXT_COL    = (226, 232, 240)
ACCENT      = (99,  102, 241)
DYN_OBS_COL = (220, 38,  38 )   # bright red for dynamically spawned obstacles

FPS = 30
ANIM_DELAY  = 40   # ms between animation frames

# ─────────────────────────── Grid Node ───────────────────────────
class Node:
    __slots__ = ('row', 'col', 'f', 'g', 'h', 'parent')
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.f = self.g = self.h = 0
        self.parent: Optional['Node'] = None

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.row == other.row and self.col == other.col

    def __hash__(self):
        return hash((self.row, self.col))

# ─────────────────────────── Heuristics ──────────────────────────
def manhattan(r1, c1, r2, c2):
    return abs(r1 - r2) + abs(c1 - c2)

def euclidean(r1, c1, r2, c2):
    return math.sqrt((r1 - r2) ** 2 + (c1 - c2) ** 2)

# ─────────────────────────── Search Algorithms ───────────────────
def get_neighbors(grid, row, col, rows, cols):
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    result = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < rows and 0 <= nc < cols and not grid[nr][nc]:
            result.append((nr, nc))
    return result

def reconstruct_path(node):
    path = []
    cur = node
    while cur:
        path.append((cur.row, cur.col))
        cur = cur.parent
    return list(reversed(path))

def gbfs(grid, start, goal, rows, cols, heuristic_fn):
    """Greedy Best-First Search: f(n) = h(n)"""
    gr, gc = goal
    open_heap = []
    start_node = Node(*start)
    start_node.h = heuristic_fn(start[0], start[1], gr, gc)
    start_node.f = start_node.h
    heapq.heappush(open_heap, (start_node.f, id(start_node), start_node))
    visited = set()
    frontier_set = {start}
    visited_order = []
    frontier_order = [start]

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        pos = (current.row, current.col)
        if pos in visited:
            continue
        visited.add(pos)
        frontier_set.discard(pos)
        visited_order.append(pos)

        if pos == goal:
            return reconstruct_path(current), visited_order, frontier_order

        for nr, nc in get_neighbors(grid, current.row, current.col, rows, cols):
            npos = (nr, nc)
            if npos not in visited:
                n = Node(nr, nc)
                n.parent = current
                n.h = heuristic_fn(nr, nc, gr, gc)
                n.f = n.h
                heapq.heappush(open_heap, (n.f, id(n), n))
                if npos not in frontier_set:
                    frontier_set.add(npos)
                    frontier_order.append(npos)

    return None, visited_order, frontier_order  # no path

def astar(grid, start, goal, rows, cols, heuristic_fn):
    """A* Search: f(n) = g(n) + h(n)"""
    gr, gc = goal
    open_heap = []
    start_node = Node(*start)
    start_node.g = 0
    start_node.h = heuristic_fn(start[0], start[1], gr, gc)
    start_node.f = start_node.g + start_node.h
    heapq.heappush(open_heap, (start_node.f, id(start_node), start_node))
    g_score = {start: 0}
    visited_order = []
    frontier_order = [start]
    frontier_set = {start}
    closed = set()

    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        pos = (current.row, current.col)
        if pos in closed:
            continue
        closed.add(pos)
        frontier_set.discard(pos)
        visited_order.append(pos)

        if pos == goal:
            return reconstruct_path(current), visited_order, frontier_order

        for nr, nc in get_neighbors(grid, current.row, current.col, rows, cols):
            npos = (nr, nc)
            if npos in closed:
                continue
            tentative_g = g_score.get(pos, float('inf')) + 1
            if tentative_g < g_score.get(npos, float('inf')):
                n = Node(nr, nc)
                n.parent = current
                n.g = tentative_g
                n.h = heuristic_fn(nr, nc, gr, gc)
                n.f = n.g + n.h
                g_score[npos] = tentative_g
                heapq.heappush(open_heap, (n.f, id(n), n))
                if npos not in frontier_set:
                    frontier_set.add(npos)
                    frontier_order.append(npos)

    return None, visited_order, frontier_order

# ─────────────────────────── Button Widget ───────────────────────
class Button:
    def __init__(self, rect, label, tooltip=''):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.tooltip = tooltip
        self.active = False
        self.hovered = False

    def draw(self, surf, font):
        col = BTN_ACT if self.active else (BTN_HOV if self.hovered else BTN_COL)
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, ACCENT if self.active else GRAY, self.rect, 1, border_radius=6)
        txt = font.render(self.label, True, TEXT_COL)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def handle(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

# ─────────────────────────── InputBox Widget ─────────────────────
class InputBox:
    def __init__(self, rect, label, value=''):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.value = str(value)
        self.active = False

    def draw(self, surf, font, label_font):
        lbl = label_font.render(self.label, True, GRAY)
        surf.blit(lbl, (self.rect.x, self.rect.y - 18))
        col = ACCENT if self.active else DARK_GRAY
        pygame.draw.rect(surf, (20, 30, 50), self.rect, border_radius=4)
        pygame.draw.rect(surf, col, self.rect, 1, border_radius=4)
        txt = font.render(self.value, True, TEXT_COL)
        surf.blit(txt, (self.rect.x + 6, self.rect.y + 5))

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_BACKSPACE:
                self.value = self.value[:-1]
            elif event.unicode.isdigit() or (event.unicode == '.' and '.' not in self.value):
                self.value += event.unicode

    def get_int(self, default):
        try:
            return max(2, int(self.value))
        except:
            return default

    def get_float(self, default):
        try:
            return max(0.0, min(1.0, float(self.value)))
        except:
            return default

# ─────────────────────────── Main App ────────────────────────────
class PathfindingApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption("Dynamic Pathfinding Agent")
        self.clock = pygame.time.Clock()

        self.font_lg  = pygame.font.SysFont('Segoe UI', 18, bold=True)
        self.font_md  = pygame.font.SysFont('Segoe UI', 14)
        self.font_sm  = pygame.font.SysFont('Segoe UI', 12)
        self.font_xs  = pygame.font.SysFont('Segoe UI', 11)

        # Grid state
        self.rows = 20
        self.cols = 20
        self.grid = [[False]*self.cols for _ in range(self.rows)]  # False=open, True=wall
        self.dynamic_grid = [[False]*self.cols for _ in range(self.rows)]  # dynamically spawned
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)

        # Algorithm & heuristic selection
        self.algo      = 'astar'     # 'astar' | 'gbfs'
        self.heuristic = 'manhattan' # 'manhattan' | 'euclidean'

        # Edit mode: 'wall' | 'start' | 'goal'
        self.edit_mode = 'wall'
        self.drawing_wall = None   # True=placing, False=erasing

        # Search result
        self.path         = []
        self.visited      = []
        self.frontier     = []
        self.path_cost    = 0
        self.nodes_visited= 0
        self.exec_time_ms = 0
        self.replans      = 0
        self.no_path      = False

        # Animation
        self.animating       = False
        self.anim_step       = 0
        self.anim_visited_done = False
        self.last_anim_time  = 0

        # Agent traversal (dynamic mode)
        self.traversing      = False
        self.agent_pos       = None
        self.trav_step       = 0
        self.dynamic_mode    = False
        self.spawn_prob      = 0.03
        self.last_trav_time  = 0
        self.trav_delay      = 120  # ms per step

        self._build_ui()

    def _build_ui(self):
        sx = GRID_AREA_W + PANEL_PAD
        y = PANEL_PAD

        self.btn_run   = Button((sx, y, SIDEBAR_W-2*PANEL_PAD, 36), "▶  Run Search")
        y += 44
        self.btn_clear = Button((sx, y, (SIDEBAR_W-2*PANEL_PAD)//2 - 4, 30), "Clear Path")
        self.btn_reset = Button((sx + (SIDEBAR_W-2*PANEL_PAD)//2 + 4, y, (SIDEBAR_W-2*PANEL_PAD)//2, 30), "Reset Grid")
        y += 40

        # Algorithm toggle
        w2 = (SIDEBAR_W-2*PANEL_PAD)//2 - 2
        self.btn_astar = Button((sx, y, w2, 30), "A*")
        self.btn_gbfs  = Button((sx+w2+4, y, w2, 30), "GBFS")
        self.btn_astar.active = True
        y += 40

        # Heuristic toggle
        self.btn_manh  = Button((sx, y, w2, 30), "Manhattan")
        self.btn_eucl  = Button((sx+w2+4, y, w2, 30), "Euclidean")
        self.btn_manh.active = True
        y += 40

        # Edit mode
        w3 = (SIDEBAR_W-2*PANEL_PAD)//3 - 2
        self.btn_wall  = Button((sx, y, w3, 28), "Wall")
        self.btn_sedit = Button((sx+w3+3, y, w3, 28), "Start")
        self.btn_gedit = Button((sx+2*(w3+3), y, w3, 28), "Goal")
        self.btn_wall.active = True
        y += 38

        # Grid size inputs
        half = (SIDEBAR_W-2*PANEL_PAD)//2 - 4
        self.inp_rows = InputBox((sx, y+20, half, 28), "Rows", self.rows)
        self.inp_cols = InputBox((sx+half+8, y+20, half, 28), "Cols", self.cols)
        y += 58

        # Obstacle density
        self.inp_dens = InputBox((sx, y+20, SIDEBAR_W-2*PANEL_PAD, 28), "Obstacle Density (0-1)", '0.30')
        y += 58
        self.btn_rand  = Button((sx, y, SIDEBAR_W-2*PANEL_PAD, 30), "🎲 Random Map")
        y += 40

        # Dynamic mode
        self.btn_dynmode = Button((sx, y, SIDEBAR_W-2*PANEL_PAD, 30), "Dynamic Mode: OFF")
        y += 40
        self.inp_spawn = InputBox((sx, y+20, SIDEBAR_W-2*PANEL_PAD, 28), "Spawn Prob (0-1)", '0.03')
        y += 58
        self.inp_speed = InputBox((sx, y+20, SIDEBAR_W-2*PANEL_PAD, 28), "Agent Speed (ms/step)", '120')
        y += 52
        self.btn_traverse = Button((sx, y, SIDEBAR_W-2*PANEL_PAD, 36), "▶  Start Traversal")
        y += 46
        self.btn_resize = Button((sx, y, SIDEBAR_W-2*PANEL_PAD, 30), "Apply Grid Size")

        self.all_buttons = [
            self.btn_run, self.btn_clear, self.btn_reset,
            self.btn_astar, self.btn_gbfs,
            self.btn_manh, self.btn_eucl,
            self.btn_wall, self.btn_sedit, self.btn_gedit,
            self.btn_rand, self.btn_dynmode, self.btn_traverse, self.btn_resize
        ]
        self.all_inputs = [self.inp_rows, self.inp_cols, self.inp_dens, self.inp_spawn, self.inp_speed]

    # ── Helpers ──────────────────────────────────────────────────
    def cell_size(self):
        return min(GRID_AREA_W // self.cols, SCREEN_H // self.rows)

    def grid_offset(self):
        cs = self.cell_size()
        ox = (GRID_AREA_W - cs * self.cols) // 2
        oy = (SCREEN_H - cs * self.rows) // 2
        return ox, oy

    def pixel_to_cell(self, px, py):
        cs = self.cell_size()
        ox, oy = self.grid_offset()
        col = (px - ox) // cs
        row = (py - oy) // cs
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return int(row), int(col)
        return None

    def combined_wall(self, r, c):
        return self.grid[r][c] or self.dynamic_grid[r][c]

    def heuristic_fn(self):
        return manhattan if self.heuristic == 'manhattan' else euclidean

    def merged_grid(self):
        """Merge static + dynamic walls for search."""
        return [[self.grid[r][c] or self.dynamic_grid[r][c] for c in range(self.cols)]
                for r in range(self.rows)]

    def run_search(self, from_pos=None):
        t0 = time.perf_counter()
        mg = self.merged_grid()
        start = from_pos if from_pos else self.start
        fn = self.heuristic_fn()
        if self.algo == 'astar':
            path, visited, frontier = astar(mg, start, self.goal, self.rows, self.cols, fn)
        else:
            path, visited, frontier = gbfs(mg, start, self.goal, self.rows, self.cols, fn)
        elapsed = (time.perf_counter() - t0) * 1000

        self.path          = path or []
        self.visited       = visited
        self.frontier      = frontier
        self.nodes_visited = len(visited)
        self.path_cost     = len(self.path) - 1 if self.path else 0
        self.exec_time_ms  = elapsed
        self.no_path       = (path is None)
        return path

    def start_animation(self):
        self.animating         = True
        self.anim_step         = 0
        self.anim_visited_done = False
        self.last_anim_time    = pygame.time.get_ticks()

    def clear_path(self):
        self.path = []
        self.visited = []
        self.frontier = []
        self.animating = False
        self.traversing = False
        self.agent_pos = None
        self.no_path = False
        self.dynamic_grid = [[False]*self.cols for _ in range(self.rows)]

    def reset_grid(self):
        self.clear_path()
        self.grid = [[False]*self.cols for _ in range(self.rows)]
        self.nodes_visited = self.path_cost = self.exec_time_ms = self.replans = 0

    def resize_grid(self):
        self.rows = self.inp_rows.get_int(self.rows)
        self.cols = self.inp_cols.get_int(self.cols)
        self.start = (0, 0)
        self.goal  = (self.rows-1, self.cols-1)
        self.inp_rows.value = str(self.rows)
        self.inp_cols.value = str(self.cols)
        self.reset_grid()

    def random_map(self):
        density = self.inp_dens.get_float(0.30)
        self.clear_path()
        self.grid = [[False]*self.cols for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) not in (self.start, self.goal):
                    self.grid[r][c] = random.random() < density
        self.nodes_visited = self.path_cost = self.exec_time_ms = self.replans = 0

    def spawn_dynamic_obstacle(self):
        """Spawn one obstacle randomly (not on path, start, goal)."""
        path_set = set(self.path)
        for _ in range(100):
            r = random.randint(0, self.rows-1)
            c = random.randint(0, self.cols-1)
            pos = (r, c)
            if pos == self.start or pos == self.goal:
                continue
            if self.grid[r][c] or self.dynamic_grid[r][c]:
                continue
            self.dynamic_grid[r][c] = True
            return pos
        return None

    # ── Drawing ──────────────────────────────────────────────────
    def draw_grid(self):
        cs = self.cell_size()
        ox, oy = self.grid_offset()

        visited_set  = set(self.visited[:self.anim_step] if self.animating and not self.anim_visited_done else self.visited)
        frontier_set = set(self.frontier)
        path_set     = set(self.path)

        if self.animating and not self.anim_visited_done:
            frontier_set = set()
            path_set = set()
        elif self.animating and self.anim_visited_done:
            shown = self.anim_step - len(self.visited)
            path_set = set(self.path[:shown])

        for r in range(self.rows):
            for c in range(self.cols):
                x = ox + c * cs
                y = oy + r * cs
                rect = pygame.Rect(x, y, cs-1, cs-1)
                pos = (r, c)

                if self.combined_wall(r, c):
                    col = DYN_OBS_COL if self.dynamic_grid[r][c] else WALL_COL
                elif pos == self.start:
                    col = START_COL
                elif pos == self.goal:
                    col = GOAL_COL
                elif self.agent_pos and pos == self.agent_pos:
                    col = AGENT_COL
                elif pos in path_set:
                    col = PATH_COL
                elif pos in visited_set:
                    col = VISITED_COL
                elif pos in frontier_set and not self.animating:
                    col = FRONTIER_COL
                else:
                    col = (25, 35, 55)

                pygame.draw.rect(self.screen, col, rect, border_radius=2)

        # Grid lines
        for r in range(self.rows+1):
            pygame.draw.line(self.screen, (30,40,60), (ox, oy+r*cs), (ox+self.cols*cs, oy+r*cs))
        for c in range(self.cols+1):
            pygame.draw.line(self.screen, (30,40,60), (ox+c*cs, oy), (ox+c*cs, oy+self.rows*cs))

        if self.no_path and not self.animating:
            txt = self.font_lg.render("NO PATH FOUND", True, (255, 80, 80))
            self.screen.blit(txt, txt.get_rect(center=(GRID_AREA_W//2, SCREEN_H//2)))

    def draw_sidebar(self):
        pygame.draw.rect(self.screen, SIDEBAR_COL, (GRID_AREA_W, 0, SIDEBAR_W, SCREEN_H))
        pygame.draw.line(self.screen, ACCENT, (GRID_AREA_W, 0), (GRID_AREA_W, SCREEN_H), 2)

        # Title
        sx = GRID_AREA_W + PANEL_PAD
        title = self.font_lg.render("Pathfinding Agent", True, ACCENT)
        self.screen.blit(title, (sx, PANEL_PAD - 4))

        # Section labels
        def section(y, text):
            lbl = self.font_xs.render(text, True, (100, 120, 150))
            self.screen.blit(lbl, (sx, y))

        section(88, "──── ALGORITHM ────────────────")
        section(128, "──── HEURISTIC ────────────────")
        section(168, "──── EDIT MODE ────────────────")
        section(208, "──── GRID CONFIG ───────────────")
        section(350, "──── DYNAMIC MODE ──────────────")

        for btn in self.all_buttons:
            btn.draw(self.screen, self.font_md)
        for inp in self.all_inputs:
            inp.draw(self.screen, self.font_md, self.font_xs)

        # Metrics
        my = SCREEN_H - 155
        pygame.draw.rect(self.screen, (20,30,50), (sx-2, my-8, SIDEBAR_W-PANEL_PAD+2, 145), border_radius=6)
        pygame.draw.rect(self.screen, ACCENT, (sx-2, my-8, SIDEBAR_W-PANEL_PAD+2, 145), 1, border_radius=6)
        self.screen.blit(self.font_md.render("── Metrics ──────────────", True, ACCENT), (sx+4, my))
        my += 22
        metrics = [
            ("Algorithm",   self.algo.upper()),
            ("Heuristic",   self.heuristic.capitalize()),
            ("Nodes Visited", str(self.nodes_visited)),
            ("Path Cost",   str(self.path_cost)),
            ("Exec Time",   f"{self.exec_time_ms:.2f} ms"),
            ("Re-plans",    str(self.replans)),
        ]
        for k, v in metrics:
            kt = self.font_xs.render(k + ":", True, GRAY)
            vt = self.font_xs.render(v, True, TEXT_COL)
            self.screen.blit(kt, (sx+4, my))
            self.screen.blit(vt, (sx + 120, my))
            my += 18

        # Legend
        legend = [
            (START_COL, "Start"), (GOAL_COL, "Goal"), (AGENT_COL, "Agent"),
            (PATH_COL, "Path"), (VISITED_COL, "Visited"), (FRONTIER_COL, "Frontier"),
            (WALL_COL, "Wall"), (DYN_OBS_COL, "Dyn.Obs"),
        ]
        lx, ly = sx, my + 4
        for i, (col, lbl) in enumerate(legend):
            if i % 2 == 0 and i > 0:
                ly += 16
                lx = sx
            pygame.draw.rect(self.screen, col, (lx, ly, 10, 10), border_radius=2)
            t = self.font_xs.render(lbl, True, GRAY)
            self.screen.blit(t, (lx+13, ly))
            lx += 80

    def draw(self):
        self.screen.fill(BG_COL)
        self.draw_grid()
        self.draw_sidebar()
        pygame.display.flip()

    # ── Animation tick ───────────────────────────────────────────
    def tick_animation(self):
        if not self.animating:
            return
        now = pygame.time.get_ticks()
        if now - self.last_anim_time < ANIM_DELAY:
            return
        self.last_anim_time = now
        total_visited = len(self.visited)

        if not self.anim_visited_done:
            self.anim_step += 1
            if self.anim_step >= total_visited:
                self.anim_visited_done = True
                self.anim_step = total_visited
        else:
            path_step = self.anim_step - total_visited
            if path_step >= len(self.path):
                self.animating = False
            else:
                self.anim_step += 1

    # ── Traversal tick ───────────────────────────────────────────
    def tick_traversal(self):
        if not self.traversing or not self.path:
            return
        now = pygame.time.get_ticks()
        if now - self.last_trav_time < self.trav_delay:
            return
        self.last_trav_time = now

        # Possibly spawn a new obstacle
        if self.dynamic_mode and random.random() < self.spawn_prob:
            new_obs = self.spawn_dynamic_obstacle()
            if new_obs and new_obs in set(self.path[self.trav_step:]):
                # Re-plan from current agent position
                old_step = self.trav_step
                new_path = self.run_search(from_pos=self.agent_pos)
                self.replans += 1
                self.trav_step = 0
                if not new_path:
                    self.traversing = False
                    self.no_path = True
                return

        if self.trav_step >= len(self.path):
            self.traversing = False
            self.agent_pos = self.goal
            return

        self.agent_pos = self.path[self.trav_step]
        self.trav_step += 1

    # ── Event handling ───────────────────────────────────────────
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            for inp in self.all_inputs:
                inp.handle(event)

            for btn in self.all_buttons:
                btn.handle(event)

            # Button clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_button_clicks(event.pos)
                # Grid interaction
                if event.pos[0] < GRID_AREA_W:
                    cell = self.pixel_to_cell(*event.pos)
                    if cell:
                        self._handle_cell_click(cell, event.pos)

            if event.type == pygame.MOUSEBUTTONUP:
                self.drawing_wall = None

            if event.type == pygame.MOUSEMOTION:
                if event.pos[0] < GRID_AREA_W and self.drawing_wall is not None and self.edit_mode == 'wall':
                    cell = self.pixel_to_cell(*event.pos)
                    if cell and cell != self.start and cell != self.goal:
                        self.grid[cell[0]][cell[1]] = self.drawing_wall
                        self.clear_path()

    def _handle_button_clicks(self, pos):
        sx = GRID_AREA_W + PANEL_PAD

        if self.btn_run.rect.collidepoint(pos):
            self.clear_path()
            self.run_search()
            self.start_animation()

        elif self.btn_clear.rect.collidepoint(pos):
            self.clear_path()
            self.nodes_visited = self.path_cost = 0
            self.exec_time_ms = 0; self.replans = 0

        elif self.btn_reset.rect.collidepoint(pos):
            self.reset_grid()
            self.nodes_visited = self.path_cost = 0
            self.exec_time_ms = 0; self.replans = 0

        elif self.btn_astar.rect.collidepoint(pos):
            self.algo = 'astar'
            self.btn_astar.active = True; self.btn_gbfs.active = False

        elif self.btn_gbfs.rect.collidepoint(pos):
            self.algo = 'gbfs'
            self.btn_gbfs.active = True; self.btn_astar.active = False

        elif self.btn_manh.rect.collidepoint(pos):
            self.heuristic = 'manhattan'
            self.btn_manh.active = True; self.btn_eucl.active = False

        elif self.btn_eucl.rect.collidepoint(pos):
            self.heuristic = 'euclidean'
            self.btn_eucl.active = True; self.btn_manh.active = False

        elif self.btn_wall.rect.collidepoint(pos):
            self.edit_mode = 'wall'
            self.btn_wall.active = True; self.btn_sedit.active = False; self.btn_gedit.active = False

        elif self.btn_sedit.rect.collidepoint(pos):
            self.edit_mode = 'start'
            self.btn_sedit.active = True; self.btn_wall.active = False; self.btn_gedit.active = False

        elif self.btn_gedit.rect.collidepoint(pos):
            self.edit_mode = 'goal'
            self.btn_gedit.active = True; self.btn_wall.active = False; self.btn_sedit.active = False

        elif self.btn_rand.rect.collidepoint(pos):
            self.random_map()

        elif self.btn_dynmode.rect.collidepoint(pos):
            self.dynamic_mode = not self.dynamic_mode
            self.btn_dynmode.label = f"Dynamic Mode: {'ON' if self.dynamic_mode else 'OFF'}"
            self.btn_dynmode.active = self.dynamic_mode

        elif self.btn_traverse.rect.collidepoint(pos):
            if not self.path:
                self.run_search()
            if self.path:
                self.spawn_prob = self.inp_spawn.get_float(0.03)
                self.trav_delay = max(20, int(self.inp_speed.value) if self.inp_speed.value else 120)
                self.animating = False
                self.trav_step = 0
                self.agent_pos = self.start
                self.traversing = True
                self.replans = 0
                self.dynamic_grid = [[False]*self.cols for _ in range(self.rows)]

        elif self.btn_resize.rect.collidepoint(pos):
            self.resize_grid()

    def _handle_cell_click(self, cell, pos):
        r, c = cell
        if self.edit_mode == 'wall':
            if cell not in (self.start, self.goal):
                placing = not self.grid[r][c]
                self.drawing_wall = placing
                self.grid[r][c] = placing
                self.clear_path()
        elif self.edit_mode == 'start':
            if not self.grid[r][c] and cell != self.goal:
                self.start = cell
                self.clear_path()
        elif self.edit_mode == 'goal':
            if not self.grid[r][c] and cell != self.start:
                self.goal = cell
                self.clear_path()

    # ── Main loop ────────────────────────────────────────────────
    def run(self):
        while True:
            self.clock.tick(FPS)
            self.handle_events()
            self.tick_animation()
            self.tick_traversal()
            self.draw()


# ─────────────────────────── Entry Point ─────────────────────────
if __name__ == '__main__':
    app = PathfindingApp()
    app.run()
