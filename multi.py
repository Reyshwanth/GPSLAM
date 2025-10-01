import math
import matplotlib.pyplot as plt
import numpy as np
import heapq
from scipy import ndimage
import matplotlib.image as mpimg
from pathlib import Path
import threading

# removed IPython display/widget dependencies to make the animation non-interactive

class AlgoEnvironment:
    def __init__(self, alpha=0.5):
        self.x_range = 50.0
        self.y_range = 50.0
        self.time_step = 0.25
        self.velocity = 1.0
        self.instances = 5
        self.agent_pos = [0.4, 0.4]
        self.agent_heading = math.pi / 6
        self.path_index = 0
        self.turn_rate = 1
        self.radius = self.velocity / self.turn_rate
        self.max_time = 5
        self.locx = 5
        self.locy = 7
        self.alpha = alpha
        try:
            relative = Path('fighter_top_view_processed.png')
            absolute = relative.absolute()
            self.original_image = mpimg.imread(str(absolute))
            print("all good")
        except Exception:
            print("except")
            self.original_image = None
        self.rotated_image = self.original_image
        self.image = None

class Node:
    def __init__(self, x, y, theta, parent=None, g=0, h=0, timer=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = parent
        self.g = g
        self.h = h
        self.f = g + h
        self.timer = timer
    def __lt__(self, other):
        return self.f < other.f

def a_star_search(env, src, dest):
    grid_res = 0.1
    heading_res = math.radians(10)
    def heuristic(x, y, gx, gy):
        return math.hypot(x - gx, y - gy)
    def local_h(x, y, lx, ly):
        return abs(env.radius - math.hypot(x - lx, y - ly))
    turn_options = np.linspace(-env.turn_rate, env.turn_rate, env.instances)
    goal_x, goal_y = dest[0], dest[1]
    start_theta = env.agent_heading
    start_node = Node(src[0], src[1], start_theta, None, 0, heuristic(src[0], src[1], goal_x, goal_y))
    open_set = []
    heapq.heappush(open_set, start_node)
    closed = set()
    def state_key(x, y, theta):
        return (round(x/grid_res), round(y/grid_res), round(theta/heading_res))
    g_scores = {state_key(src[0], src[1], start_theta): 0}
    min_f = math.inf
    min_node = None
    while open_set:
        current = heapq.heappop(open_set)
        curr_key = state_key(current.x, current.y, current.theta)
        if current.timer >= env.max_time:
            continue
        # closest point to requested path-time
        if current.f <= min_f and abs(current.timer - (env.max_time-env.time_step)) < 1e-5:
            min_node = current
            min_f = current.f
        if curr_key in closed:
            continue
        closed.add(curr_key)
        for turn_rate in turn_options:
            alpha = turn_rate * env.time_step
            if alpha != 0:
                new_theta = current.theta + alpha
                new_theta = (new_theta + 2*math.pi) % (2*math.pi)
                dx = 2 * env.velocity * env.time_step / abs(alpha) * math.sin(abs(alpha)/2) * math.cos(current.theta + alpha/2)
                dy = 2 * env.velocity * env.time_step / abs(alpha) * math.sin(abs(alpha)/2) * math.sin(current.theta + alpha/2)
            else:
                new_theta = current.theta
                dx = env.velocity * env.time_step * math.cos(current.theta)
                dy = env.velocity * env.time_step * math.sin(current.theta)
            nx = current.x + dx
            ny = current.y + dy
            if not (0 <= nx <= env.x_range and 0 <= ny <= env.y_range):
                continue
            nkey = state_key(nx, ny, new_theta)
            cost = env.velocity * env.time_step
            tentative_g = current.g + cost
            if (nkey not in g_scores or tentative_g < g_scores[nkey]):
                g_scores[nkey] = tentative_g
                h1 = env.alpha*heuristic(nx, ny, goal_x, goal_y) + (1-env.alpha)*local_h(nx, ny, env.locx, env.locy)
                neighbor = Node(nx, ny, new_theta, parent=current, g=tentative_g, h=h1, timer=env.time_step + current.timer)
                heapq.heappush(open_set, neighbor)
    if min_node is not None:
        path = []
        n = min_node
        env.agent_heading = n.theta
        while n is not None:
            path.append((n.x, n.y, n.theta))
            n = n.parent
        path.reverse()
        return path
    else:
        return None

# --- Animation loop logic ---
class PathAnimator:
    def __init__(self):
        self.running = False
        self.thread = None
        self.env = None
        self.src = [0., 0.]
        self.dest = [40, 10]
        self.path_full = []
        self.alpha = 0.5
        self.lock = threading.Lock()
    def reset(self, alpha=None):
        if alpha is not None:
            self.alpha = alpha
        self.env = AlgoEnvironment(alpha=self.alpha)
        self.src = [0., 0.]
        self.path_full = []
    def start(self, alpha):
        self.reset(alpha)
        self.running = True
        self.thread = threading.Thread(target=self._worker_fn, daemon=True)
        self.thread.start()
    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2)
            self.thread = None
    def _worker_fn(self):
        plt.ioff()
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot goal and customer
        ax.plot(self.src[0], self.src[1], 'go', label="Start", markersize=10)
        ax.plot(self.dest[0], self.dest[1], 'bo', label="Goal", markersize=10)
        ax.plot(self.env.locx, self.env.locy, 'yo', label="Customer", markersize=10)
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 51)
        ax.set_xlabel("x axis (m)")
        ax.set_ylabel("y axis (m)")
        ax.legend()
        #display(fig)
        src = list(self.src)
        env = self.env
        while self.running:
            # Compute one segment
            path_segment = a_star_search(env, src, self.dest)
            if path_segment is None or len(path_segment) < 2:
                print("End of animation / no path")
                self.running = False
                break
            # Don't repeat last point if extending
            if self.path_full and path_segment:
                path_segment = path_segment[1:]
            self.path_full.extend(path_segment)
            src = [self.path_full[-1][0], self.path_full[-1][1]]
            # Plot updated trajectory
            xs, ys, thetas = zip(*self.path_full)
            ax.plot(xs, ys, '-', lw=1.5, color='black', zorder=1)
            # Optional: aircraft image at tip
            if env.original_image is not None:
                size = 1.2
                angle = np.degrees(self.path_full[-1][2]) - 90
                rotated_image = ndimage.rotate(env.original_image, angle=angle, reshape=False).astype('uint8')
                ax.imshow((rotated_image * 255).astype(np.uint8),
                          extent=[xs[-1] - size, xs[-1] + size, ys[-1] - size, ys[-1] + size],
                          alpha=1, zorder=10)
            plt.pause(0.4)  # Animation speed
        plt.close(fig)

    def animate_until(self, alpha, max_time, pause=0.4):
        """Run the animation non-interactively until `max_time` (seconds).

        This function mirrors the logic in `_worker_fn` but runs synchronously
        (no threading, no widget controls) and stops naturally when no further
        path segments are available or when the planner cannot extend the path.
        """
        # prepare environment
        self.reset(alpha)
        self.env.max_time = max_time

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot goal and customer
        ax.plot(self.src[0], self.src[1], 'go', label="Start", markersize=10)
        ax.plot(self.dest[0], self.dest[1], 'bo', label="Goal", markersize=10)
        ax.plot(self.env.locx, self.env.locy, 'yo', label="Customer", markersize=10)
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_xlabel("x axis (m)")
        ax.set_ylabel("y axis (m)")
        ax.legend()

        src = list(self.src)
        env = self.env

        # run until planner cannot extend path further
        current_time=0
        while current_time<max_time:
            current_time+=0.25
            path_segment = a_star_search(env, src, self.dest)
            if path_segment is None or len(path_segment) < 2:
                print("End of animation / no path")
                break
            # Don't repeat last point if extending
            if self.path_full and path_segment:
                path_segment = path_segment[1:]
            self.path_full.extend(path_segment)
            src = [self.path_full[-1][0], self.path_full[-1][1]]
            # Plot updated trajectory
            xs, ys, thetas = zip(*self.path_full)
            ax.plot(xs, ys, '-', lw=1.5, color='black', zorder=1)
            # Optional: aircraft image at tip
            if env.original_image is not None:
                size = 1.2
                angle = np.degrees(self.path_full[-1][2]) - 90
                rotated_image = ndimage.rotate(env.original_image, angle=angle, reshape=False).astype('uint8')
                ax.imshow((rotated_image * 255).astype(np.uint8),
                          extent=[xs[-1] - size, xs[-1] + size, ys[-1] - size, ys[-1] + size],
                          alpha=1, zorder=10)
            plt.pause(pause)  # Animation speed

        # keep the final figure displayed for inspection (interactive mode on)
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    # Non-interactive entry point: run the animation up to a preset time.
    # Change these values as needed.
    PRESET_ALPHA = 0.5
    PRESET_MAX_TIME = 5  # seconds (planner's time horizon)
    PAUSE = 0.4  # seconds between frames

    # Multi-agent runner: three agents, each with their own AlgoEnvironment
    class MultiAgentAnimator:
        def __init__(self, agent_configs):
            """agent_configs: list of dicts with keys: src, dest, alpha, color, label"""
            self.agents = []
            for cfg in agent_configs:
                a = {
                    'env': AlgoEnvironment(alpha=cfg.get('alpha', PRESET_ALPHA)),
                    'src': list(cfg.get('src', [0.0, 0.0])),
                    'dest': list(cfg.get('dest', [8.0, 2.0])),
                    'path_full': [],
                    'color': cfg.get('color', 'black'),
                    'label': cfg.get('label', ''),
                    'finished': False
                }
                # apply per-agent customer location if provided
                if 'locx' in cfg:
                    a['env'].locx = cfg['locx']
                if 'locy' in cfg:
                    a['env'].locy = cfg['locy']
                self.agents.append(a)
            self._figure_closed = False
            # Try to load a precomputed costmap (costmap.npz) for background
            self.background_map = None
            try:
                data = np.load('costmap.npz')
                # Expect arrays: xs, ys, cost
                if 'xs' in data and 'ys' in data and 'cost' in data:
                    self.background_map = {
                        'xs': data['xs'],
                        'ys': data['ys'],
                        'cost': data['cost']
                    }
            except Exception:
                # no precomputed map available; will fall back to dynamic coverage
                self.background_map = None

        def animate_until(self, max_time, pause=0.4):
            plt.ion()
            fig, ax = plt.subplots(figsize=(8, 8))

            def _on_close(event):
                self._figure_closed = True

            fig.canvas.mpl_connect('close_event', _on_close)

            # initial plot of start/goal/customer markers
            for a in self.agents:
                ax.plot(a['src'][0], a['src'][1], 'o', color=a['color'], label=f"Start {a['label']}", markersize=8)
                ax.plot(a['dest'][0], a['dest'][1], '^', color=a['color'], label=f"Goal {a['label']}", markersize=8)
                # per-agent customer marker (distinct marker + text label)
                try:
                    ax.plot(a['env'].locx, a['env'].locy, marker='X', linestyle='None', color=a['color'],
                            label=f"Customer {a['label']}", markersize=8)
                    ax.text(a['env'].locx + 0.15, a['env'].locy + 0.15, f"C{a['label']}", color=a['color'], fontsize=8)
                except Exception:
                    pass
            # deduplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            unique = {}
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h
            ax.legend(unique.values(), unique.keys())
            # Optional: show global customer(s) from first agent env
            # (keep axes fixed to original ranges)
            ax.set_xlim(-1, 51)
            ax.set_ylim(-1, 51)
            ax.set_xlabel("x axis (m)")
            ax.set_ylabel("y axis (m)")
            ax.legend()

            current_time = 0.0
            time_step = self.agents[0]['env'].time_step if self.agents else 0.25

            while current_time < max_time and not self._figure_closed:
                current_time += time_step
                all_finished = True
                for a in self.agents:
                    if a['finished']:
                        continue
                    env = a['env']
                    src = a['src']
                    path_segment = a_star_search(env, src, a['dest'])
                    if path_segment is None or len(path_segment) < 2:
                        a['finished'] = True
                        continue
                    all_finished = False
                    if a['path_full'] and path_segment:
                        path_segment = path_segment[1:]
                    a['path_full'].extend(path_segment)
                    a['src'] = [a['path_full'][-1][0], a['path_full'][-1][1]]

                # Plot updated trajectories for all agents
                ax.clear()
                # Background: either load precomputed costmap or compute dynamic coverage
                try:
                    grid_size = float(self.agents[0]['env'].x_range) if self.agents else 50.0
                    # If a precomputed static background map exists, draw it first (zorder 0)
                    if self.background_map is not None:
                        bx = self.background_map['xs']
                        by = self.background_map['ys']
                        bcost = self.background_map['cost']
                        ax.imshow(bcost, origin='lower', extent=[bx[0], bx[-1], by[0], by[-1]],
                                  cmap='viridis', alpha=0.8, interpolation='nearest', zorder=0, vmin=0, vmax=np.max(bcost))

                    # Always compute dynamic coverage overlay (1m resolution grid)
                    grid_res = 0.1
                    rx = np.arange(grid_res/2.0, grid_size, grid_res)
                    ry = np.arange(grid_res/2.0, grid_size, grid_res)
                    GX, GY = np.meshgrid(rx, ry)
                    coverage = np.zeros_like(GX, dtype=float)
                    cover_radius = 5.0
                    # mark cells covered by any agent's visited path points
                    for a in self.agents:
                        for pt in a.get('path_full', []):
                            px, py = pt[0], pt[1]
                            d = np.hypot(GX - px, GY - py)
                            coverage[d <= cover_radius] = 1.0
                    # display dynamic coverage overlay with slightly stronger alpha and above static map
                        ax.imshow(coverage, origin='lower', extent=[0, grid_size, 0, grid_size],
                                  cmap='Greens', alpha=0.8, interpolation='nearest', zorder=1, vmin=0, vmax=1)
                except Exception:
                    # if anything fails, continue without background/overlay
                    pass
                # re-draw markers, paths and customers
                for a in self.agents:
                    if a['path_full']:
                        xs, ys, thetas = zip(*a['path_full'])
                        ax.plot(xs, ys, '-', lw=1.5, color=a['color'], zorder=1, label=f"Path {a['label']}")
                        # aircraft icon for last pose if available
                        if a['env'].original_image is not None:
                            size = 1.2
                            angle = np.degrees(a['path_full'][-1][2]) - 90
                            try:
                                rotated_image = ndimage.rotate(a['env'].original_image, angle=angle, reshape=False).astype('uint8')
                                ax.imshow((rotated_image * 255).astype(np.uint8),
                                          extent=[xs[-1] - size, xs[-1] + size, ys[-1] - size, ys[-1] + size],
                                          alpha=1, zorder=10)
                            except Exception:
                                pass
                    # start and goal markers
                    ax.plot(a['src'][0], a['src'][1], 'o', color=a['color'], markersize=6)
                    ax.plot(a['dest'][0], a['dest'][1], '^', color=a['color'], markersize=6)
                    # customer marker + small label
                    try:
                        ax.plot(a['env'].locx, a['env'].locy, marker='X', linestyle='None', color=a['color'], markersize=7)
                        ax.text(a['env'].locx + 0.12, a['env'].locy + 0.12, f"C{a['label']}", color=a['color'], fontsize=8)
                    except Exception:
                        pass

                # deduplicate legend entries
                handles, labels = ax.get_legend_handles_labels()
                unique = {}
                for h, l in zip(handles, labels):
                    if l not in unique:
                        unique[l] = h
                ax.legend(unique.values(), unique.keys())

                ax.set_xlim(-1, 51)
                ax.set_ylim(-1, 51)
                ax.set_xlabel("x axis (m)")
                ax.set_ylabel("y axis (m)")
                ax.legend()
                plt.pause(pause)

                if all_finished:
                    print('All agents finished / no further path')
                    break

            plt.ioff()
            try:
                plt.show()
            except Exception:
                pass

    # configure three agents with different starts and goals
    agent_configs = [
        {'src': [0.0, 0.0], 'dest': [40.0, 10.0], 'alpha': 0.6, 'color': 'black', 'label': 'A', 'locx':40, 'locy':35},
        {'src': [5.0, 45.0], 'dest': [35.0, 5.0], 'alpha': 0.1, 'color': 'red',   'label': 'B', 'locx':10, 'locy':15},
        {'src': [45.0, 5.0], 'dest': [10.0, 40.0], 'alpha': 0.3, 'color': 'blue',  'label': 'C', 'locx':40, 'locy':30},
    ]

    multi_anim = MultiAgentAnimator(agent_configs)
    multi_anim.animate_until(max_time=PRESET_MAX_TIME, pause=PAUSE)