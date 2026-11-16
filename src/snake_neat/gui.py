"""Tkinter graphical interface (standard library) for watching evolution.

Panels:
  * board: the champion of the latest generation plays live, with its
    8 vision rays (what the network actually "sees");
  * network: the champion's topology (layers = topological depth), edges
    colored by weight sign and magnitude, nodes colored by activation;
  * curves: best / mean fitness per generation;
  * speciation: stacked chart of species sizes over time.

Evolution runs in a dedicated thread (which delegates evaluation to a process
pool) and communicates with the Tk loop through a thread-safe queue.
"""

from __future__ import annotations

import colorsys
import math
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter import font as tkfont

import numpy as np

from .config import GameConfig, NEATConfig
from .evaluation import Evaluator
from .game import ACTION_NAMES, DIRS, SnakeGame, input_labels
from .genome import HIDDEN, INPUT, OUTPUT, Genome
from .network import FeedForwardNetwork
from .persistence import load_genome, save_genome
from .population import GenerationStats, Population

# ----------------------------------------------------------------- palette
BG = "#101218"
PANEL = "#171a22"
GRID = "#1d212b"
FG = "#e2e8f0"
MUTED = "#8b93a5"
ACCENT = "#a3e635"
FOOD = "#fb7185"
NEUTRAL = "#2a2f3b"
POS = "#34d399"
NEG = "#f472b6"
ACT_NEG = "#60a5fa"
ACT_POS = "#fbbf24"
BEST = "#a3e635"
MEAN = "#60a5fa"
FONT: tuple = ("TkDefaultFont", 10)
FONT_S: tuple = ("TkDefaultFont", 8)
FONT_B: tuple = ("TkDefaultFont", 11, "bold")
FONT_L: tuple = ("TkDefaultFont", 18, "bold")
MONO: tuple = ("TkFixedFont", 10)


def _setup_fonts(root: tk.Tk) -> None:
    """Pick the first installed font from a list of preferences."""
    global FONT, FONT_S, FONT_B, FONT_L, MONO
    available = set(tkfont.families(root))
    sans = next((f for f in ("Inter", "Cantarell", "Noto Sans", "DejaVu Sans", "Liberation Sans",
                             "Helvetica") if f in available), "TkDefaultFont")
    mono = next((f for f in ("JetBrains Mono", "Noto Sans Mono", "DejaVu Sans Mono", "Liberation Mono",
                             "Source Code Pro") if f in available), "TkFixedFont")
    FONT, FONT_S, FONT_B, FONT_L = (sans, 10), (sans, 8), (sans, 11, "bold"), (sans, 18, "bold")
    MONO = (mono, 9)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(1, c)) * 255) for c in rgb)


def _rgb(h: str) -> tuple[float, float, float]:
    return tuple(int(h[i:i + 2], 16) / 255 for i in (1, 3, 5))  # type: ignore[return-value]


def mix(a: str, b: str, t: float) -> str:
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(x + (y - x) * t for x, y in zip(ra, rb)))  # type: ignore[arg-type]


def diverging(v: float) -> str:
    v = max(-1.0, min(1.0, v))
    return mix(NEUTRAL, ACT_POS, v) if v >= 0 else mix(NEUTRAL, ACT_NEG, -v)


def species_color(sid: int) -> str:
    h = (sid * 0.61803398875) % 1.0
    return _hex(colorsys.hsv_to_rgb(h, 0.55, 0.85))


def fmt(x: float) -> str:
    if abs(x) >= 1e6:
        return f"{x / 1e6:.1f}M"
    if abs(x) >= 1e3:
        return f"{x / 1e3:.1f}k"
    return f"{x:.0f}" if abs(x) >= 10 or x == int(x) else f"{x:.1f}"


def nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    t = math.ceil(lo / step) * step
    out = []
    while t <= hi + 1e-9:
        out.append(t)
        t += step
    return out


# ============================================================ evolution
class EvolutionThread(threading.Thread):
    def __init__(self, game_cfg: GameConfig, neat_cfg: NEATConfig, seed: int, episodes: int,
                 workers: int, out: queue.Queue) -> None:
        super().__init__(daemon=True)
        self.game_cfg, self.neat_cfg = game_cfg, neat_cfg
        self.seed, self.episodes, self.workers = seed, episodes, workers
        self.out = out
        self.running = threading.Event()
        self.stopping = threading.Event()
        self.evaluator: Evaluator | None = None
        self.population: Population | None = None

    def run(self) -> None:
        try:
            pop = self.population = Population(self.neat_cfg, self.game_cfg.n_inputs,
                                               self.game_cfg.n_outputs, seed=self.seed)
            ev = self.evaluator = Evaluator(self.game_cfg, self.episodes, self.workers, seed=self.seed)
            while not self.stopping.is_set():
                if not self.running.wait(0.2):
                    continue
                t0 = time.perf_counter()
                stats = pop.run_generation(lambda gs: ev(gs, pop.generation))
                stats.elapsed = time.perf_counter() - t0
                self.out.put(stats)
        except Exception as exc:  # forwarded to the GUI
            if not self.stopping.is_set():
                self.out.put(exc)
        finally:
            if self.evaluator is not None:
                self.evaluator.close()

    def stop(self) -> None:
        self.stopping.set()
        self.running.set()


# ============================================================ plots
class LinePlot:
    def __init__(self, canvas: tk.Canvas, title: str) -> None:
        self.c = canvas
        self.title = title

    def draw(self, xs: list[int], series: list[tuple[str, list[float], str]]) -> None:
        c = self.c
        c.delete("all")
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 120)
        L, R, T, B = 52, 12, 28, 26
        c.create_text(L, 12, text=self.title, fill=FG, font=FONT_B, anchor="w")
        if len(xs) < 2:
            c.create_text(w / 2, h / 2, text="waiting for data…", fill=MUTED, font=FONT)
            return
        ys = [v for _, vals, _ in series for v in vals]
        lo, hi = 0.0, max(ys) * 1.05 or 1.0
        x0, x1 = xs[0], xs[-1]
        X = lambda x: L + (x - x0) / max(1, x1 - x0) * (w - L - R)  # noqa: E731
        Y = lambda y: h - B - (y - lo) / (hi - lo) * (h - T - B)    # noqa: E731
        for t in nice_ticks(lo, hi):
            c.create_line(L, Y(t), w - R, Y(t), fill=GRID)
            c.create_text(L - 6, Y(t), text=fmt(t), fill=MUTED, font=FONT_S, anchor="e")
        for t in nice_ticks(x0, x1, 5):
            c.create_text(X(t), h - B + 12, text=f"{t:.0f}", fill=MUTED, font=FONT_S)
        lx = w - R
        for name, vals, col in reversed(series):
            pts = [p for x, v in zip(xs, vals) for p in (X(x), Y(v))]
            c.create_line(*pts, fill=col, width=2)
            c.create_text(X(xs[-1]) - 4, Y(vals[-1]) - 9, text=fmt(vals[-1]), fill=col, font=FONT_S, anchor="e")
            item = c.create_text(lx, 12, text=f"● {name}", fill=col, font=FONT_S, anchor="e")
            lx = c.bbox(item)[0] - 10


class SpeciesPlot:
    def __init__(self, canvas: tk.Canvas) -> None:
        self.c = canvas

    def draw(self, gens: list[int], sizes: list[dict[int, int]], total: int) -> None:
        c = self.c
        c.delete("all")
        w, h = max(c.winfo_width(), 200), max(c.winfo_height(), 120)
        L, R, T, B = 40, 12, 28, 26
        n_now = len(sizes[-1]) if sizes else 0
        c.create_text(L, 12, text="Speciation", fill=FG, font=FONT_B, anchor="w")
        c.create_text(w - R, 12, text=f"{n_now} living species", fill=MUTED, font=FONT_S, anchor="e")
        if len(gens) < 2:
            c.create_text(w / 2, h / 2, text="waiting for data…", fill=MUTED, font=FONT)
            return
        x0, x1 = gens[0], gens[-1]
        X = lambda i: L + (gens[i] - x0) / max(1, x1 - x0) * (w - L - R)  # noqa: E731
        Y = lambda y: h - B - y / total * (h - T - B)                        # noqa: E731
        bottom = np.zeros(len(gens))
        ids = sorted(set().union(*sizes))
        for sid in ids:
            vals = np.array([s.get(sid, 0) for s in sizes], dtype=float)
            top = bottom + vals
            nz = np.flatnonzero(vals)
            a, b = max(nz[0] - 1, 0), min(nz[-1] + 1, len(gens) - 1)
            pts = [p for i in range(a, b + 1) for p in (X(i), Y(top[i]))]
            pts += [p for i in range(b, a - 1, -1) for p in (X(i), Y(bottom[i]))]
            c.create_polygon(*pts, fill=species_color(sid), outline="")
            bottom = top
        for t in nice_ticks(x0, x1, 5):
            c.create_text(L + (t - x0) / max(1, x1 - x0) * (w - L - R), h - B + 12,
                          text=f"{t:.0f}", fill=MUTED, font=FONT_S)
        for t in (0, total // 2, total):
            c.create_text(L - 6, Y(t), text=str(t), fill=MUTED, font=FONT_S, anchor="e")


# ============================================================ replay
MAX_SPECIES_VIEW = 16


class Runner:
    """A snake replayed live: genome, compiled network, current game and current decision."""

    def __init__(self, genome: Genome, cfg: GameConfig, label: str, sid: int | None = None) -> None:
        self.cfg, self.sid = cfg, sid
        self.game: SnakeGame | None = None
        self.next_action = 1
        self.death_time: float | None = None
        self.set_genome(genome, label)

    def set_genome(self, genome: Genome, label: str) -> None:
        self.genome, self.label = genome, label
        self.net = FeedForwardNetwork.from_genome(genome)

    def new_episode(self, seed: int) -> None:
        self.game = SnakeGame(self.cfg, seed)
        self.death_time = None
        self.think()

    def think(self) -> None:
        """Decision for the displayed state (the activations shown correspond to this state)."""
        self.next_action = int(np.argmax(self.net.activate(self.game.observe())))

    def advance(self, n: int) -> None:
        for _ in range(n):
            self.game.step(self.next_action)
            if not self.game.alive:
                return
            self.think()

    def ready_to_restart(self, now: float, hold: float = 1.2) -> bool:
        """True once the game has been over for `hold` seconds (time to see the death)."""
        if self.game.alive:
            return False
        if self.death_time is None:
            self.death_time = now
        return now - self.death_time > hold


# ============================================================ application
class App:
    def __init__(self, root: tk.Tk, game_cfg: GameConfig, neat_cfg: NEATConfig, *, seed: int,
                 episodes: int, workers: int, genome_path: str | None, view: str = "champion") -> None:
        self.root = root
        self._initial_view = view
        self.game_cfg, self.neat_cfg = game_cfg, neat_cfg
        self.seed, self.episodes, self.workers = seed, episodes, workers
        self.queue: queue.Queue = queue.Queue()
        self.evo: EvolutionThread | None = None
        self.history: list[GenerationStats] = []
        self.latest: GenerationStats | None = None
        self.rng = np.random.default_rng()

        # replay state
        self.replay_cfg = game_cfg
        self.runner: Runner | None = None               # "champion" view
        self.pending: tuple[Genome, str] | None = None
        self.runners: list[Runner] = []                 # "species" view
        self.focus_sid: int | None = None
        self.panels: list[tuple[Runner, tuple[float, float, float, float]]] = []
        self._net_owner: Runner | None = None
        self._net_genome: Genome | None = None
        self.node_items: dict[int, int] = {}
        self.output_items: dict[int, int] = {}

        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._poll)
        self.root.after(30, self._tick)

        if genome_path:
            self._load(genome_path)
        else:
            self.toggle_evolution()

    # ---------------------------------------------------------- widgets
    def _build(self) -> None:
        r = self.root
        r.title("Snake × NEAT — neuroevolution")
        r.configure(bg=BG)
        st = ttk.Style(r)
        st.theme_use("clam")
        st.configure(".", background=BG, foreground=FG, font=FONT)
        st.configure("TButton", background=PANEL, foreground=FG, borderwidth=1, focusthickness=0, padding=(10, 4))
        st.map("TButton", background=[("active", NEUTRAL)])
        st.configure("Accent.TButton", background="#3f6212", foreground="#f7fee7")
        st.map("Accent.TButton", background=[("active", "#4d7c0f")])
        st.configure("TCheckbutton", background=BG, foreground=FG)
        st.map("TCheckbutton", background=[("active", BG)])
        st.configure("TRadiobutton", background=BG, foreground=FG)
        st.map("TRadiobutton", background=[("active", BG)])
        st.configure("TLabel", background=BG, foreground=FG)
        st.configure("Muted.TLabel", foreground=MUTED, font=FONT_S)
        st.configure("Horizontal.TScale", background=BG, troughcolor=PANEL)

        self.cell, self.ox, self.oy = 20, 0, 0
        self.layout_mode: str | None = None

        self.bar = ttk.Frame(r, padding=(10, 8, 10, 4))
        bar_l = ttk.Frame(self.bar)
        bar_r = ttk.Frame(self.bar)
        self.bar_l, self.bar_r = bar_l, bar_r
        self.btn_evo = ttk.Button(bar_l, text="Start evolution", style="Accent.TButton",
                                  command=self.toggle_evolution, width=19)
        self.btn_evo.pack(side="left")
        ttk.Label(bar_l, text="  Speed").pack(side="left")
        self.speed = tk.DoubleVar(value=40)
        ttk.Scale(bar_l, from_=2, to=400, variable=self.speed, length=120).pack(side="left", padx=4)
        self.speed_lbl = ttk.Label(bar_l, text="40 steps/s", width=11)
        self.speed_lbl.pack(side="left")
        self.follow = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar_l, text="Follow champion", variable=self.follow).pack(side="left", padx=6)
        self.show_rays = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar_l, text="Vision", variable=self.show_rays).pack(side="left", padx=4)
        self.view = tk.StringVar(value=self._initial_view)
        ttk.Label(bar_r, text="View:").pack(side="left")
        ttk.Radiobutton(bar_r, text="Champion", value="champion", variable=self.view,
                        command=self._on_view_change).pack(side="left", padx=3)
        ttk.Radiobutton(bar_r, text="All species", value="species", variable=self.view,
                        command=self._on_view_change).pack(side="left", padx=(3, 12))
        ttk.Button(bar_r, text="New game", command=self._new_episode).pack(side="left", padx=4)
        ttk.Button(bar_r, text="Save shown genome…", command=self._ask_save).pack(side="left", padx=4)
        ttk.Button(bar_r, text="Load…", command=self._ask_load).pack(side="left", padx=4)

        canvas = dict(bg=PANEL, highlightthickness=0, width=320, height=200)
        self.left = ttk.Frame(r)
        self.board = tk.Canvas(self.left, **canvas)
        self.board.pack(fill="both", expand=True)
        self.game_lbl = ttk.Label(self.left, text="", font=MONO, justify="left")
        self.game_lbl.pack(anchor="w", pady=(6, 0))
        self.replay_lbl = ttk.Label(self.left, text="", style="Muted.TLabel")
        self.replay_lbl.pack(anchor="w")
        self.netc = tk.Canvas(r, **canvas)
        self.fitc = tk.Canvas(r, **canvas)
        self.spc = tk.Canvas(r, **canvas)
        self.fit_plot = LinePlot(self.fitc, "Fitness  F = 1000·apples + steps")
        self.sp_plot = SpeciesPlot(self.spc)
        self.status = ttk.Label(r, text="", font=MONO, padding=(10, 0, 10, 8))

        self.board.bind("<Configure>", lambda e: self._on_board_resize())
        self.board.bind("<Button-1>", self._on_board_click)
        self.netc.bind("<Configure>", lambda e: self._on_net_resize())
        self.fitc.bind("<Configure>", lambda e: self._redraw_plots())
        self.spc.bind("<Configure>", lambda e: self._redraw_plots())
        r.bind("<Configure>", self._on_root_resize)

        sw, sh = r.winfo_screenwidth(), r.winfo_screenheight()
        r.geometry(f"{min(1380, sw - 40)}x{min(940, sh - 80)}")
        r.minsize(640, 600)
        self._relayout(self._mode_for(min(1380, sw - 40), min(940, sh - 80)))

        r.bind("<space>", lambda e: self.toggle_evolution())
        r.bind("<v>", lambda e: self._toggle_view())
        r.bind("<plus>", lambda e: self.speed.set(min(400, self.speed.get() * 1.5)))
        r.bind("<minus>", lambda e: self.speed.set(max(2, self.speed.get() / 1.5)))
        r.bind("<KP_Add>", lambda e: self.speed.set(min(400, self.speed.get() * 1.5)))
        r.bind("<KP_Subtract>", lambda e: self.speed.set(max(2, self.speed.get() / 1.5)))

    def _on_root_resize(self, event: tk.Event) -> None:
        if event.widget is self.root:
            self._relayout(self._mode_for(event.width, event.height))

    @staticmethod
    def _mode_for(w: int, h: int) -> str:
        if w >= 2.0 * h and w >= 1500:
            return "ultra"
        return "wide" if w >= 1.15 * h and w >= 1000 else "tall"

    def _relayout(self, mode: str) -> None:
        """Three columns (ultra-wide), two (landscape) or one (portrait / tiled window)."""
        if mode == self.layout_mode:
            return
        self.layout_mode = mode
        r = self.root
        for w in (self.bar, self.bar_l, self.bar_r, self.left, self.netc, self.fitc, self.spc, self.status):
            w.grid_forget()
        for i in range(6):
            r.rowconfigure(i, weight=0)
            r.columnconfigure(i, weight=0)
        self.bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        pad = dict(padx=10, pady=5)
        if mode == "ultra":
            self.bar_l.grid(row=0, column=0, sticky="w")
            self.bar_r.grid(row=0, column=1, sticky="e")
            self.bar.columnconfigure(1, weight=1)
            self.bar.grid(columnspan=3)
            self.left.grid(row=1, column=0, rowspan=2, sticky="nsew", **pad)
            self.netc.grid(row=1, column=1, rowspan=2, sticky="nsew", **pad)
            self.fitc.grid(row=1, column=2, sticky="nsew", **pad)
            self.spc.grid(row=2, column=2, sticky="nsew", **pad)
            self.status.grid(row=3, column=0, columnspan=3, sticky="w")
            r.rowconfigure(1, weight=1)
            r.rowconfigure(2, weight=1)
            for col, wgt in ((0, 3), (1, 4), (2, 4)):
                r.columnconfigure(col, weight=wgt)
        elif mode == "wide":
            self.bar_l.grid(row=0, column=0, sticky="w")
            self.bar_r.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.left.grid(row=1, column=0, sticky="nsew", **pad)
            self.netc.grid(row=1, column=1, sticky="nsew", **pad)
            self.fitc.grid(row=2, column=0, sticky="nsew", **pad)
            self.spc.grid(row=2, column=1, sticky="nsew", **pad)
            self.status.grid(row=3, column=0, columnspan=2, sticky="w")
            r.rowconfigure(1, weight=3)
            r.rowconfigure(2, weight=1)
            r.columnconfigure(0, weight=2)
            r.columnconfigure(1, weight=3)
        else:
            self.bar_l.grid(row=0, column=0, sticky="w")
            self.bar_r.grid(row=1, column=0, sticky="w", pady=(4, 0))
            self.left.grid(row=1, column=0, sticky="nsew", **pad)
            self.netc.grid(row=2, column=0, sticky="nsew", **pad)
            self.fitc.grid(row=3, column=0, sticky="nsew", **pad)
            self.spc.grid(row=4, column=0, sticky="nsew", **pad)
            self.status.grid(row=5, column=0, sticky="w")
            for i, wgt in ((1, 4), (2, 4), (3, 2), (4, 2)):
                r.rowconfigure(i, weight=wgt)
            r.columnconfigure(0, weight=1)

    def _on_board_resize(self) -> None:
        self._draw_grid()
        self._draw_game()

    def _on_net_resize(self) -> None:
        self._net_owner = None
        self._refresh_network()

    # ---------------------------------------------------------- evolution
    def toggle_evolution(self) -> None:
        if self.evo is None:
            self.evo = EvolutionThread(self.game_cfg, self.neat_cfg, self.seed, self.episodes,
                                       self.workers, self.queue)
            self.evo.start()
        if self.evo.running.is_set():
            self.evo.running.clear()
            self.btn_evo.configure(text="Resume evolution")
        else:
            self.evo.running.set()
            self.btn_evo.configure(text="Pause")

    def _poll(self) -> None:
        got = False
        try:
            while True:
                msg = self.queue.get_nowait()
                if isinstance(msg, Exception):
                    messagebox.showerror("Evolution error", repr(msg))
                    continue
                self._on_generation(msg)
                got = True
        except queue.Empty:
            pass
        if got:
            self._redraw_plots()
        self.root.after(150, self._poll)

    def _on_generation(self, st: GenerationStats) -> None:
        self.latest = st
        self.history.append(st)
        if len(self.history) > 1:  # free memory: only the genomes of the latest state are used
            self.history[-2].champion = None  # type: ignore[assignment]
            self.history[-2].species_champions = {}
        label = f"champion of generation {st.generation} (F={fmt(st.best_fitness)}, " \
                f"{st.best_apples:.1f} apples on average in training)"
        if self.follow.get():
            if self.runner is None or self.replay_cfg != self.game_cfg:
                self._set_champion(st.champion, label, self.game_cfg)
            else:
                self.pending = (st.champion, label)
        if self.view.get() == "species":
            self._sync_species()
        state = "running" if self.evo and self.evo.running.is_set() else "paused"
        self.status.configure(text=(
            f"Generation {st.generation:4d} ({state}) │ best F {fmt(st.best_fitness):>6} │ mean F {fmt(st.mean_fitness):>6} │ "
            f"max apples {st.max_apples:3d} │ species {st.n_species:2d} (δt={st.threshold:.2f}) │ "
            f"champion: {st.champion_hidden} hidden, {st.champion_conns} conn. │ {st.elapsed:.2f} s/gen"))

    def _redraw_plots(self) -> None:
        h = self.history
        if len(h) > 600:  # downsample to keep plotting smooth
            stride = math.ceil(len(h) / 600)
            h = h[::stride] + ([h[-1]] if (len(h) - 1) % stride else [])
        xs = [s.generation for s in h]
        self.fit_plot.draw(xs, [("best", [s.best_fitness for s in h], BEST),
                                ("mean", [s.mean_fitness for s in h], MEAN)])
        self.sp_plot.draw(xs, [s.species_sizes for s in h], self.neat_cfg.population_size)

    # ------------------------------------------------------- replay
    def _seed(self) -> int:
        return int(self.rng.integers(2**31))

    @property
    def species_view(self) -> bool:
        return self.view.get() == "species"

    @property
    def focus(self) -> Runner | None:
        """Snake whose network is displayed."""
        if self.species_view:
            for r in self.runners:
                if r.sid == self.focus_sid:
                    return r
            return self.runners[0] if self.runners else None
        return self.runner

    def _toggle_view(self) -> None:
        self.view.set("champion" if self.species_view else "species")
        self._on_view_change()

    def _on_view_change(self) -> None:
        if self.species_view:
            self._sync_species()
        self._draw_grid()
        self._draw_game()
        self._refresh_network()

    def _set_champion(self, genome: Genome, label: str, cfg: GameConfig) -> None:
        if cfg != self.replay_cfg:
            self.replay_cfg = cfg
            if not self.species_view:
                self._draw_grid()
        self.runner = Runner(genome, cfg, label)
        self.runner.new_episode(self._seed())
        self._draw_game()
        self._refresh_network()

    def _sync_species(self) -> None:
        """One snake per living species (the species champion), in order of species appearance.

        A snake from a species that is still alive is not interrupted: it receives its species' new
        champion when its game ends. The board of an extinct species disappears immediately.
        """
        st = self.latest
        if st is None:
            return
        shown = dict(list(st.species_champions.items())[:MAX_SPECIES_VIEW])
        current = {r.sid: r for r in self.runners}
        runners = []
        for sid, g in shown.items():
            r = current.get(sid)
            if r is None:
                r = Runner(g, self.game_cfg, f"species {sid}", sid)
                r.new_episode(self._seed())
            runners.append(r)
        self.runners = sorted(runners, key=lambda r: r.sid or 0)

    def _new_episode(self) -> None:
        for r in (self.runners if self.species_view else [self.runner]):
            if r is not None:
                r.new_episode(self._seed())
        self._draw_game()

    def _tick(self) -> None:
        sps = self.speed.get()
        self.speed_lbl.configure(text=f"{sps:.0f} steps/s")
        frame_ms = max(16 if not self.species_view else 33, int(1000 / sps))
        n_steps = max(1, round(sps * frame_ms / 1000))
        now = time.monotonic()
        if self.species_view:
            for r in list(self.runners):
                if r.game.alive:
                    r.advance(n_steps)
                elif r.ready_to_restart(now):
                    new = self.latest.species_champions.get(r.sid) if self.latest else None
                    if new is not None and new is not r.genome:
                        r.set_genome(new, f"species {r.sid}")
                    r.new_episode(self._seed())
        elif self.runner is not None:
            r = self.runner
            if r.game.alive:
                r.advance(n_steps)
            elif r.ready_to_restart(now):
                if self.pending is not None and self.follow.get():
                    g, label = self.pending
                    self.pending = None
                    self._set_champion(g, label, self.game_cfg)
                else:
                    r.new_episode(self._seed())
        self._draw_game()
        self._refresh_network()
        self._update_activations()
        self.root.after(frame_ms, self._tick)

    def _on_board_click(self, e: tk.Event) -> None:
        if not self.species_view:
            return
        for r, (x0, y0, x1, y1) in self.panels:
            if x0 - 4 <= e.x <= x1 + 4 and y0 - 22 <= e.y <= y1 + 4:
                self.focus_sid = r.sid
                self._refresh_network()
                self._draw_game()
                return

    # ------------------------------------------------------------ drawing
    def _draw_grid(self) -> None:
        b, cfg = self.board, self.replay_cfg
        b.delete("all")
        if self.species_view:
            return
        w, h = max(b.winfo_width(), 50), max(b.winfo_height(), 50)
        self.cell = c = max(4, min(w // cfg.width, h // cfg.height))
        self.ox, self.oy = (w - c * cfg.width) // 2, (h - c * cfg.height) // 2
        for x in range(cfg.width):
            for y in range(cfg.height):
                x0, y0 = self._corner(x, y)
                b.create_rectangle(x0, y0, x0 + c, y0 + c, fill=GRID if (x + y) % 2 == 0 else BG, outline="")

    def _corner(self, x: int, y: int) -> tuple[int, int]:
        return self.ox + x * self.cell, self.oy + y * self.cell

    def _center(self, x: int, y: int) -> tuple[float, float]:
        return self.ox + (x + 0.5) * self.cell, self.oy + (y + 0.5) * self.cell

    def _draw_game(self) -> None:
        if self.species_view:
            self._draw_species()
        else:
            self._draw_single()

    def _draw_single(self) -> None:
        b, c = self.board, self.cell
        b.delete("dyn")
        if self.runner is None or self.runner.game is None:
            return
        g = self.runner.game
        if self.show_rays.get() and g.alive:
            hx, hy = self._center(*g.head)
            for dx, dy, tw, tb, tf in g.cast_rays():
                ex, ey = hx + dx * (tw - 0.5) * c, hy + dy * (tw - 0.5) * c
                if tf:
                    b.create_line(hx, hy, hx + dx * tf * c, hy + dy * tf * c, fill=FOOD, width=2, tags="dyn")
                else:
                    b.create_line(hx, hy, ex, ey, fill="#556070", dash=(3, 3), tags="dyn")
                if tb:
                    px, py = hx + dx * tb * c, hy + dy * tb * c
                    b.create_oval(px - 4, py - 4, px + 4, py + 4, outline="#f59e0b", width=2, tags="dyn")
        if g.food is not None:
            x0, y0 = self._corner(*g.food)
            p = c * 0.18
            b.create_oval(x0 + p, y0 + p, x0 + c - p, y0 + c - p, fill=FOOD, outline="", tags="dyn")
        n = len(g.body)
        dead = not g.alive
        for i, (x, y) in enumerate(g.body):
            t = i / max(1, n - 1)
            col = mix(ACCENT, "#166534", t) if not dead else mix("#9ca3af", "#374151", t)
            p = 1 if i == 0 else 2
            x0, y0 = self._corner(x, y)
            b.create_rectangle(x0 + p, y0 + p, x0 + c - p, y0 + c - p, fill=col, outline="", tags="dyn")
        hx, hy = self._center(*g.head)  # eyes oriented along the heading
        fx, fy = DIRS[g.heading]
        rx, ry = -fy, fx
        for s in (-1, 1):
            ex = hx + fx * c * 0.18 + s * rx * c * 0.2
            ey = hy + fy * c * 0.18 + s * ry * c * 0.2
            b.create_oval(ex - 2.5, ey - 2.5, ex + 2.5, ey + 2.5, fill=BG, outline="", tags="dyn")
        if dead:
            item = b.create_text(b.winfo_width() / 2, b.winfo_height() / 2, text=f"Dead: {g.death}",
                                 fill=FG, font=FONT_L, tags="dyn")
            x0, y0, x1, y1 = b.bbox(item)
            b.create_rectangle(x0 - 14, y0 - 8, x1 + 14, y1 + 8, fill=BG, outline=FOOD, width=2, tags="dyn")
            b.tag_raise(item)
        self.game_lbl.configure(text=f"Apples {g.apples:3d}   Length {g.length:3d}   Steps {g.steps:5d}   "
                                     f"Hunger {g.hunger:3d}/{g.cfg.hunger_limit}")
        self.replay_lbl.configure(text=self.runner.label)

    def _draw_species(self) -> None:
        """Mosaic: one board per species, bordered with the species color (see the speciation chart)."""
        b = self.board
        b.delete("all")
        self.panels = []
        W, H = max(b.winfo_width(), 100), max(b.winfo_height(), 100)
        if not self.runners:
            b.create_text(W / 2, H / 2, text="Species will appear at the end of the first generation",
                          fill=MUTED, font=FONT)
            return
        cfg = self.game_cfg
        k, head, pad = len(self.runners), 18, 10
        cell, cols, rows = max(((min((W / c - pad) / cfg.width, (H / math.ceil(k / c) - pad - head) / cfg.height),
                                 c, math.ceil(k / c)) for c in range(1, k + 1)))
        cell = max(2.0, math.floor(cell))
        pw, ph = W / cols, H / rows
        bw, bh = cell * cfg.width, cell * cfg.height
        sizes = self.latest.species_sizes if self.latest else {}
        focus = self.focus
        for i, r in enumerate(self.runners):
            g = r.game
            ox = (i % cols) * pw + (pw - bw) / 2
            oy = (i // cols) * ph + head + (ph - head - bh) / 2
            col = species_color(r.sid or 0)
            focused = r is focus
            b.create_rectangle(ox - 2, oy - 2, ox + bw + 2, oy + bh + 2, fill=BG, outline=col if focused else mix(BG, col, 0.5),
                               width=3 if focused else 1)
            if pw >= 210:
                title = f"Species {r.sid} · {sizes.get(r.sid, 0)} ind. · {g.apples} apples"
            else:
                title = f"{r.sid} · {sizes.get(r.sid, 0)} ind. · {g.apples} a."
            b.create_text(ox, oy - 5, text=title, anchor="sw", fill=FG if focused else col,
                          font=FONT_B if focused and cell >= 10 else FONT_S)
            if g.food is not None:
                fx, fy = g.food
                p = cell * 0.15
                b.create_oval(ox + fx * cell + p, oy + fy * cell + p, ox + (fx + 1) * cell - p, oy + (fy + 1) * cell - p,
                              fill=FOOD, outline="")
            body_col = col if g.alive else "#4b5563"
            pts = [v for x, y in g.body for v in (ox + (x + 0.5) * cell, oy + (y + 0.5) * cell)]
            b.create_line(*pts, fill=mix(body_col, BG, 0.35), width=max(1.0, cell - 2), capstyle="projecting",
                          joinstyle="miter")
            hx, hy = g.head
            b.create_rectangle(ox + hx * cell + 1, oy + hy * cell + 1, ox + (hx + 1) * cell - 1, oy + (hy + 1) * cell - 1,
                               fill=body_col if g.alive else "#6b7280", outline="")
            if not g.alive:
                b.create_text(ox + bw / 2, oy + bh / 2, text=g.death, fill=FG, font=FONT_B)
            self.panels.append((r, (ox, oy, ox + bw, oy + bh)))
        if focus is not None and focus.game is not None:
            f = focus.game
            self.game_lbl.configure(text=f"Species {focus.sid}   Apples {f.apples:3d}   Length {f.length:3d}   "
                                         f"Steps {f.steps:5d}")
        self.replay_lbl.configure(text="Each board shows the best individual of a species (same color as the "
                                       "speciation chart). Click a board to show its network.")

    def _draw_network_placeholder(self) -> None:
        self.netc.delete("all")
        self.netc.create_text(self.netc.winfo_width() / 2, self.netc.winfo_height() / 2,
                              text="The champion's network will appear here", fill=MUTED, font=FONT)

    def _refresh_network(self) -> None:
        f = self.focus
        if f is None:
            if self._net_owner is not None or not self.netc.find_all():
                self._net_owner = None
                self._draw_network_placeholder()
        elif f is not self._net_owner or f.genome is not self._net_genome:
            self._draw_network(f)
            self._update_activations()

    def _draw_network(self, runner: Runner) -> None:
        cv, g, net = self.netc, runner.genome, runner.net
        self._net_owner, self._net_genome = runner, runner.genome
        cv.delete("all")
        W, H = max(cv.winfo_width(), 300), max(cv.winfo_height(), 250)
        top, bottom = 44, H - 10
        x_in, x_out = 100, W - 105
        labels = input_labels(runner.cfg)

        inputs = sorted(n for n, ng in g.nodes.items() if ng.kind == INPUT)
        outputs = sorted(n for n, ng in g.nodes.items() if ng.kind == OUTPUT)
        hidden = [n for n, ng in g.nodes.items() if ng.kind == HIDDEN]
        cols = sorted({net.depth[n] for n in hidden})
        pos: dict[int, tuple[float, float]] = {}
        for i, n in enumerate(inputs):
            pos[n] = (x_in, top + (i + 0.5) * (bottom - top) / len(inputs))
        for j, d in enumerate(cols):
            members = sorted((n for n in hidden if net.depth[n] == d))
            x = x_in + (j + 1) * (x_out - x_in) / (len(cols) + 1)
            gap = min(46.0, (bottom - top) / max(1, len(members)))
            y0 = (top + bottom) / 2 - gap * (len(members) - 1) / 2
            for k, n in enumerate(members):
                pos[n] = (x + (12 if k % 2 else -12) * (len(members) > 6), y0 + k * gap)
        gap = 70
        y0 = (top + bottom) / 2 - gap * (len(outputs) - 1) / 2
        for k, n in enumerate(outputs):
            pos[n] = (x_out, y0 + k * gap)

        enabled = sorted((c for c in g.conns.values() if c.enabled), key=lambda c: abs(c.weight))
        for c in enabled:
            a = min(1.0, abs(c.weight) / 3.0)
            col = mix(PANEL, POS if c.weight > 0 else NEG, 0.15 + 0.85 * a)
            (x1, y1), (x2, y2) = pos[c.src], pos[c.dst]
            cv.create_line(x1, y1, x2, y2, fill=col, width=0.6 + 2.4 * a)

        self.node_items.clear()
        self.output_items.clear()
        for n, (x, y) in pos.items():
            kind = g.nodes[n].kind
            r = 5 if kind == INPUT else 8 if kind == HIDDEN else 11
            self.node_items[n] = cv.create_oval(x - r, y - r, x + r, y + r, fill=NEUTRAL, outline="#4b5563")
        for i, n in enumerate(inputs):
            x, y = pos[n]
            cv.create_text(x - 10, y, text=labels[i] if i < len(labels) else f"x{i}", fill=MUTED,
                           font=FONT_S, anchor="e")
        for k, n in enumerate(outputs):
            x, y = pos[n]
            self.output_items[n] = cv.create_text(x + 16, y, text=ACTION_NAMES[k], fill=MUTED, font=FONT, anchor="w")
        who = "Champion" if runner.sid is None else f"Species {runner.sid}"
        cv.create_text(12, 14, anchor="w", fill=FG if runner.sid is None else species_color(runner.sid), font=FONT_B,
                       text=f"{who} network — {g.n_hidden} hidden node(s), {g.n_enabled} active connections")
        cv.create_text(12, 30, anchor="w", fill=MUTED, font=FONT_S,
                       text="edges: green = positive weight, pink = negative, thickness ∝ |w|  —  nodes: blue (−1) to orange (+1)")

    def _update_activations(self) -> None:
        f = self.focus
        if f is None or f is not self._net_owner or not self.node_items:
            return
        vals, idx = f.net.values, f.net.index
        for n, item in self.node_items.items():
            self.netc.itemconfigure(item, fill=diverging(float(vals[idx[n]])))
        for k, n in enumerate(sorted(self.output_items)):
            chosen = k == f.next_action
            self.netc.itemconfigure(self.output_items[n], fill=ACCENT if chosen else MUTED,
                                    font=FONT_B if chosen else FONT)
            self.netc.itemconfigure(self.node_items[n], outline=ACCENT if chosen else "#4b5563",
                                    width=3 if chosen else 1)

    # --------------------------------------------------- files / exit
    def _ask_save(self) -> None:
        f = self.focus
        genome = f.genome if f else (self.latest.champion if self.latest else None)
        if genome is None:
            messagebox.showinfo("Nothing to save", "No genome is displayed yet.")
            return
        gen = self.latest.generation if self.latest else 0
        name = f"species{f.sid}_g{gen}.json" if f is not None and f.sid is not None else f"champion_g{gen}.json"
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Genome JSON", "*.json")],
                                            initialfile=name)
        if path:
            meta = {"generation": gen, "fitness": genome.fitness, "species": f.sid if f else None}
            save_genome(path, genome, f.cfg if f else self.game_cfg, meta)

    def _ask_load(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Genome JSON", "*.json")])
        if path:
            self._load(path)

    def _load(self, path: str) -> None:
        try:
            genome, cfg, meta = load_genome(path)
        except Exception as exc:
            messagebox.showerror("Cannot load", str(exc))
            return
        self.follow.set(False)
        self.pending = None
        if self.species_view:
            self.view.set("champion")
        label = f"loaded genome: {path.rsplit('/', 1)[-1]}" + (
            f" (generation {meta['generation']})" if "generation" in meta else "")
        self.replay_cfg = cfg
        self._draw_grid()
        self._set_champion(genome, label, cfg)

    def on_close(self) -> None:
        if self.evo is not None:
            self.evo.stop()
            if self.evo.evaluator is not None:
                self.evo.evaluator.close()
        self.root.destroy()


def run_gui(game_cfg: GameConfig, neat_cfg: NEATConfig, *, seed: int = 0, episodes: int = 3,
            workers: int = 0, genome_path: str | None = None, view: str = "champion") -> None:
    root = tk.Tk()
    _setup_fonts(root)
    App(root, game_cfg, neat_cfg, seed=seed, episodes=episodes, workers=workers, genome_path=genome_path,
        view="species" if view == "species" else "champion")
    root.mainloop()
