"""RViz markers for what the obstacle monitor's checks see, and the plan being followed.

For the latest check (the motion the trajectory server asked about, about 1 s ahead):
  - lines through the wrist (the wrist_yaw box's center) and the fingertip (the gripper box's
    far end) at every step, colored by what the check said about the step: green clear,
    yellow slowed for an obstacle's distance, orange sweeping unsafe space (contact speed),
    red blocked;
  - at the step that matters most (the first blocked one, else the first unsafe one, else the
    slowest), a ghost of just the shapes that hit something there (or the limiting shape, if
    only slowed), and a label by the arm with the state and how far ahead it is, e.g.
    "BLOCKED: 0.4 s";
  - the voxels behind it, in the same color, as cubes of a voxel's average size: the obstacle
    voxels a blocked step sweeps, or the unsafe voxels an unsafe step sweeps plus the obstacles
    within contamination reach of its contaminated shadow (what contaminated it).
Obstacle voxels turn grey and fade out over fade_time once checks stop showing them, so one
that flickers for one check stays visible long enough to look at. Plans are
drawn as lines through the wrist and fingertip with a faint ghost at the end; a plan that's
replaced (a new goal, or later a replan) turns grey and fades out over plan_fade_time.

Everything is kept with the base heading it was recorded at and drawn in base_footprint
turned by the heading change since, so turning the base doesn't drag old markers along.
"""

import collections
import dataclasses

import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

import stretch_obstacle_grid as og

CLEAR, SLOWED, UNSAFE, BLOCKED = range(4)
STATUS_NAMES = {CLEAR: "clear", SLOWED: "slowed", UNSAFE: "unsafe", BLOCKED: "blocked"}
STEP_COLORS = {CLEAR: (0.2, 0.8, 0.2), SLOWED: (1.0, 0.85, 0.1), UNSAFE: (1.0, 0.5, 0.0), BLOCKED: (0.9, 0.1, 0.1)}
STALE_COLOR = (0.55, 0.55, 0.55)
PLAN_COLOR = (0.4, 0.6, 1.0)
LINE_WIDTH = 0.008  # m
TEXT_HEIGHT = 0.05  # m
TEXT_ABOVE_ARM = 0.1  # m
SHAPE_KINDS = {"box": Marker.CUBE, "cylinder": Marker.CYLINDER, "capsule": Marker.CYLINDER, "sphere": Marker.SPHERE}


# --- Geometry ------------------------------------------------------------------------------

def rotation_to_quaternion(r):
    """(x, y, z, w) of a rotation matrix."""
    w = np.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return float(x), float(y), float(z), float(w)


def turn(yaw):
    """4x4 rotation about z by yaw (rad)."""
    c, s = np.cos(yaw), np.sin(yaw)
    out = np.eye(4)
    out[:2, :2] = [[c, -s], [s, c]]
    return out


def wrist_and_fingertip(body, names):
    """The wrist (wrist_yaw box's center) and fingertip (the middle of the gripper box's face
    farthest from the wrist) of a posed body, in base_footprint."""
    shapes = body.shapes()
    wrist = np.asarray(shapes[names.index("wrist_yaw")].pose, dtype=float)[:3, 3]
    gripper = shapes[names.index("gripper_box")]
    pose, half = np.asarray(gripper.pose, dtype=float), np.asarray(gripper.dims, dtype=float)
    away = pose[:3, :3].T @ (pose[:3, 3] - wrist)  # wrist to gripper center, in the box's axes
    axis = int(np.argmax(np.abs(away) / half))
    tip = pose[:3, 3] + pose[:3, axis] * np.sign(away[axis]) * half[axis]
    return wrist, tip


def path_points(bodies, names):
    """(wrist, fingertip) arrays (n, 3) through posed bodies."""
    points = [wrist_and_fingertip(b, names) for b in bodies]
    return np.array([w for w, _ in points]), np.array([t for _, t in points])


def mean_voxel_size(grid, reach=1.1):
    """Edge (m) of a cube with the average volume of the grid's voxels within `reach` m of
    the axis: cell deep and tall, and an arc between one and two cells wide."""
    config = grid.config
    volumes, counts = [], []
    for band in grid.bands:
        for ri in range(band.r_begin, band.r_end):
            r = config.r_min + (ri + 0.5) * config.cell
            if r <= reach:
                volumes.append(config.cell * config.cell * r * band.dtheta)
                counts.append(band.theta_bins)
    return float(np.average(volumes, weights=counts) ** (1.0 / 3.0))


# --- What each check said ------------------------------------------------------------------

def step_statuses(steps, durations):
    """CLEAR, SLOWED, UNSAFE or BLOCKED per step, from check()'s StepLimits and each step's
    planned duration (s)."""
    out = []
    for step, duration in zip(steps, durations):
        if step.blocked:
            out.append(BLOCKED)
        elif step.unsafe:
            out.append(UNSAFE)
        elif step.min_duration > duration * 1.001:
            out.append(SLOWED)
        else:
            out.append(CLEAR)
    return out


def choose_focus(statuses, steps, durations):
    """(step, status) worth showing: the first blocked step, else the first unsafe one, else
    the slowest, if it's slowed at all; None if everything is clear."""
    for status in (BLOCKED, UNSAFE):
        if status in statuses:
            return statuses.index(status), status
    if SLOWED in statuses:
        caps = [d / s.min_duration if s.min_duration > 0 else 1.0 for s, d in zip(steps, durations)]
        k = int(np.argmin(caps))
        return k, SLOWED
    return None


@dataclasses.dataclass
class Focus:
    """The step a check is limited by, and what limits it."""
    step: int
    status: int
    shapes: list            # og.SelfShapes that hit something there (or the limiting one), at the step's end
    text: str
    anchor: np.ndarray      # where the label goes: the arm's center now, base_footprint
    voxels: np.ndarray      # (n,) voxel indices behind it (none if only slowed)
    centers: np.ndarray     # (n, 3) their centers, base_footprint
    alphas: np.ndarray      # (n,) fainter for voxels at the edge (StepHit.in_both false)
    fades: np.ndarray       # (n,) bool: an obstacle voxel, left fading when it stops showing


@dataclasses.dataclass
class CheckRecord:
    stamp: float            # s, when the check was answered
    heading: float          # rad, the base's heading then (odometry)
    wrist: np.ndarray       # (n, 3) at each sample
    tip: np.ndarray
    statuses: list          # per step
    focus: Focus = None


def focus_text(status, start_time):
    """The label: the state and how far ahead (s of planned motion), e.g. "BLOCKED: 0.4 s"."""
    return f"{STATUS_NAMES[status].upper()}: {start_time:.1f} s"


def make_focus(k, status, bodies, names, steps, start_times, hits=(), centers=None, sources=(), max_voxels=3000):
    """The Focus of a check at step k. hits: the step's StepHits (blocked or unsafe); sources:
    voxel indices of obstacles behind contaminated shadow among them. At most max_voxels voxels."""
    end = bodies[k + 1].shapes()
    if hits:
        shapes = [end[i] for i in sorted({h.shape for h in hits})]
    else:
        shapes = [end[steps[k].shape]] if steps[k].shape >= 0 else []
    arm = np.asarray(bodies[0].shapes()[names.index("arm")].pose, dtype=float)[:3, 3]
    hits = list(hits)[:max_voxels]
    sources = np.asarray(sources, dtype=np.int64)[:max(0, max_voxels - len(hits))]
    voxels = np.r_[np.array([h.voxel for h in hits], dtype=np.int64), sources]
    alphas = np.r_[[0.9 if h.in_both else 0.4 for h in hits], np.full(len(sources), 0.9)]
    fades = np.r_[np.full(len(hits), status == BLOCKED), np.ones(len(sources), dtype=bool)]
    return Focus(step=k, status=status, shapes=shapes, text=focus_text(status, start_times[k]),
                 anchor=arm + [0.0, 0.0, TEXT_ABOVE_ARM], voxels=voxels,
                 centers=centers[voxels] if len(voxels) else np.zeros((0, 3)), alphas=alphas, fades=fades)


def contamination_sources(contaminated, obstacles, reach):
    """Indices into `obstacles` ((m, 3) centers) within `reach` m of any of `contaminated`
    ((n, 3) centers): the obstacles a contaminated shadow can have spread from. Contamination
    spreads through the shadow, so its path is at least this straight-line distance."""
    contaminated, obstacles = np.asarray(contaminated, dtype=float), np.asarray(obstacles, dtype=float)
    if not len(contaminated) or not len(obstacles):
        return np.zeros(0, dtype=np.int64)
    lo, hi = contaminated.min(axis=0) - reach, contaminated.max(axis=0) + reach
    near = np.flatnonzero(np.all((obstacles >= lo) & (obstacles <= hi), axis=1))
    found = np.zeros(len(near), dtype=bool)
    for chunk in np.array_split(contaminated, max(1, len(contaminated) // 256)):
        d2 = ((obstacles[near, None, :] - chunk[None, :, :]) ** 2).sum(axis=2)
        found |= (d2 <= reach * reach).any(axis=1)
    return near[found]


# --- Markers -------------------------------------------------------------------------------

def color(rgb, alpha):
    return ColorRGBA(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(alpha))


def points_msg(points, transform):
    p = np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]
    return [Point(x=float(x), y=float(y), z=float(z)) for x, y, z in p]


def shape_marker(header, ns, marker_id, shape, rgba, transform=None):
    """One of the robot's shapes (og.SelfShape) as a marker; a capsule as its bounding cylinder."""
    pose = np.asarray(shape.pose, dtype=float)
    if transform is not None:
        pose = transform @ pose
    d = np.asarray(shape.dims, dtype=float)
    m = Marker(header=header, ns=ns, id=marker_id, type=SHAPE_KINDS[shape.kind], action=Marker.ADD, color=rgba)
    m.pose.position.x, m.pose.position.y, m.pose.position.z = (float(v) for v in pose[:3, 3])
    m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = \
        rotation_to_quaternion(pose[:3, :3])
    if shape.kind == "box":
        m.scale.x, m.scale.y, m.scale.z = (float(v) for v in 2 * d)
    elif shape.kind == "sphere":
        m.scale.x = m.scale.y = m.scale.z = float(2 * d[0])
    else:
        m.scale.x = m.scale.y = float(2 * d[0])
        m.scale.z = float(2 * d[1] + (2 * d[0] if shape.kind == "capsule" else 0.0))
    return m


def shape_markers(header, shapes, ns="robot_shapes", rgba=None, transform=None, first_id=0):
    """The robot's shapes as translucent markers."""
    rgba = rgba or color((0.2, 0.6, 1.0), 0.35)
    return [shape_marker(header, ns, first_id + i, s, rgba, transform) for i, s in enumerate(shapes)]


def path_lines(header, ns, marker_id, wrist, tip, transform, colors):
    """LINE_LIST through the wrist and fingertip points, segment k in colors[k] (ColorRGBA)."""
    m = Marker(header=header, ns=ns, id=marker_id, type=Marker.LINE_LIST, action=Marker.ADD)
    m.pose.orientation.w = 1.0
    m.scale.x = LINE_WIDTH
    for track in (wrist, tip):
        p = points_msg(track, transform)
        for k in range(len(p) - 1):
            m.points += [p[k], p[k + 1]]
            m.colors += [colors[k], colors[k]]
    return m


def voxel_cubes(header, ns, marker_id, centers, colors, size, transform):
    m = Marker(header=header, ns=ns, id=marker_id, type=Marker.CUBE_LIST, action=Marker.ADD)
    m.pose.orientation.w = 1.0
    m.scale.x = m.scale.y = m.scale.z = size
    m.points = points_msg(centers, transform)
    m.colors = list(colors)
    return m


def text_marker(header, ns, marker_id, text, at, rgba, transform):
    """Text at `at` (base_footprint)."""
    # RViz (jazzy) draws a space in a text marker several letters wide; a tab reads as a space.
    m = Marker(header=header, ns=ns, id=marker_id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD,
               color=rgba, text=text.replace(" ", "\t"))
    m.pose.position = points_msg([at], transform)[0]
    m.pose.orientation.w = 1.0
    m.scale.z = TEXT_HEIGHT
    return m


@dataclasses.dataclass
class PlanRecord:
    heading: float          # rad, the base's heading its rotation is measured from
    wrist: np.ndarray
    tip: np.ndarray
    end_shapes: list
    replaced_at: float = None


class CheckDisplay:
    """Keeps the latest check, fading leftovers of blocks, and plans; draws them on demand."""

    def __init__(self, voxel_size, fade_time=3.0, plan_fade_time=10.0, check_timeout=0.5):
        self.voxel_size = voxel_size
        self.fade_time = fade_time
        self.plan_fade_time = plan_fade_time
        self.check_timeout = check_timeout  # s; an older check is no longer current
        self.check = None
        self.stale_voxels = {}  # voxel index -> (center, heading, since)
        self.plans = []         # the newest last; earlier ones fading

    def add_check(self, record):
        old = self.check
        self.check = record
        if old is not None:
            self._retire(old, record.stamp, still=record.focus)
        if record.focus is not None:
            for v in record.focus.voxels[record.focus.fades]:
                self.stale_voxels.pop(int(v), None)

    def add_plan(self, record, now):
        for plan in self.plans:
            if plan.replaced_at is None:
                plan.replaced_at = now
        self.plans.append(record)

    def _retire(self, record, now, still=None):
        """A check, no longer current: its obstacle voxels that `still` (the new check's focus)
        doesn't show start fading."""
        focus = record.focus
        if focus is None:
            return
        current = set() if still is None else {int(v) for v in still.voxels[still.fades]}
        for v, c in zip(focus.voxels[focus.fades], focus.centers[focus.fades]):
            if int(v) not in current:
                self.stale_voxels[int(v)] = (c, record.heading, now)

    def markers(self, header, now, heading_now):
        """Markers for everything still showing, after a DELETEALL (so nothing lingers)."""
        out = [Marker(header=header, action=Marker.DELETEALL)]
        if self.check is not None and now - self.check.stamp > self.check_timeout:
            self._retire(self.check, now)
            self.check = None

        def placed(heading):
            return turn(heading - heading_now)

        def fading(since, fade_time):
            return max(0.0, 1.0 - (now - since) / fade_time)

        # Plans: the current one, and replaced ones fading.
        self.plans = [p for p in self.plans if p.replaced_at is None or fading(p.replaced_at, self.plan_fade_time) > 0]
        for i, plan in enumerate(self.plans):
            if plan.replaced_at is None:
                rgb, line_alpha, ghost_alpha = PLAN_COLOR, 0.8, 0.12
            else:
                fade = fading(plan.replaced_at, self.plan_fade_time)
                rgb, line_alpha, ghost_alpha = STALE_COLOR, 0.6 * fade, 0.08 * fade
            transform = placed(plan.heading)
            n = len(plan.wrist) - 1
            out.append(path_lines(header, "plan", i, plan.wrist, plan.tip, transform, [color(rgb, line_alpha)] * n))
            out += shape_markers(header, plan.end_shapes, f"plan_end_{i}", color(rgb, ghost_alpha), transform)

        # Obstacle voxels no longer shown, fading.
        self.stale_voxels = {v: s for v, s in self.stale_voxels.items() if fading(s[2], self.fade_time) > 0}
        if self.stale_voxels:
            by_heading = collections.defaultdict(list)
            for c, heading, since in self.stale_voxels.values():
                by_heading[heading].append((c, color(STALE_COLOR, 0.8 * fading(since, self.fade_time))))
            for i, (heading, items) in enumerate(by_heading.items()):
                out.append(voxel_cubes(header, "stale_voxels", i, [c for c, _ in items], [k for _, k in items],
                                       self.voxel_size, placed(heading)))

        # The current check.
        check = self.check
        if check is not None:
            transform = placed(check.heading)
            colors = [color(STEP_COLORS[s], 0.9) for s in check.statuses]
            out.append(path_lines(header, "check_path", 0, check.wrist, check.tip, transform, colors))
            focus = check.focus
            if focus is not None:
                rgb = STEP_COLORS[focus.status]
                out += shape_markers(header, focus.shapes, "check_ghost", color(rgb, 0.35), transform)
                out.append(text_marker(header, "check_text", 0, focus.text, focus.anchor, color(rgb, 1.0), transform))
                if len(focus.voxels):
                    out.append(voxel_cubes(header, "check_voxels", 0, focus.centers,
                                           [color(rgb, a) for a in focus.alphas], self.voxel_size, transform))
        return out

