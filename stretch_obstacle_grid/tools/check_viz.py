"""RViz markers for what the obstacle monitor's checks see, and the plan being followed.

For the latest check (the motion the trajectory server asked about, about 1 s ahead):
  - lines through the wrist (the wrist_yaw box's center) and the fingertip (the gripper box's
    far end) at every step, colored by what the check said about the step: green clear,
    yellow slowed for an obstacle's distance, orange sweeping unsafe space (contact speed),
    red blocked;
  - at the step that matters most (the first blocked one, else the first unsafe one, else the
    slowest), a ghost of the robot's shapes that have moved by then (or move in the step) and a
    label saying why;
  - the voxels behind a blocked or unsafe step, as cubes of a voxel's average size: a block's
    colored by which LiDAR saw them, unsafe space's by label.
A block's voxels and ghost turn grey and fade out over fade_time once checks stop showing
them, so a block that flickers for one check stays visible long enough to look at. Plans are
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
# Block voxels by which LiDARs' latest frames have a return in them (bit i: LiDAR i). Neither:
# an obstacle kept from older frames (static).
SEEN_BY_COLORS = {0: (1.0, 1.0, 1.0), 1: (0.9, 0.2, 0.9), 2: (0.1, 0.8, 0.9), 3: (0.9, 0.1, 0.1)}
UNSAFE_COLORS = {int(og.UNKNOWN): (0.6, 0.6, 0.8), int(og.CONTAMINATED_SHADOW): (0.96, 0.65, 0.51)}
STALE_COLOR = (0.55, 0.55, 0.55)
PLAN_COLOR = (0.4, 0.6, 1.0)
LINE_WIDTH = 0.008  # m
TEXT_HEIGHT = 0.04  # m
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
    shapes: list            # og.SelfShapes at the end of the step
    text: str
    voxels: np.ndarray      # (n,) voxel indices behind it (none if only slowed)
    centers: np.ndarray     # (n, 3) their centers, base_footprint
    colors: list            # (r, g, b, a) per voxel; voxels at the edge (StepHit.in_both false) fainter


@dataclasses.dataclass
class CheckRecord:
    stamp: float            # s, when the check was answered
    heading: float          # rad, the base's heading then (odometry)
    wrist: np.ndarray       # (n, 3) at each sample
    tip: np.ndarray
    statuses: list          # per step
    focus: Focus = None


def focus_text(status, step, start_time, shape, hits=None, seen_by=None, labels=None, lidar_names=("left", "right"),
               cap=1.0):
    """What a check's focus says, on two lines, e.g. "BLOCKED 0.40 s ahead: gripper_box" and
    "6 voxels (right 6)"."""
    head = f"{STATUS_NAMES[status].upper()} {start_time:.2f} s ahead: {shape}\n"
    if status == SLOWED:
        return (f"{head}{100 * cap:.0f}% speed: {step.clearance_moving:.2f} m from a moving obstacle, "
                f"{step.clearance_occupied:.2f} m from any")
    n = len(hits)
    edge = sum(not h.in_both for h in hits)
    edge_text = f", {edge} at the edge" if edge else ""
    if status == BLOCKED:
        counts = collections.Counter(int(b) for b in seen_by)
        parts = [f"{' + '.join(lidar_names[i] for i in range(len(lidar_names)) if bits >> i & 1) or 'older frames'} "
                 f"{count}" for bits, count in sorted(counts.items())]
        return f"{head}{n} voxels ({', '.join(parts)}{edge_text})"
    counts = collections.Counter(int(v) for v in labels)
    return (f"{head}{n} voxels unsafe ({counts.get(int(og.UNKNOWN), 0)} unknown, "
            f"{counts.get(int(og.CONTAMINATED_SHADOW), 0)} contaminated{edge_text})")


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


def text_marker(header, ns, marker_id, text, shapes, rgba, transform, fallback=(0.0, 0.0, 1.0)):
    """Text above a ghost's shapes (or at `fallback`, base_footprint, if there are none)."""
    tops = np.array([np.asarray(s.pose, dtype=float)[:3, 3] for s in shapes]) if shapes else np.array([fallback])
    at = np.r_[tops[:, :2].mean(axis=0), tops[:, 2].max() + 0.12]
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


@dataclasses.dataclass
class StaleBlock:
    since: float            # s, when checks stopped showing it
    heading: float
    shapes: list
    text: str


class CheckDisplay:
    """Keeps the latest check, fading leftovers of blocks, and plans; draws them on demand."""

    def __init__(self, voxel_size, fade_time=3.0, plan_fade_time=10.0, check_timeout=0.5):
        self.voxel_size = voxel_size
        self.fade_time = fade_time
        self.plan_fade_time = plan_fade_time
        self.check_timeout = check_timeout  # s; an older check is no longer current
        self.check = None
        self.stale_voxels = {}  # voxel index -> (center, heading, since)
        self.stale_blocks = []
        self.plans = []         # the newest last; earlier ones fading

    def add_check(self, record):
        old = self.check
        self.check = record
        if old is not None:
            self._retire(old, record.stamp, still=record.focus)
        if record.focus is not None and record.focus.status == BLOCKED:
            for v in record.focus.voxels:
                self.stale_voxels.pop(int(v), None)

    def add_plan(self, record, now):
        for plan in self.plans:
            if plan.replaced_at is None:
                plan.replaced_at = now
        self.plans.append(record)

    def _retire(self, record, now, still=None):
        """A check's block, no longer current, starts fading: its voxels that `still` (the new
        check's focus) doesn't show, and its ghost unless the new check is blocked too."""
        focus = record.focus
        if focus is None or focus.status != BLOCKED:
            return
        current = set() if still is None or still.status != BLOCKED else {int(v) for v in still.voxels}
        for v, c in zip(focus.voxels, focus.centers):
            if int(v) not in current:
                self.stale_voxels[int(v)] = (c, record.heading, now)
        if still is None or still.status != BLOCKED:
            self.stale_blocks.append(StaleBlock(now, record.heading, focus.shapes, focus.text))

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

        # Blocks no longer shown, fading.
        self.stale_voxels = {v: s for v, s in self.stale_voxels.items() if fading(s[2], self.fade_time) > 0}
        if self.stale_voxels:
            by_heading = collections.defaultdict(list)
            for c, heading, since in self.stale_voxels.values():
                by_heading[heading].append((c, color(STALE_COLOR, 0.8 * fading(since, self.fade_time))))
            for i, (heading, items) in enumerate(by_heading.items()):
                out.append(voxel_cubes(header, "stale_voxels", i, [c for c, _ in items], [k for _, k in items],
                                       self.voxel_size, placed(heading)))
        self.stale_blocks = [b for b in self.stale_blocks if fading(b.since, self.fade_time) > 0][-5:]
        for i, block in enumerate(self.stale_blocks):
            fade = fading(block.since, self.fade_time)
            transform = placed(block.heading)
            out += shape_markers(header, block.shapes, f"stale_ghost_{i}", color(STALE_COLOR, 0.25 * fade), transform)
            out.append(text_marker(header, "stale_text", i, block.text, block.shapes,
                                   color(STALE_COLOR, fade), transform))

        # The current check.
        check = self.check
        if check is not None:
            transform = placed(check.heading)
            colors = [color(STEP_COLORS[s], 0.9) for s in check.statuses]
            out.append(path_lines(header, "check_path", 0, check.wrist, check.tip, transform, colors))
            focus = check.focus
            if focus is not None:
                rgb = STEP_COLORS[focus.status]
                out += shape_markers(header, focus.shapes, "check_ghost", color(rgb, 0.3), transform)
                out.append(text_marker(header, "check_text", 0, focus.text, focus.shapes, color(rgb, 1.0), transform))
                if len(focus.voxels):
                    out.append(voxel_cubes(header, "check_voxels", 0, focus.centers,
                                           [color(c[:3], c[3]) for c in focus.colors], self.voxel_size, transform))
        return out


def make_focus(k, status, bodies, names, steps, durations, start_times, hits=None, seen_by=None, labels=None,
               centers=None, lidar_names=("left", "right"), max_voxels=3000):
    """The Focus of a check at step k: StepHits with their LiDAR bits (seen_by), labels and
    centers for a blocked or unsafe step. At most max_voxels voxels are drawn."""
    step = steps[k]
    cap = durations[k] / step.min_duration if step.min_duration > 0 else 1.0
    hits = hits or []
    text = focus_text(status, step, start_times[k], names[step.shape] if step.shape >= 0 else "?", hits,
                      seen_by, labels, lidar_names, cap)
    voxels = np.array([h.voxel for h in hits][:max_voxels], dtype=np.int64)
    colors = []
    for i, hit in enumerate(hits[:max_voxels]):
        rgb = SEEN_BY_COLORS.get(int(seen_by[i]), SEEN_BY_COLORS[3]) if status == BLOCKED else \
            UNSAFE_COLORS.get(int(labels[i]), STALE_COLOR)
        colors.append((*rgb, 0.9 if hit.in_both else 0.4))
    return Focus(step=k, status=status, shapes=moved_shapes(bodies, k), text=text, voxels=voxels,
                 centers=centers[voxels] if len(voxels) else np.zeros((0, 3)), colors=colors)


def moved_shapes(bodies, k, threshold=0.005):
    """The shapes at the end of step k that have moved since the first sample or move during
    the step (any corner more than threshold m): the ghost leaves out what stays put."""
    def corners(shape):
        return np.array(og.piece_corners(shape))
    first, start, end = bodies[0].shapes(), bodies[k].shapes(), bodies[k + 1].shapes()
    return [e for f, s, e in zip(first, start, end)
            if max(np.abs(corners(e) - corners(f)).max(), np.abs(corners(e) - corners(s)).max()) > threshold]
