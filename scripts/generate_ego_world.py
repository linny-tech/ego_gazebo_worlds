#!/usr/bin/env python3
"""Generate and validate a deterministic Gazebo 11 obstacle world."""

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


MAP_SIZE = 50.0
MAP_HEIGHT = 4.0
HALF_MAP = MAP_SIZE / 2.0
WALL_THICKNESS = 0.4
WALL_INNER_EDGE = HALF_MAP - WALL_THICKNESS

# Use margin above the user's strict 1.5 m requirement.
REQUIRED_CLEARANCE = 1.5
PLACEMENT_CLEARANCE = 1.75
WALL_CLEARANCE = 1.75
CENTER_CLEAR_RADIUS = 5.0
DEFAULT_SEED = 3601133

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PACKAGE_DIR / "worlds" / "ego_50x50.world"
DEFAULT_LAYOUT = PACKAGE_DIR / "config" / "obstacle_layout.csv"
DEFAULT_REPORT = PACKAGE_DIR / "config" / "validation_report.txt"

COLORS: Sequence[Tuple[float, float, float, float]] = (
    (0.72, 0.25, 0.20, 1.0),
    (0.20, 0.42, 0.72, 1.0),
    (0.23, 0.62, 0.36, 1.0),
    (0.84, 0.56, 0.18, 1.0),
    (0.52, 0.31, 0.66, 1.0),
    (0.20, 0.63, 0.66, 1.0),
)


@dataclass(frozen=True)
class Obstacle:
    name: str
    kind: str
    x: float
    y: float
    width: float
    depth: float
    height: float
    yaw: float
    color_index: int

    @property
    def radius(self) -> float:
        """Conservative footprint bounding radius."""
        if self.kind == "cylinder":
            return self.width / 2.0
        return math.hypot(self.width / 2.0, self.depth / 2.0)

    @property
    def is_large(self) -> bool:
        return max(self.width, self.depth) > 4.0


def surface_clearance(a: Obstacle, b: Obstacle) -> float:
    """Conservative clearance using footprint bounding circles."""
    return math.hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius


def center_clearance(obstacle: Obstacle) -> float:
    return math.hypot(obstacle.x, obstacle.y) - obstacle.radius


def wall_clearance(obstacle: Obstacle) -> float:
    return min(
        WALL_INNER_EDGE - abs(obstacle.x) - obstacle.radius,
        WALL_INNER_EDGE - abs(obstacle.y) - obstacle.radius,
    )


def valid_candidate(candidate: Obstacle, placed: Iterable[Obstacle]) -> bool:
    if center_clearance(candidate) < CENTER_CLEAR_RADIUS:
        return False
    if wall_clearance(candidate) < WALL_CLEARANCE:
        return False
    return all(
        surface_clearance(candidate, other) >= PLACEMENT_CLEARANCE
        for other in placed
    )


def candidate_position(rng: random.Random) -> Tuple[float, float]:
    """Mixture distribution produces an irregular, non-grid layout."""
    cluster_centers = (
        (-15.0, -12.0),
        (-14.0, 12.5),
        (-4.0, 17.0),
        (12.5, -13.0),
        (16.0, 8.5),
        (8.0, 17.0),
    )

    if rng.random() < 0.72:
        cx, cy = rng.choice(cluster_centers)
        return rng.gauss(cx, 5.0), rng.gauss(cy, 5.0)

    return rng.uniform(-21.0, 21.0), rng.uniform(-21.0, 21.0)


def size_for_class(rng: random.Random, size_class: str, kind: str) -> Tuple[float, float, float]:
    if size_class == "large":
        width = rng.uniform(4.15, 5.0)
        depth = width if kind == "cylinder" else rng.uniform(4.0, 5.0)
        height = rng.uniform(2.6, MAP_HEIGHT)
    elif size_class == "medium":
        width = rng.uniform(2.25, 3.8)
        depth = width if kind == "cylinder" else rng.uniform(2.0, 3.8)
        height = rng.uniform(1.7, MAP_HEIGHT)
    else:
        width = rng.uniform(1.0, 2.2)
        depth = width if kind == "cylinder" else rng.uniform(1.0, 2.2)
        height = rng.uniform(1.0, 3.4)

    return width, depth, height


def generate_layout(seed: int) -> List[Obstacle]:
    rng = random.Random(seed)

    # Place large objects first. Only three of 27 obstacles are large.
    size_classes = ["large"] * 3 + ["medium"] * 8 + ["small"] * 16
    placed: List[Obstacle] = []
    box_count = 0
    cylinder_count = 0

    for size_class in size_classes:
        kind = "cylinder" if rng.random() < 0.38 else "box"
        if size_class == "large" and sum(o.kind == "cylinder" for o in placed if o.is_large) >= 1:
            kind = "box"

        width, depth, height = size_for_class(rng, size_class, kind)

        for _ in range(30000):
            x, y = candidate_position(rng)
            yaw = rng.uniform(-math.pi, math.pi) if kind == "box" else 0.0

            sequence = cylinder_count + 1 if kind == "cylinder" else box_count + 1
            candidate = Obstacle(
                name=f"{kind}_{sequence:02d}",
                kind=kind,
                x=x,
                y=y,
                width=width,
                depth=depth,
                height=height,
                yaw=yaw,
                color_index=len(placed) % len(COLORS),
            )

            if valid_candidate(candidate, placed):
                placed.append(candidate)
                if kind == "cylinder":
                    cylinder_count += 1
                else:
                    box_count += 1
                break
        else:
            raise RuntimeError(
                f"Unable to place {size_class} {kind}; reduce obstacle count or clearance"
            )

    return placed


def validate_layout(obstacles: Sequence[Obstacle]) -> str:
    if not obstacles:
        raise ValueError("Layout contains no obstacles")

    min_pair = math.inf
    min_pair_names = ("", "")
    for index, obstacle in enumerate(obstacles):
        if obstacle.kind not in {"box", "cylinder"}:
            raise ValueError(f"Unsupported obstacle kind: {obstacle.kind}")
        if not (1.0 <= obstacle.width <= 5.0):
            raise ValueError(f"{obstacle.name}: width/diameter out of range")
        if not (1.0 <= obstacle.depth <= 5.0):
            raise ValueError(f"{obstacle.name}: depth out of range")
        if not (1.0 <= obstacle.height <= MAP_HEIGHT):
            raise ValueError(f"{obstacle.name}: height out of range")
        if center_clearance(obstacle) < CENTER_CLEAR_RADIUS:
            raise ValueError(f"{obstacle.name}: enters center takeoff area")
        if wall_clearance(obstacle) < WALL_CLEARANCE:
            raise ValueError(f"{obstacle.name}: too close to boundary wall")

        for other in obstacles[index + 1 :]:
            clearance = surface_clearance(obstacle, other)
            if clearance < min_pair:
                min_pair = clearance
                min_pair_names = (obstacle.name, other.name)

    if min_pair <= REQUIRED_CLEARANCE:
        raise ValueError(
            f"Minimum clearance {min_pair:.3f} m is not strictly above "
            f"{REQUIRED_CLEARANCE:.3f} m"
        )

    large_count = sum(obstacle.is_large for obstacle in obstacles)
    box_count = sum(obstacle.kind == "box" for obstacle in obstacles)
    cylinder_count = len(obstacles) - box_count

    return "\n".join(
        (
            "EGO Gazebo map validation: PASS",
            f"Map volume: {MAP_SIZE:.1f} x {MAP_SIZE:.1f} x {MAP_HEIGHT:.1f} m",
            f"Obstacle count: {len(obstacles)}",
            f"Boxes: {box_count}",
            f"Cylinders: {cylinder_count}",
            f"Large-footprint obstacles: {large_count}",
            f"Center clear radius: {CENTER_CLEAR_RADIUS:.3f} m",
            f"Required pair clearance: > {REQUIRED_CLEARANCE:.3f} m",
            f"Conservative minimum pair clearance: {min_pair:.3f} m",
            f"Closest pair: {min_pair_names[0]} / {min_pair_names[1]}",
            f"Minimum wall clearance: {min(wall_clearance(o) for o in obstacles):.3f} m",
        )
    ) + "\n"


def rgba_text(color: Tuple[float, float, float, float]) -> str:
    return " ".join(f"{component:.3f}" for component in color)


def static_box_model(
    name: str,
    x: float,
    y: float,
    z: float,
    width: float,
    depth: float,
    height: float,
    color: Tuple[float, float, float, float],
    yaw: float = 0.0,
) -> str:
    color_text = rgba_text(color)
    return f"""    <model name='{name}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 {yaw:.4f}</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{width:.3f} {depth:.3f} {height:.3f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{width:.3f} {depth:.3f} {height:.3f}</size></box></geometry>
          <material><ambient>{color_text}</ambient><diffuse>{color_text}</diffuse></material>
        </visual>
      </link>
    </model>"""


def obstacle_model(obstacle: Obstacle) -> str:
    color_text = rgba_text(COLORS[obstacle.color_index])
    z = obstacle.height / 2.0

    if obstacle.kind == "cylinder":
        radius = obstacle.width / 2.0
        geometry = (
            f"<cylinder><radius>{radius:.3f}</radius>"
            f"<length>{obstacle.height:.3f}</length></cylinder>"
        )
    else:
        geometry = (
            f"<box><size>{obstacle.width:.3f} {obstacle.depth:.3f} "
            f"{obstacle.height:.3f}</size></box>"
        )

    return f"""    <model name='{obstacle.name}'>
      <static>true</static>
      <pose>{obstacle.x:.3f} {obstacle.y:.3f} {z:.3f} 0 0 {obstacle.yaw:.4f}</pose>
      <link name='link'>
        <collision name='collision'><geometry>{geometry}</geometry></collision>
        <visual name='visual'>
          <geometry>{geometry}</geometry>
          <material><ambient>{color_text}</ambient><diffuse>{color_text}</diffuse></material>
        </visual>
      </link>
    </model>"""


def world_text(obstacles: Sequence[Obstacle]) -> str:
    models = [
        static_box_model(
            "floor", 0.0, 0.0, -0.05, MAP_SIZE, MAP_SIZE, 0.1,
            (0.32, 0.34, 0.36, 1.0),
        ),
        static_box_model(
            "north_wall", 0.0, HALF_MAP - WALL_THICKNESS / 2.0,
            MAP_HEIGHT / 2.0, MAP_SIZE, WALL_THICKNESS, MAP_HEIGHT,
            (0.16, 0.18, 0.20, 1.0),
        ),
        static_box_model(
            "south_wall", 0.0, -HALF_MAP + WALL_THICKNESS / 2.0,
            MAP_HEIGHT / 2.0, MAP_SIZE, WALL_THICKNESS, MAP_HEIGHT,
            (0.16, 0.18, 0.20, 1.0),
        ),
        static_box_model(
            "east_wall", HALF_MAP - WALL_THICKNESS / 2.0, 0.0,
            MAP_HEIGHT / 2.0, WALL_THICKNESS, MAP_SIZE, MAP_HEIGHT,
            (0.16, 0.18, 0.20, 1.0),
        ),
        static_box_model(
            "west_wall", -HALF_MAP + WALL_THICKNESS / 2.0, 0.0,
            MAP_HEIGHT / 2.0, WALL_THICKNESS, MAP_SIZE, MAP_HEIGHT,
            (0.16, 0.18, 0.20, 1.0),
        ),
    ]

    # Visual-only center marker: no collision and therefore not an obstacle.
    center_marker = """    <model name='center_takeoff_marker'>
      <static>true</static>
      <pose>0 0 0.006 0 0 0</pose>
      <link name='link'>
        <visual name='visual'>
          <geometry><cylinder><radius>2.0</radius><length>0.01</length></cylinder></geometry>
          <material><ambient>0.12 0.50 0.16 1</ambient><diffuse>0.12 0.50 0.16 1</diffuse></material>
        </visual>
      </link>
    </model>"""
    models.append(center_marker)
    models.extend(obstacle_model(obstacle) for obstacle in obstacles)

    joined_models = "\n".join(models)
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='ego_50x50'>
    <gravity>0 0 -9.8066</gravity>
    <magnetic_field>6.0e-06 2.3e-05 -4.2e-05</magnetic_field>
    <atmosphere type='adiabatic'/>

    <physics name='px4_ode' default='1' type='ode'>
      <max_step_size>0.004</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>250</real_time_update_rate>
      <ode>
        <solver><type>quick</type><iters>20</iters><sor>1.3</sor></solver>
        <constraints><cfm>0</cfm><erp>0.2</erp><contact_max_correcting_vel>100</contact_max_correcting_vel></constraints>
      </ode>
    </physics>

    <scene>
      <ambient>0.45 0.45 0.45 1</ambient>
      <background>0.72 0.80 0.90 1</background>
      <shadows>false</shadows>
      <grid>true</grid>
    </scene>

    <spherical_coordinates>
      <surface_model>EARTH_WGS84</surface_model>
      <latitude_deg>47.397742</latitude_deg>
      <longitude_deg>8.545594</longitude_deg>
      <elevation>488.0</elevation>
      <heading_deg>0</heading_deg>
    </spherical_coordinates>

    <include>
      <uri>model://sun</uri>
    </include>

{joined_models}
  </world>
</sdf>
"""


def write_layout_csv(path: Path, obstacles: Sequence[Obstacle]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ("name", "kind", "x", "y", "width", "depth", "height", "yaw", "color_index")
        )
        for obstacle in obstacles:
            writer.writerow(
                (
                    obstacle.name,
                    obstacle.kind,
                    f"{obstacle.x:.6f}",
                    f"{obstacle.y:.6f}",
                    f"{obstacle.width:.6f}",
                    f"{obstacle.depth:.6f}",
                    f"{obstacle.height:.6f}",
                    f"{obstacle.yaw:.6f}",
                    obstacle.color_index,
                )
            )


def read_layout_csv(path: Path) -> List[Obstacle]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            Obstacle(
                name=row["name"],
                kind=row["kind"],
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["width"]),
                depth=float(row["depth"]),
                height=float(row["height"]),
                yaw=float(row["yaw"]),
                color_index=int(row["color_index"]),
            )
            for row in csv.DictReader(stream)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.validate_only:
        obstacles = read_layout_csv(args.layout)
    else:
        obstacles = generate_layout(args.seed)
        write_layout_csv(args.layout, obstacles)
        args.world.parent.mkdir(parents=True, exist_ok=True)
        args.world.write_text(world_text(obstacles), encoding="utf-8")

    report = validate_layout(obstacles)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
