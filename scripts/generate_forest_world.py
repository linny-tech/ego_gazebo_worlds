#!/usr/bin/env python3
"""Generate and validate a deterministic 50 x 50 x 5 m forest world."""

import argparse
import csv
import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


MAP_SIZE = 50.0
MAP_HEIGHT = 5.0
HALF_MAP = MAP_SIZE / 2.0
WALL_THICKNESS = 0.3
WALL_INNER_EDGE = HALF_MAP - WALL_THICKNESS

TREE_COUNT = 72
REQUIRED_TREE_CLEARANCE = 1.8
PLACEMENT_TREE_CLEARANCE = 2.05
WALL_CLEARANCE = 1.5
START_POSITION = (-21.0, 0.0)
START_CLEAR_RADIUS = 3.0
GOAL_POSITION = (21.0, 0.0)
GOAL_CLEAR_RADIUS = 2.5
NAVIGATION_CLEARANCE = 0.65
NAVIGATION_GRID_RESOLUTION = 0.5
DEFAULT_SEED = 3605055

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PACKAGE_DIR / "worlds" / "ego_forest_50x50x5.world"
DEFAULT_LAYOUT = PACKAGE_DIR / "config" / "forest_layout.csv"
DEFAULT_REPORT = PACKAGE_DIR / "config" / "forest_validation_report.txt"

TRUNK_COLORS: Sequence[Tuple[float, float, float, float]] = (
    (0.29, 0.16, 0.08, 1.0),
    (0.36, 0.21, 0.10, 1.0),
    (0.24, 0.13, 0.06, 1.0),
    (0.42, 0.27, 0.13, 1.0),
)

CROWN_COLORS: Sequence[Tuple[float, float, float, float]] = (
    (0.08, 0.30, 0.10, 1.0),
    (0.12, 0.38, 0.13, 1.0),
    (0.16, 0.43, 0.16, 1.0),
    (0.09, 0.34, 0.16, 1.0),
)


@dataclass(frozen=True)
class Tree:
    name: str
    x: float
    y: float
    trunk_radius: float
    trunk_height: float
    crown_radius: float
    crown_z: float
    color_index: int


def surface_clearance(a: Tree, b: Tree) -> float:
    return math.hypot(a.x - b.x, a.y - b.y) - a.trunk_radius - b.trunk_radius


def point_clearance(tree: Tree, point: Tuple[float, float]) -> float:
    return math.hypot(tree.x - point[0], tree.y - point[1]) - tree.trunk_radius


def wall_clearance(tree: Tree) -> float:
    return min(
        WALL_INNER_EDGE - abs(tree.x) - tree.trunk_radius,
        WALL_INNER_EDGE - abs(tree.y) - tree.trunk_radius,
    )


def valid_candidate(candidate: Tree, placed: Iterable[Tree]) -> bool:
    if wall_clearance(candidate) < WALL_CLEARANCE:
        return False
    if point_clearance(candidate, START_POSITION) < START_CLEAR_RADIUS:
        return False
    if point_clearance(candidate, GOAL_POSITION) < GOAL_CLEAR_RADIUS:
        return False
    return all(
        surface_clearance(candidate, other) >= PLACEMENT_TREE_CLEARANCE
        for other in placed
    )


def candidate_position(rng: random.Random) -> Tuple[float, float]:
    """Mix uniform samples and loose clusters for a natural, non-grid forest."""
    cluster_centers = (
        (-13.0, -14.0),
        (-11.0, 13.5),
        (0.0, -9.0),
        (1.5, 12.0),
        (13.0, -13.0),
        (14.0, 11.0),
    )
    if rng.random() < 0.62:
        center_x, center_y = rng.choice(cluster_centers)
        return rng.gauss(center_x, 4.6), rng.gauss(center_y, 4.6)
    return rng.uniform(-22.5, 22.5), rng.uniform(-22.5, 22.5)


def generate_layout(seed: int) -> List[Tree]:
    rng = random.Random(seed)
    placed: List[Tree] = []

    for index in range(1, TREE_COUNT + 1):
        trunk_radius = rng.uniform(0.18, 0.38)
        trunk_height = rng.uniform(4.15, MAP_HEIGHT)
        crown_radius = rng.uniform(0.72, 1.05)
        crown_z = min(MAP_HEIGHT - crown_radius, trunk_height - 0.55)

        for _ in range(40000):
            x, y = candidate_position(rng)
            candidate = Tree(
                name=f"tree_{index:03d}",
                x=x,
                y=y,
                trunk_radius=trunk_radius,
                trunk_height=trunk_height,
                crown_radius=crown_radius,
                crown_z=crown_z,
                color_index=(index - 1) % len(TRUNK_COLORS),
            )
            if valid_candidate(candidate, placed):
                placed.append(candidate)
                break
        else:
            raise RuntimeError(
                "Unable to place all trees; reduce tree count or clearance"
            )

    return placed


def navigation_path_exists(trees: Sequence[Tree]) -> bool:
    """Check a conservative 2-D route from the west start to the east goal."""
    resolution = NAVIGATION_GRID_RESOLUTION
    margin = NAVIGATION_CLEARANCE
    minimum = -WALL_INNER_EDGE + margin
    maximum = WALL_INNER_EDGE - margin
    cell_count = int(math.floor((maximum - minimum) / resolution)) + 1

    def to_cell(point: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(round((point[0] - minimum) / resolution)),
            int(round((point[1] - minimum) / resolution)),
        )

    def cell_point(cell: Tuple[int, int]) -> Tuple[float, float]:
        return minimum + cell[0] * resolution, minimum + cell[1] * resolution

    def is_free(cell: Tuple[int, int]) -> bool:
        if not (0 <= cell[0] < cell_count and 0 <= cell[1] < cell_count):
            return False
        x, y = cell_point(cell)
        return all(
            math.hypot(x - tree.x, y - tree.y)
            > tree.trunk_radius + NAVIGATION_CLEARANCE
            for tree in trees
        )

    start = to_cell(START_POSITION)
    goal = to_cell(GOAL_POSITION)
    if not is_free(start) or not is_free(goal):
        return False

    queue = deque([start])
    visited = {start}
    # Four-connected search avoids accepting diagonal corner cuts between trees.
    neighbors = ((-1, 0), (0, -1), (0, 1), (1, 0))

    while queue:
        current = queue.popleft()
        if current == goal:
            return True
        for offset_x, offset_y in neighbors:
            candidate = current[0] + offset_x, current[1] + offset_y
            if candidate not in visited and is_free(candidate):
                visited.add(candidate)
                queue.append(candidate)
    return False


def validate_layout(trees: Sequence[Tree]) -> str:
    if len(trees) != TREE_COUNT:
        raise ValueError(f"Expected {TREE_COUNT} trees, got {len(trees)}")

    minimum_pair = math.inf
    closest_pair = ("", "")
    for index, tree in enumerate(trees):
        if not (0.18 <= tree.trunk_radius <= 0.38):
            raise ValueError(f"{tree.name}: trunk radius out of range")
        if not (4.15 <= tree.trunk_height <= MAP_HEIGHT):
            raise ValueError(f"{tree.name}: trunk height out of range")
        if wall_clearance(tree) < WALL_CLEARANCE:
            raise ValueError(f"{tree.name}: too close to boundary wall")
        if point_clearance(tree, START_POSITION) < START_CLEAR_RADIUS:
            raise ValueError(f"{tree.name}: enters the west-edge takeoff area")
        if point_clearance(tree, GOAL_POSITION) < GOAL_CLEAR_RADIUS:
            raise ValueError(f"{tree.name}: enters the east-edge goal area")

        for other in trees[index + 1 :]:
            clearance = surface_clearance(tree, other)
            if clearance < minimum_pair:
                minimum_pair = clearance
                closest_pair = tree.name, other.name

    if minimum_pair <= REQUIRED_TREE_CLEARANCE:
        raise ValueError(
            f"Minimum tree clearance {minimum_pair:.3f} m is not above "
            f"{REQUIRED_TREE_CLEARANCE:.3f} m"
        )
    if not navigation_path_exists(trees):
        raise ValueError("No conservative route exists from start to goal")

    minimum_start = min(point_clearance(tree, START_POSITION) for tree in trees)
    minimum_goal = min(point_clearance(tree, GOAL_POSITION) for tree in trees)
    return "\n".join(
        (
            "EGO Gazebo forest map validation: PASS",
            f"Map volume: {MAP_SIZE:.1f} x {MAP_SIZE:.1f} x {MAP_HEIGHT:.1f} m",
            f"Tree count: {len(trees)}",
            f"PX4 spawn position: ({START_POSITION[0]:.1f}, {START_POSITION[1]:.1f}, 0.2) m",
            f"Suggested EGO goal: ({GOAL_POSITION[0]:.1f}, {GOAL_POSITION[1]:.1f}) m",
            f"Required tree clearance: > {REQUIRED_TREE_CLEARANCE:.3f} m",
            f"Conservative minimum tree clearance: {minimum_pair:.3f} m",
            f"Closest pair: {closest_pair[0]} / {closest_pair[1]}",
            f"Minimum wall clearance: {min(wall_clearance(tree) for tree in trees):.3f} m",
            f"Minimum spawn-area clearance: {minimum_start:.3f} m",
            f"Minimum goal-area clearance: {minimum_goal:.3f} m",
            f"West-to-east route ({NAVIGATION_CLEARANCE:.2f} m inflation): PASS",
            "Tree crowns: visual only; trunks provide LiDAR/collision obstacles",
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
    transparency: float = 0.0,
) -> str:
    color_text = rgba_text(color)
    return f"""    <model name='{name}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
      <link name='link'>
        <collision name='collision'>
          <geometry><box><size>{width:.3f} {depth:.3f} {height:.3f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{width:.3f} {depth:.3f} {height:.3f}</size></box></geometry>
          <material><ambient>{color_text}</ambient><diffuse>{color_text}</diffuse></material>
          <transparency>{transparency:.2f}</transparency>
        </visual>
      </link>
    </model>"""


def marker_model(name: str, x: float, y: float, color: str) -> str:
    return f"""    <model name='{name}'>
      <static>true</static>
      <pose>{x:.3f} {y:.3f} 0.006 0 0 0</pose>
      <link name='link'>
        <visual name='visual'>
          <geometry><cylinder><radius>1.0</radius><length>0.01</length></cylinder></geometry>
          <material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>
        </visual>
      </link>
    </model>"""


def tree_model(tree: Tree) -> str:
    trunk_color = rgba_text(TRUNK_COLORS[tree.color_index])
    crown_color = rgba_text(CROWN_COLORS[tree.color_index])
    return f"""    <model name='{tree.name}'>
      <static>true</static>
      <pose>{tree.x:.3f} {tree.y:.3f} 0 0 0 0</pose>
      <link name='tree_link'>
        <collision name='trunk_collision'>
          <pose>0 0 {tree.trunk_height / 2.0:.3f} 0 0 0</pose>
          <geometry>
            <cylinder><radius>{tree.trunk_radius:.3f}</radius><length>{tree.trunk_height:.3f}</length></cylinder>
          </geometry>
        </collision>
        <visual name='trunk_visual'>
          <pose>0 0 {tree.trunk_height / 2.0:.3f} 0 0 0</pose>
          <geometry>
            <cylinder><radius>{tree.trunk_radius:.3f}</radius><length>{tree.trunk_height:.3f}</length></cylinder>
          </geometry>
          <material><ambient>{trunk_color}</ambient><diffuse>{trunk_color}</diffuse></material>
        </visual>
        <visual name='crown_visual'>
          <pose>0 0 {tree.crown_z:.3f} 0 0 0</pose>
          <geometry><sphere><radius>{tree.crown_radius:.3f}</radius></sphere></geometry>
          <material><ambient>{crown_color}</ambient><diffuse>{crown_color}</diffuse></material>
        </visual>
      </link>
    </model>"""


def world_text(trees: Sequence[Tree]) -> str:
    models = [
        static_box_model(
            "forest_floor", 0.0, 0.0, -0.05, MAP_SIZE, MAP_SIZE, 0.1,
            (0.12, 0.26, 0.09, 1.0),
        ),
        static_box_model(
            "north_boundary", 0.0, HALF_MAP - WALL_THICKNESS / 2.0,
            MAP_HEIGHT / 2.0, MAP_SIZE, WALL_THICKNESS, MAP_HEIGHT,
            (0.08, 0.16, 0.07, 1.0), 0.35,
        ),
        static_box_model(
            "south_boundary", 0.0, -HALF_MAP + WALL_THICKNESS / 2.0,
            MAP_HEIGHT / 2.0, MAP_SIZE, WALL_THICKNESS, MAP_HEIGHT,
            (0.08, 0.16, 0.07, 1.0), 0.35,
        ),
        static_box_model(
            "east_boundary", HALF_MAP - WALL_THICKNESS / 2.0, 0.0,
            MAP_HEIGHT / 2.0, WALL_THICKNESS, MAP_SIZE, MAP_HEIGHT,
            (0.08, 0.16, 0.07, 1.0), 0.35,
        ),
        static_box_model(
            "west_boundary", -HALF_MAP + WALL_THICKNESS / 2.0, 0.0,
            MAP_HEIGHT / 2.0, WALL_THICKNESS, MAP_SIZE, MAP_HEIGHT,
            (0.08, 0.16, 0.07, 1.0), 0.35,
        ),
        marker_model("west_edge_spawn_marker", *START_POSITION, "0.12 0.40 0.90 1"),
        marker_model("east_edge_goal_marker", *GOAL_POSITION, "0.95 0.68 0.08 1"),
    ]
    models.extend(tree_model(tree) for tree in trees)
    joined_models = "\n".join(models)

    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='ego_forest_50x50x5'>
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
      <ambient>0.42 0.46 0.40 1</ambient>
      <background>0.67 0.78 0.88 1</background>
      <shadows>false</shadows>
      <grid>false</grid>
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


def write_layout_csv(path: Path, trees: Sequence[Tree]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "name", "x", "y", "trunk_radius", "trunk_height",
                "crown_radius", "crown_z", "color_index",
            )
        )
        for tree in trees:
            writer.writerow(
                (
                    tree.name,
                    f"{tree.x:.6f}",
                    f"{tree.y:.6f}",
                    f"{tree.trunk_radius:.6f}",
                    f"{tree.trunk_height:.6f}",
                    f"{tree.crown_radius:.6f}",
                    f"{tree.crown_z:.6f}",
                    tree.color_index,
                )
            )


def read_layout_csv(path: Path) -> List[Tree]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            Tree(
                name=row["name"],
                x=float(row["x"]),
                y=float(row["y"]),
                trunk_radius=float(row["trunk_radius"]),
                trunk_height=float(row["trunk_height"]),
                crown_radius=float(row["crown_radius"]),
                crown_z=float(row["crown_z"]),
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
        trees = read_layout_csv(args.layout)
    else:
        trees = generate_layout(args.seed)
        write_layout_csv(args.layout, trees)
        args.world.parent.mkdir(parents=True, exist_ok=True)
        args.world.write_text(world_text(trees), encoding="utf-8")

    report = validate_layout(trees)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
