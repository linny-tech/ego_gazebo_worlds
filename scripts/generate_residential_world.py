#!/usr/bin/env python3
"""Generate and validate a 50 x 50 x 10 m residential Gazebo world."""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


MAP_SIZE = 50.0
MAP_HEIGHT = 10.0
HALF_MAP = MAP_SIZE / 2.0
WALL_THICKNESS = 0.3
WALL_INNER_EDGE = HALF_MAP - WALL_THICKNESS
MIN_BUILDING_GAP = 2.5
CENTRAL_ROUTE_HALF_WIDTH = 3.0
START_POSITION = (-22.0, 0.0)
GOAL_POSITION = (22.0, 0.0)
PILLARS_PER_RESIDENCE = 4

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PACKAGE_DIR / "worlds" / "ego_residential_50x50x10.world"
DEFAULT_LAYOUT = PACKAGE_DIR / "config" / "residential_layout.csv"
DEFAULT_REPORT = PACKAGE_DIR / "config" / "residential_validation_report.txt"

WALL_COLORS: Sequence[Tuple[float, float, float, float]] = (
    (0.78, 0.68, 0.53, 1.0),
    (0.72, 0.76, 0.70, 1.0),
    (0.82, 0.71, 0.62, 1.0),
    (0.68, 0.73, 0.80, 1.0),
)
ROOF_COLORS: Sequence[Tuple[float, float, float, float]] = (
    (0.42, 0.12, 0.10, 1.0),
    (0.32, 0.16, 0.13, 1.0),
    (0.46, 0.20, 0.12, 1.0),
    (0.28, 0.25, 0.24, 1.0),
)


@dataclass(frozen=True)
class Residence:
    name: str
    x: float
    y: float
    width: float
    depth: float
    height: float
    front_sign: int
    color_index: int


def generate_layout() -> List[Residence]:
    """Arrange two residential rows around a wide central community road."""
    heights = (8.8, 7.8, 9.6, 8.4, 8.4, 9.2, 7.6, 8.8)
    residences: List[Residence] = []
    index = 0
    for y, front_sign in ((14.0, -1), (-14.0, 1)):
        for x in (-15.0, -5.0, 5.0, 15.0):
            residences.append(
                Residence(
                    name=f"residence_{index + 1:02d}",
                    x=x,
                    y=y,
                    width=7.2,
                    depth=7.6,
                    height=heights[index],
                    front_sign=front_sign,
                    color_index=index % len(WALL_COLORS),
                )
            )
            index += 1
    return residences


def footprint_gap(a: Residence, b: Residence) -> float:
    gap_x = abs(a.x - b.x) - (a.width + b.width) / 2.0
    gap_y = abs(a.y - b.y) - (a.depth + b.depth) / 2.0
    if gap_x < 0.0 and gap_y < 0.0:
        return -min(-gap_x, -gap_y)
    return math.hypot(max(gap_x, 0.0), max(gap_y, 0.0))


def validate_layout(residences: Sequence[Residence], world: str = "") -> str:
    if len(residences) != 8:
        raise ValueError(f"Expected 8 residences, got {len(residences)}")

    minimum_gap = math.inf
    closest_pair = ("", "")
    for index, residence in enumerate(residences):
        if residence.front_sign not in (-1, 1):
            raise ValueError(f"{residence.name}: invalid front direction")
        if residence.height > MAP_HEIGHT or residence.height < 6.0:
            raise ValueError(f"{residence.name}: roof height outside map volume")
        if abs(residence.x) + residence.width / 2.0 > WALL_INNER_EDGE - 1.0:
            raise ValueError(f"{residence.name}: exceeds east/west residential zone")
        if abs(residence.y) + residence.depth / 2.0 > WALL_INNER_EDGE - 1.0:
            raise ValueError(f"{residence.name}: exceeds north/south residential zone")
        if abs(residence.y) - residence.depth / 2.0 <= CENTRAL_ROUTE_HALF_WIDTH + 2.0:
            raise ValueError(f"{residence.name}: intrudes into central road")

        for other in residences[index + 1 :]:
            gap = footprint_gap(residence, other)
            if gap < minimum_gap:
                minimum_gap = gap
                closest_pair = residence.name, other.name

    if minimum_gap <= MIN_BUILDING_GAP:
        raise ValueError(
            f"Minimum building gap {minimum_gap:.3f} m is not above "
            f"{MIN_BUILDING_GAP:.3f} m"
        )

    if world:
        lowered = world.lower()
        if "<cylinder" in lowered or "<sphere" in lowered:
            raise ValueError("Residential world must use box geometry only")
        roof_count = world.count("roof_collision")
        pillar_count = world.count("pillar_") // 2
        if roof_count != len(residences):
            raise ValueError(f"Expected {len(residences)} collision roofs, got {roof_count}")
        if pillar_count != len(residences) * PILLARS_PER_RESIDENCE:
            raise ValueError(
                f"Expected {len(residences) * PILLARS_PER_RESIDENCE} pillars, "
                f"got {pillar_count}"
            )

    return "\n".join(
        (
            "EGO Gazebo residential map validation: PASS",
            f"Map volume: {MAP_SIZE:.1f} x {MAP_SIZE:.1f} x {MAP_HEIGHT:.1f} m",
            f"Residence count: {len(residences)}",
            f"Collision roof count: {len(residences)}",
            f"Ground-to-roof rectangular pillars: {len(residences) * PILLARS_PER_RESIDENCE}",
            "Primitive obstacle geometry: boxes only (no cylinders or spheres)",
            f"Minimum residence gap: {minimum_gap:.3f} m",
            f"Closest residences: {closest_pair[0]} / {closest_pair[1]}",
            f"Central road clear width: {CENTRAL_ROUTE_HALF_WIDTH * 2.0:.1f} m",
            f"PX4 spawn position: ({START_POSITION[0]:.1f}, {START_POSITION[1]:.1f}, 0.2) m",
            f"Suggested EGO goal: ({GOAL_POSITION[0]:.1f}, {GOAL_POSITION[1]:.1f}) m",
            "West-to-east ground-level route through community gates: PASS",
        )
    ) + "\n"


def rgba(color: Tuple[float, float, float, float]) -> str:
    return " ".join(f"{value:.3f}" for value in color)


def box_elements(
    name: str,
    x: float,
    y: float,
    z: float,
    width: float,
    depth: float,
    height: float,
    color: Tuple[float, float, float, float],
) -> str:
    color_text = rgba(color)
    geometry = f"<geometry><box><size>{width:.3f} {depth:.3f} {height:.3f}</size></box></geometry>"
    return f"""        <collision name='{name}_collision'>
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          {geometry}
        </collision>
        <visual name='{name}_visual'>
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          {geometry}
          <material><ambient>{color_text}</ambient><diffuse>{color_text}</diffuse></material>
        </visual>"""


def residence_model(residence: Residence) -> str:
    wall_thickness = 0.34
    roof_thickness = 0.40
    pillar_size = 0.48
    door_width = 1.60
    roof_bottom = residence.height - roof_thickness
    front_y = residence.front_sign * (residence.depth / 2.0 - wall_thickness / 2.0)
    back_y = -front_y
    side_x = residence.width / 2.0 - wall_thickness / 2.0
    pillar_x = residence.width / 2.0 - pillar_size / 2.0
    pillar_y = residence.depth / 2.0 - pillar_size / 2.0
    facade_width = (residence.width - door_width) / 2.0
    facade_x = door_width / 2.0 + facade_width / 2.0

    wall_color = WALL_COLORS[residence.color_index]
    roof_color = ROOF_COLORS[residence.color_index]
    pillar_color = tuple(max(component - 0.10, 0.0) for component in wall_color[:3]) + (1.0,)

    parts = [
        box_elements(
            "roof", 0.0, 0.0, residence.height - roof_thickness / 2.0,
            residence.width + 0.6, residence.depth + 0.6, roof_thickness, roof_color,
        ),
        box_elements(
            "back_wall", 0.0, back_y, roof_bottom / 2.0,
            residence.width, wall_thickness, roof_bottom, wall_color,
        ),
        box_elements(
            "left_wall", -side_x, 0.0, roof_bottom / 2.0,
            wall_thickness, residence.depth, roof_bottom, wall_color,
        ),
        box_elements(
            "right_wall", side_x, 0.0, roof_bottom / 2.0,
            wall_thickness, residence.depth, roof_bottom, wall_color,
        ),
        box_elements(
            "front_left", -facade_x, front_y, roof_bottom / 2.0,
            facade_width, wall_thickness, roof_bottom, wall_color,
        ),
        box_elements(
            "front_right", facade_x, front_y, roof_bottom / 2.0,
            facade_width, wall_thickness, roof_bottom, wall_color,
        ),
        box_elements(
            "door_lintel", 0.0, front_y, roof_bottom - 0.55,
            door_width, wall_thickness, 1.10, wall_color,
        ),
    ]
    for pillar_index, (x, y) in enumerate(
        ((-pillar_x, -pillar_y), (-pillar_x, pillar_y), (pillar_x, -pillar_y), (pillar_x, pillar_y)),
        start=1,
    ):
        parts.append(
            box_elements(
                f"pillar_{pillar_index}", x, y, roof_bottom / 2.0,
                pillar_size, pillar_size, roof_bottom, pillar_color,
            )
        )

    return f"""    <model name='{residence.name}'>
      <static>true</static>
      <pose>{residence.x:.3f} {residence.y:.3f} 0 0 0 0</pose>
      <link name='building_link'>
{chr(10).join(parts)}
      </link>
    </model>"""


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
    color_text = rgba(color)
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


def gate_models(prefix: str, x: float) -> List[str]:
    color = (0.34, 0.37, 0.40, 1.0)
    return [
        static_box_model(f"{prefix}_left_post", x, -4.0, 2.25, 0.55, 0.75, 4.5, color),
        static_box_model(f"{prefix}_right_post", x, 4.0, 2.25, 0.55, 0.75, 4.5, color),
        static_box_model(f"{prefix}_roof", x, 0.0, 4.65, 0.85, 8.75, 0.35, color),
    ]


def world_text(residences: Sequence[Residence]) -> str:
    models = [
        static_box_model("community_floor", 0.0, 0.0, -0.05, 50.0, 50.0, 0.1, (0.25, 0.31, 0.24, 1.0)),
        static_box_model("central_road", 0.0, 0.0, 0.01, 50.0, 6.0, 0.02, (0.16, 0.17, 0.18, 1.0)),
        static_box_model("north_sidewalk", 0.0, 4.2, 0.04, 50.0, 1.6, 0.08, (0.48, 0.48, 0.46, 1.0)),
        static_box_model("south_sidewalk", 0.0, -4.2, 0.04, 50.0, 1.6, 0.08, (0.48, 0.48, 0.46, 1.0)),
        static_box_model("north_boundary", 0.0, 24.85, 5.0, 50.0, 0.3, 10.0, (0.20, 0.25, 0.28, 1.0), 0.55),
        static_box_model("south_boundary", 0.0, -24.85, 5.0, 50.0, 0.3, 10.0, (0.20, 0.25, 0.28, 1.0), 0.55),
        static_box_model("east_boundary", 24.85, 0.0, 5.0, 0.3, 50.0, 10.0, (0.20, 0.25, 0.28, 1.0), 0.55),
        static_box_model("west_boundary", -24.85, 0.0, 5.0, 0.3, 50.0, 10.0, (0.20, 0.25, 0.28, 1.0), 0.55),
        static_box_model("west_spawn_pad", -22.0, 0.0, 0.03, 2.0, 2.0, 0.06, (0.12, 0.40, 0.90, 1.0)),
        static_box_model("east_goal_pad", 22.0, 0.0, 0.03, 2.0, 2.0, 0.06, (0.95, 0.68, 0.08, 1.0)),
    ]
    models.extend(gate_models("west_gate", -20.0))
    models.extend(gate_models("east_gate", 20.0))

    planter_color = (0.34, 0.23, 0.13, 1.0)
    for row_y in (-6.4, 6.4):
        for index, x in enumerate((-12.0, 0.0, 12.0), start=1):
            row_name = "south" if row_y < 0 else "north"
            models.append(
                static_box_model(
                    f"{row_name}_planter_{index}", x, row_y, 0.28,
                    3.0, 1.2, 0.56, planter_color,
                )
            )

    models.extend(residence_model(residence) for residence in residences)
    joined_models = "\n".join(models)
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='ego_residential_50x50x10'>
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
      <ambient>0.48 0.49 0.47 1</ambient>
      <background>0.68 0.78 0.88 1</background>
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

    <include><uri>model://sun</uri></include>

{joined_models}
  </world>
</sdf>
"""


def write_layout(path: Path, residences: Sequence[Residence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("name", "x", "y", "width", "depth", "height", "front_sign", "color_index"))
        for residence in residences:
            writer.writerow(
                (
                    residence.name,
                    f"{residence.x:.3f}",
                    f"{residence.y:.3f}",
                    f"{residence.width:.3f}",
                    f"{residence.depth:.3f}",
                    f"{residence.height:.3f}",
                    residence.front_sign,
                    residence.color_index,
                )
            )


def read_layout(path: Path) -> List[Residence]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            Residence(
                name=row["name"],
                x=float(row["x"]),
                y=float(row["y"]),
                width=float(row["width"]),
                depth=float(row["depth"]),
                height=float(row["height"]),
                front_sign=int(row["front_sign"]),
                color_index=int(row["color_index"]),
            )
            for row in csv.DictReader(stream)
        ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    parser.add_argument("--layout", type=Path, default=DEFAULT_LAYOUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        residences = read_layout(args.layout)
        world = args.world.read_text(encoding="utf-8")
    else:
        residences = generate_layout()
        world = world_text(residences)
        write_layout(args.layout, residences)
        args.world.parent.mkdir(parents=True, exist_ok=True)
        args.world.write_text(world, encoding="utf-8")

    report = validate_layout(residences, world)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
