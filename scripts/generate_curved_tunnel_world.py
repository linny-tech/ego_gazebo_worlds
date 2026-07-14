#!/usr/bin/env python3
"""Generate and validate a narrow curved tunnel world for EGO testing."""

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


MAP_SIZE = 50.0
MAP_HEIGHT = 10.0
HALF_MAP = MAP_SIZE / 2.0
WALL_THICKNESS = 0.45
FLOOR_THICKNESS = 0.14
CEILING_THICKNESS = 0.35
SEGMENT_OVERLAP = 1.20
START_POSITION = (-22.0, -8.0)
GOAL_POSITION = (22.0, -5.0)

CENTERLINE: Sequence[Tuple[float, float]] = (
    START_POSITION,
    (-16.0, -8.0),
    (-11.0, -5.0),
    (-7.0, 1.0),
    (-1.0, 5.0),
    (6.0, 5.0),
    (12.0, 1.0),
    (16.0, -5.0),
    GOAL_POSITION,
)
SEGMENT_WIDTHS = (4.4, 4.6, 4.2, 4.8, 4.4, 4.6, 4.2, 4.8)
SEGMENT_HEIGHTS = (3.8, 4.2, 3.5, 4.0, 3.6, 4.4, 3.7, 4.1)

PACKAGE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORLD = PACKAGE_DIR / "worlds" / "ego_curved_tunnel_50x50x10.world"
DEFAULT_LAYOUT = PACKAGE_DIR / "config" / "tunnel_layout.csv"
DEFAULT_REPORT = PACKAGE_DIR / "config" / "tunnel_validation_report.txt"

ROCK_COLORS: Sequence[Tuple[float, float, float, float]] = (
    (0.24, 0.22, 0.20, 1.0),
    (0.30, 0.27, 0.23, 1.0),
    (0.20, 0.22, 0.23, 1.0),
    (0.33, 0.29, 0.24, 1.0),
)


@dataclass(frozen=True)
class TunnelSegment:
    name: str
    index: int
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    center_x: float
    center_y: float
    length: float
    yaw: float
    width: float
    height: float


def generate_layout() -> List[TunnelSegment]:
    segments: List[TunnelSegment] = []
    for index, ((start_x, start_y), (end_x, end_y), width, height) in enumerate(
        zip(CENTERLINE[:-1], CENTERLINE[1:], SEGMENT_WIDTHS, SEGMENT_HEIGHTS),
        start=1,
    ):
        dx = end_x - start_x
        dy = end_y - start_y
        segments.append(
            TunnelSegment(
                name=f"tunnel_segment_{index:02d}",
                index=index,
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                center_x=(start_x + end_x) / 2.0,
                center_y=(start_y + end_y) / 2.0,
                length=math.hypot(dx, dy),
                yaw=math.atan2(dy, dx),
                width=width,
                height=height,
            )
        )
    return segments


def angle_difference(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(second - first), math.cos(second - first)))


def validate_layout(segments: Sequence[TunnelSegment], world: str = "") -> str:
    if len(segments) != 8:
        raise ValueError(f"Expected 8 tunnel segments, got {len(segments)}")

    for index, segment in enumerate(segments):
        if not (4.0 <= segment.width <= 5.0):
            raise ValueError(f"{segment.name}: width outside narrow-tunnel range")
        if not (3.2 <= segment.height <= 4.6):
            raise ValueError(f"{segment.name}: height outside tunnel range")
        if segment.length < 4.0:
            raise ValueError(f"{segment.name}: segment too short")
        if max(abs(segment.start_x), abs(segment.end_x)) > 23.0:
            raise ValueError(f"{segment.name}: exceeds east/west map bounds")
        if max(abs(segment.start_y), abs(segment.end_y)) > 23.0:
            raise ValueError(f"{segment.name}: exceeds north/south map bounds")
        if index:
            previous = segments[index - 1]
            if (previous.end_x, previous.end_y) != (segment.start_x, segment.start_y):
                raise ValueError(f"{segment.name}: disconnected centerline")

    bend_angles = [
        math.degrees(angle_difference(first.yaw, second.yaw))
        for first, second in zip(segments[:-1], segments[1:])
    ]
    significant_bends = sum(angle >= 15.0 for angle in bend_angles)
    if significant_bends < 6:
        raise ValueError(f"Tunnel is not curved enough: {significant_bends} bends")
    if max(bend_angles) > 65.0:
        raise ValueError("Tunnel contains an excessively sharp bend")

    if world:
        lowered = world.lower()
        if "<cylinder" in lowered or "<sphere" in lowered:
            raise ValueError("Tunnel world must use box geometry only")
        expected = len(segments)
        counts = {
            "floors": world.count("name='floor_collision'"),
            "ceilings": world.count("name='ceiling_collision'"),
            "left_walls": world.count("name='left_wall_collision'"),
            "right_walls": world.count("name='right_wall_collision'"),
        }
        if any(count != expected for count in counts.values()):
            raise ValueError(f"Incomplete tunnel shell: {counts}")
        if world.count("name='ceiling_drop_collision'") != 4:
            raise ValueError("Expected four low-ceiling constrictions")
        if world.count("name='wall_protrusion_collision'") != 4:
            raise ValueError("Expected four side-wall protrusions")

    path_length = sum(segment.length for segment in segments)
    minimum_width = min(segment.width for segment in segments)
    minimum_height = min(segment.height for segment in segments)
    minimum_lateral_clearance = minimum_width / 2.0 - 0.50
    minimum_vertical_clearance = minimum_height - 0.45
    return "\n".join(
        (
            "EGO Gazebo curved tunnel validation: PASS",
            f"Map volume: {MAP_SIZE:.1f} x {MAP_SIZE:.1f} x {MAP_HEIGHT:.1f} m",
            f"Tunnel segment count: {len(segments)}",
            f"Centerline length: {path_length:.3f} m",
            f"Significant bends (>= 15 deg): {significant_bends}",
            f"Maximum bend angle: {max(bend_angles):.3f} deg",
            f"Tunnel inner-width range: {minimum_width:.1f} - {max(item.width for item in segments):.1f} m",
            f"Tunnel clear-height range: {minimum_height:.1f} - {max(item.height for item in segments):.1f} m",
            f"Minimum lateral clearance from center after protrusion: {minimum_lateral_clearance:.2f} m",
            f"Minimum vertical clearance below low ceiling: {minimum_vertical_clearance:.2f} m",
            "Low-ceiling constrictions: 4",
            "Side-wall protrusions: 4",
            "Primitive obstacle geometry: boxes only (no cylinders or spheres)",
            f"PX4 spawn position: ({START_POSITION[0]:.1f}, {START_POSITION[1]:.1f}, 0.3) m",
            f"Suggested EGO goal: ({GOAL_POSITION[0]:.1f}, {GOAL_POSITION[1]:.1f}) m",
            "Continuous curved centerline route: PASS",
        )
    ) + "\n"


def rgba(color: Tuple[float, float, float, float]) -> str:
    return " ".join(f"{component:.3f}" for component in color)


def box_elements(
    name: str,
    x: float,
    y: float,
    z: float,
    length: float,
    depth: float,
    height: float,
    color: Tuple[float, float, float, float],
) -> str:
    color_text = rgba(color)
    geometry = f"<geometry><box><size>{length:.3f} {depth:.3f} {height:.3f}</size></box></geometry>"
    return f"""        <collision name='{name}_collision'>
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          {geometry}
        </collision>
        <visual name='{name}_visual'>
          <pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose>
          {geometry}
          <material><ambient>{color_text}</ambient><diffuse>{color_text}</diffuse></material>
        </visual>"""


def tunnel_segment_model(segment: TunnelSegment) -> str:
    shell_length = segment.length + SEGMENT_OVERLAP
    outer_width = segment.width + 2.0 * WALL_THICKNESS
    rock_color = ROCK_COLORS[(segment.index - 1) % len(ROCK_COLORS)]
    ceiling_color = tuple(max(value - 0.05, 0.0) for value in rock_color[:3]) + (1.0,)
    floor_color = (0.16, 0.15, 0.14, 1.0)
    wall_offset = segment.width / 2.0 + WALL_THICKNESS / 2.0

    parts = [
        box_elements(
            "floor", 0.0, 0.0, -FLOOR_THICKNESS / 2.0,
            shell_length, outer_width, FLOOR_THICKNESS, floor_color,
        ),
        box_elements(
            "ceiling", 0.0, 0.0, segment.height + CEILING_THICKNESS / 2.0,
            shell_length, outer_width, CEILING_THICKNESS, ceiling_color,
        ),
        box_elements(
            "left_wall", 0.0, wall_offset, segment.height / 2.0,
            shell_length, WALL_THICKNESS, segment.height, rock_color,
        ),
        box_elements(
            "right_wall", 0.0, -wall_offset, segment.height / 2.0,
            shell_length, WALL_THICKNESS, segment.height, rock_color,
        ),
    ]

    # Alternate constrained sections to mimic irregular cave walls and roof.
    if segment.index % 2 == 0:
        parts.append(
            box_elements(
                "ceiling_drop", 0.35, 0.0, segment.height - 0.225,
                1.20, segment.width, 0.45, ceiling_color,
            )
        )
    else:
        side = 1.0 if segment.index in (1, 5) else -1.0
        parts.append(
            box_elements(
                "wall_protrusion", -0.40, side * (segment.width / 2.0 - 0.25), 1.15,
                1.50, 0.50, 2.30, rock_color,
            )
        )

    return f"""    <model name='{segment.name}'>
      <static>true</static>
      <pose>{segment.center_x:.3f} {segment.center_y:.3f} 0 0 0 {segment.yaw:.6f}</pose>
      <link name='tunnel_link'>
{chr(10).join(parts)}
      </link>
    </model>"""


def static_box_model(
    name: str,
    x: float,
    y: float,
    z: float,
    length: float,
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
          <geometry><box><size>{length:.3f} {depth:.3f} {height:.3f}</size></box></geometry>
        </collision>
        <visual name='visual'>
          <geometry><box><size>{length:.3f} {depth:.3f} {height:.3f}</size></box></geometry>
          <material><ambient>{color_text}</ambient><diffuse>{color_text}</diffuse></material>
          <transparency>{transparency:.2f}</transparency>
        </visual>
      </link>
    </model>"""


def junction_models(segments: Sequence[TunnelSegment]) -> List[str]:
    models: List[str] = []
    for index, (first, second) in enumerate(zip(segments[:-1], segments[1:]), start=1):
        x, y = first.end_x, first.end_y
        width = max(first.width, second.width) + 2.0 * WALL_THICKNESS
        height = min(first.height, second.height)
        models.append(
            static_box_model(
                f"junction_{index:02d}_floor", x, y, -FLOOR_THICKNESS / 2.0,
                width, width, FLOOR_THICKNESS, (0.16, 0.15, 0.14, 1.0),
            )
        )
        models.append(
            static_box_model(
                f"junction_{index:02d}_ceiling", x, y,
                height + CEILING_THICKNESS / 2.0,
                width, width, CEILING_THICKNESS, (0.20, 0.19, 0.18, 1.0),
            )
        )
    return models


def world_text(segments: Sequence[TunnelSegment]) -> str:
    models = [
        static_box_model("map_ground", 0.0, 0.0, -0.22, 50.0, 50.0, 0.20, (0.10, 0.10, 0.10, 1.0)),
        static_box_model("north_boundary", 0.0, 24.85, 5.0, 50.0, 0.3, 10.0, (0.12, 0.13, 0.14, 1.0), 0.65),
        static_box_model("south_boundary", 0.0, -24.85, 5.0, 50.0, 0.3, 10.0, (0.12, 0.13, 0.14, 1.0), 0.65),
        static_box_model("east_boundary", 24.85, 0.0, 5.0, 0.3, 50.0, 10.0, (0.12, 0.13, 0.14, 1.0), 0.65),
        static_box_model("west_boundary", -24.85, 0.0, 5.0, 0.3, 50.0, 10.0, (0.12, 0.13, 0.14, 1.0), 0.65),
        static_box_model("tunnel_spawn_pad", START_POSITION[0], START_POSITION[1], 0.025, 1.5, 1.5, 0.05, (0.10, 0.34, 0.78, 1.0)),
        static_box_model("tunnel_goal_pad", GOAL_POSITION[0], GOAL_POSITION[1], 0.025, 1.5, 1.5, 0.05, (0.88, 0.58, 0.08, 1.0)),
    ]
    models.extend(junction_models(segments))
    models.extend(tunnel_segment_model(segment) for segment in segments)
    joined_models = "\n".join(models)
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <world name='ego_curved_tunnel_50x50x10'>
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
      <ambient>0.20 0.20 0.19 1</ambient>
      <background>0.08 0.09 0.10 1</background>
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


def write_layout(path: Path, segments: Sequence[TunnelSegment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "name", "index", "start_x", "start_y", "end_x", "end_y",
                "center_x", "center_y", "length", "yaw", "width", "height",
            )
        )
        for segment in segments:
            writer.writerow(
                (
                    segment.name,
                    segment.index,
                    f"{segment.start_x:.6f}",
                    f"{segment.start_y:.6f}",
                    f"{segment.end_x:.6f}",
                    f"{segment.end_y:.6f}",
                    f"{segment.center_x:.6f}",
                    f"{segment.center_y:.6f}",
                    f"{segment.length:.6f}",
                    f"{segment.yaw:.6f}",
                    f"{segment.width:.6f}",
                    f"{segment.height:.6f}",
                )
            )


def read_layout(path: Path) -> List[TunnelSegment]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            TunnelSegment(
                name=row["name"],
                index=int(row["index"]),
                start_x=float(row["start_x"]),
                start_y=float(row["start_y"]),
                end_x=float(row["end_x"]),
                end_y=float(row["end_y"]),
                center_x=float(row["center_x"]),
                center_y=float(row["center_y"]),
                length=float(row["length"]),
                yaw=float(row["yaw"]),
                width=float(row["width"]),
                height=float(row["height"]),
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
        segments = read_layout(args.layout)
        world = args.world.read_text(encoding="utf-8")
    else:
        segments = generate_layout()
        world = world_text(segments)
        write_layout(args.layout, segments)
        args.world.parent.mkdir(parents=True, exist_ok=True)
        args.world.write_text(world, encoding="utf-8")

    report = validate_layout(segments, world)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
