#!/usr/bin/env python3
"""Generate a mobile-base (mecanum / omni) + lead-screw lift MJCF for the OpenArm bimanual robot.

Reads the position-control robot XML (openarm_bimanual_position_control.xml),
lifts ``openarm_body_link0`` out of the world body and re-mounts it on a lift
platform attached to a square omnidirectional mobile base. Wheels are plain
cylinder geoms with decorative rollers; the base is driven kinematically
(qpos integration) by mujoco_cartesian_control_fk.py --mobile / --mobile2,
so no physics contacts are required for locomotion.

Two wheel styles are supported:
  - ``mecanum``: wheel axles along chassis y (sideways); rollers at 45 deg to
    the hub axis (roller motion 45 deg to hub motion).
  - ``omni``: X-layout - wheels yawed +/-45 deg to sit on the chassis diagonal
    corners (adjacent wheel axles at 90 deg to each other); dual-row rollers
    (2 x 12 = 24 per wheel) with axes tangent to the rim, i.e. at 0 deg to the
    hub rolling direction, so rollers roll freely along the hub axle
    (90 deg to the hub motion).

Geometry (SI units):
  - chassis box: 0.52 x 0.52 x 0.11 m (square base)
  - wheels: cylinder diameter 0.152 m (r=0.076), width 0.04 m
  - lift: lead screw + 2 guide rods, slide joint stroke 0 ~ 0.80 m

Base joint names (used by the control program):
  base_x / base_y / base_yaw  - planar chassis joints (kinematic omni-drive)
  lift_z                      - lead-screw lift slide joint, range 0~0.8 m
  wheel_fl / wheel_fr / wheel_rl / wheel_rr - visual wheel hinge joints

Usage:
    conda activate yolo
    python generate_mobile_base_xml.py                    # mecanum version
    python generate_mobile_base_xml.py --style omni       # omni X-layout version
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# ------------------------------- parameters -------------------------------
ROBOT_BODY = "openarm_body_link0"

CHASSIS_HALF = (0.26, 0.26, 0.055)      # 520 x 520 x 110 mm box
AXLE_Z = 0.076                          # chassis origin height (= wheel radius)
WHEEL_RADIUS = 0.076                    # 152 mm diameter
WHEEL_HALF_WIDTH = 0.02                 # 40 mm wide wheels
WHEEL_X = 0.185                         # half wheelbase (front-back offset)
WHEEL_Y = 0.225                         # half track (left-right offset)

LIFT_STROKE = 0.8                       # 800 mm lead-screw stroke
LIFT_BASE_Z = 0.13                      # lift joint origin (on chassis top, base frame)
SCREW_X = -0.12                         # lead screw behind torso center
SCREW_RADIUS = 0.012
ROD_RADIUS = 0.008
ROD_Y = 0.055

ROLLER_COUNT = 8
ROLLER_RADIUS = 0.011
ROLLER_HALF_LEN = 0.028
ROLLER_RIM = 0.074                      # roller center circle radius

# omni dual-row rollers: 2 rows x 12 = 24 per wheel
OMNI_ROLLER_COUNT = 12                  # rollers per row
OMNI_ROLLER_ROWS = 2
OMNI_ROLLER_RADIUS = 0.008
OMNI_ROLLER_HALF_LEN = 0.012
OMNI_ROW_OFFSET = 0.011                 # row offset from wheel mid-plane
OMNI_ROLLER_RIM = WHEEL_RADIUS - OMNI_ROLLER_RADIUS  # flush with hub surface

# wheel layout: (name, x, y, mirrored / yaw-plus-45deg)
WHEELS = (
    ("wheel_fl", +WHEEL_X, +WHEEL_Y, False),
    ("wheel_fr", +WHEEL_X, -WHEEL_Y, True),
    ("wheel_rl", -WHEEL_X, +WHEEL_Y, True),
    ("wheel_rr", -WHEEL_X, -WHEEL_Y, False),
)

# omni X-layout: wheel body yaw about z (deg). Axle = local y after yaw, so
# FL/RR axles point along (1,1)/sqrt(2) and FR/RL along (-1,1)/sqrt(2);
# adjacent wheel axles are at 90 deg to each other.
OMNI_WHEEL_YAW = {
    "wheel_fl": -45.0,
    "wheel_fr": +45.0,
    "wheel_rl": +45.0,
    "wheel_rr": -45.0,
}

# --------------------------- mobile3 parameters ---------------------------
# torso stand: cut off the lower 2/3 (and its big base); keep the shoulders
TORSO_CUT_Z_MM = 515.0        # raw mesh coords are mm (mesh scale=0.001)
TORSO_CUT_Z = TORSO_CUT_Z_MM / 1000.0
TORSO_KEEP_MESHES = ("body_link0_3", "body_link0_5")  # span the cut plane / above
# chassis: cut the 4 corners with 115 mm legs, wheels on the corner diagonals
CHAMFER_LEG = 0.115
MOBILE3_CHASSIS_HALF = (0.26, 0.26, 0.055)   # same 520x520x110 mm box
# wheel centers on the corner diagonals, rim exactly 20 mm clear of the chamfer:
# chamfer line is x + y = 2*hx - leg; solve (2t - line)/sqrt(2) = R + 0.020
_CHAMFER_LINE = 2.0 * MOBILE3_CHASSIS_HALF[0] - CHAMFER_LEG  # 0.405
MOBILE3_WHEEL_XY = (_CHAMFER_LINE + np.sqrt(2.0) * (WHEEL_RADIUS + 0.020)) / 2.0  # ~0.2704
# lift housing: 900 mm tall shell, bottom face flush with the chassis top
HOUSING_X = -0.15             # housing center x (behind torso, rear at -0.22)
HOUSING_HALF = (0.07, 0.10, 0.45)
HOUSING_Z = MOBILE3_CHASSIS_HALF[2] + HOUSING_HALF[2]  # 0.055+0.45 -> 0.055..0.955
# waist joints between lift platform and the (cut) torso
WAIST_PITCH_Z = 0.08          # pitch axis height above lift body origin
WAIST_PITCH_RANGE = (-0.35, 1.5708)   # -20 deg .. +90 deg (lean forward)
WAIST_YAW_RANGE = (-0.5236, 0.5236)   # +/-30 deg horizontal rotation


def _quat_from_z(direction: np.ndarray) -> str:
    """Quaternion (w x y z string) rotating +z onto the given direction."""
    d = np.asarray(direction, dtype=float)
    d /= np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, d)
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return "1 0 0 0" if d[2] > 0 else "0 1 0 0"
    axis /= n
    angle = np.arccos(np.clip(z @ d, -1.0, 1.0))
    s = np.sin(angle / 2.0)
    q = np.array([np.cos(angle / 2.0), *(axis * s)])
    return " ".join(f"{v:.6f}" for v in q)


def _yaw_quat(yaw_deg: float) -> str:
    """Quaternion (w x y z) for a rotation about z."""
    half = np.radians(yaw_deg) / 2.0
    return f"{np.cos(half):.6f} 0 0 {np.sin(half):.6f}"


def _add_wheel(chassis: ET.Element, name: str, x: float, y: float, mirrored: bool) -> None:
    """Cylinder mecanum wheel body + decorative 45-degree rollers (visual only)."""
    wheel = ET.SubElement(chassis, "body", name=name, pos=f"{x} {y} 0")
    ET.SubElement(
        wheel, "inertial",
        pos="0 0 0", mass="0.6", diaginertia="0.000789 0.001444 0.000789",
    )
    ET.SubElement(wheel, "joint", name=name, type="hinge", axis="0 1 0")
    # hub cylinder (axis along y), collidable so a future dynamics mode still works
    ET.SubElement(
        wheel, "geom",
        name=f"{name}_hub", type="cylinder",
        size=f"{WHEEL_RADIUS} {WHEEL_HALF_WIDTH}",
        quat="0.7071068 0.7071068 0 0",
        rgba="0.10 0.10 0.10 1",
        condim="3", friction="1.2 0.01 0.01",
    )
    # decorative rollers at +/-45 deg around the rim (non-colliding)
    for i in range(ROLLER_COUNT):
        theta = i * 2.0 * np.pi / ROLLER_COUNT
        px, pz = ROLLER_RIM * np.cos(theta), ROLLER_RIM * np.sin(theta)
        tangent = np.array([-np.sin(theta), 0.0, np.cos(theta)])
        lateral = np.array([0.0, -1.0 if mirrored else 1.0, 0.0])
        axis = (tangent + lateral) / np.sqrt(2.0)
        ET.SubElement(
            wheel, "geom",
            name=f"{name}_roller{i}", type="capsule",
            pos=f"{px:.6f} 0 {pz:.6f}",
            quat=_quat_from_z(axis),
            size=f"{ROLLER_RADIUS} {ROLLER_HALF_LEN}",
            rgba="0.85 0.35 0.05 1",
            contype="0", conaffinity="0",
        )


def _add_omni_wheel(chassis: ET.Element, name: str, x: float, y: float) -> None:
    """Cylinder omni wheel body, yawed +/-45 deg (X-layout on chassis corners).

    Rollers have their axes parallel to the hub axle (local y), i.e. the roller
    rolling direction is at 90 deg to the hub rolling direction.
    """
    wheel = ET.SubElement(
        chassis, "body", name=name, pos=f"{x} {y} 0", quat=_yaw_quat(OMNI_WHEEL_YAW[name])
    )
    ET.SubElement(
        wheel, "inertial",
        pos="0 0 0", mass="0.6", diaginertia="0.000789 0.001444 0.000789",
    )
    ET.SubElement(wheel, "joint", name=name, type="hinge", axis="0 1 0")
    # hub cylinder (axis along local y)
    ET.SubElement(
        wheel, "geom",
        name=f"{name}_hub", type="cylinder",
        size=f"{WHEEL_RADIUS} {WHEEL_HALF_WIDTH}",
        quat="0.7071068 0.7071068 0 0",
        rgba="0.10 0.10 0.10 1",
        condim="3", friction="1.2 0.01 0.01",
    )
    # dual-row omni rollers: 2 rows x 12 = 24 barrels around the rim.
    # Roller axes are TANGENT to the rim circle, i.e. at 0 deg to the hub
    # rolling direction, so the rollers roll freely along the hub axle
    # (90 deg to the hub motion). Second row staggered by half a pitch so
    # ground contact stays continuous (classic dual-row omni wheel).
    for row in range(OMNI_ROLLER_ROWS):
        sgn = +1.0 if row == 0 else -1.0
        stagger = 0.0 if row == 0 else np.pi / OMNI_ROLLER_COUNT
        for i in range(OMNI_ROLLER_COUNT):
            theta = i * 2.0 * np.pi / OMNI_ROLLER_COUNT + stagger
            px, pz = OMNI_ROLLER_RIM * np.cos(theta), OMNI_ROLLER_RIM * np.sin(theta)
            tangent = np.array([-np.sin(theta), 0.0, np.cos(theta)])
            ET.SubElement(
                wheel, "geom",
                name=f"{name}_roller_r{row}_{i}", type="capsule",
                pos=f"{px:.6f} {sgn * OMNI_ROW_OFFSET} {pz:.6f}",
                quat=_quat_from_z(tangent),
                size=f"{OMNI_ROLLER_RADIUS} {OMNI_ROLLER_HALF_LEN}",
                rgba="0.10 0.60 0.30 1",
                contype="0", conaffinity="0",
            )


def build_mobile_base_xml(src_xml: Path, dst_xml: Path, wheel_style: str = "mecanum") -> None:
    if wheel_style not in ("mecanum", "omni"):
        raise ValueError(f"unknown wheel style: {wheel_style}")
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("worldbody not found")

    robot_body = worldbody.find(f"body[@name='{ROBOT_BODY}']")
    if robot_body is None:
        raise ValueError(f"body '{ROBOT_BODY}' not found")
    worldbody.remove(robot_body)

    # ------------------------------- chassis -------------------------------
    hx, hy, hz = CHASSIS_HALF
    chassis = ET.SubElement(worldbody, "body", name="base_chassis", pos=f"0 0 {AXLE_Z}")
    ET.SubElement(chassis, "joint", name="base_x", type="slide", axis="1 0 0")
    ET.SubElement(chassis, "joint", name="base_y", type="slide", axis="0 1 0")
    ET.SubElement(chassis, "joint", name="base_yaw", type="hinge", axis="0 0 1")
    ET.SubElement(
        chassis, "inertial",
        pos="0 0 0.075", mass="25.0", diaginertia="0.6 0.6 1.1",
    )
    ET.SubElement(
        chassis, "geom",
        name="base_chassis_box", type="box",
        pos=f"0 0 {hz + 0.02}",
        size=f"{hx} {hy} {hz}",
        rgba="0.16 0.16 0.16 1",
        condim="3", friction="1 0.01 0.01",
    )
    # lead screw + guide rods (static visuals on the chassis)
    screw_z = LIFT_BASE_Z + LIFT_STROKE / 2.0
    ET.SubElement(
        chassis, "geom",
        name="lift_lead_screw", type="cylinder",
        pos=f"{SCREW_X} 0 {screw_z}",
        size=f"{SCREW_RADIUS} {LIFT_STROKE / 2.0}",
        rgba="0.80 0.80 0.82 1",
        contype="0", conaffinity="0",
    )
    for sgn in (+1.0, -1.0):
        ET.SubElement(
            chassis, "geom",
            name=f"lift_guide_rod_{'p' if sgn > 0 else 'n'}", type="cylinder",
            pos=f"{SCREW_X} {sgn * ROD_Y} {screw_z}",
            size=f"{ROD_RADIUS} {LIFT_STROKE / 2.0}",
            rgba="0.55 0.55 0.58 1",
            contype="0", conaffinity="0",
        )

    for name, x, y, mirrored in WHEELS:
        if wheel_style == "omni":
            _add_omni_wheel(chassis, name, x, y)
        else:
            _add_wheel(chassis, name, x, y, mirrored)

    # -------------------------------- lift ---------------------------------
    lift = ET.SubElement(chassis, "body", name="base_lift", pos=f"0 0 {LIFT_BASE_Z}")
    ET.SubElement(
        lift, "joint",
        name="lift_z", type="slide", axis="0 0 1",
        range=f"0 {LIFT_STROKE}", limited="true",
    )
    ET.SubElement(
        lift, "inertial",
        pos="0 0 0.05", mass="4.0", diaginertia="0.02 0.02 0.01",
    )
    # carriage wrapping the screw / rods
    ET.SubElement(
        lift, "geom",
        name="lift_carriage", type="box",
        pos=f"{SCREW_X} 0 0.06", size="0.05 0.10 0.06",
        rgba="0.25 0.25 0.28 1",
        contype="0", conaffinity="0",
    )
    # platform carrying the torso
    ET.SubElement(
        lift, "geom",
        name="lift_platform", type="box",
        pos="0 0 0.015", size="0.13 0.13 0.015",
        rgba="0.20 0.20 0.22 1",
        contype="0", conaffinity="0",
    )

    # re-mount the dual-arm torso on the lift platform
    robot_body.set("pos", "0 0 0.03")
    lift.append(robot_body)

    ET.indent(tree, space="  ", level=0)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
    print(f"Wrote mobile-base XML: {dst_xml}")


# ---------------------------------------------------------------------------
# pure-numpy mesh utilities (OBJ I/O, plane slicing, prism building)
# ---------------------------------------------------------------------------

def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load an OBJ file -> (vertices (N,3), faces (M,3) zero-indexed)."""
    verts, faces = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                for i in range(1, len(idx) - 1):  # fan-triangulate polygons
                    faces.append([idx[0], idx[i], idx[i + 1]])
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64) - 1


def _save_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("# generated by generate_mobile_base_xml.py\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def _slice_keep_above_z(vertices: np.ndarray, faces: np.ndarray, z_cut: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep the part of a triangle mesh with z >= z_cut, capping the cut.

    Per-triangle plane clipping + centroid-fan cap of the (convex-ish) cut
    section. Pure numpy, sufficient for the torso stand meshes here.
    """
    dz = vertices[:, 2] - z_cut
    new_verts: list[np.ndarray] = []
    new_faces: list[list[int]] = []
    cache: dict[int, int] = {}
    cap_points: list[np.ndarray] = []

    def add_vertex(v: np.ndarray) -> int:
        new_verts.append(v)
        return len(new_verts) - 1

    def reuse(idx: int) -> int:
        if idx not in cache:
            cache[idx] = add_vertex(vertices[idx].copy())
        return cache[idx]

    for tri in faces:
        poly = [vertices[i] for i in tri]
        keep = [dz[i] >= 0.0 for i in tri]
        if all(keep):
            new_faces.append([reuse(i) for i in tri])
            continue
        if not any(keep):
            continue
        # clip polygon against the plane z = z_cut (Sutherland-Hodgman)
        clipped: list[np.ndarray] = []
        n = len(poly)
        for k in range(n):
            cur, prv = poly[k], poly[(k - 1) % n]
            cur_in, prv_in = cur[2] >= z_cut, prv[2] >= z_cut
            if cur_in != prv_in:
                t = (z_cut - prv[2]) / (cur[2] - prv[2])
                pt = prv + t * (cur - prv)
                pt[2] = z_cut
                clipped.append(pt)
                cap_points.append(pt.copy())
            if cur_in:
                clipped.append(cur)
        if len(clipped) >= 3:
            ids = [add_vertex(v) for v in clipped]
            for k in range(1, len(ids) - 1):
                new_faces.append([ids[0], ids[k], ids[k + 1]])

    if not new_faces:
        raise ValueError(f"nothing left above z={z_cut}")
    v_out = np.asarray(new_verts)
    f_out = np.asarray(new_faces, dtype=np.int64)

    # cap the cut section with a centroid fan (normal facing down)
    if cap_points:
        pts = np.unique(np.round(np.asarray(cap_points), 6), axis=0)
        if len(pts) >= 3:
            center = pts.mean(axis=0)
            ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
            pts = pts[np.argsort(ang)]
            cid = len(v_out)
            v_out = np.vstack([v_out, pts, center[None, :]])
            m = len(pts)
            for k in range(m):
                f_out = np.vstack([f_out, [[cid + m, cid + (k + 1) % m, cid + k]]])
    return v_out, f_out


def _chamfered_box_obj(path: Path, half: tuple[float, float, float], leg: float) -> None:
    """Write an OBJ of a box whose 4 vertical corners are chamfered (45 deg,
    triangle legs = `leg`), centered at the origin."""
    hx, hy, hz = half
    a = hx - leg  # chamfer start inset along each edge
    ring = [  # octagon, CCW seen from +z
        (hx, -a), (a, -hy), (-a, -hy), (-hx, -a),
        (-hx, a), (-a, hy), (a, hy), (hx, a),
    ]
    verts = np.array([[x, y, s * hz] for s in (-1.0, 1.0) for x, y in ring])
    faces: list[list[int]] = []
    for k in range(8):  # side quads -> 2 triangles each
        k2 = (k + 1) % 8
        faces.append([k, k2, 8 + k2])
        faces.append([k, 8 + k2, 8 + k])
    for k in range(1, 7):  # bottom (-z) and top (+z) fans
        faces.append([0, k + 1, k])
        faces.append([8, 8 + k, 8 + k + 1])
    _save_obj(path, verts, np.asarray(faces, dtype=np.int64))


def prepare_mobile3_assets(meshes_dir: Path) -> None:
    """Generate the cut torso meshes and the chamfered chassis mesh."""
    # torso: cut the stand at 2/3 of its height, keep the shoulder section
    for name in TORSO_KEEP_MESHES:
        src = meshes_dir / "visual" / "body" / f"{name}.obj"
        dst = meshes_dir / "visual" / "body" / f"{name}_cut.obj"
        verts, faces = _load_obj(src)
        v_out, f_out = _slice_keep_above_z(verts, faces, TORSO_CUT_Z_MM)
        _save_obj(dst, v_out, f_out)
        print(f"  cut {src.name}: z>={TORSO_CUT_Z_MM:.0f} mm -> {dst.name} "
              f"({len(faces)} -> {len(f_out)} tris)")

    # chamfered chassis (meters, no mesh scale)
    dst = meshes_dir / "visual" / "body" / "chassis_chamfered.obj"
    _chamfered_box_obj(dst, MOBILE3_CHASSIS_HALF, CHAMFER_LEG)
    print(f"  chamfered chassis (legs {CHAMFER_LEG * 1000:.0f} mm) -> {dst.name}")


def build_mobile3_xml(src_xml: Path, dst_xml: Path) -> None:
    """mobile3: chamfered chassis + corner omni wheels + 900 mm lift housing
    + 2 waist joints (pitch, yaw) + torso stand cut to its upper 1/3."""
    tree = ET.parse(src_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("worldbody not found")

    robot_body = worldbody.find(f"body[@name='{ROBOT_BODY}']")
    if robot_body is None:
        raise ValueError(f"body '{ROBOT_BODY}' not found")
    worldbody.remove(robot_body)

    # ---- cut torso: drop geoms fully below the cut, retarget kept meshes ----
    asset = root.find("asset")
    drop = {"body_link0_0.obj", "body_link0_1.obj", "body_link0_2.obj", "body_link0_4.obj"}
    remap = {f"{n}.obj": f"{n}_cut.obj" for n in TORSO_KEEP_MESHES}
    for name, new_name in remap.items():
        src_mesh = asset.find(f"mesh[@name='{name}']")
        if src_mesh is None:
            raise ValueError(f"mesh asset '{name}' not found")
        ET.SubElement(
            asset, "mesh",
            name=new_name,
            file=f"visual/body/{new_name}",
            scale=src_mesh.get("scale", "0.001 0.001 0.001"),
        )
    for geom in list(robot_body.iter("geom")):
        mesh_name = geom.get("mesh")
        if mesh_name in drop:
            robot_body.remove(geom)
        elif mesh_name in remap:
            geom.set("mesh", remap[mesh_name])
    # lighter inertial for the cut torso (kept section CoM ~0.64 m up)
    inertial = robot_body.find("inertial")
    if inertial is not None:
        inertial.set("pos", "0 0 0.64")
        inertial.set("mass", "4.0")
        inertial.set("diaginertia", "0.03 0.03 0.01")

    # ------------------------------- chassis -------------------------------
    chassis = ET.SubElement(worldbody, "body", name="base_chassis", pos=f"0 0 {AXLE_Z}")
    ET.SubElement(chassis, "joint", name="base_x", type="slide", axis="1 0 0")
    ET.SubElement(chassis, "joint", name="base_y", type="slide", axis="0 1 0")
    ET.SubElement(chassis, "joint", name="base_yaw", type="hinge", axis="0 0 1")
    ET.SubElement(chassis, "inertial", pos="0 0 0", mass="25.0", diaginertia="0.6 0.6 1.1")
    # chamfered chassis mesh, centered so the wheel axles sit at mid-height
    ET.SubElement(
        asset, "mesh",
        name="chassis_chamfered", file="visual/body/chassis_chamfered.obj",
    )
    ET.SubElement(
        chassis, "geom",
        name="base_chassis_box", type="mesh", mesh="chassis_chamfered",
        pos="0 0 0",
        rgba="0.16 0.16 0.16 1",
        condim="3", friction="1 0.01 0.01",
    )
    # lead screw + guide rods (inside the housing, bottom flush with chassis top)
    screw_lo = MOBILE3_CHASSIS_HALF[2]              # chassis top (base frame)
    screw_hi = LIFT_BASE_Z + LIFT_STROKE            # full stroke top (0.93)
    screw_z = (screw_lo + screw_hi) / 2.0
    screw_half = (screw_hi - screw_lo) / 2.0
    ET.SubElement(
        chassis, "geom",
        name="lift_lead_screw", type="cylinder",
        pos=f"{SCREW_X} 0 {screw_z}",
        size=f"{SCREW_RADIUS} {screw_half}",
        rgba="0.80 0.80 0.82 1",
        contype="0", conaffinity="0",
    )
    for sgn in (+1.0, -1.0):
        ET.SubElement(
            chassis, "geom",
            name=f"lift_guide_rod_{'p' if sgn > 0 else 'n'}", type="cylinder",
            pos=f"{SCREW_X} {sgn * ROD_Y} {screw_z}",
            size=f"{ROD_RADIUS} {screw_half}",
            rgba="0.55 0.55 0.58 1",
            contype="0", conaffinity="0",
        )
    # 900 mm lift housing shell (collidable: arms can hit the mast)
    ET.SubElement(
        chassis, "geom",
        name="lift_housing", type="box",
        pos=f"{HOUSING_X} 0 {HOUSING_Z}",
        size=f"{HOUSING_HALF[0]} {HOUSING_HALF[1]} {HOUSING_HALF[2]}",
        rgba="0.72 0.72 0.75 1",
        margin="0.01",
    )

    # omni wheels on the chamfered corner diagonals (clear of the chassis)
    for name, _, _, _ in WHEELS:
        fr = name.split("_")[1]  # fl / fr / rl / rr
        x = MOBILE3_WHEEL_XY if fr[0] == "f" else -MOBILE3_WHEEL_XY
        y = MOBILE3_WHEEL_XY if fr[1] == "l" else -MOBILE3_WHEEL_XY
        _add_omni_wheel(chassis, name, x, y)

    # -------------------------------- lift ---------------------------------
    lift = ET.SubElement(chassis, "body", name="base_lift", pos=f"0 0 {LIFT_BASE_Z}")
    ET.SubElement(
        lift, "joint",
        name="lift_z", type="slide", axis="0 0 1",
        range=f"0 {LIFT_STROKE}", limited="true",
    )
    ET.SubElement(lift, "inertial", pos="0 0 0.05", mass="4.0", diaginertia="0.02 0.02 0.01")
    # carriage wrapping the screw / rods (collidable, part of the lift shell)
    ET.SubElement(
        lift, "geom",
        name="lift_carriage", type="box",
        pos=f"{SCREW_X} 0 0.06", size="0.05 0.10 0.06",
        rgba="0.25 0.25 0.28 1",
        margin="0.01",
    )
    # platform carrying the waist assembly (halved in length/width: 130x130 mm)
    ET.SubElement(
        lift, "geom",
        name="lift_platform", type="box",
        pos="0 0 0.015", size="0.065 0.065 0.015",
        rgba="0.20 0.20 0.22 1",
        margin="0.01",
    )

    # --------------------- waist joints + cut torso -------------------------
    waist_pitch = ET.SubElement(lift, "body", name="waist_pitch", pos=f"0 0 {WAIST_PITCH_Z}")
    ET.SubElement(
        waist_pitch, "joint",
        name="waist_pitch", type="hinge", axis="0 1 0",
        range=f"{WAIST_PITCH_RANGE[0]} {WAIST_PITCH_RANGE[1]}", limited="true",
    )
    ET.SubElement(waist_pitch, "inertial", pos="0 0 0", mass="1.0", diaginertia="0.002 0.002 0.002")
    ET.SubElement(  # pitch hinge drum
        waist_pitch, "geom",
        name="waist_pitch_drum", type="cylinder",
        size="0.05 0.045", quat="0.7071068 0.7071068 0 0",
        rgba="0.30 0.30 0.33 1",
        contype="0", conaffinity="0",
    )
    waist_yaw = ET.SubElement(waist_pitch, "body", name="waist_yaw", pos="0 0 0")
    ET.SubElement(
        waist_yaw, "joint",
        name="waist_yaw", type="hinge", axis="0 0 1",
        range=f"{WAIST_YAW_RANGE[0]} {WAIST_YAW_RANGE[1]}", limited="true",
    )
    ET.SubElement(waist_yaw, "inertial", pos="0 0 0", mass="0.8", diaginertia="0.001 0.001 0.001")
    ET.SubElement(  # yaw post connecting drum to the torso bottom
        waist_yaw, "geom",
        name="waist_yaw_post", type="cylinder",
        pos="0 0 -0.025", size="0.04 0.025",
        rgba="0.42 0.42 0.45 1",
        contype="0", conaffinity="0",
    )

    # re-mount the cut torso so the mesh bottom sits at the waist joint
    robot_body.set("pos", f"0 0 {-TORSO_CUT_Z}")
    waist_yaw.append(robot_body)

    # contact excludes for parts in permanent (by-design) contact
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for b1, b2 in (
        ("base_chassis", "base_lift"),      # carriage slides inside the housing
        ("base_lift", "waist_pitch"),       # pitch drum rests on the platform
        ("base_lift", "waist_yaw"),         # yaw post meets the platform top
        ("waist_pitch", "openarm_body_link0"),  # drum merges into torso bottom
    ):
        ET.SubElement(contact, "exclude", body1=b1, body2=b2)

    ET.indent(tree, space="  ", level=0)
    tree.write(dst_xml, encoding="utf-8", xml_declaration=True)
    print(f"Wrote mobile3 XML: {dst_xml}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="src/openarm_description/mujoco/openarm_bimanual_position_control.xml",
        help="Position-control robot MJCF",
    )
    parser.add_argument(
        "--dst",
        default=None,
        help="Output mobile-base MJCF (default depends on --style)",
    )
    parser.add_argument(
        "--style",
        choices=("mecanum", "omni", "mobile3"),
        default="mecanum",
        help="Wheel/base type: mecanum (45-deg rollers), omni (90-deg dual-row rollers, "
             "X-layout), or mobile3 (omni + chamfered chassis + housing + waist + cut torso)",
    )
    args = parser.parse_args()
    dst = args.dst
    if args.style == "mobile3":
        if dst is None:
            dst = "src/openarm_description/mujoco/openarm_bimanual_mobile3.xml"
        meshes_dir = Path(args.src).parent / "meshes"
        prepare_mobile3_assets(meshes_dir)
        build_mobile3_xml(Path(args.src), Path(dst))
        return
    if dst is None:
        suffix = "mobile2" if args.style == "omni" else "mobile"
        dst = f"src/openarm_description/mujoco/openarm_bimanual_{suffix}.xml"
    build_mobile_base_xml(Path(args.src), Path(dst), wheel_style=args.style)


if __name__ == "__main__":
    main()
