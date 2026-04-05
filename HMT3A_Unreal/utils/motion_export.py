from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from process.motion_repair import repair_motion_frames
from utils.math_utils import quaternion_identity, quaternion_inverse, quaternion_multiply, quaternion_to_euler_degrees


COMMON_BLENDER_PATHS = (
    Path("C:/Program Files/Blender Foundation/Blender 5.0/blender.exe"),
    Path("C:/Program Files/Blender Foundation/Blender 4.4/blender.exe"),
    Path("C:/Program Files/Blender Foundation/Blender 4.3/blender.exe"),
)


class MotionExporter:
    def __init__(self, output_dir, enabled: bool, source_fps: float, base_name: str = "fused_motion"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.source_fps = float(source_fps)
        self.base_name = base_name
        self.frames = []

    def record_frame(self, frame_index: int, timestamp_ms: int, persons: list[dict]) -> None:
        if not self.enabled:
            return

        from network.packet_builder import build_person_payload

        self.frames.append(
            {
                "frame": int(frame_index),
                "timestamp_ms": int(timestamp_ms),
                "persons": [build_person_payload(person) for person in persons],
            }
        )

    def close(self, export_json: bool = True, export_bvh: bool = True, export_fbx: bool = True) -> list[Path]:
        if not self.enabled or not self.frames:
            return []

        repaired_frames = repair_motion_frames(self.frames, source_fps=self.source_fps)
        outputs = []

        if export_json:
            outputs.append(self._write_json(self.frames, suffix="_raw", export_type="fused_motion_raw"))
            outputs.append(self._write_json(repaired_frames, suffix="", export_type="fused_motion_repaired"))

        bvh_paths = []
        if export_bvh or export_fbx:
            bvh_paths = self._write_bvh_files(repaired_frames)
            if export_bvh:
                outputs.extend(bvh_paths)

        if export_fbx and bvh_paths:
            outputs.extend(self._write_fbx_files(bvh_paths))

        return outputs

    def _next_output_path(self, stem: str, suffix: str) -> Path:
        candidate = self.output_dir / f"{stem}{suffix}"
        if not candidate.exists():
            return candidate

        index = 1
        while True:
            candidate = self.output_dir / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def _write_json(self, frames: list[dict], suffix: str, export_type: str) -> Path:
        output_path = self._next_output_path(f"{self.base_name}{suffix}", ".json")
        payload = {
            "metadata": {
                "source_fps": round(self.source_fps, 6),
                "frame_count": len(frames),
                "export_type": export_type,
            },
            "frames": frames,
        }
        output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"Motion JSON saved -> {output_path}")
        return output_path

    def _write_bvh_files(self, frames: list[dict]) -> list[Path]:
        person_ids = sorted({person["id"] for frame in frames for person in frame.get("persons", [])})
        outputs = []
        for person_id in person_ids:
            output_path = self._write_bvh_for_person(frames, person_id)
            if output_path is not None:
                outputs.append(output_path)
        return outputs

    def _write_fbx_files(self, bvh_paths: list[Path]) -> list[Path]:
        blender_executable = self._find_blender_executable()
        if blender_executable is None:
            print("FBX export skipped -> Blender executable not found.")
            return []

        outputs = []
        for bvh_path in bvh_paths:
            fbx_path = self._convert_bvh_to_fbx(blender_executable, bvh_path)
            if fbx_path is not None:
                outputs.append(fbx_path)
        return outputs

    def _find_blender_executable(self) -> Path | None:
        blender_path = shutil.which("blender")
        if blender_path:
            return Path(blender_path)

        for candidate in COMMON_BLENDER_PATHS:
            if candidate.exists():
                return candidate
        return None

    def _convert_bvh_to_fbx(self, blender_executable: Path, bvh_path: Path) -> Path | None:
        fbx_path = self._next_output_path(bvh_path.stem, ".fbx")
        script_content = """
import bpy
import sys
from pathlib import Path

marker = sys.argv.index("--")
bvh_path = Path(sys.argv[marker + 1])
fbx_path = Path(sys.argv[marker + 2])

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_anim.bvh(filepath=str(bvh_path))
imported_objects = list(bpy.context.selected_objects)
if not imported_objects:
    raise RuntimeError(f"No objects imported from {bvh_path}")

for scene_object in bpy.context.scene.objects:
    scene_object.select_set(False)
for imported_object in imported_objects:
    imported_object.select_set(True)

bpy.context.view_layer.objects.active = imported_objects[0]
bpy.ops.export_scene.fbx(
    filepath=str(fbx_path),
    use_selection=True,
    add_leaf_bones=False,
    bake_anim=True,
    object_types={'ARMATURE'},
)
"""

        temp_script_path = None
        try:
            with tempfile.NamedTemporaryFile("w", suffix="_hmt3a_fbx.py", delete=False, encoding="utf-8") as handle:
                handle.write(script_content)
                temp_script_path = Path(handle.name)

            subprocess.run(
                [
                    str(blender_executable),
                    "--background",
                    "--factory-startup",
                    "--python",
                    str(temp_script_path),
                    "--",
                    str(bvh_path),
                    str(fbx_path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"FBX export failed for {bvh_path.name}: {exc.stderr.strip() or exc.stdout.strip()}")
            return None
        finally:
            if temp_script_path is not None and temp_script_path.exists():
                temp_script_path.unlink(missing_ok=True)

        print(f"Motion FBX saved -> {fbx_path}")
        return fbx_path

    def _write_bvh_for_person(self, frames: list[dict], person_id: int) -> Path | None:
        person_frames = []
        last_frame = None
        for frame in frames:
            current = next((person for person in frame.get("persons", []) if person["id"] == person_id), None)
            if current is None:
                current = last_frame
            if current is None:
                continue
            person_frames.append(
                {
                    "frame": frame["frame"],
                    "timestamp_ms": frame["timestamp_ms"],
                    "person": current,
                }
            )
            last_frame = current

        if not person_frames:
            return None

        rest_person = person_frames[0]["person"]
        rest_skeleton = rest_person["skeleton"]
        joints = rest_skeleton.get("joints", {})
        root_name = rest_skeleton.get("root", "root")
        joint_order = []
        hierarchy_lines = ["HIERARCHY"]
        self._append_joint_hierarchy(joints, root_name, hierarchy_lines, joint_order, indent_level=0)

        motion_lines = [
            "MOTION",
            f"Frames: {len(person_frames)}",
            f"Frame Time: {1.0 / max(self.source_fps, 1e-6):.8f}",
        ]
        for frame in person_frames:
            motion_lines.append(self._build_bvh_frame_line(frame["person"], rest_person, joint_order, root_name))

        output_path = self._next_output_path(f"{self.base_name}_person_{person_id}", ".bvh")
        output_path.write_text("\n".join(hierarchy_lines + motion_lines) + "\n", encoding="ascii")
        print(f"Motion BVH saved -> {output_path}")
        return output_path

    def _append_joint_hierarchy(self, joints: dict, joint_name: str, lines: list[str], joint_order: list[str], indent_level: int) -> None:
        joint = joints.get(joint_name)
        if joint is None:
            return

        indent = "  " * indent_level
        joint_label = "ROOT" if indent_level == 0 else "JOINT"
        lines.append(f"{indent}{joint_label} {joint_name}")
        lines.append(f"{indent}{{")

        offset = joint.get("local_offset") or [0.0, 0.0, 0.0]
        if indent_level == 0:
            offset = [0.0, 0.0, 0.0]
        lines.append(f"{indent}  OFFSET {offset[0]:.6f} {offset[1]:.6f} {offset[2]:.6f}")

        if indent_level == 0:
            lines.append(f"{indent}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation")
        else:
            lines.append(f"{indent}  CHANNELS 3 Xrotation Yrotation Zrotation")

        joint_order.append(joint_name)
        children = list(joint.get("children", []))
        if not children:
            end_offset = self._estimate_end_site_offset(joint)
            lines.append(f"{indent}  End Site")
            lines.append(f"{indent}  {{")
            lines.append(f"{indent}    OFFSET {end_offset[0]:.6f} {end_offset[1]:.6f} {end_offset[2]:.6f}")
            lines.append(f"{indent}  }}")
        else:
            for child_name in children:
                self._append_joint_hierarchy(joints, child_name, lines, joint_order, indent_level + 1)

        lines.append(f"{indent}}}")

    def _estimate_end_site_offset(self, joint: dict) -> list[float]:
        offset = joint.get("local_offset")
        if offset is not None:
            magnitude = sum(component * component for component in offset) ** 0.5
            if magnitude > 1e-6:
                scale = 0.25
                return [round(float(component) * scale, 6) for component in offset]
        return [0.0, 0.05, 0.0]

    def _build_bvh_frame_line(self, person: dict, rest_person: dict, joint_order: list[str], root_name: str) -> str:
        current_joints = person["skeleton"]["joints"]
        current_rotations = person["rotations"]
        rest_joints = rest_person["skeleton"]["joints"]
        rest_rotations = rest_person["rotations"]
        channels = []

        current_root_position = current_joints.get(root_name, {}).get("position")
        rest_root_position = rest_joints.get(root_name, {}).get("position")
        channels.extend(self._serialize_root_translation_delta(current_root_position, rest_root_position))

        for joint_name in joint_order:
            current_rotation = current_rotations.get(joint_name)
            rest_rotation = rest_rotations.get(joint_name)
            euler = self._compute_rotation_delta_euler(current_rotation, rest_rotation)
            channels.extend(f"{float(value):.6f}" for value in euler)

        return " ".join(channels)

    def _serialize_root_translation_delta(self, current_position, rest_position) -> list[str]:
        if current_position is None or rest_position is None:
            return ["0.000000", "0.000000", "0.000000"]

        return [
            f"{float(current_position['x']) - float(rest_position['x']):.6f}",
            f"{float(current_position['y']) - float(rest_position['y']):.6f}",
            f"{float(current_position['z']) - float(rest_position['z']):.6f}",
        ]

    def _compute_rotation_delta_euler(self, current_rotation, rest_rotation) -> list[float]:
        current_quaternion = self._local_quaternion(current_rotation)
        rest_quaternion = self._local_quaternion(rest_rotation)
        delta_quaternion = quaternion_multiply(quaternion_inverse(rest_quaternion), current_quaternion)
        return quaternion_to_euler_degrees(delta_quaternion)

    def _local_quaternion(self, rotation) -> tuple[float, float, float, float]:
        if rotation is None:
            return quaternion_identity()

        raw_quaternion = rotation.get("local_quaternion")
        if raw_quaternion is None or len(raw_quaternion) != 4:
            return quaternion_identity()

        return tuple(float(component) for component in raw_quaternion)
