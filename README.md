<div align="center">

<img src="./assets/logo.png" alt="Kinara Logo" width="130" />

# KINARA
### Markerless Motion Capture & Biomechanical Inverse Kinematics Engine

[![Language](https://img.shields.io/badge/Language-Rust_%7C_Python-dea584?style=flat-square&logo=rust&logoColor=white)](https://www.rust-lang.org)
[![GPU Acceleration](https://img.shields.io/badge/Inference-CUDA_%7C_DirectML_%7C_CPU-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-zone)
[![Blender Ready](https://img.shields.io/badge/DCC_Target-Blender_4.x+-f5792a?style=flat-square&logo=blender&logoColor=white)](https://www.blender.org)
[![Rig Architecture](https://img.shields.io/badge/Skeleton-78--Joint_Armature-6366f1?style=flat-square)](./docs/ik_and_skeleton.md)
[![License](https://img.shields.io/badge/License-MIT-3b82f6?style=flat-square)](./LICENSE.md)

<p align="center">
  <b>Kinara</b> is a markerless motion capture engine designed to extract 3D skeletal and finger-articulated animation from standard video feeds.<br/>
  It enforces biomechanical constraints, maintains distance-invariant character scaling, and outputs clean animation data ready for Blender and game engines.
</p>

<p align="center">
  <a href="#-system-architecture">Architecture</a> •
  <a href="#-biomechanical-ik-engine">IK Constraints</a> •
  <a href="#-feature-matrix">Features</a> •
  <a href="#-blender-pipeline">Blender Workflow</a> •
  <a href="#-multi-camera--multi-person">Multi-Cam & Tracking</a> •
  <a href="#-development-status">Status</a> •
  <a href="./docs/">Documentation</a>
</p>

</div>

---

> [!NOTE]
> **Active Engineering Pivot**: Kinara is currently transitioning its proven Python pipeline architecture into a high-performance native Rust core. Track module-by-module implementation status in [progress.md](./progress.md).

---

## ✦ System Architecture

Kinara processes video frames through a staged pipeline designed to separate high-level body localization from fine-grained finger articulation:

```mermaid
flowchart LR
    A["Input Video / Feed<br/>(File or Webcam)"] --> B["Dual-Scale Ingest"]
    B -->|Proxy Resolution| C["Body Pose Estimator<br/>(17 COCO Keypoints)"]
    B -->|Native Pixel Crop| D["Forearm-Guided Hand ROI<br/>(Wrist Crop)"]
    C --> D
    D --> E["Hand Pose Inference<br/>(21 Hand Keypoints)"]
    C & E --> F["Biomechanical IK Solver<br/>(Tendon Lock & Planar Hinges)"]
    F --> G["Temporal Stabilization<br/>(Filtering & Planted Foot Lock)"]
    G --> H["Scale & Ground Normalizer<br/>(Torso Metric & Z=0 Align)"]
    H --> I["Export Targets<br/>(Blender JSON, FBX, Live UDP)"]

    classDef stage fill:#0f172a,stroke:#3b82f6,stroke-width:1.5px,color:#f8fafc;
    classDef highlight fill:#1e1b4b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;
    classDef export fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    class A,B,C,D,E stage;
    class F,G,H highlight;
    class I export;
```

---

## ◈ Biomechanical IK Engine

Standard neural pose estimators predict unconstrained Cartesian points $(x, y, z)$. Without anatomical priors, raw landmark outputs suffer from backwards joint hyperextension, bone stretching, and fingers clipping through each other. Kinara enforces physical constraints at the kinematic solver stage:

<table>
<tr>
<td width="50%" valign="top">

### 1. DIP-PIP Tendon Coupling Ratio
In the human hand, the *Distal Interphalangeal* (DIP) and *Proximal Interphalangeal* (PIP) joints share the *Flexor Digitorum Profundus* tendon. Humans cannot voluntarily articulate the fingertip without flexing the middle knuckle.
$$\theta_{\text{DIP}} \approx 0.67 \times \theta_{\text{PIP}}$$
* **Impact**: Eliminates redundant degrees of freedom, removes fingertip jitter, and prevents backwards-bending tips.

</td>
<td width="50%" valign="top">

### 2. Dynamic Fist Abduction Clamping
When fingers are extended, they can freely splay laterally (abduction). As the fingers curl inward to form a fist, ligaments tighten and lateral spread vanishes:
$$\text{Max Abduction}(\theta_{\text{MCP}}) = \text{Base Spread} \times \left(1.0 - \frac{\theta_{\text{MCP}}}{90^\circ}\right)$$
* **Impact**: Eliminates adjacent finger meshes clipping through each other when closing into a fist.

</td>
</tr>
<tr>
<td width="50%" valign="top">

### 3. Strict 1-DOF Planar Knuckle Hinges
The PIP and DIP joints are physiological revolute hinges:
* Lateral deviation: **$0^\circ$ locked** (no sideways bend)
* Axial longitudinal twist: **$0^\circ$ locked**
* Hyperextension: **Hard limit at $0^\circ$** (knuckles never bend in reverse).

</td>
<td width="50%" valign="top">

### 4. Rigid Bone Length Invariance
Smoothing raw Cartesian coordinates causes limbs to artificially shrink and stretch during perspective changes. Kinara enforces invariant segment lengths and smooths **joint orientation quaternions**:
$$\mathbf{q}_{\text{stabilized}} = \text{Slerp}(\mathbf{q}_{\text{prev}}, \mathbf{q}_{\text{new}}, \alpha)$$

</td>
</tr>
</table>

> [!TIP]
> For the complete mathematical formulation and proof of joint limits, see [docs/ik_and_skeleton.md](./docs/ik_and_skeleton.md).

---

## ❖ Feature Matrix

| Capability | Unconstrained AI Trackers | Kinara Engine | Technical Advantage |
| :--- | :---: | :---: | :--- |
| **Finger Tracking** | 2D loose points | **Anatomical 21-Point IK** | Tendon coupling and knuckle planar hinge limits |
| **Distant Footage** | Fails (Lost in downsampling) | **Wrist-Oriented Crop Extraction** | Evaluates hands on native-resolution crops |
| **Joint Rotations** | None (Positions only) | **Quaternion Rig Transforms** | Eliminates axial twist and rubber-bone artifacts |
| **Scale Stability** | Shrinks/expands with distance | **Torso Metric Normalization** | Bidirectional zero-lag smoothing of torso scale |
| **Ground Interaction** | Floating / Foot skate | **Planted Foot Velocity Lock** | Automatic ground plane alignment ($Z = 0.0$) |
| **DCC Export** | Raw CSV or loose coordinates | **Blender Motion JSON + FBX 7.4** | Direct import into Blender 4.x+ character armatures |

---

## ⬡ Blender 4.x+ Pipeline

Kinara structures animation data into standard humanoid hierarchies for Blender, Maya, and Unreal Engine:

```
Kinara Motion Output
 ├── motion.json (kinara-motion-json-v1: 78-joint canonical hierarchy + rest offsets)
 └── motion.fbx  (ASCII FBX 7.4: Hierarchical LimbNodes + baked translation curves)
```

### Option A: Direct FBX Import
1. In Blender, navigate to `File` ▸ `Import` ▸ `FBX (.fbx)`.
2. Select your exported `.fbx` file.
3. The 78-joint armature imports with baked keyframe curves and feet grounded to the floor plane ($Z = 0.0$).

### Option B: Headless JSON Importer
Execute the automated importer script to build armatures, attach constraint empties, and bake pose actions:

```bash
blender --background --python tools/import_kinara_motion.py -- --input outputs/motion.json --fps 60
```

---

## ◈ Multi-Camera & Multi-Person

### Single-Camera Multi-Person Tracking
* **Track Association**: Tracks multiple individuals simultaneously using bounding box overlap, motion prediction, and torso clothing color profiles to minimize identity swaps during actor crossovers.
* **Hand Ownership Disambiguation**: Enforces wrist proximity and boundary penalties to prevent hands from being incorrectly reassigned between adjacent actors.

### Multi-Camera 3D Triangulation
* **ChArUco Calibration**: Calibrates multi-camera setups using ChArUco checkerboards to compute camera intrinsic matrices ($\mathbf{K}$) and extrinsic transformations ($[\mathbf{R} \mid \mathbf{t}]$).
* **Direct Linear Transformation (DLT)**: Reconstructs true metric 3D joint positions from synchronized 2D detections across 2 to 4 camera angles.

---

## ⬢ Technical Architecture & Directory Layout

```
Kinara/
 ├── docs/
 │    ├── programflow.mmd      # High-fidelity Mermaid architecture diagram
 │    ├── functions.md        # Technical specification of all engine functions
 │    └── ik_and_skeleton.md  # Biomechanical IK theory & Blender export spec
 ├── assets/                  # Branding and visual assets
 ├── models/                  # ONNX / TensorRT model weights
 ├── outputs/                 # Exported motion JSON, FBX, and MP4 previews
 ├── src/
 │    ├── export/             # JSON, FBX, and Blender armature exporters
 │    ├── inference/          # Body pose and hand landmark inference
 │    ├── solver/             # Biomechanical IK constraints and temporal filtering
 │    └── video/              # Video ingestion and hardware-accelerated decoders
 ├── progress.md              # Systematic function-by-function Rust port ledger
 └── README.md                # Project documentation
```

---

## ⚙ System Requirements

### Hardware
* **Camera**: Standard webcam, action camera, or pre-recorded video (`.mp4`, `.mov`, `.mkv`).
* **GPU**: NVIDIA GPU with CUDA support recommended for accelerated inference; DirectML and CPU execution supported as fallbacks.
* **Multi-Camera (Optional)**: 2 to 4 synchronized camera views for calibrated 3D triangulation.

### Supported Software & DCC Tools
* **Blender**: 4.0+ (Armature import, action baking)
* **Unreal Engine**: 5.x (Live UDP / OSC stream receiver)
* **Rust Toolchain**: 1.78+ (`cargo`, `rustc`)

---

## 📚 Technical Documentation

* **[Progress & Implementation Ledger](./progress.md)** — Detailed checklist of every function being ported to Rust.
* **[Program Flow Architecture Diagram](./docs/programflow.mmd)** — Mermaid flowchart illustrating multi-scale ingest and kinematic solving.
* **[IK Constraints & Blender Export Specification](./docs/ik_and_skeleton.md)** — Mathematical proofs and Blender integration methods.
* **[Engine Function Reference](./docs/functions.md)** — Architectural index of all module functions, signatures, and behaviors.

---

## 📄 License & Attribution

Kinara is licensed under the **[MIT License](./LICENSE.md)**.
Built for open biomechanics, markerless motion tracking, and real-time graphics pipelines.
