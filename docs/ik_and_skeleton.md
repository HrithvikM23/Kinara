# Biomechanical IK Constraints & Blender Skeleton Export Specification

---

## Executive Overview

Markerless motion capture from mono or multi-view RGB video fundamentally solves an **ill-posed inverse problem**: recovering full 3D skeletal poses and high-degree-of-freedom finger articulations from noisy, occluded 2D image projections.

Without strict biomechanical domain knowledge, neural landmark estimators produce anatomically impossible artifacts:
* Fingertips fluttering independently or bending backward like rubber.
* Adjacent fingers passing through each other when clenching a fist.
* Limbs stretching and shrinking as the actor moves toward or away from the camera.
* Joints twisting uncontrollably around their longitudinal axis (the "rubber-bone" roll problem).

This document details the **Inverse Kinematics (IK) constraints** engineered into the Kinara pipeline to guarantee anatomical validity, alongside the **pipeline methods used to output production-ready skeleton files for Blender 4.x+**.

---

## Part 1: What Are Inverse Kinematics (IK) Constraints?

### 1. Forward vs. Inverse Kinematics

In computer graphics and robotics, skeletal motion is governed by two dual principles:

1. **Forward Kinematics (FK)**:
   Given known joint angles and segment lengths, FK calculates the world-space Cartesian coordinates of the end-effectors (e.g., fingertips, wrists, feet):
   $$\mathbf{p}_{\text{end}} = \prod_{i=1}^{N} \mathbf{T}_i(\theta_i) \cdot \mathbf{p}_{\text{local}}$$

2. **Inverse Kinematics (IK)**:
   Given observed target positions in space (such as 2D/3D landmarks detected by a neural network), IK computes the internal joint rotations $(\theta_1, \theta_2, \dots, \theta_N)$ required for the kinematic chain to reach those positions:
   $$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} \|\mathbf{f}(\boldsymbol{\theta}) - \mathbf{p}_{\text{observed}}\|^2 \quad \text{subject to } \boldsymbol{\theta}_{\text{min}} \le \boldsymbol{\theta} \le \boldsymbol{\theta}_{\text{max}}$$

```
┌────────────────────────────────────────────────────────┐
│  Raw Video Frame -> Neural Estimator (Noisy Points)    │
└──────────────────────────┬─────────────────────────────┘
                           │ 2D / 3D Coordinate Observations
                           ▼
┌────────────────────────────────────────────────────────┐
│           Biomechanical IK Constraint Engine          │
│  - Locks bone lengths (Rigid Segment Invariance)       │
│  - Couples dependent joints (Tendon Coupling)          │
│  - Clamps out-of-plane splay (Fist Abduction Clamp)    │
│  - Restricts joint DOF (1-DOF Planar Hinges)           │
└──────────────────────────┬─────────────────────────────┘
                           │ Validated Joint Orientations (Quaternions)
                           ▼
┌────────────────────────────────────────────────────────┐
│        Anatomically Rigid 3D Character Armature        │
└────────────────────────────────────────────────────────┘
```

### 2. Why Video Pose Requires Hard Biomechanical Constraints

Standard neural pose estimators (such as MediaPipe, YOLO-Pose, or RTMPose) output independent spatial points $(x, y, z)$. These points lack:
* **Rotational Constraints**: A coordinate point carries zero information about bone roll, pitch, or yaw. When mapped directly to a 3D character armature, bones twist randomly along their axes.
* **Rigid Segment Preservation**: Points fluctuate frame-to-frame, causing limbs to stretch or compress artificially.
* **Physical Plausibility**: Estimators have no inherent understanding that bones cannot collide or bend backwards.

**IK Constraints** act as hard biological guardrails. Rather than accepting raw coordinates as ground truth, the solver treats neural detections as noisy pull targets, projecting them into a constrained biomechanical manifold.

---

### 3. Hand & Finger IK Constraints

The human hand possesses over 20 degrees of freedom within a compact, easily occluded space. Kinara enforces five physiological constraints:

#### Constraint 3.1: DIP-PIP Tendon Coupling Ratio
* **Biological Principle**: In human hand anatomy, the *Distal Interphalangeal* (DIP) joint and the *Proximal Interphalangeal* (PIP) joint share the *Flexor Digitorum Profundus* tendon. Humans cannot voluntarily bend the distal fingertip without simultaneously flexing the middle knuckle.
* **Mathematical Law**:
  $$\theta_{\text{DIP}} \approx \kappa \cdot \theta_{\text{PIP}} \quad \text{where } \kappa \approx 0.67 \quad (\text{physiological bounds: } [0.60, 0.77])$$
* **Impact**: Eliminates 4 redundant degrees of freedom across the four fingers, stops fingertip trembling, and guarantees tips never bend backwards.

#### Constraint 3.2: Dynamic Fist Abduction Clamping
* **Biological Principle**: When the fingers are fully extended, they can splay laterally (abduction/adduction). However, as the fingers curl inward at the metacarpophalangeal (MCP) knuckle to form a fist, the transverse metacarpal ligaments tighten and the proximal phalanges contact one another, driving lateral spread to zero.
* **Mathematical Law**:
  $$\text{Abduction}_{\text{max}}(\theta_{\text{MCP}}) = \text{Abduction}_{\text{base}} \times \max\left(0.0, 1.0 - \frac{\theta_{\text{MCP}}}{90^\circ}\right)$$
* **Impact**: Fully eliminates inter-finger mesh clipping and intersection when clenching a fist.

#### Constraint 3.3: Strict 1-DOF Planar Hinge Knuckles
* **Biological Principle**: The PIP (middle knuckle) and DIP (tip knuckle) joints are pure anatomical revolute hinges. They possess:
  * Lateral deviation (Abduction/Adduction): $0^\circ$ (strictly locked).
  * Longitudinal twist (Axial Rotation): $0^\circ$ (strictly locked).
  * Flexion range: $0^\circ \le \theta_{\text{PIP}} \le 105^\circ$, $0^\circ \le \theta_{\text{DIP}} \le 70^\circ$.
  * Hyperextension: Hard stop at $0^\circ$ (knuckles never bend in reverse).
* **Impact**: Prevents broken, rubbery finger artifacts common in markerless trackers.

#### Constraint 3.4: Phalange Segment Length Invariance
* **Mathematical Law**:
  $$\forall (i, j) \in \text{Bones}, \quad \|\mathbf{p}_j - \mathbf{p}_i\| = L_{ij}^{\text{target}}$$
  If perspective foreshortening in a 2D projection suggests a bone has contracted, the solver reprojects the child joint along the unit vector from the parent, maintaining the invariant segment length $L_{ij}^{\text{target}}$ established from calibration.

#### Constraint 3.5: Finger Collision Lanes & Radial Envelope
* **Mathematical Law**: Hand landmarks are projected onto an orthonormal basis defined by the palm plane $(\hat{\mathbf{u}}_{\text{forward}}, \hat{\mathbf{v}}_{\text{lateral}}, \hat{\mathbf{w}}_{\text{normal}})$. Each finger is assigned a non-overlapping lateral interval $[v_k^{\text{min}}, v_k^{\text{max}}]$ such that:
  $$v_{\text{thumb}} < v_{\text{index}} < v_{\text{middle}} < v_{\text{ring}} < v_{\text{pinky}}$$
  Fingers cannot cross lateral bounds, preventing finger swapping during complex hand gestures.

---

### 4. Full-Body IK Constraints

| Joint Group | Degrees of Freedom | Constraint Model | Physiological Range |
| :--- | :--- | :--- | :--- |
| **Knees** | 1 DOF (Pure Hinge) | Planar hinge, zero hyperextension | $0^\circ \le \theta_{\text{flex}} \le 150^\circ$, Lateral $= 0^\circ$ |
| **Elbows** | 1 DOF (Pure Hinge) | Planar hinge, zero hyperextension | $0^\circ \le \theta_{\text{flex}} \le 155^\circ$, Lateral $= 0^\circ$ |
| **Hips** | 3 DOF (Ball & Socket) | Swing-Twist cone limit | Flexion: $-30^\circ \text{ to } 120^\circ$, Abduction: $-45^\circ \text{ to } 45^\circ$ |
| **Shoulders**| 3 DOF (Ball & Socket) | Elliptical cone constraint | Clamped to prevent arm penetrating torso |
| **Spine/Chest**| Distributed Curvature | Spline continuity limit | Maximum relative bend $< 15^\circ$ per vertebral segment |
| **Feet/Ankles**| Contact State Constraint | Foot-lock velocity thresholding | When $|v_z| < \epsilon$ and $z \approx 0$, lock $(x, y, z)$ coordinates |

---

## Part 2: Methods to Output a Skeleton File Suitable for Blender

To interface seamlessly with Blender 4.x+ without requiring complex third-party addons, Kinara implements two complementary export pathways:

```
┌─────────────────────────────────────────────────────────────┐
│                 Kinara Constrained Motion                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │  Kinara Motion JSON  │        │   ASCII FBX 7.4.0    │
    │ (kinara-motion-json) │        │ (Universal 3D Asset) │
    └──────────┬───────────┘        └──────────┬───────────┘
               │                               │
               ▼                               ▼
    ┌──────────────────────┐        ┌──────────────────────┐
    │ Blender Python Script│        │ Direct Native Import │
    │ (import_kinara.py)   │        │ (File -> Import FBX) │
    └──────────┬───────────┘        └──────────┬───────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
        ┌──────────────────────────────────────────────┐
        │ 78-Joint Blender Armature with Baked Actions │
        └──────────────────────────────────────────────┘
```

---

### 1. The Canonical 78-Joint Hierarchy

Both export formats adhere to Kinara's standard **78-joint humanoid skeleton hierarchy**:

```
HipsRoot (Root)
 ├── Chest
 │    ├── Neck
 │    │    └── Head
 │    ├── LeftShoulder
 │    │    └── LeftElbow
 │    │         └── LeftWrist
 │    │              ├── LeftThumb1  -> LeftThumb2  -> LeftThumb3  -> LeftThumb4
 │    │              ├── LeftIndex1  -> LeftIndex2  -> LeftIndex3  -> LeftIndex4
 │    │              ├── LeftMiddle1 -> LeftMiddle2 -> LeftMiddle3 -> LeftMiddle4
 │    │              ├── LeftRing1   -> LeftRing2   -> LeftRing3   -> LeftRing4
 │    │              └── LeftPinky1  -> LeftPinky2  -> LeftPinky3  -> LeftPinky4
 │    └── RightShoulder
 │         └── RightElbow
 │              └── RightWrist
 │                   ├── RightThumb1  -> RightThumb2  -> RightThumb3  -> RightThumb4
 │                   ├── RightIndex1  -> RightIndex2  -> RightIndex3  -> RightIndex4
 │                   ├── RightMiddle1 -> RightMiddle2 -> RightMiddle3 -> RightMiddle4
 │                   ├── RightRing1   -> RightRing2   -> RightRing3   -> RightRing4
 │                   └── RightPinky1  -> RightPinky2  -> RightPinky3  -> RightPinky4
 ├── LeftHip
 │    └── LeftKnee -> LeftAnkle -> LeftFoot -> LeftToeBase
 └── RightHip
      └── RightKnee -> RightAnkle -> RightFoot -> RightToeBase
```

---

### 2. Coordinate System Transformation

OpenCV image space and Blender world space use different basis orientations:

| Coordinate System | $+X$ Axis | $+Y$ Axis | $+Z$ Axis |
| :--- | :--- | :--- | :--- |
| **OpenCV Camera Space** | Screen Right | Screen Down | Forward into scene (Depth) |
| **Blender World Space** | Character Right | Forward into scene (Depth) | Character Up (Altitude) |

Before writing export data, coordinates are transformed via the orthogonal basis rotation:
$$\begin{pmatrix} X_{\text{blender}} \\ Y_{\text{blender}} \\ Z_{\text{blender}} \end{pmatrix} = \begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix} \begin{pmatrix} X_{\text{cv}} \\ Y_{\text{cv}} \\ Z_{\text{cv}} \end{pmatrix}$$

---

### 3. Method A: Kinara Motion JSON (`kinara-motion-json-v1`)

The JSON export is an open, human-readable specification designed for programmatic parsing by Blender's Python API (`bpy`).

#### JSON Payload Structure:
```json
{
  "version": "kinara-motion-json-v1",
  "fps": 60.0,
  "frame_count": 1800,
  "coordinate_system": {
    "space": "kinara_normalized",
    "right_axis": "X",
    "forward_axis": "Y",
    "up_axis": "Z",
    "grounded_axis": "Z"
  },
  "skeleton": [
    {"name": "HipsRoot", "parent": null},
    {"name": "Chest", "parent": "HipsRoot"},
    {"name": "Neck", "parent": "Chest"},
    {"name": "Head", "parent": "Neck"},
    {"name": "LeftShoulder", "parent": "Chest"}
  ],
  "rest_joints": {
    "HipsRoot": {"x": 0.0, "y": 0.0, "z": 0.95},
    "Chest":    {"x": 0.0, "y": 0.0, "z": 1.25}
  },
  "frames": [
    {
      "frame_index": 0,
      "joints": {
        "HipsRoot": {"x": 0.012, "y": -0.004, "z": 0.951, "confidence": 0.98},
        "LeftWrist": {"x": -0.421, "y": 0.112, "z": 0.984, "confidence": 0.95}
      }
    }
  ]
}
```

#### How Blender Consumes This File (`import_kinara_motion.py`):
1. **Armature Instantiation**: Creates an Armature object `bpy.data.armatures.new("KinaraArmature")`.
2. **Rest Pose Generation**: In `EDIT` mode, creates `EditBone` instances for each joint, setting bone head coordinates to the parent's rest position and bone tail coordinates to the child's rest position.
3. **Target Constraint Rigging**: For each joint, generates a hidden Empty keyed to the joint's world-space trajectories, attaching `IK` or `Copy Location / Track To` bone constraints.
4. **Action Baking**: Calls `bpy.ops.nla.bake()` to bake the resulting motion directly into bone quaternion rotation curves (`rotation_quaternion`), deleting the temporary target empties.

---

### 4. Method B: Lightweight ASCII FBX 7.4 Export

For direct drag-and-drop into Blender (`File -> Import -> FBX (.fbx)`), Kinara generates standalone, standard-compliant ASCII FBX files without requiring Python execution:

1. **Header & Definitions**:
   Specifies `FBXHeaderVersion: 1003` and `FBXVersion: 7400`. Declares templates for `Model::LimbNode` (armature bones) and `NodeAttribute::Skeleton`.
2. **Armature Hierarchy Construction**:
   Emits `Model: "Model::LeftWrist", "LimbNode"` with relative transform vectors linking child bones to parent bones in bind pose.
3. **Connections Section**:
   Defines `OO` (Object-to-Object) links establishing the scene hierarchy:
   ```
   ; Model::LeftWrist -> Model::LeftElbow
   C: "OO", 100021, 100020
   ```
4. **Animation Takes & Curves**:
   Emits `AnimationStack` and `AnimationCurveNode` records mapping frame ticks to translation/rotation curves:
   $$\text{FBX Tick} = \text{round}\left(\frac{\text{frame\_index}}{\text{FPS}} \times 46,186,158,000\right)$$

---

### 5. Distance-Invariant Torso Metric Scale Normalization

A critical problem in single-camera markerless capture is that character scale changes as the actor approaches or walks away from the camera. Kinara solves this with zero-lag scale normalization:

1. **Torso Measurement**:
   Measures Euclidean distance between `HipsRoot` and `Neck` on each frame $t$:
   $$L_{\text{torso}}(t) = \|\mathbf{p}_{\text{Neck}}(t) - \mathbf{p}_{\text{HipsRoot}}(t)\|$$
2. **Zero-Lag Bidirectional EMA Smoothing**:
   Applies a forward and backward exponential moving average across the entire timeline to filter out frame-to-frame detection noise while eliminating latency:
   $$S_{\text{smoothed}}(t) = \text{EMA}_{\text{backward}}\left(\text{EMA}_{\text{forward}}(L_{\text{torso}}(t))\right)$$
3. **Canonical Normalization**:
   Scales all joint vectors relative to `HipsRoot` by target anatomical scale $S_{\text{target}}$ (default $0.50\text{ m}$ torso):
   $$\mathbf{p}_{\text{normalized}}(t) = \mathbf{p}_{\text{HipsRoot}}(t) + \left(\mathbf{p}(t) - \mathbf{p}_{\text{HipsRoot}}(t)\right) \times \frac{S_{\text{target}}}{S_{\text{smoothed}}(t)}$$

---

### 6. Ground Plane Alignment ($Z = 0$)

To ensure character feet contact the floor naturally without floating or penetrating the ground:
1. Calculates the minimum vertical coordinate among all confident foot/toe joints across the animation sequence:
   $$Z_{\text{floor}} = \min_{t, j \in \{\text{Ankle}, \text{Foot}, \text{ToeBase}\}} Z_j(t)$$
2. Offsets all joint coordinates by $-Z_{\text{floor}}$:
   $$\forall j, \quad Z_j^*(t) = Z_j(t) - Z_{\text{floor}}$$
   Guaranteeing that at maximum plant, the character's footwear aligns precisely with the Blender ground plane ($Z = 0.0$).
