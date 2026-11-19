# Vision + Ollama pick-and-place — setup, use, and how to improve it step by step

## 0. What you have now

```
you type: "pick the mustard bottle and put it in the bin"
   │  Ollama (or built-in rules)  → which object, where to
   ▼
 RealSense D435 (sim) ──► height map ──► segmentation ──► size/colour recognition  (vision/)
   ▼
 grasp planner: closing direction, width, height, collision check of the open fingers (grasping/)
   ▼
 motion planner: IK → approach, descend, grasp, lift, carry, release, home (motion/)
   ▼
 arm_controller / gripper_controller  +  Gazebo "virtual grasp" (DetachableJoint)
   ▼
 verification with Gazebo ground truth → retry if the object is not in the bin
```

| Your "future work" item | Where it is now |
|---|---|
| D435 camera in the simulation | `urdf/d435_camera.urdf.xacro`, bridged in `gazebo/launch/bin_picking.launch.py` |
| Object detection + pose estimation | `bin_picking_ai/vision/` (OpenCV, no training) |
| Grasp planning strategy | `bin_picking_ai/grasping/planner.py` |
| MoveIt | `src/ur10e_moveit_config` (config for `move_group`); step 7 below connects it |
| Test and optimise the pipeline | `tools/benchmark.py`, `test/`, steps 1–8 below |
| Ollama | `bin_picking_ai/llm/` (model is auto-detected; falls back to rules if Ollama is off) |

## 1. Install and build (ROS 2 Jazzy + Gazebo Harmonic)

```bash
sudo apt install ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-controllers \
                 ros-jazzy-xacro ros-jazzy-tf2-ros python3-opencv python3-scipy python3-requests
# optional MoveIt:  sudo apt install ros-jazzy-moveit
cd <your workspace>            # the folder that contains src/
colcon build --symlink-install
source install/setup.bash
```

Ollama (already installed on your machine): make sure it runs (`ollama serve`) and a text model exists
(`ollama list`; e.g. `ollama pull llama3.2`). The node picks an installed model by itself, or force one with
`ollama_model:=qwen2.5`. No Ollama? Everything still works with the built-in rule parser (`use_llm:=false`).

## 2. Run

```bash
ros2 launch gazebo bin_picking.launch.py            # Gazebo + robot + camera + AI node (≈15 s to be ready)
ros2 run bin_picking_ai chat                        # second terminal: type commands
```

Wait for `robot> Ready for commands.` Examples:

* `pick the mustard bottle and put it in the bin`
* `grab the spam can` · `take the tallest object and throw it away` · `put the pink box on the right side`
* `bring me the red thing` · `clear the table` (all objects, tall first) · `what do you see` · `go home` · `stop`

Look at what the robot sees: `ros2 run rqt_image_view rqt_image_view /bin_picking/debug_image`
(left/right in commands = what you see in this picture).

Without Gazebo/ROS you can test the whole logic: `python3 -m bin_picking_ai.tools.offline_demo "pick the cereal"`
(from `src/bin_picking_ai`), `python3 -m pytest -q test`, `python3 -m bin_picking_ai.tools.benchmark`.

## 3. How a grasp works here (important to understand before tuning)

The fingers close around the object using the real gripper joints, then a Gazebo `DetachableJoint`
welds the object to the gripper (“virtual grasp”). Scanned meshes with concave triangle collision are
unreliable for pure friction grasping in simulation; the weld makes the demo dependable and still tests
perception, grasp planning, IK and motion. Step 5 shows how to remove the weld.

## 4. Improvement path — do these in order, one at a time

Each step has a **check** that tells you it worked before you move on.

**Step 1 – Verify the camera.** `ros2 topic hz /d435/image` (~15 Hz), open `/bin_picking/debug_image`.
Boxes + names must sit on the objects. *If the picture is empty:* the bridge or sensors plugin is not running
(`gz topic -l | grep d435`). *If boxes are shifted:* the camera extrinsics are wrong — compare
`ros2 run tf2_ros tf2_echo base_link camera_color_optical_frame` with `camera_xyz` in `ur_macro.xacro`.

**Step 2 – Verify names.** In `chat` type `what do you see`. Every object should have the right name.
If two objects are confused, tune `Catalog.cost_matrix_row` weights (`classifier.py`) or add the
distinguishing alias/colour in `tools/build_catalog.py` and re-run it. Optional: add a vision LLM
(`ollama pull llava`) to label uncertain blobs — hook it in `Detector` after `catalog.assign` using
`OllamaClient.describe_image` (already implemented) for blobs whose `label_cost > 0.45`.

**Step 3 – Verify one pick by hand.** Command `pick the sprinkles`. Watch the fingers: they should straddle the
object, close, lift, and the object should follow. If the arm stops at "arm step … failed": raise the
`constraints` tolerances in `config/ur10e_controllers.yaml`. If it follows the wrong object: `tcp_z` in
`config/gripper_<type>.yaml` must equal the xacro arg `tcp_z`, and the world positions in `data/world_objects.json`
must match the world (re-generate: `ros2 run bin_picking_ai gen_attach_xacro`).

**Step 4 – Run the benchmark in Gazebo.** Type the 14 object commands (or script them with
`ros2 topic pub`) and count successes; write the number down. Everything below is judged against it.
Offline reference (kinematic emulator): 14/14, ~26 s per pick.

**Step 5 – Remove the weld (real grasping).** Edit `orchestrator._execute`: skip `attach`/`detach` steps.
Then tune, in this order: finger `effort` and `mu1/mu2` in `gripper.urdf.xacro`; `squeeze` and `close_theta`
in `config/gripper_<type>.yaml`; object `<mass>`/`<inertia>` in each model.sdf (they are all 1 kg / inertia 1, far
too heavy and stiff — use realistic values, e.g. 0.3 kg); replace the mesh `<collision>` of each object by a
box or cylinder primitive. Re-run the benchmark after each change.

**Step 6 – Speed.** `config/places.yaml → motion`: raise `joint_speed`, `line_speed`; lower `approach_height`.
Check the benchmark does not drop.

**Step 7 – MoveIt.** `ros2 launch ur10e_moveit_config move_group.launch.py`, open RViz → MotionPlanning,
verify the arm plans to `home`. Then replace the joint-space transit segments (`carry_to_bin`,
`to_pregrasp`, `to_home` in `motion/plan.py`) by calls to the `/move_action` action with joint goals computed
by `UR10e.solve`; add the table, bin and detected objects to the planning scene so OMPL avoids them.
Keep the straight-line descend/lift segments as they are.

**Step 8 – Robustness.** Cluttered scenes: add a "push apart" move in `GraspPlanner` when all grasps have
negative clearance. Noisy depth: enable `<noise>` in `d435_camera.urdf.xacro`. Wrist camera: change `camera_xyz`
/`parent` to `tool0` for an eye-in-hand view and scan from several poses.

## 5. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `no camera images received` | `ros2 topic list \| grep d435`; check `gz-sim-sensors-system` is in the world and a GPU/OpenGL is available (`LIBGL_ALWAYS_SOFTWARE=1` helps in VMs) |
| Objects invisible / grey in Gazebo | fixed `map_Kd` texture paths in each `meshes/model.mtl`; restart Gazebo |
| `trajectory goal rejected` | controllers not active: `ros2 control list_controllers` must show `arm_controller`, `gripper_controller` active |
| Robot does not start at home | `initial_value` params in `control/ur10e_ros2_control.urdf.xacro` |
| LLM answers slowly | use a smaller model (`ollama_model:=llama3.2:1b`) or `use_llm:=false` |
| "no free space on the table" | table is crowded; remove objects first (`clear the table`) |

## Robotiq grippers: 2F-140 (default) and 2F-85

The workspace uses a Robotiq gripper instead of the T42 hand; everything else (camera, vision, LLM/rule parser,
grasp planner, motion planner, virtual grasp, bin placement) is unchanged.

| | 2F-140 (default) | 2F-85 |
|---|---|---|
| select | nothing (or `export UR_GRIPPER=2f140`) | `export UR_GRIPPER=2f85` before launching |
| stroke | 140 mm (128.5 mm between the collision pads) | 85 mm |
| objects it can grasp | all 14 | 10 of 14 (whey protein, mustard bottle, 2 potted meat cans are too wide) |
| tcp (tool0 -> pad centre) | 0.205 m | 0.15 m |
| config | `config/gripper_2f140.yaml` | `config/gripper_2f85.yaml` |

* One actuated joint `gripper_joint` (0 = open, 2F-140: ~0.695 closed, 2F-85: 0.7929 closed); the other finger joints are `<mimic>` joints driven by gz_ros2_control.
* Too-wide objects are reported ("no feasible grasp ... too wide") and skipped, never attempted.
* Tune in `config/gripper_<type>.yaml` (`squeeze`, `max_object_width`, pad geometry) and the friction/contact values in `gripper*.urdf.xacro`.
* Mimic-joint behaviour, pad friction and release in Gazebo are verified only statically; if a pick slips, raise `mu1/mu2` or `squeeze`.
