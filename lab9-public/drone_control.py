import numpy as np
import pandas as pd
import tyro
import mujoco
from mujoco import viewer
from scipy.spatial.transform import Rotation as R

from drone_simulator import DroneSimulator
from pid import PID


SIM_TIME = 5000  # Maximum simulation time in steps


def xquat_to_euler(xquat):
    return R.from_quat([xquat[1], xquat[2], xquat[3], xquat[0]]).as_euler('xyz', degrees=True)


def build_world(fixed_track: bool, rotated_gates: bool) -> str:
    world = open("xml/scene.xml").read()
    if not fixed_track:
        world = world.replace(
            '<body name="red_gate" pos="-2 0 1">',
            f'<body name="red_gate" pos="-2 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">'
        )
        world = world.replace(
            '<body name="green_gate" pos="-4 -0.6 1.3">',
            f'<body name="green_gate" pos="-4 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">'
        )
        world = world.replace(
            '<body name="blue_gate" pos="-6 0.6 0.7">',
            f'<body name="blue_gate" pos="-6 {np.random.uniform(-0.6, 0.6)} {np.random.uniform(0.7, 1.3)}">'
        )

    if rotated_gates:
        world = world.replace(
            '<body name="red_gate"',
            f'<body name="red_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else -15}"'
        )
        world = world.replace(
            '<body name="green_gate"',
            f'<body name="green_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else -30}"'
        )
        world = world.replace(
            '<body name="blue_gate"',
            f'<body name="blue_gate" euler="0 0 {np.random.uniform(-45, 45) if not fixed_track else 45}"'
        )
    return world


def run_single_task(*, wind: bool, rotated_gates: bool, rendering_freq: float, fixed_track: bool) -> None:
    world = build_world(fixed_track, rotated_gates)
    model = mujoco.MjModel.from_xml_string(world)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    view = viewer.launch_passive(model, data)
    view.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    view.cam.fixedcamid = model.camera("track").id

    pos_targets = [
        [0, 0, 1], # drone initialization
        data.body("red_gate").xpos.copy().tolist(),
        data.body("green_gate").xpos.copy().tolist(),
        data.body("blue_gate").xpos.copy().tolist(),
        [-8, 0, 1] # drone end position
    ]

    yaw_quat_targets = [
        [1, 0, 0, 0],
        data.body("red_gate").xquat.copy().tolist(),
        data.body("green_gate").xquat.copy().tolist(),
        data.body("blue_gate").xquat.copy().tolist(),
        [1, 0, 0, 0]
    ]

    yaw_angle_targets = [xquat_to_euler(xquat)[2] for xquat in yaw_quat_targets]

    # TODO: Design PID control
    # Altitude
    pid_altitude = PID(
        gain_prop=10.0, gain_int=0.1, gain_der=5.0,
        sensor_period=model.opt.timestep, output_limits=(-10, 10)
    )

    # Position
    pid_x = PID(
        gain_prop=1.5, gain_int=0.01, gain_der=0.4,
        sensor_period=model.opt.timestep, output_limits=(-3, 3)
    )

    pid_y = PID(
        gain_prop=1.5, gain_int=0.01, gain_der=0.4,
        sensor_period=model.opt.timestep, output_limits=(-7, 7)
    )

    # Attitude (inner loop)
    pid_roll = PID(
        gain_prop=0.6, gain_int=0.0, gain_der=0.1,
        sensor_period=model.opt.timestep, output_limits=(-2, 2)
    )

    pid_pitch = PID(
        gain_prop=0.6, gain_int=0.0, gain_der=0.1,
        sensor_period=model.opt.timestep, output_limits=(-2, 2)
    )

    pid_yaw = PID(
        gain_prop=0.8, gain_int=0.0, gain_der=0.2,
        sensor_period=model.opt.timestep, output_limits=(-1, 1)
    )
    # END OF TODO

    task_label = f"rotated={'yes' if rotated_gates else 'no'}, wind={'yes' if wind else 'no'}"
    print(f"Starting task ({task_label})")
    data.qpos[0:3] = pos_targets[0]
    data.qpos[3:7] = [1, 0, 0, 0]  # no rotation
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    wind_change_prob = 0.1 if wind else 0

    # If you want the simulation to be displayed more slowly, decrease rendering_freq
    # Note that this DOES NOT change the timestep used to approximate the physics of the simulation!
    drone_simulator = DroneSimulator(
        model, data, view, wind_change_prob = wind_change_prob, rendering_freq = rendering_freq
    )

    # TODO: Define additional variables if needed
    next_target_i = 1
    BASE_THRUST = 3.2496
    REACH_THRESHOLD = 1.2
    ROTATE_THRESHOLD = 1.2
    # END OF TODO

    try:
        for _ in range(SIM_TIME):
            current_pos, previous_pos = drone_simulator.position_sensor()
            current_orien, previous_orien = drone_simulator.orientation_sensor()

            if np.linalg.norm(np.array(current_pos) - np.array(pos_targets[-1])) < REACH_THRESHOLD:
                break

            # TODO: define the current target position
            distance_to_target = np.linalg.norm(np.array(current_pos[:2]) - np.array(pos_targets[next_target_i][:2]))
            if distance_to_target < REACH_THRESHOLD:
                print("Reached target", next_target_i)
                next_target_i += 1
            pos_target = pos_targets[next_target_i].copy()
            yaw_angle_target = yaw_angle_targets[next_target_i].copy()
            # END OF TODO

            # TODO: use PID controllers to steer the drone

            def to_drone_coordinate_system(target_pos_error, drone_angle):
                x_err, y_err, z_err = target_pos_error
                delta_xy_absolute = np.linalg.norm((x_err, y_err))
                # Angle order is xyz, i.e. over the x-axis (roll), over y (pitch), over z (yaw)
                drone_yaw = drone_angle[2]
                target_err_yaw = np.degrees(np.arctan2(y_err, x_err))
                angle_to_target = target_err_yaw - drone_yaw
                # Yes, this does not account for drone's angle to z-axis. It's fine as long as it doesn't flip over.
                delta_x_rel = np.cos(np.radians(angle_to_target)) * delta_xy_absolute
                delta_y_rel = -np.sin(np.radians(angle_to_target)) * delta_xy_absolute
                print("delta_y_rel:", delta_y_rel)
                return delta_x_rel, delta_y_rel, z_err

            def wrap_angle(angle):
                return (angle + 180) % 360 - 180

            error_world = np.array(current_pos) - np.array(pos_target)
            error_body = to_drone_coordinate_system(error_world, current_orien)

            prev_error_world = np.array(previous_pos) - np.array(pos_target)
            prev_error_body = to_drone_coordinate_system(prev_error_world, current_orien)

            desired_thrust = BASE_THRUST + pid_altitude.output_signal(0, [error_body[2], prev_error_body[2]])

            desired_roll = pid_y.output_signal(0, [error_body[1], prev_error_body[1]])
            desired_pitch = pid_x.output_signal(0, [error_body[0], prev_error_body[0]]) # Can't pitch too aggressively.

            if distance_to_target < ROTATE_THRESHOLD:
                desired_yaw = yaw_angle_target  # Match gate orientation
            else:
                target_direction = np.array(pos_target) - np.array(current_pos)
                desired_yaw = np.degrees(np.arctan2(target_direction[1], target_direction[0]))  # Point towards the gate

            # target_direction = np.array(pos_target) - np.array(current_pos)
            # target_yaw = np.degrees(np.arctan2(target_direction[1], target_direction[0]))  # Point towards the gate
            # weight = np.clip((distance_to_target - 0.3) / (1.5 - 0.3), 0, 1)
            # desired_yaw = (weight * target_yaw) + ((1 - weight) * yaw_angle_target)

            desired_yaw_wrapped = wrap_angle(desired_yaw - current_orien[2])
            pre_yaw_wrapped = wrap_angle(desired_yaw - previous_orien[2])

            roll_thrust = -pid_roll.output_signal(desired_roll, [current_orien[0], previous_orien[0]])
            pitch_thrust = -pid_pitch.output_signal(desired_pitch, [current_orien[1], previous_orien[1]])
            yaw_thrust = pid_yaw.output_signal(0, [desired_yaw_wrapped, pre_yaw_wrapped])
            # END OF TODO

            # For debugging purposes you can uncomment, but keep in mind that this slows down the simulation

            desired_col = pos_target + [desired_roll, desired_pitch, desired_yaw]
            current_col = np.concat([current_pos, current_orien])
            data = np.array([desired_col, current_col, desired_col - current_col]).T
            row_names = ["x", "y", "z", "roll", "pitch", "yaw"]
            headers = ["desired", "current", "difference"]
            print(pd.DataFrame(data, index=row_names, columns=headers))

            drone_simulator.sim_step(
                desired_thrust, roll_thrust=roll_thrust,
                pitch_thrust=pitch_thrust, yaw_thrust=yaw_thrust
            )

        current_pos, _ = drone_simulator.position_sensor()
        assert np.linalg.norm(
            np.array(current_pos[:2]) - np.array(pos_targets[-1][:2])) < REACH_THRESHOLD, "Drone did not reach the final target!"
        print(f"Task ({task_label}) completed successfully!")
    finally:
        # Ensure viewer is closed before the next run to avoid multiple open windows.
        try:
            view.close()
        except Exception:
            pass


def main(
        wind: bool = False,
        rotated_gates: bool = False,
        all_tasks: bool = False,
        runs: int = 10,
        rendering_freq: float = 3.0,
        fixed_track: bool = False,
) -> None:
    """
    Run the drone control simulation.

    Args:
        wind: Enable wind disturbances.
        rotated_gates: Rotate gates to create the harder variant.
        all_tasks: Run all four combinations of wind/rotated gates.
        runs: How many times to repeat each selected task.
        rendering_freq: Viewer rendering frequency multiplier (lower slows playback).
    """
    task_list = []
    if all_tasks:
        task_list = [
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ]
    else:
        task_list = [(wind, rotated_gates)]

    for wind_flag, rotated in task_list:
        for run_idx in range(runs):
            print(f"\nRun {run_idx + 1}/{runs} for wind={wind_flag}, rotated_gates={rotated}")
            run_single_task(
                wind=wind_flag,
                rotated_gates=rotated,
                rendering_freq=rendering_freq,
                fixed_track=fixed_track,
            )


if __name__ == '__main__':
    tyro.cli(main)