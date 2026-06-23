"""Preprocess Maya-style per-frame CSV into the format expected by csv_to_npz.py.

Input CSV (Maya export, "v8_perframe"):
    Frame, root_translateX[cm], root_translateY[cm], root_translateZ[cm],
    root_rotateX[deg], root_rotateY[deg], root_rotateZ[deg],
    <22 K1 joint dofs in degrees, named "*_dof">
    (header row included)

Output CSV (csv_to_npz.py compatible):
    pos_x[m], pos_y[m], pos_z[m], quat_x, quat_y, quat_z, quat_w, <22 joints in rad>
    (no header)

The Euler convention is configurable via --euler_order; default "XYZ" intrinsic
matches the most common Maya export (rotateXYZ in source order = intrinsic XYZ).
If the resulting motion looks upside-down or sideways, try a different order.
"""

import argparse
import csv
import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Maya-style CSV")
    parser.add_argument("--output", required=True, help="Output CSV (csv_to_npz format)")
    parser.add_argument("--euler_order", default="XYZ",
                        help="Euler order for root rotation (scipy convention, e.g. XYZ, ZYX). "
                             "Uppercase = intrinsic, lowercase = extrinsic.")
    parser.add_argument("--pos_scale", type=float, default=0.01,
                        help="Scale applied to root translation (default 0.01 = cm -> m).")
    parser.add_argument("--drop_first", type=int, default=0,
                        help="Drop the first N rows (useful when the source pads frame 0).")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]

    expected_dof_cols = [
        "Head_Yaw_dof", "Head_Pitch_dof",
        "Left_Shoulder_Pitch_dof", "Left_Shoulder_Roll_dof",
        "Left_Elbow_Pitch_dof", "Left_Elbow_Yaw_dof",
        "Right_Shoulder_Pitch_dof", "Right_Shoulder_Roll_dof",
        "Right_Elbow_Pitch_dof", "Right_Elbow_Yaw_dof",
        "Left_Hip_Pitch_dof", "Left_Hip_Roll_dof", "Left_Hip_Yaw_dof",
        "Left_Knee_Pitch_dof", "Left_Ankle_Pitch_dof", "Left_Ankle_Roll_dof",
        "Right_Hip_Pitch_dof", "Right_Hip_Roll_dof", "Right_Hip_Yaw_dof",
        "Right_Knee_Pitch_dof", "Right_Ankle_Pitch_dof", "Right_Ankle_Roll_dof",
    ]
    actual_dof_cols = header[7:]
    if actual_dof_cols != expected_dof_cols:
        print(f"WARNING: DOF column order mismatch.")
        print(f"  expected: {expected_dof_cols}")
        print(f"  actual:   {actual_dof_cols}")

    data = np.array([[float(x) for x in row] for row in rows])
    if args.drop_first > 0:
        data = data[args.drop_first:]

    root_pos = data[:, 1:4] * args.pos_scale
    root_euler_deg = data[:, 4:7]
    joint_deg = data[:, 7:]

    root_rot = R.from_euler(args.euler_order, root_euler_deg, degrees=True)
    quat_xyzw = root_rot.as_quat()
    joint_rad = np.deg2rad(joint_deg)

    out = np.concatenate([root_pos, quat_xyzw, joint_rad], axis=1)
    np.savetxt(args.output, out, delimiter=",", fmt="%.8f")

    print(f"Wrote {args.output}  shape={out.shape}")
    print(f"  euler_order={args.euler_order}  pos_scale={args.pos_scale}")
    print(f"  root_pos[0]={root_pos[0]}  root_euler[0]={root_euler_deg[0]}")
    print(f"  quat_xyzw[0]={quat_xyzw[0]}")


if __name__ == "__main__":
    main()
