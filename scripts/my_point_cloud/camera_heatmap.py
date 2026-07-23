"""
Isaac Lab: カメラから深度を取得し、点群に変換して「上から見た」ヒートマップを
リアルタイムに (別ウィンドウで) 表示するスクリプト

事前準備 (初回のみ):
    Isaac Labのpython環境に opencv-python が無い場合はインストールしてください。
    ./isaaclab.sh -p -m pip install opencv-python

実行例:
    # リアルタイム表示 (デフォルト)。cv2のウィンドウが立ち上がり、
    # シミュレーションと並行してヒートマップが更新され続けます。
    ./isaaclab.sh -p camera_depth_to_pointcloud_heatmap.py --enable_cameras

    # ヒートマップ更新頻度を下げて軽くしたい場合 (5ステップに1回更新)
    ./isaaclab.sh -p camera_depth_to_pointcloud_heatmap.py --enable_cameras --update_every 5

    # 1回だけ計算して画像を保存するだけでよい場合
    ./isaaclab.sh -p camera_depth_to_pointcloud_heatmap.py --enable_cameras --no-live

ウィンドウ操作:
    ヒートマップウィンドウで 'q' か Esc キーを押すか、ウィンドウを閉じるか、
    ターミナルで Ctrl+C を押すと終了します。

主なオプション:
    --live / --no-live   : リアルタイム表示のON/OFF (デフォルトON)
    --update_every N     : Nステップごとにヒートマップを再計算 (負荷調整用)
    --map_half_size M    : トップダウンマップの表示範囲 (原点から ±M [m])
    --grid_res R         : トップダウンマップのグリッド解像度 [m]
    --max_steps N        : live時、Nステップで自動終了 (未指定なら手動で閉じるまで継続)
    --output_dir DIR     : 出力先 (終了時に最後のフレームを保存)

出力 (--output_dir、デフォルトは ./output/ 以下に保存):
    - pointcloud_heatmap_topdown.png : 上から見た2Dヒートマップ (最終フレーム, 死角=地面色)
    - pointcloud.npy                 : 点群の生データ (N, 3) (最終フレーム)
"""

import argparse
import random

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------
# 1. コマンドライン引数 & Isaac Sim アプリの起動
#    (Isaac Sim を起動してから isaaclab.* を import する必要があるため、
#     必ずこの順番で書く)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Camera depth -> pointcloud -> heatmap example")
parser.add_argument("--num_steps", type=int, default=20, help="[非live時] 深度取得前に進める初期ステップ数")
parser.add_argument("--seed", type=int, default=None, help="オブジェクトのランダムスポーン用シード (未指定なら毎回ランダム)")
parser.add_argument("--output_dir", type=str, default="output", help="出力先ディレクトリ (相対 or 絶対パス)")
parser.add_argument("--live", action="store_true", default=True, help="リアルタイムでヒートマップをウィンドウ表示する (デフォルト有効)")
parser.add_argument("--no-live", dest="live", action="store_false", help="リアルタイム表示を無効にし、最後に1回だけ保存する")
parser.add_argument("--update_every", type=int, default=3, help="何ステップごとにヒートマップを再計算するか (大きいほど軽い)")
parser.add_argument("--map_half_size", type=float, default=2.5, help="トップダウンマップの表示範囲 (原点から±この値 [m])")
parser.add_argument("--grid_res", type=float, default=0.03, help="トップダウンマップのグリッド解像度 [m]")
parser.add_argument("--max_steps", type=int, default=None, help="live時に自動終了するまでの最大ステップ数 (未指定なら手動で閉じるまで継続)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.seed is not None:
    random.seed(args_cli.seed)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# 2. ここから先で isaaclab / isaacsim 系の import が可能になる
# ------------------------------------------------------------------
import os
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import unproject_depth, transform_points

# turbo カラーマップの256段階LUT (RGB, 0-255)
_TURBO_LUT = (cm.get_cmap("turbo")(np.linspace(0.0, 1.0, 256))[:, :3] * 255).astype(np.uint8)


def heightmap_to_bgr_image(heightmap: np.ndarray, vmin: float, vmax: float, upscale_to: int = 600) -> np.ndarray:
    """heightmap (ny, nx) を turbo カラーマップでBGR画像に変換する (cv2.imshow用)。
    上が +Y になるよう上下反転し、見やすいようにアップスケールする。
    """
    norm = np.clip((heightmap - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    idx = (norm * 255).astype(np.uint8)
    rgb = _TURBO_LUT[idx]  # (ny, nx, 3) RGB
    bgr = rgb[:, :, ::-1]
    bgr = np.flipud(bgr)  # 画像の上が +Y になるように反転
    if upscale_to is not None:
        bgr = cv2.resize(bgr, (upscale_to, upscale_to), interpolation=cv2.INTER_NEAREST)
    return np.ascontiguousarray(bgr)


def look_at_quat(eye: np.ndarray, target: np.ndarray, world_up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> tuple:
    """eye から target を見るカメラの向きを ROS 慣習 (X:右, Y:下, Z:前) の
    quaternion (w, x, y, z) として返す。

    CameraCfg.OffsetCfg(..., convention="ros") と組み合わせて使う。
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    forward = target - eye
    forward = forward / np.linalg.norm(forward)

    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)

    true_up = np.cross(right, forward)
    down = -true_up

    # カメラ座標系(X=right, Y=down, Z=forward) -> ワールド座標系 への回転行列
    rot_mat = np.stack([right, down, forward], axis=1)  # 3x3, 列ベクトルが camera軸

    # 回転行列 -> クォータニオン (w, x, y, z)
    m = rot_mat
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s

    return (float(w), float(x), float(y), float(z))


def make_topdown_heatmap(
    points: np.ndarray,
    grid_res: float = 0.05,
    agg: str = "max",
    x_range: Optional[tuple] = None,
    y_range: Optional[tuple] = None,
    fill_value: float = 0.0,
):
    """点群 (N,3) をXY平面に射影し、各グリッドセルの高さ(Z)を集計して
    2Dのトップダウンヒートマップ (height, y方向, x方向) を作る。

    agg: "max" (最大高さ) か "mean" (平均高さ)
    x_range / y_range: 指定すると座標範囲を固定できる (リアルタイム表示でガタつかないように)
    fill_value: 死角 (どの点も落ちなかったセル) を埋める高さ。地面の高さ(通常0.0)に合わせると
                死角が地面と同じ色になり、白飛びしなくなる。
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_min, x_max = x_range if x_range is not None else (x.min(), x.max())
    y_min, y_max = y_range if y_range is not None else (y.min(), y.max())

    nx = max(int(np.ceil((x_max - x_min) / grid_res)) + 1, 1)
    ny = max(int(np.ceil((y_max - y_min) / grid_res)) + 1, 1)

    in_range = (x >= x_min) & (x <= x_max) & (y >= y_min) & (y <= y_max)
    x, y, z = x[in_range], y[in_range], z[in_range]

    ix = np.clip(((x - x_min) / grid_res).astype(int), 0, nx - 1)
    iy = np.clip(((y - y_min) / grid_res).astype(int), 0, ny - 1)

    if agg == "max":
        heightmap = np.full((ny, nx), -np.inf, dtype=np.float64)
        if len(z) > 0:
            np.maximum.at(heightmap, (iy, ix), z)
        empty_mask = np.isneginf(heightmap)
    elif agg == "mean":
        sum_map = np.zeros((ny, nx), dtype=np.float64)
        count_map = np.zeros((ny, nx), dtype=np.float64)
        if len(z) > 0:
            np.add.at(sum_map, (iy, ix), z)
            np.add.at(count_map, (iy, ix), 1.0)
        with np.errstate(invalid="ignore"):
            heightmap = np.where(count_map > 0, sum_map / np.maximum(count_map, 1), 0.0)
        empty_mask = count_map == 0
    else:
        raise ValueError(f"未対応の agg: {agg}")

    # 死角 (どの点も落ちなかったセル) を地面の高さで埋める -> 白飛びせず地面と同じ色になる
    heightmap[empty_mask] = fill_value

    extent = (x_min, x_max, y_min, y_max)
    return heightmap, extent


# ------------------------------------------------------------------
# 3. シーン定義: 地面 + ライト + 何か適当なオブジェクト + カメラ
# ------------------------------------------------------------------
@configclass
class CameraSceneCfg(InteractiveSceneCfg):
    """深度を撮るための最小構成シーン。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ライト
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(0.9, 0.9, 0.9)),
    )

    # 深度に段差をつけるためのオブジェクト。位置は main() 内でランダムに上書きする
    cube = AssetBaseCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.5, 0.5),
            rigid_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 1.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),  # プレースホルダ
    )

    # カメラ本体。distance_to_image_plane が深度データ (メートル単位)
    # offset (pos / rot) は main() 内で (2.5, 2.5, 2.5) -> target (0,0,0) を見る向きに上書きする
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.1,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(2.5, 2.5, 2.5),
            rot=(1.0, 0.0, 0.0, 0.0),  # プレースホルダ (main()で計算した値に上書き)
            convention="ros",
        ),
    )


def compute_pointcloud(camera, camera_eye_fallback, camera_quat_fallback, verbose: bool = False):
    """カメラの深度出力を取得し、ワールド座標系の点群 (N,3) numpy配列を返す。

    センサーの pos_w / quat_w_ros が NaN の場合は、起動時に計算した固定オフセットに
    フォールバックする (Isaac Labの既知の不具合対策)。
    """
    depth_image = camera.data.output["distance_to_image_plane"]

    intrinsics = camera.data.intrinsic_matrices
    points_cam = unproject_depth(depth_image, intrinsics, is_ortho=False)

    sensor_pos = camera.data.pos_w
    sensor_quat = camera.data.quat_w_ros
    if torch.isfinite(sensor_pos).all() and torch.isfinite(sensor_quat).all():
        cam_pos_for_transform = sensor_pos
        cam_quat_for_transform = sensor_quat
    else:
        if verbose:
            print("[WARN] camera.data.pos_w / quat_w_ros にNaN/Infが含まれるため、固定オフセットにフォールバックします。")
        cam_pos_for_transform = torch.tensor(
            camera_eye_fallback, dtype=torch.float32, device=points_cam.device
        ).unsqueeze(0)
        cam_quat_for_transform = torch.tensor(
            camera_quat_fallback, dtype=torch.float32, device=points_cam.device
        ).unsqueeze(0)

    points_world = transform_points(points_cam, cam_pos_for_transform, cam_quat_for_transform)
    points_world_np = points_world[0].cpu().numpy()
    valid_mask = np.isfinite(points_world_np).all(axis=1)
    return points_world_np[valid_mask], depth_image


def main():
    # --- カメラの位置・向きを計算 (2.5, 2.5, 2.5) から原点(0,0,0)を見る ---
    camera_eye = np.array([2.5, 2.5, 2.5])
    camera_target = np.array([0.0, 0.0, 0.0])
    camera_quat = look_at_quat(camera_eye, camera_target)

    # --- オブジェクトのランダムスポーン位置 (XY平面上、原点付近) ---
    rand_x = random.uniform(-1.5, 1.5)
    rand_y = random.uniform(-1.5, 1.5)
    object_pos = (rand_x, rand_y, 0.25)
    print(f"[INFO] オブジェクトのスポーン位置: {object_pos}")

    # --- シミュレーション & シーンの生成 ---
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=tuple(camera_eye), target=tuple(camera_target))

    scene_cfg = CameraSceneCfg(num_envs=1, env_spacing=2.0)
    # プレースホルダ値を計算済みの値で上書き
    scene_cfg.camera.offset.pos = tuple(camera_eye)
    scene_cfg.camera.offset.rot = camera_quat
    scene_cfg.cube.init_state.pos = object_pos

    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[INFO] シーン初期化完了。ステップを進めます...")

    camera = scene["camera"]

    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] 出力先ディレクトリ: {output_dir}")

    # カメラのデータが有効になるまで何ステップか先に進めておく
    for _ in range(args_cli.num_steps):
        sim.step()
        scene.update(dt=sim.get_physics_dt())

    # トップダウンマップの表示範囲を固定 (原点中心 ±map_half_size) しておくと、
    # フレームごとに範囲がガタつかず見やすい
    half = args_cli.map_half_size
    x_range = (-half, half)
    y_range = (half * -1, half)
    grid_res = args_cli.grid_res
    ground_height = 0.0  # 地面のZ座標。死角セルはこの値で塗りつぶす

    if not args_cli.live or not _CV2_AVAILABLE:
        if args_cli.live and not _CV2_AVAILABLE:
            print("[WARN] opencv-python が見つからないため、リアルタイム表示をスキップし1回だけ保存します。")
            print("       (Isaac Labの python 環境で `pip install opencv-python` を実行すると有効になります)")
        # ---------------- 非ライブ: 1回だけ計算して保存 ----------------
        points_world_np, depth_image = compute_pointcloud(camera, camera_eye, camera_quat, verbose=True)
        print(f"[INFO] 点群点数: {points_world_np.shape[0]}")

        heightmap, extent = make_topdown_heatmap(
            points_world_np, grid_res=grid_res, agg="max",
            x_range=x_range, y_range=y_range, fill_value=ground_height,
        )
        plt.figure(figsize=(8, 7))
        im = plt.imshow(heightmap, origin="lower", extent=extent, cmap="turbo", aspect="equal")
        plt.colorbar(im, label="Height Z [m]")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Top-down Heatmap (max height per cell)")
        plt.tight_layout()
        pointcloud_heatmap_path = os.path.join(output_dir, "pointcloud_heatmap_topdown.png")
        plt.savefig(pointcloud_heatmap_path, dpi=150)
        plt.close()

        pointcloud_path = os.path.join(output_dir, "pointcloud.npy")
        np.save(pointcloud_path, points_world_np)
        print(f"[INFO] 保存しました: {pointcloud_heatmap_path}, {pointcloud_path}")
        return

    # ---------------- ライブ表示: cv2ウィンドウをリアルタイム更新 ----------------
    window_name = "Top-down Heatmap (live)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 600, 640)

    print("[INFO] リアルタイム表示を開始します。ヒートマップのウィンドウで 'q' か Esc を押すか、")
    print("       ウィンドウを閉じるか Ctrl+C で終了します。")

    step_count = 0
    last_points = None
    live_broken = False
    try:
        while simulation_app.is_running():
            sim.step()
            scene.update(dt=sim.get_physics_dt())
            step_count += 1

            if step_count % args_cli.update_every == 0:
                points_world_np, _ = compute_pointcloud(camera, camera_eye, camera_quat)
                last_points = points_world_np

                heightmap, _ = make_topdown_heatmap(
                    points_world_np, grid_res=grid_res, agg="max",
                    x_range=x_range, y_range=y_range, fill_value=ground_height,
                )
                finite_max = float(np.nanmax(heightmap)) if heightmap.size > 0 else 1.0
                vmax = max(finite_max, ground_height + 0.05)

                disp = heightmap_to_bgr_image(heightmap, vmin=ground_height, vmax=vmax)
                cv2.putText(
                    disp, f"step {step_count}  points={points_world_np.shape[0]}  Zmax={vmax:.2f}m",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
                )

                try:
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as e:
                    print(f"[WARN] cv2.imshow に失敗しました ({e})。"
                          " opencv-python (GUIサポートあり) がインストールされているか確認してください。"
                          " 例: pip uninstall opencv-python-headless && pip install opencv-python")
                    live_broken = True
                    break

                if key == 27 or key == ord("q"):
                    print("[INFO] 終了キーが押されたため終了します。")
                    break
                # ウィンドウが閉じられたかチェック
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] ヒートマップのウィンドウが閉じられました。終了します。")
                    break

            if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
                print(f"[INFO] max_steps ({args_cli.max_steps}) に到達したため終了します。")
                break
    except KeyboardInterrupt:
        print("[INFO] Ctrl+C を検知したため終了します。")
    finally:
        cv2.destroyAllWindows()

    if live_broken:
        print("[INFO] cv2表示に失敗しましたが、直前のフレームは通常どおり保存します。")

    # 終了時に最後のフレームを保存しておく
    if last_points is not None:
        heightmap, extent = make_topdown_heatmap(
            last_points, grid_res=grid_res, agg="max",
            x_range=x_range, y_range=y_range, fill_value=ground_height,
        )
        plt.figure(figsize=(8, 7))
        im2 = plt.imshow(heightmap, origin="lower", extent=extent, cmap="turbo", aspect="equal")
        plt.colorbar(im2, label="Height Z [m]")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.title("Top-down Heatmap (last frame)")
        plt.tight_layout()
        final_path = os.path.join(output_dir, "pointcloud_heatmap_topdown.png")
        plt.savefig(final_path, dpi=150)
        plt.close()

        np.save(os.path.join(output_dir, "pointcloud.npy"), last_points)
        print(f"[INFO] 最終フレームを {final_path} に保存しました。")


if __name__ == "__main__":
    main()
    simulation_app.close()