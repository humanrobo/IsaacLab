import torch
import torch.nn.functional as F
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt
import omni.ui as ui
import omni.kit.app


class RayHeightmapGenerator:
    def __init__(
        self,
        map_size=8.0,
        output_size=80,
        gui_enabled=True,
        gui_update_interval=10,
    ):
        self.map_size = map_size
        self.output_size = output_size

        # ============================================================
        # GUI
        # ============================================================
        self.gui_enabled = gui_enabled
        self.gui_update_interval = gui_update_interval
        self.gui_counter = 0

        self.height_window = None
        self.image_widget = None
        self.info_label = None

        if self.gui_enabled:
            self._create_gui()

    # ================================================================
    # GUI作成
    # ================================================================
    def _create_gui(self):

        self.height_window = ui.Window(
            "Ray Height Map",
            width=600,
            height=650,
        )

        with self.height_window.frame:

            with ui.VStack(spacing=5):

                self.info_label = ui.Label(
                    "Ray HeightMap: waiting..."
                )

                self.image_widget = ui.Image(
                    width=550,
                    height=550,
                )

    # ================================================================
    # Generate
    # ================================================================
    def generate(self, ray_hits_w):

        # ------------------------------------------------------------
        # ray_hits_w
        #
        # shape:
        # [num_envs, num_rays, 3]
        #
        # 例:
        # [4096, 81, 3]
        # ------------------------------------------------------------

        num_envs = ray_hits_w.shape[0]

        # ------------------------------------------------------------
        # Z座標だけ取り出す
        # ------------------------------------------------------------

        heightmap = ray_hits_w[:, :, 2]

        # ------------------------------------------------------------
        # Ray数から正方形サイズを計算
        #
        # 81 -> 9 x 9
        # ------------------------------------------------------------

        n = int(heightmap.shape[1] ** 0.5)

        if n * n != heightmap.shape[1]:
            raise ValueError(
                f"Ray count {heightmap.shape[1]} "
                f"is not a square number"
            )

        heightmap = heightmap.reshape(
            num_envs,
            n,
            n,
        )

        # ------------------------------------------------------------
        # inf / nan
        #
        # Rayが何にも当たらなかった場合は inf
        # → 0 にする
        # ------------------------------------------------------------

        heightmap = torch.nan_to_num(
            heightmap,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        # ------------------------------------------------------------
        # 9x9 -> 80x80
        # ------------------------------------------------------------

        heightmap = F.interpolate(
            heightmap.unsqueeze(1),
            size=(
                self.output_size,
                self.output_size,
            ),
            mode="nearest",
        )
        heightmap = torch.rot90(heightmap, k=2, dims=(2, 3))

        # ------------------------------------------------------------
        # UI
        # ------------------------------------------------------------

        if self.gui_enabled:

            self.update_gui(
                heightmap
            )

        return heightmap

    # ================================================================
    # HeightMap -> Jetカラー画像
    # ================================================================
    def heightmap_to_texture(
        self,
        height_map,
    ):

        # ------------------------------------------------------------
        # 0～0.5m を基本範囲として正規化
        #
        # 0.0 = 青
        # 0.5 = 赤
        # ------------------------------------------------------------

        hm = np.clip(
            height_map,
            0.0,
            0.5,
        )

        norm = hm / 0.5

        # ------------------------------------------------------------
        # jet colormap
        # ------------------------------------------------------------

        rgb = plt.get_cmap("jet")(norm)

        rgb = (
            rgb[:, :, :3] * 255
        ).astype(np.uint8)

        # ------------------------------------------------------------
        # PIL
        # ------------------------------------------------------------

        img = Image.fromarray(
            rgb,
            mode="RGB",
        )

        # ------------------------------------------------------------
        # PNG
        # ------------------------------------------------------------

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="PNG",
        )

        return buffer.getvalue()

    # ================================================================
    # Robot marker
    # ================================================================
    def add_robot_marker(
        self,
        height_map,
    ):

        H, W = height_map.shape

        # ------------------------------------------------------------
        # ロボットはヒートマップ中央
        # ------------------------------------------------------------

        ix = W // 2
        iy = H // 2

        # ------------------------------------------------------------
        # マーカーの高さ
        #
        # 現在の最大値より少し高くする
        # → jetで赤く見える
        # ------------------------------------------------------------

        marker_value = (
            height_map.max() + 0.2
        )

        # ------------------------------------------------------------
        # 7 x 7
        # ------------------------------------------------------------

        y0 = max(
            0,
            iy - 3,
        )

        y1 = min(
            H,
            iy + 4,
        )

        x0 = max(
            0,
            ix - 3,
        )

        x1 = min(
            W,
            ix + 4,
        )

        height_map[
            y0:y1,
            x0:x1
        ] = marker_value

        return height_map

    # ================================================================
    # GUI更新
    # ================================================================
    def update_gui(
        self,
        heightmap,
    ):

        if not self.gui_enabled:
            return

        # ------------------------------------------------------------
        # 更新頻度
        # ------------------------------------------------------------

        self.gui_counter += 1

        if (
            self.gui_counter
            % self.gui_update_interval
            != 0
        ):
            return

        # ------------------------------------------------------------
        # env 0だけ表示
        # ------------------------------------------------------------

        hm = (
            heightmap[0, 0]
            .detach()
            .cpu()
            .numpy()
            .copy()
        )

        # ------------------------------------------------------------
        # Robot marker
        # ------------------------------------------------------------

        # hm = self.add_robot_marker(
        #     hm
        # )

        # ------------------------------------------------------------
        # Jetカラー化
        # ------------------------------------------------------------

        texture = self.heightmap_to_texture(
            hm
        )

        # ------------------------------------------------------------
        # PNG保存
        # ------------------------------------------------------------

        texture_path = (
            f"/tmp/ray_heightmap_gui_"
            f"{self.gui_counter}.png"
        )

        with open(
            texture_path,
            "wb",
        ) as f:

            f.write(texture)

        # ------------------------------------------------------------
        # UI更新
        # ------------------------------------------------------------

        if self.image_widget is not None:

            self.image_widget.source_url = (
                texture_path
            )

        # ------------------------------------------------------------
        # 情報表示
        # ------------------------------------------------------------

        if self.info_label is not None:

            self.info_label.text = (
                f"Ray HeightMap | "
                f"env=0 | "
                f"shape={hm.shape} | "
                f"min={hm.min():.3f} | "
                f"max={hm.max():.3f}"
            )

        # ------------------------------------------------------------
        # Isaac Sim UI更新
        # ------------------------------------------------------------

        omni.kit.app.get_app().update()