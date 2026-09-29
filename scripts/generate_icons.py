from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ICON = ROOT / "assets" / "icon"
_RASTER_CACHE: dict[str, Image.Image] = {}


def raster(svg: Path, size: int) -> bytes:
    try:
        import cairosvg

        return cairosvg.svg2png(url=str(svg), output_width=size, output_height=size)
    except (ImportError, OSError):
        key = svg.name
        if key not in _RASTER_CACHE:
            _RASTER_CACHE[key] = draw_vector_fallback(key)
        image = _RASTER_CACHE[key].resize((size, size), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()


def draw_vector_fallback(name: str) -> Image.Image:
    """Antialiased Pillow renderer for this small, self-contained SVG mark."""
    scale = 3
    side = 1024 * scale
    transparent = name != "icon.svg"
    image = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    def p(x: float, y: float) -> tuple[int, int]:
        return round(x * scale), round(y * scale)

    def curve(points: tuple[tuple[float, float], ...], steps: int = 36) -> list[tuple[int, int]]:
        result = []
        for index in range(steps + 1):
            t = index / steps
            inv = 1 - t
            x = sum(weight * point[0] for weight, point in zip(
                (inv**3, 3 * inv**2 * t, 3 * inv * t**2, t**3), points
            ))
            y = sum(weight * point[1] for weight, point in zip(
                (inv**3, 3 * inv**2 * t, 3 * inv * t**2, t**3), points
            ))
            result.append(p(x, y))
        return result

    white_mode = name == "icon-monochrome.svg"
    if not transparent:
        y = np.linspace(0, 1, side, dtype=np.float32)[:, None, None]
        top = np.array((24, 36, 59), dtype=np.float32)[None, None, :]
        bottom = np.array((10, 18, 34), dtype=np.float32)[None, None, :]
        rgb = np.broadcast_to(np.rint(top * (1 - y) + bottom * y).astype(np.uint8), (side, side, 3)).copy()
        mask = Image.new("L", (side, side), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, side - 1, side - 1), radius=232 * scale, fill=255)
        image = Image.fromarray(np.dstack((rgb, np.asarray(mask))), mode="RGBA")
        draw = ImageDraw.Draw(image)

    if white_mode:
        track_a = track_b = lens_fill = lens_edge = white = (255, 255, 255, 255)
        draw_width, inner_width = 42 * scale, 30 * scale
    else:
        track_a, track_b = (64, 216, 202, 255), (92, 156, 255, 255)
        lens_fill, lens_edge, white = (28, 47, 73, 255), (217, 242, 255, 245), (245, 251, 255, 255)
        draw_width, inner_width = 42 * scale, 30 * scale

    top_arc = curve(((210, 408), (264, 301), (379, 230), (512, 230)))[:-1]
    top_arc += curve(((512, 230), (640, 230), (751, 295), (808, 396)))[1:]
    lower_arc = curve(((818, 616), (763, 724), (649, 794), (516, 794)))[:-1]
    lower_arc += curve(((516, 794), (388, 794), (277, 729), (220, 628)))[1:]
    draw.line(top_arc, fill=track_a, width=draw_width, joint="curve")
    draw.line(lower_arc, fill=track_a, width=28 * scale, joint="curve")

    lens = curve(((272, 512), (339, 412), (420, 362), (512, 362)))[:-1]
    lens += curve(((512, 362), (604, 362), (685, 412), (752, 512)))[1:-1]
    lens += curve(((752, 512), (685, 612), (604, 662), (512, 662)))[1:-1]
    lens += curve(((512, 662), (420, 662), (339, 612), (272, 512)))[1:]
    draw.polygon(lens, fill=lens_fill)
    draw.line(lens + [lens[0]], fill=lens_edge, width=inner_width, joint="curve")

    center = p(512, 512)
    draw.ellipse((center[0] - 94 * scale, center[1] - 94 * scale,
                  center[0] + 94 * scale, center[1] + 94 * scale),
                 outline=track_b, width=30 * scale)
    draw.ellipse((center[0] - 40 * scale, center[1] - 40 * scale,
                  center[0] + 40 * scale, center[1] + 40 * scale), fill=white)
    accent = white if white_mode else (255, 141, 115, 255)
    for x, y, radius in ((692, 362, 25), (332, 665, 18)):
        cx, cy = p(x, y); r = radius * scale
        draw.ellipse((cx-r, cy-r, cx+r, cy+r), fill=accent)
    if not white_mode:
        draw.line((p(742, 256), p(796, 225)), fill=(217, 242, 255, 180), width=18 * scale)
        draw.line((p(245, 835), p(190, 866)), fill=(217, 242, 255, 180), width=18 * scale)
    return image.resize((1024, 1024), Image.Resampling.LANCZOS)


def write_png(svg: Path, path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raster(svg, size))


def write_icns(path: Path, pngs: dict[str, bytes]) -> None:
    chunks = []
    for chunk_name, png in pngs.items():
        chunks.append(chunk_name.encode("ascii") + struct.pack(">I", len(png) + 8) + png)
    payload = b"".join(chunks)
    path.write_bytes(b"icns" + struct.pack(">I", len(payload) + 8) + payload)


def main() -> None:
    icon_sizes = (16, 32, 48, 64, 128, 256, 512, 1024)
    for size in icon_sizes:
        write_png(ICON / "icon.svg", ICON / "png" / f"motion-vision-{size}.png", size)

    base_pngs = {
        size: raster(ICON / "icon.svg", size)
        for size in (16, 32, 48, 64, 128, 256)
    }
    Image.open(io.BytesIO(raster(ICON / "icon.svg", 256))).save(
        ICON / "MotionVision.ico", format="ICO",
        sizes=[(size, size) for size in (16, 32, 48, 64, 128, 256)],
    )
    icns_sizes = {"icp4": 16, "icp5": 32, "icp6": 64, "ic07": 128, "ic08": 256, "ic09": 512, "ic10": 1024}
    write_icns(ICON / "MotionVision.icns", {key: raster(ICON / "icon.svg", size) for key, size in icns_sizes.items()})

    linux_sizes = (16, 24, 32, 48, 64, 128, 256, 512)
    for size in linux_sizes:
        write_png(ICON / "icon.svg", ROOT / "assets" / "platform" / "linux" / "hicolor" / f"{size}x{size}" / "apps" / "com.motionvision.app.png", size)

    android = ROOT / "android" / "app" / "src" / "main" / "res"
    densities = {"mdpi": 108, "hdpi": 162, "xhdpi": 216, "xxhdpi": 324, "xxxhdpi": 432}
    for density, size in densities.items():
        folder = android / f"mipmap-{density}"
        write_png(ICON / "icon.svg", folder / "ic_launcher.png", int(size * 48 / 108))
        write_png(ICON / "icon-foreground.svg", folder / "ic_launcher_foreground.png", size)
        write_png(ICON / "icon-monochrome.svg", folder / "ic_launcher_monochrome.png", size)
    for size in (48, 72, 96, 144, 192):
        folder = android / f"mipmap-{ {48:'mdpi',72:'hdpi',96:'xhdpi',144:'xxhdpi',192:'xxxhdpi'}[size] }"
        write_png(ICON / "icon.svg", folder / "ic_launcher_round.png", size)

    iconset = ICON / "MotionVision.iconset"
    iconset.mkdir(exist_ok=True)
    for size in (16, 32, 128, 256, 512):
        write_png(ICON / "icon.svg", iconset / f"icon_{size}x{size}.png", size)
        write_png(ICON / "icon.svg", iconset / f"icon_{size}x{size}@2x.png", size * 2)

    print(f"Generated {len(icon_sizes)} PNG sizes, ICO, ICNS, Linux and Android icon sets from {ICON / 'icon.svg'}")


if __name__ == "__main__":
    main()
