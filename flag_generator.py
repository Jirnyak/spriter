import random
from PIL import Image, ImageDraw

# -----------------------
# Configuration
# -----------------------
SIZE = 128

# -----------------------
# Color Utilities
# -----------------------

def random_color(min_v=40, max_v=220):
    return (
        random.randint(min_v, max_v),
        random.randint(min_v, max_v),
        random.randint(min_v, max_v),
    )


def pick_palette():
    base = random_color(50, 200)
    accent = random_color(80, 240)
    detail = random_color(60, 220)
    highlight = random_color(120, 255)
    return [base, accent, detail, highlight]

# -----------------------
# Shape Helpers
# -----------------------

def draw_triangle(draw, p1, p2, p3, color):
    draw.polygon([p1, p2, p3], fill=color)


def draw_diamond(draw, cx, cy, size, color):
    half = size // 2
    pts = [
        (cx, cy - half),
        (cx + half, cy),
        (cx, cy + half),
        (cx - half, cy),
    ]
    draw.polygon(pts, fill=color)


def draw_cross(draw, cx, cy, size, thickness, color):
    half = size // 2
    t = max(1, thickness)
    draw.rectangle([cx - t//2, cy - half, cx + t//2, cy + half], fill=color)
    draw.rectangle([cx - half, cy - t//2, cx + half, cy + t//2], fill=color)


def draw_circle_symbol(draw, cx, cy, size, color):
    r = size // 2
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)


def draw_square(draw, cx, cy, size, color):
    half = size // 2
    draw.rectangle([cx - half, cy - half, cx + half, cy + half], fill=color)

# -----------------------
# Banner Layers
# -----------------------

def paint_background(draw, palette):
    base = palette[0]
    draw.rectangle([0, 0, SIZE, SIZE], fill=base)

    # Optional gradient-like banding using translucent rectangles
    if random.random() > 0.5:
        band_color = palette[1]
        for i in range(0, SIZE, 16):
            if random.random() > 0.4:
                draw.rectangle([0, i, SIZE, i + 8], fill=band_color)


def paint_main_body(draw, palette):
    color = palette[1]
    mode = random.choice(["center_rect", "diagonal", "circle", "split_vertical"])

    if mode == "center_rect":
        margin = random.randint(12, 24)
        draw.rectangle([margin, margin, SIZE - margin, SIZE - margin], fill=color)

    elif mode == "diagonal":
        thickness = random.randint(16, 32)
        draw.polygon([
            (0, thickness),
            (thickness, 0),
            (SIZE, SIZE - thickness),
            (SIZE - thickness, SIZE),
        ], fill=color)

    elif mode == "circle":
        r = random.randint(24, 40)
        draw.ellipse([SIZE//2 - r, SIZE//2 - r, SIZE//2 + r, SIZE//2 + r], fill=color)

    else:  # split_vertical
        split = random.randint(48, 80)
        draw.rectangle([0, 0, split, SIZE], fill=color)


def paint_divisors_and_symbols(draw, palette):
    color = palette[2]
    mode = random.choice(["cross", "diamond", "triangles", "bars"])

    if mode == "cross":
        draw_cross(draw, SIZE//2, SIZE//2, size=64, thickness=10, color=color)

    elif mode == "diamond":
        draw_diamond(draw, SIZE//2, SIZE//2, size=56, color=color)

    elif mode == "triangles":
        draw_triangle(draw, (0, 0), (SIZE//2, SIZE//2), (0, SIZE), color)
        draw_triangle(draw, (SIZE, 0), (SIZE//2, SIZE//2), (SIZE, SIZE), color)

    else:  # bars
        for i in range(3):
            y = 24 + i * 24
            draw.rectangle([8, y, SIZE - 8, y + 8], fill=color)


def paint_additional_details(draw, palette):
    color = palette[3]
    detail_mode = random.choice(["stars", "circles", "squares", "triangles"])

    if detail_mode == "stars":
        for _ in range(random.randint(3, 6)):
            cx = random.randint(16, SIZE - 16)
            cy = random.randint(16, SIZE - 16)
            size = random.randint(6, 10)
            draw_cross(draw, cx, cy, size=size, thickness=2, color=color)

    elif detail_mode == "circles":
        for _ in range(random.randint(4, 8)):
            cx = random.randint(12, SIZE - 12)
            cy = random.randint(12, SIZE - 12)
            draw_circle_symbol(draw, cx, cy, size=random.randint(6, 12), color=color)

    elif detail_mode == "squares":
        for _ in range(random.randint(4, 8)):
            cx = random.randint(12, SIZE - 12)
            cy = random.randint(12, SIZE - 12)
            draw_square(draw, cx, cy, size=random.randint(6, 12), color=color)

    else:  # triangles
        for _ in range(random.randint(3, 6)):
            x = random.randint(8, SIZE - 8)
            y = random.randint(8, SIZE - 8)
            size = random.randint(8, 14)
            draw_triangle(draw, (x, y - size), (x - size, y + size), (x + size, y + size), color)

# -----------------------
# Generation
# -----------------------

def generate_flag(seed=None):
    if seed is not None:
        random.seed(seed)

    img = Image.new("RGB", (SIZE, SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    palette = pick_palette()

    # Step 1: Background
    paint_background(draw, palette)

    # Step 2: Main body
    paint_main_body(draw, palette)

    # Step 3: Divisors and symbols
    paint_divisors_and_symbols(draw, palette)

    # Step 4: Additional details
    paint_additional_details(draw, palette)

    return img


if __name__ == "__main__":
    flag = generate_flag()
    flag.save("faction_flag.png")
    print("Generated: faction_flag.png")
