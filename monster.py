import random
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import colorsys

# -----------------------
# Configuration
# -----------------------
SIZE = 128
CENTER_X, CENTER_Y = SIZE // 2, SIZE // 2

# -----------------------
# Color Generation
# -----------------------
def generate_palette():
    """Generate harmonious color palette for creature"""
    hue = random.random()
    saturation = random.uniform(0.5, 0.9)
    
    base_value = random.uniform(0.4, 0.7)
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, base_value)
    base_color = (int(r*255), int(g*255), int(b*255))
    
    # Lighter shade
    r, g, b = colorsys.hsv_to_rgb(hue, saturation*0.6, min(base_value+0.2, 0.9))
    light_color = (int(r*255), int(g*255), int(b*255))
    
    # Darker shade
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, base_value*0.6)
    dark_color = (int(r*255), int(g*255), int(b*255))
    
    # Accent color (complementary)
    accent_hue = (hue + 0.5) % 1.0
    r, g, b = colorsys.hsv_to_rgb(accent_hue, saturation*0.8, 0.8)
    accent_color = (int(r*255), int(g*255), int(b*255))
    
    return base_color, light_color, dark_color, accent_color

# -----------------------
# Organic Shape Functions
# -----------------------
def draw_organic_blob(draw, cx, cy, radius, color, points=8):
    """Draw an organic blob shape using random vertices"""
    angles = np.linspace(0, 2*np.pi, points, endpoint=False)
    vertices = []
    for angle in angles:
        r = radius * random.uniform(0.7, 1.3)
        x = cx + r * np.cos(angle)
        y = cy + r * np.sin(angle)
        vertices.append((x, y))
    draw.polygon(vertices, fill=color)
    return vertices

def draw_limb(draw, start_x, start_y, angle, length, thickness, color):
    """Draw a connected limb with segments"""
    segments = random.randint(2, 3)
    current_x, current_y = start_x, start_y
    current_angle = angle
    
    for i in range(segments):
        seg_length = length / segments
        end_x = current_x + seg_length * np.cos(current_angle)
        end_y = current_y + seg_length * np.sin(current_angle)
        
        # Thicker at base, thinner at end
        seg_thickness = thickness * (1 - i * 0.3 / segments)
        
        draw.line([(current_x, current_y), (end_x, end_y)], 
                 fill=color, width=int(seg_thickness))
        
        # Draw joint blob
        draw.ellipse([end_x-seg_thickness//2, end_y-seg_thickness//2,
                     end_x+seg_thickness//2, end_y+seg_thickness//2],
                    fill=color)
        
        current_x, current_y = end_x, end_y
        current_angle += random.uniform(-0.3, 0.3)  # Slight bend
    
    return current_x, current_y

def draw_symmetrical_feature(draw, cx, cy, offset_x, offset_y, size, color, shape='ellipse'):
    """Draw feature on both sides symmetrically"""
    if shape == 'ellipse':
        draw.ellipse([cx + offset_x - size//2, cy + offset_y - size//2,
                     cx + offset_x + size//2, cy + offset_y + size//2], fill=color)
        draw.ellipse([cx - offset_x - size//2, cy + offset_y - size//2,
                     cx - offset_x + size//2, cy + offset_y + size//2], fill=color)
    elif shape == 'circle':
        draw.ellipse([cx + offset_x - size//2, cy + offset_y - size//2,
                     cx + offset_x + size//2, cy + offset_y + size//2], fill=color)
        draw.ellipse([cx - offset_x - size//2, cy + offset_y - size//2,
                     cx - offset_x + size//2, cy + offset_y + size//2], fill=color)

def add_texture_layer(img, base_regions, color, density='medium'):
    """Add organic texture details like scales, spots, or fur"""
    draw = ImageDraw.Draw(img)
    texture_type = random.choice(['scales', 'spots', 'stripes', 'bumps'])
    
    if texture_type == 'scales':
        scale_size = random.randint(3, 6)
        spacing = scale_size + 1
        for region in base_regions:
            cx, cy, radius = region
            for y in range(int(cy - radius), int(cy + radius), spacing):
                for x in range(int(cx - radius), int(cx + radius), spacing):
                    if random.random() > 0.3:
                        offset_x = random.randint(-1, 1)
                        offset_y = random.randint(-1, 1)
                        draw.ellipse([x+offset_x, y+offset_y, 
                                    x+offset_x+scale_size, y+offset_y+scale_size//2],
                                   outline=color, width=1)
    
    elif texture_type == 'spots':
        for region in base_regions:
            cx, cy, radius = region
            num_spots = random.randint(int(radius), int(radius*2))
            for _ in range(num_spots):
                angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(0, radius*0.8)
                spot_x = cx + dist * np.cos(angle)
                spot_y = cy + dist * np.sin(angle)
                spot_size = random.randint(2, 5)
                draw.ellipse([spot_x-spot_size//2, spot_y-spot_size//2,
                            spot_x+spot_size//2, spot_y+spot_size//2], fill=color)
    
    elif texture_type == 'stripes':
        for region in base_regions:
            cx, cy, radius = region
            num_stripes = random.randint(3, 6)
            for i in range(num_stripes):
                y_offset = (i - num_stripes//2) * 5
                draw.arc([cx-radius, cy+y_offset-radius, cx+radius, cy+y_offset+radius],
                        start=0, end=180, fill=color, width=2)
    
    else:  # bumps
        for region in base_regions:
            cx, cy, radius = region
            num_bumps = random.randint(int(radius//2), int(radius))
            for _ in range(num_bumps):
                angle = random.uniform(0, 2*np.pi)
                dist = random.uniform(0, radius*0.9)
                bump_x = cx + dist * np.cos(angle)
                bump_y = cy + dist * np.sin(angle)
                bump_size = random.randint(2, 4)
                draw_organic_blob(draw, bump_x, bump_y, bump_size, color, points=4)

def add_edge_highlights(img, base_color):
    """Add highlights and depth to edges"""
    # Create a lighter highlight color
    highlight = tuple(min(c + 60, 255) for c in base_color)
    
    # Edge detection for highlights
    img_array = np.array(img)
    
    # Find edges where we'll add highlights
    draw = ImageDraw.Draw(img)
    
    for y in range(1, SIZE-1):
        for x in range(1, SIZE-1):
            current = img_array[y, x]
            neighbor = img_array[y-1, x-1]
            
            # If there's a color transition, add highlight
            if np.sum(current) > 30 and np.sum(neighbor) < 30:
                if random.random() > 0.7:
                    draw.point((x, y), fill=highlight)
    
    return img

def add_organic_detail_pass(draw, regions, detail_color, accent_color):
    """Add multiple passes of organic details"""
    for region in regions:
        cx, cy, radius = region
        
        # Veins/tendrils
        if random.random() > 0.5:
            num_veins = random.randint(2, 5)
            for _ in range(num_veins):
                start_angle = random.uniform(0, 2*np.pi)
                start_x = cx + (radius*0.5) * np.cos(start_angle)
                start_y = cy + (radius*0.5) * np.sin(start_angle)
                
                length = random.randint(5, 15)
                segments = random.randint(2, 4)
                
                prev_x, prev_y = start_x, start_y
                angle = start_angle
                for i in range(segments):
                    angle += random.uniform(-0.5, 0.5)
                    seg_len = length / segments
                    end_x = prev_x + seg_len * np.cos(angle)
                    end_y = prev_y + seg_len * np.sin(angle)
                    draw.line([(prev_x, prev_y), (end_x, end_y)], 
                            fill=detail_color, width=1)
                    prev_x, prev_y = end_x, end_y
        
        # Organic circles clusters
        num_clusters = random.randint(1, 3)
        for _ in range(num_clusters):
            cluster_angle = random.uniform(0, 2*np.pi)
            cluster_dist = random.uniform(0, radius*0.6)
            cluster_x = cx + cluster_dist * np.cos(cluster_angle)
            cluster_y = cy + cluster_dist * np.sin(cluster_angle)
            
            cluster_size = random.randint(2, 5)
            for j in range(cluster_size):
                offset = random.randint(-4, 4)
                draw.ellipse([cluster_x+offset-1, cluster_y+offset-1,
                            cluster_x+offset+2, cluster_y+offset+2],
                           fill=accent_color)

# -----------------------
# Main Generation
# -----------------------
def generate_monster():
    img = Image.new('RGBA', (SIZE, SIZE), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Generate color palette
    base_color, light_color, dark_color, accent_color = generate_palette()

    # --- BODY STRUCTURE ---
    body_regions = []
    body_type = random.choice(['blob', 'segmented', 'tall'])

    if body_type == 'blob':
        body_radius = random.randint(24, 34)
        body_y = CENTER_Y + random.randint(-5, 6)
        draw_organic_blob(draw, CENTER_X, body_y, body_radius, base_color, points=10)
        body_regions.append((CENTER_X, body_y, body_radius))

        # Light texture blobs
        for _ in range(random.randint(3, 6)):
            blob_offset_x = random.randint(-body_radius // 2, body_radius // 2)
            blob_offset_y = random.randint(-body_radius // 2, body_radius // 2)
            blob_size = random.randint(8, 14)
            draw_organic_blob(
                draw,
                CENTER_X + blob_offset_x,
                body_y + blob_offset_y,
                blob_size,
                light_color,
                points=6,
            )

        # Dark texture blobs (shadow layer)
        for _ in range(random.randint(2, 4)):
            blob_offset_x = random.randint(-body_radius // 3, body_radius // 3)
            blob_offset_y = random.randint(0, body_radius // 2)
            blob_size = random.randint(6, 12)
            draw_organic_blob(
                draw,
                CENTER_X + blob_offset_x,
                body_y + blob_offset_y,
                blob_size,
                dark_color,
                points=6,
            )

        limb_attachment_y = body_y + body_radius // 3
        head_y = body_y - body_radius // 2

    elif body_type == 'segmented':
        segments = random.randint(2, 4)
        seg_radius = random.randint(18, 24)
        segment_spacing = random.randint(14, 18)
        start_y = CENTER_Y - (segments - 1) * segment_spacing // 2

        for i in range(segments):
            seg_y = start_y + i * segment_spacing
            draw_organic_blob(draw, CENTER_X, seg_y, seg_radius, base_color, points=8)
            body_regions.append((CENTER_X, seg_y, seg_radius))

            # Segment highlights
            highlight_size = max(4, seg_radius // 3)
            draw_organic_blob(
                draw,
                CENTER_X - seg_radius // 3,
                seg_y - seg_radius // 3,
                highlight_size,
                light_color,
                points=5,
            )

            # Connecting tissue
            if i < segments - 1:
                for j in range(3):
                    offset = (j - 1) * 4
                    draw.line(
                        [
                            (CENTER_X + offset, seg_y + seg_radius),
                            (CENTER_X + offset, seg_y + seg_radius + segment_spacing),
                        ],
                        fill=dark_color,
                        width=2,
                    )

        limb_attachment_y = start_y + segments * segment_spacing // 2
        head_y = start_y - seg_radius

    else:  # tall
        body_height = random.randint(50, 70)
        body_width = random.randint(22, 32)
        draw.ellipse(
            [
                CENTER_X - body_width // 2,
                CENTER_Y - body_height // 2,
                CENTER_X + body_width // 2,
                CENTER_Y + body_height // 2,
            ],
            fill=base_color,
        )
        body_regions.append((CENTER_X, CENTER_Y, body_width // 2))

        for _ in range(random.randint(3, 6)):
            stripe_x = CENTER_X + random.randint(-body_width // 3, body_width // 3)
            draw.line(
                [
                    (stripe_x, CENTER_Y - body_height // 3),
                    (stripe_x, CENTER_Y + body_height // 3),
                ],
                fill=dark_color,
                width=2,
            )

        draw.ellipse(
            [
                CENTER_X - body_width // 3,
                CENTER_Y - body_height // 3,
                CENTER_X,
                CENTER_Y,
            ],
            fill=light_color,
        )

        limb_attachment_y = CENTER_Y + body_height // 4
        head_y = CENTER_Y - body_height // 2 - 8

    # --- HEAD ---
    head_size = random.randint(18, 26)
    draw_organic_blob(draw, CENTER_X, head_y, head_size, base_color, points=8)
    body_regions.append((CENTER_X, head_y, head_size))

    # Head shadow (bottom)
    shadow_size = max(5, head_size // 2)
    draw_organic_blob(
        draw,
        CENTER_X,
        head_y + head_size // 3,
        shadow_size,
        dark_color,
        points=5,
    )

    # Head highlights
    for _ in range(random.randint(2, 4)):
        offset_x = random.randint(-head_size // 3, head_size // 3)
        offset_y = random.randint(-head_size // 2, 0)
        highlight_size = random.randint(5, 10)
        draw_organic_blob(
            draw,
            CENTER_X + offset_x,
            head_y + offset_y,
            highlight_size,
            light_color,
            points=5,
        )

    # Head horns/protrusions (optional)
    if random.random() > 0.6:
        for side in (-1, 1):
            horn_x = CENTER_X + side * head_size // 2
            horn_base_y = head_y - head_size // 3
            horn_points = []
            for i in range(4):
                px = horn_x + side * i * 3
                py = horn_base_y - i * 4 + random.randint(-2, 2)
                horn_points.append((px, py))
            for i in range(len(horn_points) - 1):
                draw.line(
                    [horn_points[i], horn_points[i + 1]],
                    fill=dark_color,
                    width=2,
                )

    # --- EYES (Symmetrical) ---
    eye_type = random.choice(['simple', 'complex', 'multiple'])
    eye_offset_x = random.randint(head_size // 3, head_size // 2)
    eye_offset_y = random.randint(-5, 5)

    if eye_type == 'simple':
        eye_size = random.randint(6, 10)
        draw_symmetrical_feature(
            draw,
            CENTER_X,
            head_y,
            eye_offset_x,
            eye_offset_y,
            eye_size,
            (255, 255, 255),
            'ellipse',
        )
        draw_symmetrical_feature(
            draw,
            CENTER_X,
            head_y,
            eye_offset_x,
            eye_offset_y,
            eye_size // 2,
            (0, 0, 0),
            'circle',
        )
    elif eye_type == 'complex':
        outer_size = random.randint(8, 12)
        draw_symmetrical_feature(
            draw,
            CENTER_X,
            head_y,
            eye_offset_x,
            eye_offset_y,
            outer_size,
            accent_color,
            'ellipse',
        )
        draw_symmetrical_feature(
            draw,
            CENTER_X,
            head_y,
            eye_offset_x,
            eye_offset_y,
            outer_size // 2,
            (255, 255, 255),
            'ellipse',
        )
        draw_symmetrical_feature(
            draw,
            CENTER_X,
            head_y,
            eye_offset_x,
            eye_offset_y,
            outer_size // 4,
            (0, 0, 0),
            'circle',
        )
    else:  # multiple
        for i in range(2):
            eye_y_offset = eye_offset_y + i * 6 - 3
            eye_size = random.randint(4, 6)
            draw_symmetrical_feature(
                draw,
                CENTER_X,
                head_y,
                eye_offset_x,
                eye_y_offset,
                eye_size,
                (255, 255, 255),
                'circle',
            )
            draw_symmetrical_feature(
                draw,
                CENTER_X,
                head_y,
                eye_offset_x,
                eye_y_offset,
                eye_size // 2,
                (0, 0, 0),
                'circle',
            )

    # --- LIMBS ---
    num_limb_pairs = random.randint(1, 3)
    for i in range(num_limb_pairs):
        limb_y = limb_attachment_y + i * 10
        limb_length = random.randint(20, 35)
        limb_thickness = random.randint(4, 8)

        angle_left = random.uniform(0.3, 1.5)
        draw_limb(
            draw,
            CENTER_X - 10,
            limb_y,
            angle_left,
            limb_length,
            limb_thickness,
            dark_color,
        )

        angle_right = np.pi - angle_left
        draw_limb(
            draw,
            CENTER_X + 10,
            limb_y,
            angle_right,
            limb_length,
            limb_thickness,
            dark_color,
        )

    # --- TENTACLES/APPENDAGES (Optional) ---
    if random.random() > 0.5:
        num_tentacles = random.randint(2, 4)
        for i in range(num_tentacles):
            angle = random.uniform(0.5, 2.6)
            if i % 2 == 0:
                angle = np.pi - angle

            tentacle_start_y = CENTER_Y + random.randint(-10, 20)
            tentacle_length = random.randint(15, 30)
            tentacle_thickness = random.randint(2, 5)
            draw_limb(
                draw,
                CENTER_X,
                tentacle_start_y,
                angle,
                tentacle_length,
                tentacle_thickness,
                light_color,
            )

    # --- LAYER 2: ORGANIC TEXTURE DETAILS ---
    add_texture_layer(img, body_regions, dark_color)

    # --- LAYER 3: FINE DETAILS (veins, clusters, patterns) ---
    draw = ImageDraw.Draw(img)
    add_organic_detail_pass(draw, body_regions, dark_color, accent_color)

    # --- LAYER 4: SURFACE PATTERNS ---
    for _ in range(random.randint(5, 12)):
        spot_x = CENTER_X + random.randint(-40, 40)
        spot_y = CENTER_Y + random.randint(-40, 40)
        spot_size = random.randint(2, 6)
        draw_organic_blob(draw, spot_x, spot_y, spot_size, accent_color, points=5)

    # Bioluminescent spots (optional)
    if random.random() > 0.6:
        for _ in range(random.randint(3, 6)):
            glow_x = CENTER_X + random.randint(-35, 35)
            glow_y = CENTER_Y + random.randint(-35, 35)
            glow_size = random.randint(2, 5)
            draw_organic_blob(draw, glow_x, glow_y, glow_size, light_color, points=5)

    # Edge highlights
    add_edge_highlights(img, base_color)

    return img


# Generate and process
img = generate_monster()

# Convert to RGB for filters
img_rgb = Image.new('RGB', (SIZE, SIZE), (0, 0, 0))
img_rgb.paste(img, (0, 0), img)

# Layer 1: Film grain texture
arr = np.array(img_rgb)
grain = np.random.randint(-12, 12, arr.shape, dtype=np.int16)
arr = np.clip(arr + grain, 0, 255).astype(np.uint8)
img_rgb = Image.fromarray(arr)

# Layer 2: Slight blur for organic blending
img_rgb = img_rgb.filter(ImageFilter.GaussianBlur(radius=0.6))

# Layer 3: Subtle sharpening to bring back details
img_rgb = img_rgb.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))

# Layer 4: Final noise pass for texture
arr = np.array(img_rgb)
fine_noise = np.random.randint(-8, 8, arr.shape, dtype=np.int16)
arr = np.clip(arr + fine_noise, 0, 255).astype(np.uint8)
img_final = Image.fromarray(arr)

# Save
img_final.save('monster_sprite.png')
print("🎨 Procedural monster generated: monster_sprite.png")
print("   Multi-layered rendering:")
print("   • Connected anatomical structure")
print("   • Organic texture details (scales/spots/stripes/bumps)")
print("   • Surface patterns and veins")
print("   • Edge highlights and depth")
print("   • Harmonious color palette with accents")