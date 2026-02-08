#!/usr/bin/env python3
"""
Proof-of-concept procedural city map generator.

Usage:
  python city_map_generator.py params.json

Example params.json:
{
  "size": [1024, 768],
  "population": 12000,
  "seed": 42,
  "output": "city.png"
}
"""

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Params:
    width: int
    height: int
    population: int
    seed: int
    output: str


def calculate_size_from_population(population: int) -> Tuple[int, int]:
    """Calculate map size based on population. Returns roughly square dimensions."""
    # Medieval walled cities: pop^0.4 for constrained, dense growth
    # Walls limit expansion, forcing higher density than modern cities
    base = int((population ** 0.4) * 35)
    base = max(400, min(2048, base))
    return base, base


def _parse_size(value) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, (int, float)):
        s = int(value)
        return s, s
    raise ValueError("size must be int or [width, height]")


def load_params(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    population = int(data.get("population", 5000))
    
    if "size" in data:
        width, height = _parse_size(data["size"])
    else:
        width, height = calculate_size_from_population(population)
    
    seed = int(data.get("seed", 0))
    output = data.get("output", "city.png")
    return Params(width=width, height=height, population=population, seed=seed, output=output)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def make_grid(width: int, height: int, cell: int) -> Tuple[int, int, List[List[int]]]:
    gw = max(8, width // cell)
    gh = max(8, height // cell)
    grid = [[0 for _ in range(gw)] for _ in range(gh)]
    return gw, gh, grid


def mark_rect(grid: List[List[int]], x: int, y: int, w: int, h: int, value: int) -> None:
    gh = len(grid)
    gw = len(grid[0])
    for yy in range(y, y + h):
        if 0 <= yy < gh:
            row = grid[yy]
            for xx in range(x, x + w):
                if 0 <= xx < gw:
                    row[xx] = value


def draw_rect(draw: ImageDraw.ImageDraw, cell: int, x: int, y: int, w: int, h: int, fill: Tuple[int, int, int]) -> None:
    x0 = x * cell
    y0 = y * cell
    x1 = (x + w) * cell - 1
    y1 = (y + h) * cell - 1
    draw.rectangle([x0, y0, x1, y1], fill=fill)


def generate_polygon_wall(draw: ImageDraw.ImageDraw, grid: List[List[int]], cell: int, 
                         cx: int, cy: int, radius: int, segments: int, 
                         wall_thickness: int, tower_radius: int, color: Tuple[int, int, int]) -> None:
    """Generate organic polygon wall with circular towers at nodes."""
    nodes = []
    angle_step = 2 * math.pi / segments
    
    for i in range(segments):
        angle = i * angle_step
        # Add small random variation (low probability)
        if random.random() < 0.1:
            angle += random.uniform(-0.1, 0.1)
        r = radius
        if random.random() < 0.08:
            r += random.randint(-1, 1)
        
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        nodes.append((x, y))
    
    # Draw wall segments between nodes
    for i in range(len(nodes)):
        x1, y1 = nodes[i]
        x2, y2 = nodes[(i + 1) % len(nodes)]
        
        # Draw thick line as wall segment
        for offset in range(-wall_thickness, wall_thickness + 1):
            draw.line([(x1 * cell, y1 * cell), (x2 * cell, y2 * cell)], 
                     fill=color, width=wall_thickness * cell)
        
        # Mark grid along the line
        steps = max(abs(x2 - x1), abs(y2 - y1))
        if steps > 0:
            for t in range(steps + 1):
                fx = x1 + (x2 - x1) * t / steps
                fy = y1 + (y2 - y1) * t / steps
                mark_rect(grid, int(fx) - wall_thickness, int(fy) - wall_thickness, 
                         wall_thickness * 2, wall_thickness * 2, 1)
    
    # Draw circular towers at nodes
    tower_color = (min(255, color[0] + 20), min(255, color[1] + 20), min(255, color[2] + 20))
    for (x, y) in nodes:
        px = x * cell
        py = y * cell
        radius_px = tower_radius * cell
        draw.ellipse([px - radius_px, py - radius_px, px + radius_px, py + radius_px], 
                    fill=tower_color)
        mark_rect(grid, x - tower_radius, y - tower_radius, 
                 tower_radius * 2, tower_radius * 2, 1)


def generate_walls(params: Params, draw: ImageDraw.ImageDraw, grid: List[List[int]], cell: int) -> None:
    pop = params.population
    gw = len(grid[0])
    gh = len(grid)

    # No walls for population < 1000
    if pop < 1000:
        return

    wall_thickness = clamp(1 + pop // 15000, 1, 3)
    cx = gw // 2
    cy = gh // 2
    
    # Population 1000-10000: one inner wall polygon
    if 1000 <= pop < 10000:
        radius = min(gw, gh) // 4
        segments = 6
        tower_radius = 2
        generate_polygon_wall(draw, grid, cell, cx, cy, radius, segments, 
                            wall_thickness, tower_radius, (70, 70, 70))
    
    # Population 10000-100000: inner + outer wall polygons
    elif 10000 <= pop <= 100000:
        # Inner wall
        radius_inner = min(gw, gh) // 4
        segments_inner = 6
        tower_radius_inner = 2
        generate_polygon_wall(draw, grid, cell, cx, cy, radius_inner, segments_inner, 
                            wall_thickness, tower_radius_inner, (70, 70, 70))
        
        # Outer wall
        radius_outer = min(gw, gh) // 2 - 5
        segments_outer = 8
        tower_radius_outer = 3
        generate_polygon_wall(draw, grid, cell, cx, cy, radius_outer, segments_outer, 
                            wall_thickness, tower_radius_outer, (60, 60, 60))


def generate_main_streets(params: Params, draw: ImageDraw.ImageDraw, grid: List[List[int]], cell: int) -> None:
    gw = len(grid[0])
    gh = len(grid)
    cx = gw // 2
    cy = gh // 2
    pop = params.population

    main_width = 1
    
    # No main streets for very small cities without walls
    if pop < 1000:
        return
    
    # Use SAME random calculations as wall generation to ensure alignment
    # Save random state
    state = random.getstate()
    random.seed(params.seed)  # Reset to same seed as walls
    
    # Calculate wall gate positions - MUST match wall generation exactly
    if 1000 <= pop < 10000:
        radius = min(gw, gh) // 4
        segments = 6
    elif 10000 <= pop <= 100000:
        # For outer wall in large cities
        radius = min(gw, gh) // 2 - 5
        segments = 8
    else:
        return
    
    # Generate wall polygon nodes - SAME algorithm as generate_polygon_wall
    angle_step = 2 * math.pi / segments
    nodes = []
    for i in range(segments):
        angle = i * angle_step
        if random.random() < 0.1:
            angle += random.uniform(-0.1, 0.1)
        r = radius
        if random.random() < 0.08:
            r += random.randint(-1, 1)
        x = int(cx + r * math.cos(angle))
        y = int(cy + r * math.sin(angle))
        nodes.append((x, y))
    
    # Restore random state for subsequent generation
    random.setstate(state)
    
    # Calculate gate positions (midpoints between nodes)
    gates = []
    for i in range(len(nodes)):
        n1 = nodes[i]
        n2 = nodes[(i + 1) % len(nodes)]
        gate_x = (n1[0] + n2[0]) // 2
        gate_y = (n1[1] + n2[1]) // 2
        gates.append((gate_x, gate_y))
    
    # Select 4 gates (roughly cardinal directions)
    if len(gates) >= 4:
        selected_gates = [
            gates[0],
            gates[len(gates) // 4],
            gates[len(gates) // 2],
            gates[3 * len(gates) // 4]
        ]
    else:
        selected_gates = gates
    
    # Draw streets through gates: from edge through gate through center to opposite edge
    for gate_x, gate_y in selected_gates:
        # Direction from center to gate (outward)
        dx_out = gate_x - cx
        dy_out = gate_y - cy
        dist = math.sqrt(dx_out**2 + dy_out**2)
        if dist < 1:
            continue
        
        # Normalize direction
        dx_out /= dist
        dy_out /= dist
        
        # Extend from edge to edge through center
        max_dist = max(gw, gh)
        
        for d in range(-max_dist, max_dist):
            px = cx + dx_out * d
            py = cy + dy_out * d
            
            ix = int(px)
            iy = int(py)
            
            # Check bounds
            if ix < 1 or ix >= gw - 1 or iy < 1 or iy >= gh - 1:
                continue
            
            # Small organic variation
            if random.random() < 0.08 and abs(d) > 5:
                ix += random.choice([-1, 0, 1])
                iy += random.choice([-1, 0, 1])
                ix = clamp(ix, 1, gw - 2)
                iy = clamp(iy, 1, gh - 2)
            
            draw_rect(draw, cell, ix - main_width // 2, iy - main_width // 2, 
                     main_width, main_width, (150, 150, 150))
            mark_rect(grid, ix - main_width // 2, iy - main_width // 2, 
                     main_width, main_width, 1)


def generate_central_square(params: Params, draw: ImageDraw.ImageDraw, grid: List[List[int]], cell: int) -> None:
    gw = len(grid[0])
    gh = len(grid)
    size = clamp(6 + params.population // 4000, 6, min(gw, gh) // 3)
    x = gw // 2 - size // 2
    y = gh // 2 - size // 2
    draw_rect(draw, cell, x, y, size, size, (190, 190, 190))
    mark_rect(grid, x, y, size, size, 1)


def generate_street_network(params: Params, draw: ImageDraw.ImageDraw, grid: List[List[int]], cell: int) -> None:
    gw = len(grid[0])
    gh = len(grid)

    # Sigmoid function for smooth growth
    # Max=20 for adequate coverage in large cities
    pop = params.population
    x = math.log10(pop + 1)
    # Sigmoid parameters: max=20, steepness=1.5, midpoint=3.5
    nodes_count = int(6 + 14 / (1 + math.exp(-1.5 * (x - 3.5))))
    
    nodes_x = nodes_count
    nodes_y = nodes_count

    # Build grid of nodes with randomized positions for organic layout
    nodes = []
    node_positions = {}
    spacing = min(gw // nodes_x, gh // nodes_y)
    
    for i in range(nodes_x):
        for j in range(nodes_y):
            base_x = int((i + 0.5) * gw / nodes_x)
            base_y = int((j + 0.5) * gh / nodes_y)
            # Add significant random offset for organic feel
            offset_x = random.randint(-spacing // 2, spacing // 2)
            offset_y = random.randint(-spacing // 2, spacing // 2)
            x = clamp(base_x + offset_x, 2, gw - 2)
            y = clamp(base_y + offset_y, 2, gh - 2)
            node_id = (i, j)
            nodes.append(node_id)
            node_positions[node_id] = (x, y)

    # Randomized Prim to create a street graph
    start = (nodes_x // 2, nodes_y // 2)
    visited = {start}
    edges = []
    frontier = [start]

    def neighbors(n):
        i, j = n
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < nodes_x and 0 <= nj < nodes_y:
                yield (ni, nj)

    while frontier:
        current = frontier.pop(random.randrange(len(frontier)))
        for n in neighbors(current):
            if n not in visited:
                if random.random() < 0.85:  # Higher probability to ensure connectivity
                    visited.add(n)
                    frontier.append(n)
                    edges.append((current, n))
    
    # Connect any unvisited nodes to ensure full connectivity
    unvisited = [n for n in nodes if n not in visited]
    for unvisited_node in unvisited:
        # Find nearest visited node
        best_neighbor = None
        for n in neighbors(unvisited_node):
            if n in visited:
                best_neighbor = n
                break
        if best_neighbor:
            edges.append((unvisited_node, best_neighbor))
            visited.add(unvisited_node)

    # Add extra edges for loops and better connectivity
    for _ in range((nodes_x * nodes_y) // 4):
        a = random.choice(nodes)
        b = random.choice(list(neighbors(a)))
        edges.append((a, b))

    street_width = 1  # Constant width

    for a, b in edges:
        ax, ay = node_positions[a]
        bx, by = node_positions[b]
        
        # Draw organic connecting path
        steps = max(abs(bx - ax), abs(by - ay), 1)
        prev_x, prev_y = ax, ay
        for t in range(1, steps + 1):
            fx = ax + (bx - ax) * t / steps
            fy = ay + (by - ay) * t / steps
            # Add slight curvature
            if random.random() < 0.1:
                fx += random.randint(-1, 1)
                fy += random.randint(-1, 1)
            fx = clamp(int(fx), 0, gw - 1)
            fy = clamp(int(fy), 0, gh - 1)
            
            draw_rect(draw, cell, fx - street_width // 2, fy - street_width // 2, 
                     street_width, street_width, (170, 170, 170))
            mark_rect(grid, fx - street_width // 2, fy - street_width // 2, 
                     street_width, street_width, 1)
            prev_x, prev_y = fx, fy


def generate_houses(params: Params, draw: ImageDraw.ImageDraw, grid: List[List[int]], cell: int) -> None:
    gw = len(grid[0])
    gh = len(grid)

    # Houses scale with population^0.8
    target = max(30, int(params.population ** 0.8))
    max_attempts = target * 20

    min_size = clamp(2, 2, 4)
    max_size = clamp(4 - params.population // 20000, 2, 6)

    placed = 0
    for _ in range(max_attempts):
        if placed >= target:
            break
        w = random.randint(min_size, max_size)
        h = random.randint(min_size, max_size)
        x = random.randint(1, gw - w - 2)
        y = random.randint(1, gh - h - 2)

        # Check empty
        ok = True
        for yy in range(y, y + h):
            row = grid[yy]
            for xx in range(x, x + w):
                if row[xx] != 0:
                    ok = False
                    break
            if not ok:
                break

        if not ok:
            continue

        mark_rect(grid, x, y, w, h, 2)
        color = (210, 190, 160) if random.random() < 0.7 else (200, 180, 150)
        draw_rect(draw, cell, x, y, w, h, color)
        placed += 1


def generate_city(params: Params) -> Image.Image:
    random.seed(params.seed)
    width, height = params.width, params.height

    cell = clamp(int(min(width, height) / 120), 4, 10)
    gw, gh, grid = make_grid(width, height, cell)

    img = Image.new("RGB", (width, height), (230, 220, 200))
    draw = ImageDraw.Draw(img)

    generate_walls(params, draw, grid, cell)
    generate_main_streets(params, draw, grid, cell)
    generate_central_square(params, draw, grid, cell)
    generate_street_network(params, draw, grid, cell)
    generate_houses(params, draw, grid, cell)

    # Add text overlay
    text = f"Size: {width}x{height}\nPopulation: {params.population}"
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font = ImageFont.load_default()
    
    # Draw text with background for readability
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    padding = 8
    box_x = width - text_width - padding * 2 - 10
    box_y = 10
    
    draw.rectangle(
        [box_x, box_y, box_x + text_width + padding * 2, box_y + text_height + padding * 2],
        fill=(0, 0, 0, 180)
    )
    draw.text((box_x + padding, box_y + padding), text, fill=(255, 255, 255), font=font)

    return img


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python city_map_generator.py params.json")
        return 1
    params = load_params(sys.argv[1])
    img = generate_city(params)
    out_path = params.output
    img.save(out_path)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
