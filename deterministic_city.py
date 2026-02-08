#!/usr/bin/env python3
"""
Deterministic city growth simulator.

Population acts as time - city grows from seed through stages.
Based on central place theory and organic urban growth models.

Usage:
  python deterministic_city.py params.json
"""

import json
import math
import random
import sys
from dataclasses import dataclass
from typing import List, Tuple, Set

from PIL import Image, ImageDraw, ImageFont


@dataclass
class Params:
    population: int
    seed: int
    output: str


@dataclass
class StreetNode:
    x: float
    y: float
    generation: int  # When this node was created
    parent_id: int  # -1 for root nodes
    is_main_road: bool  # True for main cross roads


class CitySimulator:
    def __init__(self, seed: int):
        self.seed = seed
        random.seed(seed)
        
        # Fixed map size
        self.size = 4096
        self.cell_size = 8
        self.grid_w = self.size // self.cell_size
        self.grid_h = self.size // self.cell_size
        self.grid = [[0 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        
        # Density field (tracks housing density)
        self.density = [[0.0 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        
        # City center (off-center based on seed)
        self.center_x = self.grid_w // 2 + random.randint(-50, 50)
        self.center_y = self.grid_h // 2 + random.randint(-50, 50)
        
        # Growth state
        self.street_nodes: List[StreetNode] = []
        self.street_edges: List[Tuple[int, int]] = []
        self.houses: List[Tuple[int, int, int, int, float]] = []  # x, y, w, h, rotation
        self.walls: List[List[Tuple[int, int]]] = []  # List of wall polygons
        
        # Growth thresholds
        self.wall_thresholds = [1000, 10000]
        self.walls_built = set()
        
    def to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert float position to grid coordinates."""
        return (int(x), int(y))
    
    def mark_grid(self, x: int, y: int, w: int, h: int, value: int):
        """Mark rectangular area in grid."""
        for yy in range(y, y + h):
            if 0 <= yy < self.grid_h:
                for xx in range(x, x + w):
                    if 0 <= xx < self.grid_w:
                        self.grid[yy][xx] = value
    
    def is_free(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if area is free with 1-cell padding around it."""
        # Check with 1-cell padding to ensure spacing from streets
        for yy in range(y - 1, y + h + 1):
            if yy < 0 or yy >= self.grid_h:
                return False
            for xx in range(x - 1, x + w + 1):
                if xx < 0 or xx >= self.grid_w:
                    return False
                if self.grid[yy][xx] != 0:
                    return False
        return True
    
    def add_density(self, x: int, y: int, w: int, h: int, radius: int = 10):
        """Add density around a house location."""
        cx = x + w // 2
        cy = y + h // 2
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                xx = cx + dx
                yy = cy + dy
                if 0 <= xx < self.grid_w and 0 <= yy < self.grid_h:
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < radius:
                        # Gaussian-like falloff
                        strength = math.exp(-(dist*dist) / (2 * (radius/2)**2))
                        self.density[yy][xx] += strength
    
    def get_spawn_cost(self, x: int, y: int) -> float:
        """Calculate spawn cost: lower is better. Combines density and distance from center."""
        # Distance from center (normalized)
        dist_from_center = math.sqrt((x - self.center_x)**2 + (y - self.center_y)**2)
        max_dist = math.sqrt(self.grid_w**2 + self.grid_h**2) / 2
        normalized_dist = dist_from_center / max_dist
        
        # Density at this location (clamped)
        density_value = min(self.density[y][x], 10.0) / 10.0  # Normalize to 0-1
        
        # Combined cost: distance from center is MUCH more important (0.85) than density (0.15)
        # This ensures development stays near center
        cost = normalized_dist * 0.85 + density_value * 0.15
        
        return cost
    
    def initialize_main_roads(self):
        """Initialize 2 cross roads from edges to center."""
        # Horizontal road with angle variation
        y_left = self.grid_h // 2 + random.randint(-20, 20)
        y_right = self.grid_h // 2 + random.randint(-20, 20)
        
        # Vertical road with angle variation
        x_top = self.grid_w // 2 + random.randint(-20, 20)
        x_bottom = self.grid_w // 2 + random.randint(-20, 20)
        
        # Create initial nodes at edges (main roads)
        edge_nodes = [
            StreetNode(0, y_left, 0, -1, True),  # Left edge
            StreetNode(self.grid_w - 1, y_right, 0, -1, True),  # Right edge
            StreetNode(x_top, 0, 0, -1, True),  # Top edge
            StreetNode(x_bottom, self.grid_h - 1, 0, -1, True),  # Bottom edge
        ]
        
        # Center node - where roads cross
        center_node = StreetNode(float(self.center_x), float(self.center_y), 0, -1, True)
        
        self.street_nodes.append(center_node)
        center_id = 0
        
        for node in edge_nodes:
            node_id = len(self.street_nodes)
            self.street_nodes.append(node)
            self.street_edges.append((center_id, node_id))
    
    def draw_street(self, draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, 
                   color: Tuple[int, int, int], width: int = 1):
        """Draw organic street between two points."""
        # Ensure minimum length of 10px (roughly 1.25 grid cells)
        length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 1.25:  # Minimum length in grid cells
            return
        
        steps = int(max(abs(x2 - x1), abs(y2 - y1)))
        if steps < 2:
            return
        
        for i in range(steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            
            # Small organic variation
            if i > 0 and i < steps and random.random() < 0.1:
                x += random.uniform(-1.5, 1.5)
                y += random.uniform(-1.5, 1.5)
            
            gx, gy = self.to_grid(x, y)
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                # Mark grid BEFORE drawing to ensure collision detection works
                self.mark_grid(gx, gy, width, width, 1)
                px = gx * self.cell_size
                py = gy * self.cell_size
                draw.rectangle([px, py, px + self.cell_size * width, py + self.cell_size * width], 
                             fill=color)
    
    def grow_streets(self, population: int):
        """Grow streets organically using density+distance cost function."""
        # Target number of street nodes based on population
        target_nodes = int(5 + math.log10(population + 1) * 20)
        target_nodes = min(target_nodes, 250)  # Cap for performance
        
        attempts = (target_nodes - len(self.street_nodes)) * 100
        
        for _ in range(attempts):
            if len(self.street_nodes) >= target_nodes:
                break
            
            # Select any existing node as potential parent
            # Weight all nodes equally - we'll evaluate cost at the NEW location
            if len(self.street_nodes) < 10:
                # Early stage: branch from any node
                parent_id = random.randint(0, len(self.street_nodes) - 1)
            else:
                # Later stage: prefer non-edge nodes
                candidates = [i for i, n in enumerate(self.street_nodes) if not n.is_main_road or i == 0]
                if not candidates:
                    candidates = list(range(len(self.street_nodes)))
                parent_id = random.choice(candidates)
            
            parent = self.street_nodes[parent_id]
            
            # Try multiple angles to find best spawn location
            best_cost = float('inf')
            best_pos = None
            best_angle = None
            
            for attempt in range(8):  # Try 8 different directions
                angle = random.uniform(0, 2 * math.pi)
                distance = random.uniform(10, 35)
                
                new_x = parent.x + math.cos(angle) * distance
                new_y = parent.y + math.sin(angle) * distance
                
                # Check bounds
                margin = 10
                if new_x < margin or new_x >= self.grid_w - margin:
                    continue
                if new_y < margin or new_y >= self.grid_h - margin:
                    continue
                
                # Evaluate cost at NEW location
                gx, gy = int(new_x), int(new_y)
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    cost = self.get_spawn_cost(gx, gy)
                    if cost < best_cost:
                        best_cost = cost
                        best_pos = (new_x, new_y)
                        best_angle = angle
            
            if best_pos is None:
                continue
            
            new_x, new_y = best_pos
            
            # Check not too close to existing nodes
            min_node_dist = 8
            too_close = False
            for node in self.street_nodes:
                dist = math.sqrt((node.x - new_x)**2 + (node.y - new_y)**2)
                if dist < min_node_dist:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Add new node (not a main road)
            generation = len(self.street_nodes) // 10
            new_node = StreetNode(new_x, new_y, generation, parent_id, False)
            new_id = len(self.street_nodes)
            self.street_nodes.append(new_node)
            self.street_edges.append((parent_id, new_id))
            
            # Random chance to connect to nearby nodes (creates loops)
            if random.random() < 0.2:
                nearby = [i for i, node in enumerate(self.street_nodes)
                         if i != new_id and i != parent_id 
                         and not node.is_main_road
                         and math.sqrt((node.x - new_x)**2 + (node.y - new_y)**2) < 30]
                if nearby:
                    connect_to = random.choice(nearby)
                    self.street_edges.append((new_id, connect_to))
            
            # Place houses alongside this new street segment
            houses_per_segment = random.randint(2, 4)
            self._place_houses_along_segment(parent, new_node, houses_per_segment)
    
    def _place_houses_along_segment(self, node1: StreetNode, node2: StreetNode, count: int):
        """Place houses along a street segment with rotation, using density cost function."""
        # Calculate perpendicular direction to street
        dx = node2.x - node1.x
        dy = node2.y - node1.y
        length = math.sqrt(dx*dx + dy*dy)
        
        if length < 1:
            return
        
        # Normalized direction
        dir_x = dx / length
        dir_y = dy / length
        
        # Perpendicular directions (both sides of street)
        perp_x = -dir_y
        perp_y = dir_x
        
        # Try to place houses, but use cost function to decide placement probability
        attempts = count * 15
        placed = 0
        
        for _ in range(attempts):
            if placed >= count:
                break
            
            # Position along segment
            t = random.uniform(0.1, 0.9)
            seg_x = node1.x + dx * t
            seg_y = node1.y + dy * t
            
            # Offset perpendicular to street (pick a side)
            side = random.choice([-1, 1])
            offset = random.uniform(3, 8)  # Distance from street
            
            x = int(seg_x + perp_x * offset * side)
            y = int(seg_y + perp_y * offset * side)
            
            # Check bounds first
            if x < 5 or x >= self.grid_w - 5:
                continue
            if y < 5 or y >= self.grid_h - 5:
                continue
            
            # Use cost function at THIS exact location
            cost = self.get_spawn_cost(x, y)
            
            # Probabilistic placement based on cost
            # Use exponential to make low-cost areas MUCH more likely
            placement_threshold = math.exp(-cost * 3)  # Low cost = high threshold
            
            if random.random() < placement_threshold:
                # House size
                w = random.randint(2, 4)
                h = random.randint(2, 4)
                
                # House rotation (small angle relative to street direction)
                street_angle = math.atan2(dy, dx)
                rotation = street_angle + random.uniform(-0.2, 0.2)
                
                # Check bounds with size
                if x + w >= self.grid_w - 5 or y + h >= self.grid_h - 5:
                    continue
                
                # Check if free (includes collision with streets)
                if self.is_free(x, y, w, h):
                    self.mark_grid(x, y, w, h, 2)
                    self.add_density(x, y, w, h, radius=10)  # Add to density field
                    self.houses.append((x, y, w, h, rotation))
                    placed += 1
    
    def build_wall(self, radius: int, segments: int):
        """Build wall polygon with gates aligned to main roads."""
        # CRITICAL: Use exact center (node 0) where main roads cross
        if len(self.street_nodes) == 0:
            return
        
        wall_center_x = self.street_nodes[0].x
        wall_center_y = self.street_nodes[0].y
        
        # Build regular polygon
        angle_step = 2 * math.pi / segments
        nodes = []
        
        for i in range(segments):
            angle = i * angle_step + random.uniform(-0.08, 0.08)
            r = radius + random.randint(-1, 1)
            x = wall_center_x + r * math.cos(angle)
            y = wall_center_y + r * math.sin(angle)
            nodes.append((x, y))
        
        self.walls.append(nodes)
    
    def simulate(self, population: int):
        """Run simulation to given population."""
        # Stage 1: Initialize main roads
        if len(self.street_nodes) == 0:
            self.initialize_main_roads()
        
        # Stage 2: Grow streets hierarchically (houses placed during growth)
        self.grow_streets(population)
        
        # Stage 3: Build walls at thresholds
        for threshold in self.wall_thresholds:
            if population >= threshold and threshold not in self.walls_built:
                if threshold == 1000:
                    self.build_wall(min(self.grid_w, self.grid_h) // 8, 6)
                elif threshold == 10000:
                    self.build_wall(min(self.grid_w, self.grid_h) // 4, 8)
                self.walls_built.add(threshold)
    
    def render(self, population: int) -> Image.Image:
        """Render city to image."""
        img = Image.new("RGB", (self.size, self.size), (230, 220, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw regular streets first (width 1)
        for edge in self.street_edges:
            n1 = self.street_nodes[edge[0]]
            n2 = self.street_nodes[edge[1]]
            # Only draw if neither node is main road
            if not n1.is_main_road and not n2.is_main_road:
                self.draw_street(draw, n1.x, n1.y, n2.x, n2.y, (170, 170, 170), 1)
        
        # Draw main roads (width 2) on top
        for edge in self.street_edges:
            n1 = self.street_nodes[edge[0]]
            n2 = self.street_nodes[edge[1]]
            # Draw if either node is main road
            if n1.is_main_road or n2.is_main_road:
                self.draw_street(draw, n1.x, n1.y, n2.x, n2.y, (140, 140, 140), 2)
        
        # Draw houses with rotation
        for hx, hy, hw, hh, rotation in self.houses:
            cx = (hx + hw / 2) * self.cell_size
            cy = (hy + hh / 2) * self.cell_size
            pw = hw * self.cell_size
            ph = hh * self.cell_size
            
            # Create rotated rectangle
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            
            # Corner offsets from center
            corners = [
                (-pw/2, -ph/2),
                (pw/2, -ph/2),
                (pw/2, ph/2),
                (-pw/2, ph/2)
            ]
            
            # Rotate and translate corners
            rotated = []
            for ox, oy in corners:
                rx = ox * cos_r - oy * sin_r
                ry = ox * sin_r + oy * cos_r
                rotated.append((cx + rx, cy + ry))
            
            color = (210, 190, 160) if random.random() < 0.7 else (200, 180, 150)
            draw.polygon(rotated, fill=color)
        
        # Draw walls
        for wall_nodes in self.walls:
            for i in range(len(wall_nodes)):
                n1 = wall_nodes[i]
                n2 = wall_nodes[(i + 1) % len(wall_nodes)]
                x1 = int(n1[0] * self.cell_size)
                y1 = int(n1[1] * self.cell_size)
                x2 = int(n2[0] * self.cell_size)
                y2 = int(n2[1] * self.cell_size)
                draw.line([x1, y1, x2, y2], fill=(70, 70, 70), width=self.cell_size * 2)
                
                # Towers at nodes
                tower_r = self.cell_size * 3
                draw.ellipse([x1 - tower_r, y1 - tower_r, x1 + tower_r, y1 + tower_r], 
                           fill=(90, 90, 90))
        
        # Add text overlay
        text = f"Population: {population}\nSeed: {self.seed}\nStreets: {len(self.street_nodes)}\nHouses: {len(self.houses)}"
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        padding = 20
        box_x = self.size - text_width - padding * 2 - 20
        box_y = 20
        
        draw.rectangle(
            [box_x, box_y, box_x + text_width + padding * 2, box_y + text_height + padding * 2],
            fill=(0, 0, 0, 180)
        )
        draw.text((box_x + padding, box_y + padding), text, fill=(255, 255, 255), font=font)
        
        return img


def load_params(path: str) -> Params:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    population = int(data.get("population", 5000))
    seed = int(data.get("seed", 0))
    output = data.get("output", "deterministic_city.png")
    return Params(population=population, seed=seed, output=output)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python deterministic_city.py params.json")
        return 1
    
    params = load_params(sys.argv[1])
    
    simulator = CitySimulator(params.seed)
    simulator.simulate(params.population)
    img = simulator.render(params.population)
    img.save(params.output)
    
    print(f"Saved: {params.output}")
    print(f"Population: {params.population}")
    print(f"Streets: {len(simulator.street_nodes)}")
    print(f"Houses: {len(simulator.houses)}")
    print(f"Walls: {len(simulator.walls)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
