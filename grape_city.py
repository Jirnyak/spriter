#!/usr/bin/env python3
"""
Organic city growth using DLA/mycelium-like algorithms.
Population acts as time - each population increment grows the city.

Usage:
  python grape_city.py params.json
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
    is_main: bool = False


class GrapeCity:
    def __init__(self, seed: int):
        self.seed = seed
        random.seed(seed)
        
        # Fixed map size
        self.size = 4096
        self.cell_size = 8
        self.grid_w = self.size // self.cell_size
        self.grid_h = self.size // self.cell_size
        
        # Grid: 0=empty, 1=street, 2=house
        self.grid = [[0 for _ in range(self.grid_w)] for _ in range(self.grid_h)]
        
        # City center (actual center, no offset for now - can add small variation later)
        self.center_x = self.grid_w // 2
        self.center_y = self.grid_h // 2
        
        # Street network (node-based for efficient branching)
        self.street_nodes: List[StreetNode] = []
        self.street_edges: List[Tuple[int, int]] = []
        
        # Houses (x, y, w, h, rotation)
        self.houses: List[Tuple[int, int, int, int, float]] = []
        
        # Walls
        self.walls: List[List[Tuple[int, int]]] = []
        self.wall_thresholds = [1000, 10000]
        self.walls_built = set()
        
    def initialize_main_roads(self):
        """Create initial cross roads with gentle organic curves."""
        # Center node
        center_node = StreetNode(float(self.center_x), float(self.center_y), True)
        self.street_nodes.append(center_node)
        
        # Four edge nodes at exact positions
        edge_nodes = [
            StreetNode(0.0, float(self.center_y), True),  # Left
            StreetNode(float(self.grid_w - 1), float(self.center_y), True),  # Right
            StreetNode(float(self.center_x), 0.0, True),  # Top
            StreetNode(float(self.center_x), float(self.grid_h - 1), True),  # Bottom
        ]
        
        for node in edge_nodes:
            node_id = len(self.street_nodes)
            self.street_nodes.append(node)
            self.street_edges.append((0, node_id))
        
        # Mark grid for main roads - do this carefully
        for i in range(1, 5):
            n1 = self.street_nodes[0]
            n2 = self.street_nodes[i]
            
            # Horizontal roads
            if i <= 2:
                y_start = min(int(n1.y), int(n2.y))
                y_end = max(int(n1.y), int(n2.y))
                x_start = min(int(n1.x), int(n2.x))
                x_end = max(int(n1.x), int(n2.x))
                
                steps = max(abs(x_end - x_start), 1)
                for step in range(steps + 1):
                    t = step / steps
                    x = int(x_start + (x_end - x_start) * t)
                    y = int(y_start + (y_end - y_start) * t)
                    
                    if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                        self.grid[y][x] = 1
                        # Width 2 for main roads
                        if y + 1 < self.grid_h:
                            self.grid[y + 1][x] = 1
            # Vertical roads
            else:
                y_start = min(int(n1.y), int(n2.y))
                y_end = max(int(n1.y), int(n2.y))
                x_start = min(int(n1.x), int(n2.x))
                x_end = max(int(n1.x), int(n2.x))
                
                steps = max(abs(y_end - y_start), 1)
                for step in range(steps + 1):
                    t = step / steps
                    y = int(y_start + (y_end - y_start) * t)
                    x = int(x_start + (x_end - x_start) * t)
                    
                    if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                        self.grid[y][x] = 1
                        # Width 2 for main roads
                        if x + 1 < self.grid_w:
                            self.grid[y][x + 1] = 1
    
    def mark_street_segment(self, x1: float, y1: float, x2: float, y2: float):
        """Mark a street segment on the grid with 1-cell buffer, preserving houses."""
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) * 2)
        if steps < 1:
            return
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            # Mark the street cell and add 1-cell buffer
            # But NEVER overwrite houses (value 2)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    gx = x + dx
                    gy = y + dy
                    if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                        # Only mark if empty (not street and not house)
                        if self.grid[gy][gx] == 0:
                            self.grid[gy][gx] = 1
    
    def mark_street_and_remove_houses(self, x1: float, y1: float, x2: float, y2: float) -> int:
        """Mark street segment and remove any houses it overlaps. Returns count of removed houses."""
        # First, find all houses that overlap this street path
        street_cells = set()
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) * 2)
        if steps < 1:
            steps = 1
        
        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + (x2 - x1) * t)
            y = int(y1 + (y2 - y1) * t)
            
            # Add buffer zone
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    gx = x + dx
                    gy = y + dy
                    if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                        street_cells.add((gx, gy))
        
        # Remove houses that overlap street cells
        houses_to_remove = []
        for i, (hx, hy, hw, hh, rot) in enumerate(self.houses):
            # Check if any house cell overlaps street
            overlap = False
            for yy in range(hy, hy + hh):
                for xx in range(hx, hx + hw):
                    if (xx, yy) in street_cells:
                        overlap = True
                        break
                if overlap:
                    break
            
            if overlap:
                houses_to_remove.append(i)
                # Clear house from grid
                for yy in range(hy, hy + hh):
                    for xx in range(hx, hx + hw):
                        if 0 <= xx < self.grid_w and 0 <= yy < self.grid_h:
                            if self.grid[yy][xx] == 2:
                                self.grid[yy][xx] = 0
        
        # Remove houses from list (in reverse to maintain indices)
        for i in reversed(houses_to_remove):
            del self.houses[i]
        
        # Now mark street on grid
        for gx, gy in street_cells:
            self.grid[gy][gx] = 1
        
        return len(houses_to_remove)
    
    def mark_street_segment_organic(self, x1: float, y1: float, x2: float, y2: float, width: int = 1):
        """Mark a street segment with organic fluctuations."""
        steps = int(max(abs(x2 - x1), abs(y2 - y1)) * 2)
        if steps < 1:
            return
        
        for i in range(steps + 1):
            t = i / steps
            x = x1 + (x2 - x1) * t
            y = y1 + (y2 - y1) * t
            
            # Add small organic fluctuation
            if i > 0 and i < steps:
                x += random.uniform(-1.0, 1.0)
                y += random.uniform(-1.0, 1.0)
            
            gx = int(x)
            gy = int(y)
            
            if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                self.grid[gy][gx] = 1
                # For width 2, mark adjacent cell
                if width >= 2:
                    # Determine direction perpendicular to path
                    if abs(x2 - x1) > abs(y2 - y1):  # More horizontal
                        if gy + 1 < self.grid_h:
                            self.grid[gy + 1][gx] = 1
                    else:  # More vertical
                        if gx + 1 < self.grid_w:
                            self.grid[gy][gx + 1] = 1
    
    def grow_street_branch(self) -> bool:
        """Grow one street branch from existing network."""
        if len(self.street_nodes) < 5:
            return False
        
        # Simple mycelium: pick random node (excluding first 5 main road nodes)
        available = list(range(5, len(self.street_nodes)))
        if not available:
            # If no nodes yet, branch from center (node 0)
            available = [0]
        
        parent_id = random.choice(available)
        parent = self.street_nodes[parent_id]
        
        # Try multiple angles to find good branch location
        for _ in range(12):
            angle = random.uniform(0, 2 * math.pi)
            # Cap street length at 20 cells
            distance = random.uniform(10, 20)
            
            new_x = parent.x + math.cos(angle) * distance
            new_y = parent.y + math.sin(angle) * distance
            
            # Check bounds
            if not (15 <= new_x < self.grid_w - 15 and 15 <= new_y < self.grid_h - 15):
                continue
            
            # Check not too close to existing nodes
            too_close = False
            for node in self.street_nodes:
                dist = math.sqrt((node.x - new_x)**2 + (node.y - new_y)**2)
                if dist < 8:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Add new node
            new_node = StreetNode(new_x, new_y, False)
            new_id = len(self.street_nodes)
            self.street_nodes.append(new_node)
            self.street_edges.append((parent_id, new_id))
            
            # Mark street and remove any overlapped houses
            removed_count = self.mark_street_and_remove_houses(parent.x, parent.y, new_x, new_y)
            
            # Compensate by creating same number of new houses
            for _ in range(removed_count):
                self.try_place_house()
            
            # Occasional loop connections
            if random.random() < 0.15:
                nearby = [i for i, n in enumerate(self.street_nodes)
                         if i != new_id and i != parent_id
                         and not n.is_main
                         and math.sqrt((n.x - new_x)**2 + (n.y - new_y)**2) < 25]
                if nearby:
                    connect_id = random.choice(nearby)
                    self.street_edges.append((new_id, connect_id))
                    conn_node = self.street_nodes[connect_id]
                    # Mark this connection and remove overlapped houses
                    removed = self.mark_street_and_remove_houses(new_x, new_y, conn_node.x, conn_node.y)
                    # Compensate
                    for _ in range(removed):
                        self.try_place_house()
            
            return True
        
        return False
    
    def try_place_house(self) -> bool:
        """Try to place one house near a street. Returns True if successful."""
        if len(self.street_edges) < 1:
            return False
        
        # Filter out main road edges - only place houses along regular streets
        regular_edges = []
        for edge in self.street_edges:
            n1 = self.street_nodes[edge[0]]
            n2 = self.street_nodes[edge[1]]
            # Skip if both nodes are main roads
            if n1.is_main and n2.is_main:
                continue
            regular_edges.append(edge)
        
        if not regular_edges:
            return False
        
        # Pick a random regular street segment (not main road)
        edge = random.choice(regular_edges)
        n1 = self.street_nodes[edge[0]]
        n2 = self.street_nodes[edge[1]]
        
        # Pick a random point along this street segment
        t = random.random()
        street_x = n1.x + (n2.x - n1.x) * t
        street_y = n1.y + (n2.y - n1.y) * t
        
        # Try to place house near this point on the street
        for _ in range(25):  # More attempts to find valid placement
            # Place on either side of the street
            side = random.choice([-1, 1])
            
            # Calculate perpendicular direction to street
            dx = n2.x - n1.x
            dy = n2.y - n1.y
            length = math.sqrt(dx*dx + dy*dy)
            if length < 0.1:
                continue
            
            # Perpendicular vector (rotated 90 degrees)
            perp_x = -dy / length
            perp_y = dx / length
            
            # Place house at perpendicular distance from street - closer for density
            distance = random.randint(3, 7)
            x = int(street_x + perp_x * distance * side)
            y = int(street_y + perp_y * distance * side)
            
            # House size - smaller for higher density
            w = random.randint(2, 3)
            h = random.randint(2, 3)
            
            # Check bounds
            if not (5 <= x < self.grid_w - 5 and 5 <= y < self.grid_h - 5):
                continue
            if not (x + w < self.grid_w - 5 and y + h < self.grid_h - 5):
                continue
            
            # Check if house footprint is free
            # Houses can't overlap streets OR other houses
            free = True
            for yy in range(y, y + h):
                if yy < 0 or yy >= self.grid_h:
                    free = False
                    break
                for xx in range(x, x + w):
                    if xx < 0 or xx >= self.grid_w:
                        free = False
                        break
                    if self.grid[yy][xx] != 0:  # Already occupied by street or house
                        free = False
                        break
                if not free:
                    break
            
            # Also check 1px padding around house for street collision only
            if free:
                for yy in range(y - 1, y + h + 1):
                    if yy < 0 or yy >= self.grid_h:
                        continue
                    for xx in range(x - 1, x + w + 1):
                        if xx < 0 or xx >= self.grid_w:
                            continue
                        if self.grid[yy][xx] == 1:  # Street cell in padding zone
                            free = False
                            break
                    if not free:
                        break
            
            if free:
                # Mark grid
                for yy in range(y, y + h):
                    for xx in range(x, x + w):
                        self.grid[yy][xx] = 2
                
                # Calculate rotation based on street direction
                street_angle = math.atan2(dy, dx)
                rotation = street_angle + random.uniform(-0.3, 0.3)
                
                self.houses.append((x, y, w, h, rotation))
                return True
        
        return False
    
    def build_wall(self, radius: int, segments: int):
        """Build wall polygon."""
        angle_step = 2 * math.pi / segments
        nodes = []
        
        for i in range(segments):
            angle = i * angle_step + random.uniform(-0.1, 0.1)
            r = radius + random.randint(-2, 2)
            x = self.center_x + r * math.cos(angle)
            y = self.center_y + r * math.sin(angle)
            nodes.append((x, y))
        
        self.walls.append(nodes)
    
    def grow(self, target_population: int):
        """Grow city to target population (population = time)."""
        # Initialize main roads
        if len(self.street_nodes) == 0:
            self.initialize_main_roads()
        
        # Calculate targets based on population - need many streets for spreading
        target_streets = int(target_population ** 0.5)  # sqrt scaling for street network
        target_streets = min(target_streets, 800)
        
        # Target houses using population^0.8 formula (medieval scaling)
        target_houses = int(target_population ** 0.8)
        
        # Interleave street and house growth for organic development
        streets_added = len(self.street_nodes)
        houses_added = len(self.houses)
        
        max_iterations = max(target_houses, target_streets) * 50
        iteration = 0
        
        while houses_added < target_houses and iteration < max_iterations:
            iteration += 1
            
            # Grow one street branch
            if self.grow_street_branch():
                streets_added += 1
            
            # Try to place houses along the new street until no more fit
            consecutive_failures = 0
            max_failures = 50  # Try 50 times before growing new street
            
            while consecutive_failures < max_failures and houses_added < target_houses:
                if self.try_place_house():
                    houses_added += 1
                    consecutive_failures = 0  # Reset on success
                    
                    # Build walls at thresholds
                    for threshold in self.wall_thresholds:
                        if houses_added >= threshold and threshold not in self.walls_built:
                            if threshold == 1000:
                                self.build_wall(self.grid_w // 8, 6)
                            elif threshold == 10000:
                                self.build_wall(self.grid_w // 4, 8)
                            self.walls_built.add(threshold)
                else:
                    consecutive_failures += 1
    
    def render(self, population: int) -> Image.Image:
        """Render city to image."""
        img = Image.new("RGB", (self.size, self.size), (230, 220, 200))
        draw = ImageDraw.Draw(img)
        
        # Draw street edges
        # First draw regular streets
        for edge in self.street_edges:
            n1 = self.street_nodes[edge[0]]
            n2 = self.street_nodes[edge[1]]
            
            # Skip main road edges for now
            if n1.is_main and n2.is_main:
                continue
                
            x1 = int(n1.x * self.cell_size)
            y1 = int(n1.y * self.cell_size)
            x2 = int(n2.x * self.cell_size)
            y2 = int(n2.y * self.cell_size)
            
            draw.line([x1, y1, x2, y2], fill=(170, 170, 170), width=self.cell_size)
        
        # Draw main roads on top with 2px width
        for edge in self.street_edges:
            n1 = self.street_nodes[edge[0]]
            n2 = self.street_nodes[edge[1]]
            
            if n1.is_main and n2.is_main:
                x1 = int(n1.x * self.cell_size)
                y1 = int(n1.y * self.cell_size)
                x2 = int(n2.x * self.cell_size)
                y2 = int(n2.y * self.cell_size)
                
                draw.line([x1, y1, x2, y2], fill=(140, 140, 140), width=self.cell_size * 2)
        
        # Draw houses
        for hx, hy, hw, hh, rotation in self.houses:
            cx = (hx + hw / 2) * self.cell_size
            cy = (hy + hh / 2) * self.cell_size
            pw = hw * self.cell_size
            ph = hh * self.cell_size
            
            # Create rotated rectangle
            cos_r = math.cos(rotation)
            sin_r = math.sin(rotation)
            
            corners = [
                (-pw/2, -ph/2),
                (pw/2, -ph/2),
                (pw/2, ph/2),
                (-pw/2, ph/2)
            ]
            
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
    output = data.get("output", "grape_city.png")
    return Params(population=population, seed=seed, output=output)


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python grape_city.py params.json")
        return 1
    
    params = load_params(sys.argv[1])
    
    city = GrapeCity(params.seed)
    city.grow(params.population)
    img = city.render(params.population)
    img.save(params.output)
    
    print(f"Saved: {params.output}")
    print(f"Population: {params.population}")
    print(f"Streets: {len(city.street_nodes)}")
    print(f"Houses: {len(city.houses)}")
    print(f"Walls: {len(city.walls)}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
