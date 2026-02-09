import numpy as np
from PIL import Image, ImageDraw
import random
import math

class TreeGenerator:
    def __init__(self, width=128, height=256):
        self.width = width
        self.height = height
        
        # Color palettes for different tree types
        self.palettes = {
            'oak': {
                'trunk': [(101, 67, 33), (92, 64, 51), (79, 56, 41)],
                'branch': [(101, 67, 33), (92, 64, 51)],
                'leaves': [(34, 139, 34), (50, 205, 50), (107, 142, 35), (85, 107, 47)]
            },
            'cherry': {
                'trunk': [(60, 40, 30), (80, 50, 40)],
                'branch': [(70, 45, 35), (85, 55, 45)],
                'leaves': [(255, 182, 193), (255, 192, 203), (255, 105, 180), (219, 112, 147)]
            },
            'birch': {
                'trunk': [(245, 245, 245), (220, 220, 220), (200, 200, 200)],
                'branch': [(240, 240, 240), (210, 210, 210)],
                'leaves': [(144, 238, 144), (152, 251, 152), (173, 255, 47)]
            },
            'autumn': {
                'trunk': [(70, 50, 40), (90, 60, 45)],
                'branch': [(85, 60, 45), (95, 65, 50)],
                'leaves': [(255, 140, 0), (255, 69, 0), (255, 215, 0), (178, 34, 34), (210, 105, 30)]
            },
            'pine': {
                'trunk': [(90, 60, 40), (100, 70, 50)],
                'branch': [(95, 65, 45), (105, 75, 55)],
                'leaves': [(0, 100, 0), (34, 139, 34), (0, 128, 0), (25, 80, 25)]
            },
            'willow': {
                'trunk': [(101, 67, 33), (92, 64, 51)],
                'branch': [(101, 67, 33), (92, 64, 51)],
                'leaves': [(154, 205, 50), (173, 255, 47), (124, 252, 0), (144, 238, 144)]
            }
        }
    
    def seeded_random(self, seed, min_val=0.0, max_val=1.0):
        """Generate a deterministic random value based on seed."""
        random.seed(seed)
        return random.uniform(min_val, max_val)
    
    def seeded_choice(self, seed, choices):
        """Make a deterministic choice based on seed."""
        random.seed(seed)
        return random.choice(choices)
    
    def generate(self, seed, x, y):
        """Generate a tree based on seed and coordinate."""
        # Create deterministic random generator
        coord_seed = seed + x * 1024 + y
        random.seed(coord_seed)
        np.random.seed(coord_seed % (2**32))
        
        # Select palette based on topological regions (neighborhoods)
        # Divide coordinate space into regions so nearby trees have similar colors
        region_size = 128  # trees within 128x128 blocks share palette tendencies
        region_x = x // region_size
        region_y = y // region_size
        region_seed = seed + region_x * 10000 + region_y
        
        palette_names = list(self.palettes.keys())
        random.seed(region_seed)
        palette_name = random.choice(palette_names)
        palette = self.palettes[palette_name]
        
        # Reset to coord_seed for tree generation
        random.seed(coord_seed)
        
        # Create image with transparent background
        img = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Tree parameters based on seed
        trunk_height = self.height * random.uniform(0.35, 0.45)
        trunk_base_width = random.uniform(self.width * 0.08, self.width * 0.12)
        
        # Starting position (bottom center)
        start_x = self.width // 2
        start_y = self.height - 10
        
        # Generate trunk
        trunk_points = self.generate_trunk(start_x, start_y, trunk_height, trunk_base_width, coord_seed)
        
        # Draw trunk
        for i in range(len(trunk_points) - 1):
            x1, y1, w1 = trunk_points[i]
            x2, y2, w2 = trunk_points[i + 1]
            trunk_color = random.choice(palette['trunk'])
            draw.line([(x1 - w1/2, y1), (x2 - w2/2, y2)], fill=trunk_color, width=1)
            draw.line([(x1 + w1/2, y1), (x2 + w2/2, y2)], fill=trunk_color, width=1)
            draw.ellipse([x1 - w1/2, y1 - w1/4, x1 + w1/2, y1 + w1/4], fill=trunk_color)
        
        # Generate branches starting from middle-ish of trunk
        branches = []
        branch_start_ratio = random.uniform(0.45, 0.55)
        
        for i in range(int(len(trunk_points) * branch_start_ratio), len(trunk_points)):
            x, y, w = trunk_points[i]
            
            # More branches in middle, fewer at top
            progress = i / len(trunk_points)
            if progress < 0.5:
                branch_prob = 0.3
                max_branches_per_point = 2
            elif progress < 0.7:
                branch_prob = 0.4
                max_branches_per_point = 3
            else:
                branch_prob = 0.2
                max_branches_per_point = 2
            
            if random.random() < branch_prob:
                num_branches = random.randint(1, max_branches_per_point)
                for _ in range(num_branches):
                    angle = random.uniform(-60, 60)
                    length = trunk_height * random.uniform(0.15, 0.35) * (1.2 - progress)
                    depth = random.randint(2, 4)
                    branch_seed = coord_seed + i * 1000 + _
                    branch_data = self.generate_branch(x, y, angle, length, w * 0.6, depth, branch_seed)
                    branches.append(branch_data)
        
        # Draw branches
        for branch_segments, leaf_positions in branches:
            for seg in branch_segments:
                x1, y1, x2, y2, width = seg
                branch_color = random.choice(palette['branch'])
                draw.line([(x1, y1), (x2, y2)], fill=branch_color, width=max(1, int(width)))
            
            # Draw leaves at branch tips
            for lx, ly in leaf_positions:
                leaf_color = random.choice(palette['leaves'])
                leaf_size = random.uniform(4, 8)
                
                # Draw leaf as small cluster
                for _ in range(random.randint(3, 6)):
                    offset_x = random.uniform(-leaf_size, leaf_size)
                    offset_y = random.uniform(-leaf_size, leaf_size)
                    leaf_radius = random.uniform(leaf_size * 0.5, leaf_size)
                    draw.ellipse([
                        lx + offset_x - leaf_radius,
                        ly + offset_y - leaf_radius,
                        lx + offset_x + leaf_radius,
                        ly + offset_y + leaf_radius
                    ], fill=leaf_color)
        
        # Add some leaves on trunk top
        top_x, top_y, top_w = trunk_points[-1]
        for _ in range(random.randint(10, 20)):
            angle = random.uniform(0, 360)
            dist = random.uniform(top_w, top_w * 3)
            lx = top_x + dist * math.cos(math.radians(angle))
            ly = top_y + dist * math.sin(math.radians(angle))
            leaf_color = random.choice(palette['leaves'])
            leaf_size = random.uniform(5, 10)
            for _ in range(random.randint(2, 4)):
                offset_x = random.uniform(-leaf_size, leaf_size)
                offset_y = random.uniform(-leaf_size, leaf_size)
                leaf_radius = random.uniform(leaf_size * 0.5, leaf_size)
                draw.ellipse([
                    lx + offset_x - leaf_radius,
                    ly + offset_y - leaf_radius,
                    lx + offset_x + leaf_radius,
                    ly + offset_y + leaf_radius
                ], fill=leaf_color)
        
        return img
    
    def generate_trunk(self, start_x, start_y, height, base_width, seed):
        """Generate trunk points with natural curve."""
        points = []
        segments = 50
        
        for i in range(segments + 1):
            progress = i / segments
            
            # Y position (moving up)
            y = start_y - height * progress
            
            # X position with natural sway
            sway_seed = seed + i
            random.seed(sway_seed)
            sway = math.sin(progress * math.pi * 2) * base_width * 0.3
            noise = random.uniform(-base_width * 0.1, base_width * 0.1)
            x = start_x + sway + noise
            
            # Width tapers towards top
            width = base_width * (1 - progress * 0.7)
            
            points.append((x, y, width))
        
        return points
    
    def generate_branch(self, start_x, start_y, angle, length, width, depth, seed):
        """Recursively generate branch segments and leaf positions."""
        random.seed(seed)
        segments = []
        leaves = []
        
        # Current branch segment
        num_segments = random.randint(5, 10)
        current_x, current_y = start_x, start_y
        current_angle = angle
        segment_length = length / num_segments
        current_width = width
        
        for i in range(num_segments):
            # Calculate end position
            angle_variation = random.uniform(-15, 15)
            current_angle += angle_variation
            
            end_x = current_x + segment_length * math.cos(math.radians(current_angle))
            end_y = current_y + segment_length * math.sin(math.radians(current_angle))
            
            segments.append((current_x, current_y, end_x, end_y, current_width))
            
            # Taper width
            current_width *= 0.85
            
            # Move to next segment
            current_x, current_y = end_x, end_y
        
        # Add leaves at end of branch (more likely on higher branches)
        # depth parameter helps - lower depth = higher in tree
        leaf_probability = 0.3 + (1.0 / max(depth, 1)) * 0.5
        if depth <= 1 or random.random() < leaf_probability:
            leaves.append((current_x, current_y))
        
        # Generate sub-branches
        if depth > 1 and random.random() < 0.7:
            num_sub_branches = random.randint(1, 3)
            for i in range(num_sub_branches):
                # Pick a point along the branch
                branch_point = random.randint(num_segments // 2, num_segments - 1)
                bx, by, _, _, bw = segments[branch_point]
                
                # Sub-branch parameters
                sub_angle = current_angle + random.uniform(-50, 50)
                sub_length = length * random.uniform(0.5, 0.8)
                sub_seed = seed + i * 10000
                
                sub_segments, sub_leaves = self.generate_branch(
                    bx, by, sub_angle, sub_length, bw * 0.7, depth - 1, sub_seed
                )
                segments.extend(sub_segments)
                leaves.extend(sub_leaves)
        
        return segments, leaves


def generate_tree(seed, x, y, width=128, height=256):
    """
    Generate a procedural tree.
    
    Args:
        seed: Random seed for deterministic generation
        x: X coordinate (0-1024)
        y: Y coordinate (0-1024)
        width: Output width (default 128)
        height: Output height (default 256, aspect ratio 1:2)
    
    Returns:
        PIL Image with generated tree
    """
    generator = TreeGenerator(width, height)
    return generator.generate(seed, x, y)


if __name__ == "__main__":
    # Test the generator
    import sys
    
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    x = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    y = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    
    print(f"Generating tree with seed={seed}, x={x}, y={y}")
    
    img = generate_tree(seed, x, y)
    output_path = f"tree_s{seed}_x{x}_y{y}.png"
    img.save(output_path)
    print(f"Saved to {output_path}")
