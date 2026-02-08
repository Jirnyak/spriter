#!/usr/bin/env python3
"""
Run city growth from 0 to 100000 population, saving every 100 steps.
"""

import os
from grape_city import GrapeCity

def main():
    seed = 0
    city = GrapeCity(seed)
    
    # Initialize main roads
    city.initialize_main_roads()
    
    output_dir = "iterations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save initial state (pop 0)
    img = city.render(0)
    img.save(f"{output_dir}/city_{0:06d}.png")
    print(f"Saved: city_{0:06d}.png (Streets: {len(city.street_nodes)}, Houses: {len(city.houses)})")
    
    # Grow incrementally
    for pop in range(100, 100001, 100):
        # Calculate target houses for this population
        target_houses = int(pop ** 0.8)
        
        # Grow until we reach target
        max_iterations = 10000
        iteration = 0
        houses_added = len(city.houses)
        
        while houses_added < target_houses and iteration < max_iterations:
            iteration += 1
            
            # Grow one street branch
            city.grow_street_branch()
            
            # Try to place houses along the new street until no more fit
            consecutive_failures = 0
            max_failures = 50
            
            while consecutive_failures < max_failures and houses_added < target_houses:
                if city.try_place_house():
                    houses_added += 1
                    consecutive_failures = 0
                    
                    # Build walls at thresholds
                    for threshold in city.wall_thresholds:
                        if houses_added >= threshold and threshold not in city.walls_built:
                            if threshold == 1000:
                                city.build_wall(city.grid_w // 8, 6)
                            elif threshold == 10000:
                                city.build_wall(city.grid_w // 4, 8)
                            city.walls_built.add(threshold)
                else:
                    consecutive_failures += 1
        
        # Save image for this population milestone
        img = city.render(pop)
        img.save(f"{output_dir}/city_{pop:06d}.png")
        print(f"Saved: city_{pop:06d}.png (Streets: {len(city.street_nodes)}, Houses: {len(city.houses)})")

if __name__ == "__main__":
    main()
