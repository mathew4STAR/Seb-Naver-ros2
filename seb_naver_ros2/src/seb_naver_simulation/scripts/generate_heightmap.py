#!/usr/bin/env python3
import cv2
import numpy as np
import os

def generate_heightmap(filename, size=513):
    """
    Generates a grayscale heightmap for Gazebo using some procedural noise (Perlin-like).
    Gazebo prefers (2^n + 1) x (2^n + 1) for heightmaps, e.g., 513x513.
    """
    # Create an empty image
    img = np.zeros((size, size), dtype=np.float32)
    
    # Add some basic terrain features using sine waves
    for y in range(size):
        for x in range(size):
            nx = x / size * 5.0
            ny = y / size * 5.0
            
            # Simple combination of sine waves for rolling hills
            elevation = (np.sin(nx) + np.sin(ny)) * 0.5 + 1.0 # 0 to 2
            elevation += (np.sin(nx*3.1) + np.sin(ny*2.7)) * 0.25 # details
            
            img[y, x] = elevation
    
    # Normalize to 0-255 uint8
    img_min = np.min(img)
    img_max = np.max(img)
    img_norm = (img - img_min) / (img_max - img_min)
    img_uint8 = (img_norm * 255).astype(np.uint8)
    
    # Create some flat area in the center for the robot to spawn
    center = size // 2
    radius = 30
    flat_height = img_uint8[center, center]
    
    Y, X = np.ogrid[:size, :size]
    dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = dist_from_center <= radius
    
    # Apply flat center with a slight blending
    img_uint8[mask] = flat_height
    
    cv2.imwrite(filename, img_uint8)
    print(f"Generated {filename}")

if __name__ == '__main__':
    # Make sure worlds directory exists
    os.makedirs('../worlds', exist_ok=True)
    generate_heightmap('../worlds/terrain_heightmap.png')
