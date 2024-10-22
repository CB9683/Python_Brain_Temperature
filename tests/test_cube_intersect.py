import numpy as np
from code.cube_intersect import cube_intersect

# Test case
pt1 = np.array([0, 0, 0])
pt2 = np.array([2, 2, 2])

intersections = cube_intersect(pt1, pt2)
print("Intersections:", intersections)
