import numpy as np
from code.dist_point_to_line_seg import dist_point_to_line_seg


# Test case
point1 = np.array([0, 0, 0])
point2 = np.array([1, 1, 1])
new_point = np.array([1, 0, 0])

# Calculate the distance
distance, t = dist_point_to_line_seg(point1, point2, new_point)

print(f"Distance: {distance}")
print(f"t: {t}")
