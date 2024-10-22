import numpy as np

def cube_intersect(pt1, pt2):
    """
    Checks whether a line defined by two points intersects cubes in a 3D grid.

    Parameters:
    - pt1: numpy array of shape (3,), coordinates of the first point
    - pt2: numpy array of shape (3,), coordinates of the second point

    Returns:
    - intersections: List of [x, y, z] coordinates of intersected grid cells
    """
    intersections = []
    
    dvec = pt2 - pt1  # Distance vector between the two points
    
    # Creating bounding box for the line segment
    imin = int(np.floor(min(pt1[0], pt2[0])))
    imax = int(np.ceil(max(pt1[0], pt2[0])))
    jmin = int(np.floor(min(pt1[1], pt2[1])))
    jmax = int(np.ceil(max(pt1[1], pt2[1])))
    kmin = int(np.floor(min(pt1[2], pt2[2])))
    kmax = int(np.ceil(max(pt1[2], pt2[2])))

    # Iterate through the bounding box
    for i2 in range(imin, imax + 1):
        for j2 in range(jmin, jmax + 1):
            for k2 in range(kmin, kmax + 1):
                boundbox_min = np.array([i2 - 0.5, j2 - 0.5, k2 - 0.5])
                boundbox_max = np.array([i2 + 0.5, j2 + 0.5, k2 + 0.5])
                intersect = False

                # Check if line crosses the lower x boundary
                if not intersect and dvec[0] != 0:
                    t2 = (boundbox_min[0] - pt1[0]) / dvec[0]
                    if 0 < t2 < 1:
                        y_int = pt1[1] + t2 * dvec[1]
                        z_int = pt1[2] + t2 * dvec[2]
                        if boundbox_min[1] <= y_int <= boundbox_max[1] and boundbox_min[2] <= z_int <= boundbox_max[2]:
                            intersect = True

                # Check if line crosses the lower y boundary
                if not intersect and dvec[1] != 0:
                    t2 = (boundbox_min[1] - pt1[1]) / dvec[1]
                    if 0 < t2 < 1:
                        x_int = pt1[0] + t2 * dvec[0]
                        z_int = pt1[2] + t2 * dvec[2]
                        if boundbox_min[0] <= x_int <= boundbox_max[0] and boundbox_min[2] <= z_int <= boundbox_max[2]:
                            intersect = True

                # Check if line crosses the lower z boundary
                if not intersect and dvec[2] != 0:
                    t2 = (boundbox_min[2] - pt1[2]) / dvec[2]
                    if 0 < t2 < 1:
                        x_int = pt1[0] + t2 * dvec[0]
                        y_int = pt1[1] + t2 * dvec[1]
                        if boundbox_min[0] <= x_int <= boundbox_max[0] and boundbox_min[1] <= y_int <= boundbox_max[1]:
                            intersect = True

                # Check if line crosses the upper x boundary
                if not intersect and dvec[0] != 0:
                    t2 = (boundbox_max[0] - pt1[0]) / dvec[0]
                    if 0 < t2 < 1:
                        y_int = pt1[1] + t2 * dvec[1]
                        z_int = pt1[2] + t2 * dvec[2]
                        if boundbox_min[1] <= y_int <= boundbox_max[1] and boundbox_min[2] <= z_int <= boundbox_max[2]:
                            intersect = True

                # Check if line crosses the upper y boundary
                if not intersect and dvec[1] != 0:
                    t2 = (boundbox_max[1] - pt1[1]) / dvec[1]
                    if 0 < t2 < 1:
                        x_int = pt1[0] + t2 * dvec[0]
                        z_int = pt1[2] + t2 * dvec[2]
                        if boundbox_min[0] <= x_int <= boundbox_max[0] and boundbox_min[2] <= z_int <= boundbox_max[2]:
                            intersect = True

                # Check if line crosses the upper z boundary
                if not intersect and dvec[2] != 0:
                    t2 = (boundbox_max[2] - pt1[2]) / dvec[2]
                    if 0 < t2 < 1:
                        x_int = pt1[0] + t2 * dvec[0]
                        y_int = pt1[1] + t2 * dvec[1]
                        if boundbox_min[0] <= x_int <= boundbox_max[0] and boundbox_min[1] <= y_int <= boundbox_max[1]:
                            intersect = True

                if intersect:
                    intersections.append([i2, j2, k2])

    return intersections
