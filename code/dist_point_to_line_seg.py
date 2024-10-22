import numpy as np

def dist_point_to_line_seg(point1, point2, new_point):
    """
    Calculates the shortest distance from new_point to the line segment defined by point1 and point2.
    
    Parameters:
    - point1: numpy array or list of length 3, coordinates of the first point of the segment
    - point2: numpy array or list of length 3, coordinates of the second point of the segment
    - new_point: numpy array or list of length 3, coordinates of the point to check distance from the segment
    
    Returns:
    - d: The shortest distance from new_point to the line segment
    - t: The parameter t indicating the location of the closest point on the segment
    """
    point1 = np.array(point1)
    point2 = np.array(point2)
    new_point = np.array(new_point)
    
    # Vector from point1 to point2
    v = point2 - point1
    # Vector from point1 to new_point
    w = new_point - point1
    
    # Calculate the projection of w onto v
    t = np.dot(w, v) / np.dot(v, v)
    
    # Clamp t to the range [0, 1] to find the closest point on the segment
    t = max(0, min(1, t))
    
    # Find the closest point on the line segment
    closest_point = point1 + t * v
    
    # Distance from new_point to the closest point
    d = np.linalg.norm(new_point - closest_point)
    
    return d, t
