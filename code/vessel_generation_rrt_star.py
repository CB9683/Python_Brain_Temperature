import numpy as np
from code.cube_intersect import cube_intersect
from code.dist_point_to_line_seg import dist_point_to_line_seg

def vessel_generation_rrt_star(vessel_in, prob_map, domain, no_iterations, weight_factor, epsilon):
    no_points = vessel_in.shape[0]
    vessel_out = np.zeros((no_points + 2 * no_iterations, 7))
    vessel_out[:no_points, :] = vessel_in

    prob_map[~domain] = 0
    prob_map[np.isnan(prob_map)] = 0
    prob_map_logical = prob_map > 0
    prob_map_index = np.argwhere(prob_map_logical)
    prob_map_density = (prob_map[prob_map_logical] ** weight_factor) / np.sum(prob_map[prob_map_logical] ** weight_factor)
    prob_map_density_cumsum = np.cumsum(prob_map_density)

    bucket = np.empty(domain.shape, dtype=object)
    for i in range(domain.shape[0]):
        for j in range(domain.shape[1]):
            for k in range(domain.shape[2]):
                bucket[i, j, k] = []

    bucket[int(vessel_out[0, 2]), int(vessel_out[0, 3]), int(vessel_out[0, 4])].append(1)
    for m in range(1, no_points):
        intersections = cube_intersect(vessel_out[m, 2:5], vessel_out[int(vessel_out[m, 6]), 2:5])
        for intersection in intersections:
            i, j, k = np.clip(intersection, [0, 0, 0], np.array(domain.shape) - 1)
            bucket[i, j, k].append(m)
            bucket[i, j, k] = list(set(bucket[i, j, k]))

    check25, check50, check75 = False, False, False

    for n in range(no_iterations):
        idx = np.searchsorted(prob_map_density_cumsum, np.random.rand())
        new_point = prob_map_index[idx] + np.random.rand(3) - 0.5

        bucket_list = []
        range_ = 1
        check = False

        while not check:
            x_lim = [max(0, int(new_point[0]) - range_), min(domain.shape[0] - 1, int(new_point[0]) + range_)]
            y_lim = [max(0, int(new_point[1]) - range_), min(domain.shape[1] - 1, int(new_point[1]) + range_)]
            z_lim = [max(0, int(new_point[2]) - range_), min(domain.shape[2] - 1, int(new_point[2]) + range_)]

            for i in range(x_lim[0], x_lim[1] + 1):
                for j in range(y_lim[0], y_lim[1] + 1):
                    for k in range(z_lim[0], z_lim[1] + 1):
                        bucket_list.extend(bucket[i, j, k])

            if bucket_list:
                check = True
            else:
                range_ += 1
                if range_ > max(domain.shape):
                    bucket_list = list(range(no_points))
                    check = True

        bucket_list = list(set(bucket_list))
        dist = np.zeros((len(bucket_list), 2))
        for i, m in enumerate(bucket_list):
            if m == 1:
                dist[i] = [np.linalg.norm(new_point - vessel_out[0, 2:5]), 0]
            else:
                d, t = dist_point_to_line_seg(vessel_out[m, 2:5], vessel_out[int(vessel_out[m, 6]), 2:5], new_point)
                dist[i] = [d, t]

        min_vertex = bucket_list[np.argmin(dist[:, 0])]
        t = dist[np.argmin(dist[:, 0]), 1]

        point_of_connection = (
            vessel_out[min_vertex, 2:5] if t == 0 else
            vessel_out[int(vessel_out[min_vertex, 6]), 2:5] if t == 1 else
            vessel_out[min_vertex, 2:5] + t * (vessel_out[int(vessel_out[min_vertex, 6]), 2:5] - vessel_out[min_vertex, 2:5])
        )

        if np.linalg.norm(new_point - point_of_connection) > epsilon:
            new_point = point_of_connection + epsilon * (new_point - point_of_connection) / np.linalg.norm(new_point - point_of_connection)

        if t == 0 or t == 1:
            vessel_out[no_points, 0] = no_points
            vessel_out[no_points, 2:5] = new_point
            vessel_out[no_points, 6] = min_vertex if t == 0 else int(vessel_out[min_vertex, 6])
        else:
            vessel_out[no_points, 0] = no_points
            vessel_out[no_points, 2:5] = point_of_connection
            vessel_out[no_points, 6] = min_vertex

            no_points += 1
            vessel_out[no_points, 0] = no_points
            vessel_out[no_points, 2:5] = new_point
            vessel_out[no_points, 6] = no_points - 1

        no_points += 1

        intersections = cube_intersect(vessel_out[no_points - 1, 2:5], vessel_out[int(vessel_out[no_points - 1, 6]), 2:5])
        for intersection in intersections:
            i, j, k = np.clip(intersection, [0, 0, 0], np.array(domain.shape) - 1)
            bucket[i, j, k].append(no_points - 1)
            bucket[i, j, k] = list(set(bucket[i, j, k]))

        if not check25 and n / no_iterations >= 0.25:
            print("25% Complete")
            check25 = True
        elif check25 and not check50 and n / no_iterations >= 0.5:
            print("50% Complete")
            check50 = True
        elif check25 and check50 and not check75 and n / no_iterations >= 0.75:
            print("75% Complete")
            check75 = True
        elif check25 and check50 and check75 and n == no_iterations - 1:
            print("100% Complete")

    vessel_out = vessel_out[vessel_out[:, 0] > 0]
    return vessel_out, prob_map_density_cumsum
