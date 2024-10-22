import numpy as np
from code.vessel_generation_rrt_star import vessel_generation_rrt_star

def test_vessel_generation_rrt_star():
    # Define domain: a 10x10x10 cube
    domain_size = (10, 10, 10)
    domain = np.ones(domain_size, dtype=bool)
    
    # Define a probability map: higher probability in the center
    prob_map = np.zeros(domain_size)
    center = np.array([domain_size[0] // 2, domain_size[1] // 2, domain_size[2] // 2])
    prob_map[center[0]-1:center[0]+2, center[1]-1:center[1]+2, center[2]-1:center[2]+2] = 1.0
    
    # Initial vessel tree: a single point in the center of the domain
    vessel_in = np.array([[1, 0, center[0], center[1], center[2], 0, 0]])

    # Set parameters for the function
    no_iterations = 10
    weight_factor = 1.0
    epsilon = 1.0

    # Run the vessel generation function
    vessel_out, prob_map_density_cumsum = vessel_generation_rrt_star(vessel_in, prob_map, domain, no_iterations, weight_factor, epsilon)

    # Print the results
    print("Vessel Out:")
    print(vessel_out)
    print("Probability Map Cumulative Sum:")
    print(prob_map_density_cumsum)

# Run the test
if __name__ == "__main__":
    test_vessel_generation_rrt_star()
