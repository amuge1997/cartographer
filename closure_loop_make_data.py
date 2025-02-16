
import numpy as np
from slam import Slam, show_map, show_map_gray, show_map_degree, display_maps
from func import get_nonzero_coordinates, get_nonzero_coordinates_with_values
from func import get_rotation_matrix, get_t, add_gaussian_noise, downsample_map
from func import set_points_on_map, set_points_on_map_with_values

def make_data():
    
    slam = Slam()
    slam.load_map("slam_map.npy")
    # show_map(slam.map, is_save=False, is_show=True)

    vec = get_nonzero_coordinates(slam.map)

    # degree = 0
    # tx = 200
    # ty = -100
    # noise_std = 0
    degree = 45
    tx = 200
    ty = -100
    noise_std = 0

    R = get_rotation_matrix( degree)
    t = get_t( tx, ty )

    new_vec = vec @ R.T + t

    np.random.seed(0)
    new_vec = add_gaussian_noise(new_vec, std=noise_std)

    new_slam = Slam()
    set_points_on_map(new_vec, new_slam.map)
    new_slam.save_map("slam_map_new.npy".format(degree, tx, ty))

    # show_map(new_slam.map, is_save=False, is_show=True)

    
def load_data():
    slam = Slam()
    slam.load_map("slam_map.npy")

    new_slam = Slam()
    new_slam.load_map("slam_map_new.npy")

    return slam, new_slam




if __name__ == "__main__":
    make_data()
    slam, new_slam = load_data()
    show_map(new_slam.map, is_save=False, is_show=True)



