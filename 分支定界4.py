import numpy as np
from func import get_nonzero_coordinates, get_nonzero_coordinates_with_values
from func import get_rotation_matrix, get_t, add_gaussian_noise, downsample_map
from func import set_points_on_map, set_points_on_map_with_values, pad_map_with_zeros
from slam import display_maps



layer_nums = 4


rotation_spilt_num = 100

class Node:
        
    errors = None
    best_x_translation = None
    best_y_translation = None
    best_rotation = None
    def __init__(self, map_A_down, map_B_down, layer_index, x_index, y_index, rotation, rotation_end, src_raduis):
        self.map_A_down = map_A_down
        self.map_B_down = map_B_down
        self.layer_index = layer_index
        self.x_index = x_index
        self.y_index = y_index
        self.rotation = rotation
        self.rotation_end = rotation_end

        vec_b, value_b = get_nonzero_coordinates_with_values(map_B_down[layer_index])
        self.src_raduis = src_raduis
        raduis = src_raduis // 2**(layer_nums-layer_index-1)
        self.score = evaluate_transformation_1map_1vec(
            map_A_down[layer_index],
            vec_b-raduis, value_b, 
            raduis, 
            rotation, 
            x_index, 
            y_index
        )
        
        if Node.errors is None:
            if self.is_last_layer():
                Node.errors = self.score
                Node.best_x_translation = self.x_index
                Node.best_y_translation = self.y_index
                Node.best_rotation = self.rotation
                print("Best error:", Node.errors)
                print("Best rotation:", Node.best_rotation)
                print("Best x_translation:", Node.best_x_translation)
                print("Best y_translation:", Node.best_y_translation)
                print(raduis)
                print()
            else:
                self.run_with_no_rotation()
        else:
            if self.score > Node.errors:
                return
            if self.is_last_layer():
                if self.score < Node.errors:
                    Node.errors = self.score
                    Node.best_x_translation = self.x_index
                    Node.best_y_translation = self.y_index
                    Node.best_rotation = self.rotation
                    print("Best error:", Node.errors)
                    print("Best rotation:", Node.best_rotation)
                    print("Best x_translation:", Node.best_x_translation)
                    print("Best y_translation:", Node.best_y_translation)
                    print()
            else:
                self.run_with_no_rotation()

    def is_last_layer(self):
        return self.layer_index == layer_nums - 1

    def run_with_no_rotation(self):
        for xi in [0, 1]:
            for yi in [0, 1]:
                next_x_index = self.x_index*2 + xi
                next_y_index = self.y_index*2 + yi
                Node(
                    self.map_A_down, 
                    self.map_B_down, 
                    self.layer_index+1, 
                    next_x_index, 
                    next_y_index, 
                    self.rotation, 
                    self.rotation_end,
                    self.src_raduis
                )

def evaluate_transformation_2vec(vec_a, value_a, vec_b, value_b, raduis, rotation, x_translation, y_translation):
    """
    评估两个地图经过旋转和平移后的匹配误差。
    这里使用简单的重叠度来作为评估标准，可以使用ICP等其他方法。
    """
    degree = rotation
    tx = x_translation
    ty = y_translation

    R = get_rotation_matrix( degree)
    t = get_t( tx, ty )
    
    vec_b2a = vec_b @ R.T + t
    
    map_A = np.zeros((2*raduis, 2*raduis))
    map_A = set_points_on_map_with_values(vec_a+raduis, map_A, value_a)
    
    map_A2B = np.zeros_like(map_A)
    map_A2B = set_points_on_map_with_values(vec_b2a+raduis, map_A2B, value_b)
    
    corr = np.sum(map_A * map_A2B)
    corr = - corr

    # print("corr:", corr)
    
    return corr

def evaluate_transformation_1map_1vec(map_A, vec_b, value_b, raduis, rotation, x_translation, y_translation):
    """
    评估两个地图经过旋转和平移后的匹配误差。
    这里使用简单的重叠度来作为评估标准，可以使用ICP等其他方法。
    """
    degree = rotation
    tx = x_translation
    ty = y_translation

    R = get_rotation_matrix( degree)
    t = get_t( tx, ty )
    
    vec_b2a = vec_b @ R.T + t
    
    map_A2B = np.zeros_like(map_A)
    map_A2B = set_points_on_map_with_values(vec_b2a+raduis, map_A2B, value_b)
    
    corr = np.sum(map_A * map_A2B)
    corr = - corr
    
    return corr


def centralize_and_find_power(coords):
    """
    中心化坐标并计算满足条件的2的幂次。
    
    参数:
    coords (numpy.ndarray): 输入的 nx2 坐标数组。
    
    返回:
    tuple: (中心化后的数组v, 计算得到的值n)。
    """
    # Step 1: 中心化处理
    center = np.mean(coords, axis=0)
    vec_mean = coords - center
    
    # Step 2: 找到 x 最小值、x 最大值、y 最小值、y 最大值
    x_min, y_min = np.min(vec_mean, axis=0)
    x_max, y_max = np.max(vec_mean, axis=0)
    
    # Step 3: 找到这四个值中绝对值最大的值 m
    m = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    
    # Step 4: 计算 2 的幂次 n，使得 n 大于 m
    radius = 1
    while radius < m:
        radius *= 2
    
    return vec_mean, center, radius


def debug_show(vec_b_mean, value_b, raduis, map_A_down, map_B_down):
    degree = Node.best_rotation
    tx = Node.best_x_translation
    ty = Node.best_y_translation
    R = get_rotation_matrix(degree)
    t = get_t( tx, ty )
    new_vec = vec_b_mean @ R.T + t
    new_slam_map_down_to_src_slam = np.zeros((2*raduis, 2*raduis))
    set_points_on_map_with_values(new_vec+raduis, new_slam_map_down_to_src_slam, value_b)
    display_maps([map_A_down[-1], map_B_down[-1], new_slam_map_down_to_src_slam])


def debug_show2(map_A, map_B, R, t):
    vec_b, value_b = get_nonzero_coordinates_with_values(map_B)
    new_vec = vec_b @ R.T + t
    new_slam_map_down_to_src_slam = np.zeros_like(map_A)
    set_points_on_map_with_values(new_vec, new_slam_map_down_to_src_slam, value_b)
    display_maps([map_A, map_B, new_slam_map_down_to_src_slam], ["世界坐标系地图", "雷达建图", "分支定界匹配结果"])



def branch_and_bound(vec_a, value_a, vec_b, value_b):
    
    vec_a_mean, vec_a_center, raduis = centralize_and_find_power(vec_a)
    vec_b_mean, vec_b_center, _ = centralize_and_find_power(vec_b)

    # print("n:", raduis)
    # print(vec_a_mean.min(axis=0), vec_a_mean.max(axis=0))
    
    map_A = np.zeros((2*raduis, 2*raduis))
    set_points_on_map_with_values(vec_a_mean+raduis, map_A, value_a)
    map_B = np.zeros((2*raduis, 2*raduis))
    set_points_on_map_with_values(vec_b_mean+raduis, map_B, value_b)

    map_A_down = [downsample_map(map_A, 2**i) for i in range(0, layer_nums)][::-1]
    map_B_down = [downsample_map(map_B, 2**i) for i in range(0, layer_nums)][::-1]
    # display_maps([map_A, map_B])
    pi = 0
    # display_maps([map_A_down[pi], map_B_down[pi]])
    # print(map_A_down[pi].shape)

    # evaluate_transformation(vec_a_mean, value_a, vec_b_mean, value_b, raduis, 0, 0, 0)

    x_range = [-map_A_down[0].shape[1], map_A_down[0].shape[1]]
    y_range = [-map_A_down[0].shape[0], map_A_down[0].shape[0]]
    d_rotation = 360 / rotation_spilt_num
    x_list = list(range(x_range[0], x_range[1]))
    y_list = list(range(y_range[0], y_range[1]))
    x_list.remove(0)
    y_list.remove(0)
    x_list = [0] + x_list
    y_list = [0] + y_list
    for ri in np.linspace(0, 360, rotation_spilt_num+1):
        for xi in x_list:
            for yi in y_list:
                Node(
                    map_A_down, 
                    map_B_down, 
                    0, 
                    xi, 
                    yi, 
                    ri, 
                    ri+d_rotation,
                    raduis
                )
    
    # debug_show(vec_b_mean, value_b, raduis, map_A_down, map_B_down)
    
    R_ = get_rotation_matrix(Node.best_rotation)
    t_ = get_t(Node.best_x_translation, Node.best_y_translation)
    a_ = vec_a_center
    b_ = vec_b_center
    R = R_
    t = t_ + a_ - R_ @ b_

    return R, t

def run():
    from closure_loop_make_data import load_data
    slam, new_slam = load_data()

    # map_A_src = pad_map_with_zeros(slam.map, 56)
    # map_B_src = pad_map_with_zeros(new_slam.map, 56)
    map_A_src = slam.map
    map_B_src = new_slam.map

    map_A = downsample_map(map_A_src, 4)
    map_B = downsample_map(map_B_src, 4)
    
    vec_a, value_a = get_nonzero_coordinates_with_values(map_A)
    vec_b, value_b = get_nonzero_coordinates_with_values(map_B)
    
    R, t = branch_and_bound(vec_a, value_a, vec_b, value_b)
    print(R, t)
    debug_show2(map_A_src, map_B_src, R, t*4)
    # display_maps([map_A, map_B])

if __name__ == '__main__':
    run()














