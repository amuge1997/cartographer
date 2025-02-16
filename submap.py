
from make_data import per_dis, src_world_x,src_world_y,src_c
import numpy as np
from scan import scan_pil_image, show, invert_black_white
from PIL import Image


OBS = np.array(1, dtype=int)


def transform_pose(x1, y1, c1, x2, y2, c2):
    # 平移坐标
    dx = x2 - x1
    dy = y2 - y1
    
    # 旋转变换：绕点1的角度c1旋转
    x_rel = dx * np.cos(-c1) - dy * np.sin(-c1)
    y_rel = dx * np.sin(-c1) + dy * np.cos(-c1)
    
    # 计算朝向相对变化
    c_rel = c2 - c1
    
    # 角度归一化到 [-pi, pi]
    c_rel = (c_rel + np.pi) % (2 * np.pi) - np.pi
    
    c_rel = np.degrees(c_rel)
    return x_rel, y_rel, c_rel

def get_rotation_angle(matrix):
    # 提取矩阵元素
    r11 = matrix[0, 0]
    r21 = matrix[1, 0]
    
    # 计算旋转角度（弧度）
    angle_rad = np.arctan2(r21, r11)
    
    # 如果需要，转换为角度
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def estimate_rigid_transform(a, b):
    """
    Estimate the rigid transform (R, t) such that a = R * b + t.

    Parameters:
    a: np.ndarray of shape (N, D), target points.
    b: np.ndarray of shape (N, D), source points.

    Returns:
    R: np.ndarray of shape (D, D), rotation matrix.
    t: np.ndarray of shape (D,), translation vector.
    """
    # Ensure input is numpy arrays
    a = np.asarray(a)
    b = np.asarray(b)
    
    # Compute centroids
    centroid_a = np.mean(a, axis=0)
    centroid_b = np.mean(b, axis=0)
    
    # Center the points
    a_centered = a - centroid_a
    b_centered = b - centroid_b
    
    # Compute the cross-covariance matrix
    H = b_centered.T @ a_centered
    
    # Singular Value Decomposition
    U, _, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    # Ensure R is a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation vector
    t = centroid_a - R @ centroid_b

    a_estimate = b @ R + t

    def mserror(a, b):
        return np.mean((a - b) ** 2)
    
    error = mserror(a_estimate, b)
    
    return R, t, error


def rigid_transform_2d_gradient_descent(a, b, learning_rate=0.0001, max_iter=10, tolerance=1e-6):
    """
    使用梯度下降法计算二维点集的刚性变换（R 和 t），使得 b 对齐到 a，即 a = R*b + t。
    
    参数：
        a (ndarray): 目标点集，形状为 (N, 2)。
        b (ndarray): 移动点集，形状为 (N, 2)。
        learning_rate (float): 梯度下降的学习率。
        max_iter (int): 最大迭代次数。
        tolerance (float): 收敛判定的误差阈值。
    
    返回：
        R (ndarray): 旋转矩阵，形状为 (2, 2)。
        t (ndarray): 平移向量，形状为 (2,)。
    """
    # 初始化 R 和 t
    R = np.eye(2)  # 初始旋转矩阵
    t = np.zeros(2)  # 初始平移向量

    for iteration in range(max_iter):
        # 对 b 应用当前的变换
        b_transformed = (R @ b.T).T + t

        # 计算误差
        error = a - b_transformed
        loss = np.sum(np.linalg.norm(error, axis=1) ** 2)
        
        print()
        print(b_transformed)
        print(loss)

        # 计算梯度
        grad_t = -2 * np.sum(error, axis=0)  # 平移梯度
        grad_R = -2 * np.sum([np.outer(err, b[i]) for i, err in enumerate(error)], axis=0)  # 旋转梯度

        # 更新 R 和 t
        R_grad_step = R - learning_rate * grad_R  # 梯度下降更新 R
        U, _, Vt = np.linalg.svd(R_grad_step)  # 保持正交性
        R = U @ Vt
        t -= learning_rate * grad_t

        # 打印中间状态
        if iteration % 100 == 0 or iteration == max_iter - 1:
            print(f"Iter {iteration}: Loss = {loss:.6f}")

        # 检查收敛
        if np.linalg.norm(grad_t) < tolerance and np.linalg.norm(grad_R) < tolerance:
            break

    return R, t

def find_nearest_occupied_cell(slam_map, map_x, map_y, n, th):
    """
    统计以 (x, y) 为中心，范围为 n 的区域内值为 1 的栅格数量。
    如果数量大于 2，则返回这些栅格中距离 (x, y) 最近的一个栅格的坐标；否则返回 None。

    参数:
    slam_map (ndarray): 二维数组，表示地图。
    x (int): 中心点的 x 坐标。
    y (int): 中心点的 y 坐标。
    n (int): 范围大小。
    th (int): 阈值。

    返回:
    tuple 或 None: 如果数量大于 2，返回距离 (x, y) 最近的一个栅格的坐标；否则返回 None。

    提示：
    实现一个函数。输入一个二维数组slam_map。
    输入一个两个索引x,y，slam_map中的x,y坐标对应二维数组的slam_map[y,x]。
    输入一个数字n。
    统计以x,y为中心以n为范围的slam_map为1的栅格数量，
    如果数量大于2，那么找到范围n内值为1的距离x,y最近的栅格，返回其坐标，如果数量小于等于2，那么返回None
    """
    height, width = slam_map.shape
    occupied_cells = []

    # 遍历以 (x, y) 为中心，范围为 n 的区域
    for dy in range(-n, n + 1):
        for dx in range(-n, n + 1):
            # 计算当前检查的坐标
            current_x = map_x + dx
            current_y = map_y + dy

            # 检查坐标是否在地图范围内
            if 0 <= current_x < width and 0 <= current_y < height:
                # print('obs', current_y, current_x, slam_map[current_y, current_x])
                if slam_map[current_y, current_x] == OBS:
                    occupied_cells.append((current_x, current_y))

    # 统计值为 1 的栅格数量
    num_occupied_cells = len(occupied_cells)

    # 根据数量返回结果
    if num_occupied_cells >= th:
        # 计算每个栅格到中心点 (x, y) 的欧几里得距离
        distances = [(np.sqrt((cell[0] - map_x)**2 + (cell[1] - map_y)**2), cell) for cell in occupied_cells]
        # 找到距离最近的栅格
        nearest_cell = min(distances, key=lambda x: x[0])[1]
        return nearest_cell
    else:
        return None

def lidar_frame_to_distance_xy(world_x, world_y, c, lis):
    """
    将激光雷达的扫描结果转换为坐标点。

    参数:
    x (float): 起始坐标的x值
    y (float): 起始坐标的y值
    c (float): 朝向角度（以度为单位）
    lis (list): 激光雷达扫描结果，长度为n，每个元素表示对应角度的扫描距离

    返回:
    list: 转换后的坐标点列表，每个元素是一个(x, y)元组
    """
    n = len(lis)
    angle_increment = 2 * np.pi / n  # 每个扫描点的角度增量
    c_rad = np.radians(c)  # 将朝向角度从度转换为弧度
    coordinates = []

    for i, distance in enumerate(lis):
        if distance > 0:  # 只处理有效距离
            angle = c_rad + i * angle_increment  # 计算当前扫描点的绝对角度
            x_end = world_x + distance * np.cos(angle)
            y_end = world_y + distance * np.sin(angle)
            coordinates.append((x_end, y_end))

    return coordinates


def distance_xy_to_map_grid_xy(x,y, distance_per_pixel):
    return int(x / distance_per_pixel), int(y / distance_per_pixel)

def map_grid_xy_add_offset(x,y,cx,cy):
    return x + cx, y + cy

def map_xy_offset_reverse(x,y,cx,cy):
    return x - cx, y - cy

def show_map(map, is_save=False, is_show=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(['black', 'white', 'yellow', 'red'])
    
    plt.imshow(map, cmap=cmap)
    plt.axis('off')
    if is_save:
        plt.savefig("map.png")
    
    if is_show:
        plt.show()

def show_map_gray(map, is_save=False, is_show=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    plt.imshow(map, cmap='gray')
    plt.axis('off')
    if is_save:
        plt.savefig("map.png")
    
    if is_show:
        plt.show()

def show_map_degree(map, is_save=False, is_show=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap(['black', 'gray', 'green', 'blue', 'yellow', 'red'])
    plt.imshow(map, cmap=cmap)
    plt.axis('off')
    if is_save:
        plt.savefig("map.png")
    
    if is_show:
        plt.show()


def display_maps(maps):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    cmap = ListedColormap(['black', 'gray', 'green', 'blue', 'yellow', 'red'])
    """
    显示传入的地图列表，每个地图使用imshow显示。
    
    参数:
    maps (list): 包含多个地图的列表，每个地图是一个二维numpy数组。
    """
    # 创建一个子图布局，假设最多显示 2 行，每行显示 3 个图
    rows = (len(maps) + 2) // 3  # 自动计算行数，最多每行3个子图
    cols = min(3, len(maps))     # 每行最多显示 3 个图
    
    fig, axes = plt.subplots(rows, cols, figsize=(13, 5))
    
    # 如果只有一个子图的情况
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])  # 将axes转为二维数组

    # 遍历每个地图，显示它们
    for i, ax in enumerate(axes.flat):
        if i < len(maps):
            ax.imshow(maps[i], cmap=cmap)  # 显示地图，假设是灰度图像
            ax.set_title(f'Map {i + 1}')
            ax.axis('off')  # 不显示坐标轴
        else:
            ax.axis('off')  # 如果没有地图，关闭坐标轴显示
    
    # 自动调整子图间的布局
    plt.tight_layout()
    plt.show()

class SubmapManger:
    def __init__(self):
        
        self.max_frames_num = 5
        self.frames_num = 0

        self.submaps = [Submap]
        self.submap_width = 400
        self.submap_height = 400
        self.submap_center_x = self.submap_width // 2
        self.submap_center_y = self.submap_height // 2
        self.submap_distance_per_pixel = 5
    
    def add_submap(self):
        self.submaps.append(Submap())
    
    def show_merge_submap(self):
        from func import get_nonzero_coordinates_with_values, get_rotation_matrix, get_t, set_points_on_map_with_values
        mainmap_width = 400
        mainmap_height = 400
        mainmap_center_x = mainmap_width // 2
        mainmap_center_y = mainmap_height // 2
        mainmap = np.zeros((mainmap_height, mainmap_width), dtype=int)
        print("合并子图显示")
        print(len(self.submaps))

        
        submap0 = self.submaps[0]
        # x0, y0, c0 = submap0.get_center_relate_world_xyc()
        xys0, value0 = get_nonzero_coordinates_with_values(submap0.map)
        xys0[:, 0] = xys0[:,0] - submap0.center_x
        xys0[:, 1] = xys0[:,1] - submap0.center_y
        # R,y change
        xys0[:, 0] = xys0[:,0] + mainmap_center_x
        xys0[:, 1] = xys0[:,1] + mainmap_center_y
        set_points_on_map_with_values(xys0, mainmap, value0)

        for submap_iter in self.submaps:
            x, y, c = submap_iter.get_center_relate_world_xyc_or_relate_last_submap()
            print(x, y, c)
        print()

        for i in range(1, len(self.submaps)):
            
            x_iter, y_iter, c_iter = 0,0,0
            for submap_iter in self.submaps[1:i+1]:
                x, y, c = submap_iter.get_center_relate_world_xyc_or_relate_last_submap()
                x_iter += x // submap_iter.distance_per_pixel
                y_iter += y // submap_iter.distance_per_pixel
                c_iter += c
            print(x_iter, y_iter, c_iter)
            submap2 = self.submaps[i]
            xys2, value2 = get_nonzero_coordinates_with_values(submap2.map)
            xys2[:, 0] = xys2[:,0] - submap2.center_x
            xys2[:, 1] = xys2[:,1] - submap2.center_y
            R = get_rotation_matrix( c_iter )
            t = get_t( x_iter, y_iter )
            xys2 = xys2 @ R.T + t
            xys2[:, 0] = xys2[:,0] + mainmap_center_x
            xys2[:, 1] = xys2[:,1] + mainmap_center_y
            set_points_on_map_with_values(xys2, mainmap, value2)
        
        display_maps([mainmap])

    def get_last_submap(self):
        return self.submaps[-1]
    
    def set_last_submap_center_relate_world_xyc(self, x, y, c):
        self.submaps[-1].set_center_relate_world_xyc(x, y, c)
        # print(x,y,c)
        # exit()
    
    def set_last_submap_point(self, lidar_relate_submap_center_distance_x, lidar_relate_submap_center_distance_y, lidar_relate_submap_center_c, scan_frame, is_posture_optimize=False):
        self.frames_num = self.frames_num + 1
        print("帧数", self.frames_num)
        self.submaps[-1].set_point(lidar_relate_submap_center_distance_x, lidar_relate_submap_center_distance_y, lidar_relate_submap_center_c, scan_frame, is_posture_optimize)

        if self.frames_num % self.max_frames_num == 0:
            return True
        else:
            return False
    def is_last_submap_reset(self):
        return self.submaps[-1].is_reset()

    def submaps_length(self):
        return len(self.submaps)
    
    def is_empty(self):
        return len(self.submaps) == 0
    
    def reset(self):
        self.submaps = []
        self.frames_num = 0
        self.add_submap()


class Submap:
    def __init__(self):
        
        self.width = 400
        self.height = 400
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        self.map = np.zeros((self.height, self.width), dtype=int)
        self.distance_per_pixel = 5
        
        # 子地图相对世界坐标
        self.submap_relate_world_x = None
        self.submap_relate_world_y = None
        self.submap_relate_world_c = None

        self.flag_reset = True

    def is_reset(self):
        return self.flag_reset

    def set_center_relate_world_xyc(self, x, y, c):
        self.flag_reset = False
        self.submap_relate_world_x = x
        self.submap_relate_world_y = y
        self.submap_relate_world_c = c
    
    def get_center_relate_world_xyc_or_relate_last_submap(self):
        return self.submap_relate_world_x, self.submap_relate_world_y, self.submap_relate_world_c
    
    def change_grid_yx_to_world_yx(self, x, y):
        from scipy.spatial.transform import Rotation as R
        rot_matrix = R.from_euler('z', self.submap_relate_world_c).as_matrix()[:2, :2]
        xy = rot_matrix @ np.array([x, y]) + np.array([self.submap_relate_world_x, self.submap_relate_world_y])
        return xy[0] * self.distance_per_pixel, xy[1] * self.distance_per_pixel

    def save_map(self, file_name):
        np.save(file_name, self.map)
    
    def load_map(self, file_name):
        self.map = np.load(file_name)
        if self.map.shape != (self.height, self.width):
            raise Exception("地图大小不匹配")
    
    def set_point(self, lidar_relate_submap_center_distance_x, lidar_relate_submap_center_distance_y, lidar_relate_submap_center_c, scan_frame, is_posture_optimize=False):
        # if is_posture_optimize:
        #     R, t = self.posture_optimize(lidar_relate_world_x, lidar_relate_world_y, lidar_relate_world_c, scan_frame)
            
        #     print('t', t)
            
        #     if t is not None:
        #         c_est = get_rotation_angle(R)
                
        #         print('c est', c_est)
        #         lidar_relate_world_c += c_est

        #         lidar_relate_world_x += t[0]
        #         lidar_relate_world_y += t[1]
        
        map_x, map_y = map_grid_xy_add_offset(
            *distance_xy_to_map_grid_xy(
                lidar_relate_submap_center_distance_x, 
                lidar_relate_submap_center_distance_y, 
                self.distance_per_pixel
            ),
            self.center_x, 
            self.center_y)
        if map_x < 0 or map_y < 0 or map_x >= self.width or map_y >= self.height:
            raise Exception("超出地图范围")
        # self.map[map_y, map_x] = OBS
        
        lidar_relate_submap_center_distance_xys = lidar_frame_to_distance_xy(lidar_relate_submap_center_distance_x, lidar_relate_submap_center_distance_y, lidar_relate_submap_center_c, scan_frame)
        # print(2, relate_world_xys)
        for lidar_relate_submap_center_xy in lidar_relate_submap_center_distance_xys:
            map_x, map_y = map_grid_xy_add_offset(
                *distance_xy_to_map_grid_xy(
                    lidar_relate_submap_center_xy[0],
                    lidar_relate_submap_center_xy[1],
                    self.distance_per_pixel
                ),
                self.center_x, 
                self.center_y)
            
            if map_x < 0 or map_y < 0 or map_x >= self.width or map_y >= self.height:
                raise Exception("超出地图范围")
            self.map[map_y, map_x] = OBS
        return True
    
    def posture_optimize(self, relate_src_world_x, relate_src_world_y, c, scan_frame, iter_max=10):
        print('posture_optimization()')
        # print(1, relate_world_xys)

        # 临近半径
        nearest_occupied_radius = 3
        # 临近数量阈值
        nearest_occupied_th = 2
        # 匹配点数量阈值
        valid_points_th = 3

        
        R_sum = np.eye(2)  # 初始旋转矩阵
        t_sum = np.zeros(2)  # 初始平移向量

        for i in range(iter_max):
            
            relate_world_xys = lidar_frame_to_distance_xy(relate_src_world_x, relate_src_world_y, c, scan_frame)
            relate_world_xys = [list(xys) for xys in relate_world_xys]
            target_point = []
            scan_point = []
            none_sum = 0
            for relate_world_xy in relate_world_xys:
                map_x_no_offset, map_y_no_offset = distance_xy_to_map_grid_xy(relate_world_xy[0], relate_world_xy[1],self.distance_per_pixel)
                map_x, map_y = map_grid_xy_add_offset(map_x_no_offset, map_y_no_offset, self.center_x, self.center_y)
                
                occ_cell = find_nearest_occupied_cell(self.map, map_x, map_y, nearest_occupied_radius, nearest_occupied_th)
                print((map_x, map_y), occ_cell)
                if occ_cell is not None:
                    target_point += [occ_cell]
                    scan_point += [(map_x, map_y)]
                else:
                    none_sum += 1

            valid_points = len(relate_world_xys) - none_sum
            print(f"匹配: {valid_points}/{len(relate_world_xys)}")
            if valid_points < valid_points_th:
                print()
                return None, None

            target_point = np.array(target_point)
            scan_point = np.array(scan_point)
            R, t, error = estimate_rigid_transform(target_point,scan_point)

            R_sum = R @ R_sum
            t_sum += t

            print(f"{i} error {round(error, 3)}")

            old = np.array([relate_src_world_x, relate_src_world_y])
            new = old @ R + t
            relate_src_world_x = new[0]
            relate_src_world_y = new[1]

            for i in range(len(relate_world_xys)):
                old = np.array(relate_world_xys[i])
                new = old @ R + t
                relate_world_xys[i][0] = new[0]
                relate_world_xys[i][1] = new[1]

        print()
        
        return R, t
    def posture_optimize_no_iter(self, relate_src_world_x, relate_src_world_y, c, scan_frame):
        print('posture_optimization()')
        relate_world_xys = lidar_frame_to_distance_xy(relate_src_world_x, relate_src_world_y, c, scan_frame)
        # print(1, relate_world_xys)

        # 临近半径
        nearest_occupied_radius = 3
        # 临近数量阈值
        nearest_occupied_th = 2
        # 匹配点数量阈值
        valid_points_th = 3

        a = []
        b = []
        none_sum = 0
        for relate_world_xy in relate_world_xys:
            map_x_no_offset, map_y_no_offset = distance_xy_to_map_grid_xy(relate_world_xy[0], relate_world_xy[1],self.distance_per_pixel)
            map_x, map_y = map_grid_xy_add_offset(map_x_no_offset, map_y_no_offset, self.center_x, self.center_y)
            
            occ_cell = find_nearest_occupied_cell(self.map, map_x, map_y, nearest_occupied_radius, nearest_occupied_th)
            print((map_x, map_y), occ_cell)
            if occ_cell is not None:
                a += [occ_cell]
                b += [(map_x, map_y)]
            else:
                none_sum += 1

        valid_points = len(relate_world_xys) - none_sum
        print(f"匹配: {valid_points}/{len(relate_world_xys)}")
        if valid_points < valid_points_th:
            print()
            return None, None

        a = np.array(a)
        b = np.array(b)
        R, t, error = estimate_rigid_transform(a,b)
        print()
        
        return R, t
    def reset(self):
        self.flag_reset = True
        self.map = np.zeros((self.width, self.height), dtype=int)
        self.submap_relate_world_x = None
        self.submap_relate_world_y = None
        self.submap_relate_world_c = None
    




