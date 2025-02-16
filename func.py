
import numpy as np


def pad_map_with_zeros(map_array, n):
    """
    对输入的二维map在两个维度上对称地填充n个0。

    参数:
    map_array (numpy.ndarray): 输入的二维数组（地图）。
    n (int): 填充的0的数量。

    返回:
    numpy.ndarray: 填充后的二维数组。
    """
    # 使用 np.pad 在两个维度上对称填充0
    padded_map = np.pad(map_array, pad_width=((n, n), (n, n)), mode='constant', constant_values=0)
    return padded_map

def get_rotation_matrix(angle):
    """
    返回一个二维旋转矩阵。

    参数:
    angle (float): 旋转角度。

    返回:
    numpy.ndarray: 2x2 旋转矩阵。
    """
    angle = np.radians(angle)  # 将角度转换为弧度

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    # 构建旋转矩阵
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta,  cos_theta]
    ])
    return rotation_matrix

def get_t(x, y):
    return np.array([x, y])

def get_nonzero_coordinates(map):
    """
    返回二维数组中所有非零元素的坐标，按 (col, row) 顺序。

    参数:
    map (numpy.ndarray): 输入的二维数组，dtype 为 int。

    返回:
    numpy.ndarray: 形状为 (n, 2) 的数组，每行存储非零元素的 (col, row) 坐标。
    """
    # 使用 np.argwhere 找到所有非零元素的位置
    coordinates = np.argwhere(map != 0)
    
    # 交换 (row, col) 为 (col, row)
    coordinates = coordinates[:, [1, 0]]
    
    return coordinates

def get_nonzero_coordinates_with_values(map):
    """
    返回二维数组中所有非零元素的坐标，按 (col, row) 顺序。

    参数:
    map (numpy.ndarray): 输入的二维数组，dtype 为 int。

    返回:
    numpy.ndarray: 形状为 (n, 2) 的数组，每行存储非零元素的 (col, row) 坐标。
    """
    # 使用 np.argwhere 找到所有非零元素的位置
    coordinates = np.argwhere(map != 0)
    values = map[coordinates[:, 0], coordinates[:, 1]]
    
    # 交换 (row, col) 为 (col, row)
    coordinates = coordinates[:, [1, 0]]
    
    # (x,y),v
    return coordinates, values

def set_points_on_map(points, map_array):
    """
    将输入的二维坐标数组对应的点在map上置为1。

    参数:
    points (np.ndarray): 二维坐标数组，形状为 (n, 2)，每行是 (x, y)。
    map_array (np.ndarray): 二维地图数组，初始为全0，形状为 (h, w)。

    返回:
    np.ndarray: 修改后的二维地图数组。
    """
    for point in points:
        x, y = point
        x = int(x)
        y = int(y)
        # 检查是否在地图范围内，防止越界
        if 0 <= x < map_array.shape[1] and 0 <= y < map_array.shape[0]:
            map_array[y, x] = 1  # 注意：y 是行索引，x 是列索引

    return map_array

def set_points_on_map_with_values(points, map_array, values):
    """
    将输入的二维坐标数组对应的点在map上置为1。

    参数:
    points (np.ndarray): 二维坐标数组，形状为 (n, 2)，每行是 (x, y)。
    map_array (np.ndarray): 二维地图数组，初始为全0，形状为 (h, w)。

    返回:
    np.ndarray: 修改后的二维地图数组。
    """
    for i in range(points.shape[0]):
        point = points[i]
        value = values[i]
        x, y = point
        x = int(x)
        y = int(y)
        # 检查是否在地图范围内，防止越界
        if 0 <= x < map_array.shape[1] and 0 <= y < map_array.shape[0]:
            map_array[y, x] = value  # 注意：y 是行索引，x 是列索引

    return map_array

def add_gaussian_noise(points, mean=0, std=1):
    """
    给二维坐标数组添加高斯噪声。

    参数:
    points (np.ndarray): 原始二维坐标数组，形状为 (n, 2)，每行是 (x, y)。
    mean (float): 高斯噪声的均值，默认是0。
    std (float): 高斯噪声的标准差，默认是1。

    返回:
    np.ndarray: 添加噪声后的二维坐标数组，形状为 (n, 2)。
    """
    # 生成与points相同形状的高斯噪声
    noise = np.random.normal(mean, std, points.shape)
    
    # 将噪声添加到原始数据上
    noisy_points = points + noise
    
    return noisy_points


def downsample_map(map_array, n):
    """
    对二维地图进行下采样，按n倍缩小，范围内的值之和写入新地图的对应位置。

    参数:
    map_array (np.ndarray): 输入的二维地图数组，形状为 (height, width)。
    n (int): 下采样的缩小倍数。

    返回:
    np.ndarray: 下采样后的新地图。
    """
    if n == 1:
        return map_array.copy()

    # 获取原始地图的高度和宽度
    height, width = map_array.shape
    
    # 计算新地图的尺寸
    new_height = height // n
    new_width = width // n
    
    # 创建一个新的空地图用于存放结果
    downsampled_map = np.zeros((new_height, new_width), dtype=int)
    
    # 遍历原地图的每个n x n区域
    for i in range(new_height):
        for j in range(new_width):
            # 计算每个n x n区域的起始位置
            row_start = i * n
            row_end = row_start + n
            col_start = j * n
            col_end = col_start + n
            
            # 对区域内的所有值求和
            region_sum = np.sum(map_array[row_start:row_end, col_start:col_end])
            
            # 将区域内的和写入新地图
            downsampled_map[i, j] = region_sum
            
    return downsampled_map









