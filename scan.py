import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw



def scan_pil_image(image, x, y, c, n):
    # 将图片转换为像素访问对象
    pixels = image.load()
    width, height = image.size
    
    # 初始化结果列表
    ret = []
    
    angles = [math.radians(c + i * (360 / n)) for i in range(n)]
    
    for angle in angles:
        # 初始化光线长度
        distance = 0
        
        # 计算光线的步长
        step_x = math.cos(angle)
        step_y = math.sin(angle)
        
        # 从起始点开始，沿着当前方向逐步前进
        while True:
            # 计算新的坐标
            new_x = int(x + distance * step_x)
            new_y = int(y + distance * step_y)
            
            # 检查新坐标是否超出图片边界
            if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                break
            
            # 检查新坐标是否为障碍物（白色）
            if pixels[new_x, new_y] == (255, 255, 255):
                break
            
            # 增加光线长度
            distance += 1
        
        # 记录当前方向的光线长度
        ret.append(distance)
    
    return ret

def invert_black_white(image):
    # 将图片转换为像素访问对象
    pixels = image.load()
    width, height = image.size
    
    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y][:3]  # 获取RGB值
            if r == g == b:  # 如果是灰度值
                pixels[x, y] = (255 - r, 255 - g, 255 - b) + pixels[x, y][3:]
    
    return image

def show(image, x, y, c, ret):
    # 创建绘图对象
    draw = ImageDraw.Draw(image)
    
    # 定义八个方向的角度（0, 45, 90, 135, 180, 225, 270, 315）
    n = len(ret)
    angles = [math.radians(c + i * (360 / n)) for i in range(n)]
    
    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill='green')
    for i, distance in enumerate(ret):
        # 计算打击点的坐标
        hit_x = int(x + distance * math.cos(angles[i]))
        hit_y = int(y + distance * math.sin(angles[i]))
        
        # 绘制打击点
        draw.ellipse((hit_x - 2, hit_y - 2, hit_x + 2, hit_y + 2), fill='red')
        draw.line((x, y, hit_x, hit_y), fill='blue', width=1)
    
    # 显示图片
    # image.show()
    image_np = np.array(image)
    
    # 使用matplotlib显示图像
    plt.imshow(image_np)
    plt.scatter([x], [y], color='blue')  # 标记起始点
    plt.title("Scan Result")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    # 示例调用
    image = Image.open("image.png")  # 替换为你的图片路径
    x, y, c, n = 424, 281, 0, 64  # 起始坐标和朝向角度
    result = scan_pil_image(image, x, y, c, n)
    show(image, x, y, c, result)








