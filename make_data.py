import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scan import scan_pil_image, show, invert_black_white

per_dis = 10

src_world_x,src_world_y,src_c = 420, 50, 0

if __name__ == "__main__":
    image = Image.open("image.png")
    x, y, c, n = src_world_x, src_world_y, src_c, 8  # 起始坐标和朝向角度
    scan_data = []


    draw_image = image.copy()
    draw_image = invert_black_white(draw_image)
    for i in range(0, 200, per_dis):
        frame = scan_pil_image(image, x, y+i, c, n)
        scan_data.append(frame)

        show(draw_image, x, y+i, c, frame)  
        print(scan_data)


    np.save("scan_data.npy", scan_data)











