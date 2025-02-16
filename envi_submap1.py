import numpy as np
from PIL import Image
from make_data import per_dis, src_world_x,src_world_y,src_c
import tkinter as tk
import tkinter.ttk as ttk
from PIL import Image, ImageTk
from submap import Submap, show_map
import scan

Slam = Submap


class Envi:
    def __init__(self):

        self.n_scan_num = 8

        self.slam = Slam()
        self.reset_slam()

        self.reset_car_state()

        self.reset_click_list()

        # 创建主窗口
        root = tk.Tk()
        self.root = root
        root.title("Tkinter布局示例")
        root.geometry("850x510")
        root.resizable(False, False)


        # 左侧框架，宽度为100px
        left_frame = tk.Frame(root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        

        # 在左侧框架中添加三个按钮
        button1 = ttk.Button(left_frame, text="重置", command=self.reset_canvas)
        button1.pack(fill=tk.X)

        button2 = ttk.Button(left_frame, text="显示", command=self.show_map)
        button2.pack(fill=tk.X)
        
        button4 = ttk.Button(left_frame, text="slam地图保存", command=self.save_slam_map)
        button4.pack(fill=tk.X)

        button3 = ttk.Button(left_frame, text="保存click", command=self.save_click_list)
        button3.pack(fill=tk.X)

        button3 = ttk.Button(left_frame, text="使用click", command=self.load_click_list)
        button3.pack(fill=tk.X)

        
        options = ["优化", "不优化"]
        self.combo = ttk.Combobox(left_frame, values=options, width=10)
        self.combo.set("不优化")
        self.combo.pack(fill=tk.X)

        self.canvas_width = 700
        self.canvas_height = 500
        canvas_width = self.canvas_width
        canvas_height = self.canvas_height

        # 在右侧框架中添加一个空白画布
        canvas = tk.Canvas(root, bg='black', width=canvas_width, height=canvas_height)
        canvas.place(x=100, y=5)

        self.image_path = "image.png"  # 替换为你的图片路径
        pil_image = Image.open(self.image_path)
        self.pil_image = pil_image
        
        self.np_image = np.array(pil_image,dtype=int)

        # 调整图片大小以适应画布
        assert pil_image.width == canvas_width and pil_image.height == canvas_height, f"{pil_image.width} {canvas_width} {pil_image.height} {canvas_height}"

        # 将图片转换为Tkinter兼容的格式
        photo = ImageTk.PhotoImage(pil_image)

        canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        # 保持对图片的引用，防止被垃圾回收
        canvas.image = photo
        self.canvas = canvas

        # root.bind("<KeyPress>", self.on_key_press)
        canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

    def save_slam_map(self):
        self.slam.save_map("slam_map.npy")
        print("保存slam_map.npy")
    
    def save_click_list(self):
        import pickle
        with open('click_list.pkl', 'wb') as f:
            pickle.dump(self.click_list, f)
    
    def load_click_list(self):
        import pickle
        with open('click_list.pkl', 'rb') as f:
            self.click_list = pickle.load(f)
        self.is_use_click_list = True
        print(self.click_list_index, len(self.click_list))
        while self.click_list_index < len(self.click_list):
            self.on_mouse_release(None)
        # print(self.click_list)

    def reset_car_state(self):
        self.car_state = {
            'x_real': None,
            'y_real': None,
            'c_real': None,

            'x_noise': None,
            'y_noise': None,
            'c_noise': None,

            'dx_noise': None,
            'dy_noise': None,
            'dc_noise': None,
        }
    def reset_click_list(self):
        self.click_list = []
        self.click_list_index = 0
        self.is_use_click_list = False
    def reset_slam(self):
        self.slam.reset()

        # 这个是当前 雷达位置 相对于 submap原点 的姿态
        self.car_lidar_relate_submap_center_x = 0
        self.car_lidar_relate_submap_center_y = 0
        self.car_lidar_relate_submap_center_c = 0
    def reset_canvas(self):
        canvas = self.canvas
        canvas.delete("all")
        self.image_path = "image.png"  # 替换为你的图片路径
        image = Image.open(self.image_path)
        # 调整图片大小以适应画布
        canvas_width = self.canvas_width
        canvas_height = self.canvas_height
        assert image.width == canvas_width and image.height == canvas_height, f"{image.width} {canvas_width} {image.height} {canvas_height}"
        # 将图片转换为Tkinter兼容的格式
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        # 保持对图片的引用，防止被垃圾回收
        canvas.image = photo

        self.reset_car_state()
        
        self.reset_click_list()

        self.reset_slam()


    def run(self):
        # 运行主循环
        self.root.mainloop()

    def get_vector_info(self, x1, y1, x2, y2):
        """
        计算向量及其属性
        :param x1: 起点 x 坐标
        :param y1: 起点 y 坐标
        :param x2: 终点 x 坐标
        :param y2: 终点 y 坐标
        :return: (dx, dy), magnitude, angle
        """
        # 计算向量 (dx, dy)
        dx = x2 - x1
        dy = y2 - y1

        # 计算向量模长
        magnitude = np.sqrt(dx**2 + dy**2)

        # 计算角度（与正 x 轴的夹角，单位为弧度）
        angle = np.arctan2(dy, dx)  # atan2 自动处理象限
        angle = np.degrees(angle)

        return (dx, dy), magnitude, angle

    def on_mouse_release(self,event):
        """
        处理鼠标点击事件
        :param event: 鼠标事件对象
        """
        # 获取鼠标点击位置
        x, y = event.x, event.y
        
        if self.slam.is_reset():
            self.slam.set_center_relate_world_xyc(x, y, 0)

        if self.is_use_click_list:
            print("使用click list")
            if self.click_list_index >= len(self.click_list):
                return
            if self.click_list_index == 0:
                car_state = self.click_list[self.click_list_index]
                car_move_x_real = car_state['x_real']
                car_move_y_real = car_state['y_real']
                car_turn_c_real = car_state['c_real']
                car_move_x_noise = car_state['x_noise']
                car_move_y_noise = car_state['y_noise']
                car_turn_c_noise = car_state['c_noise']
                self.draw_car(car_move_x_real, car_move_y_real, 0, 'red')
            else:
                last_car_state = self.click_list[self.click_list_index-1]
                car_state = self.click_list[self.click_list_index]
                last_car_move_x_real = last_car_state['x_real']
                last_car_move_y_real = last_car_state['y_real']
                last_car_turn_c_real = last_car_state['c_real']
                car_move_x_real = car_state['x_real']
                car_move_y_real = car_state['y_real']
                car_turn_c_real = car_state['c_real']
                car_move_x_noise = car_state['x_noise']
                car_move_y_noise = car_state['y_noise']
                car_turn_c_noise = car_state['c_noise']
                car_dx_noise = car_state['dx_noise']
                car_dy_noise = car_state['dy_noise']
                car_dc_noise = car_state['dc_noise']

                self.canvas.create_line(last_car_move_x_real, last_car_move_y_real, car_move_x_real, car_move_y_real, width=1, fill='red')
                self.draw_car(car_move_x_real, car_move_y_real, car_turn_c_real, 'red')
                self.draw_car(car_move_x_noise, car_move_y_noise, car_turn_c_noise, 'yellow')

                self.run_slam(
                    car_move_x_real, car_move_y_real, car_turn_c_real, car_turn_c_noise,
                    car_dx_noise, car_dy_noise, car_dc_noise)
                    
                
            self.click_list_index+=1
            return


        print(f"鼠标点击位置: ({x}, {y})")

        if self.car_state['x_real'] is None:
            if self.np_image[y, x].all() == 0:
                print("已记录起始点")
                self.car_state['x_real'] = x
                self.car_state['y_real'] = y
                self.car_state['x_noise'] = x
                self.car_state['y_noise'] = y
                self.car_state['c_real'] = 0
                self.car_state['c_noise'] = 0
                self.draw_car(x, y, 0, 'red')
                self.click_list.append(self.car_state.copy())
            else:
                print("起始点为障碍物，请重新选择")

            return 

        (click_dx, click_dy), magnitude, angle = self.get_vector_info(self.car_state['x_real'], self.car_state['y_real'], x, y)
        print(f"dx:{click_dx},dy:{click_dy},magnitude:{magnitude},angle:{angle}")

        move_distance = 20

        # xy_std = 5 # 位置标准差
        xy_std = 0.2*10/100 # 根据使用里程计的经验估算里程计每100厘米产生0.2距离误差
        xy_std = move_distance * xy_std
        c_std = 0.02*10  # 角度标准差, 根据经验, 每100厘米产生0.02角度误差
        c_std = move_distance * c_std
        car_dx_real = click_dx / magnitude * move_distance
        car_move_x_real = self.car_state['x_real'] + car_dx_real
        car_dy_real = click_dy / magnitude * move_distance
        car_move_y_real = self.car_state['y_real'] + car_dy_real
        car_turn_c_real = angle

        car_dx_noise = car_dx_real + np.random.normal(0, xy_std)
        car_move_x_noise = self.car_state['x_real'] + car_dx_noise
        car_dy_noise = car_dy_real + np.random.normal(0, xy_std)
        car_move_y_noise = self.car_state['y_real'] + car_dy_noise
        car_dc_noise = np.random.normal(0, c_std)
        car_turn_c_noise = car_turn_c_real + car_dc_noise
        print("c real", car_turn_c_real)

        if  0<=int(car_move_y_real)<=self.np_image.shape[0]-1 and 0<=int(car_move_x_real)<=self.np_image.shape[1]-1 and \
            0<=int(car_move_y_noise)<=self.np_image.shape[0]-1 and 0<=int(car_move_x_noise)<=self.np_image.shape[1]-1 and \
            self.np_image[int(car_move_y_real), int(car_move_x_real)].all() == 0 and self.np_image[int(car_move_y_noise), int(car_move_x_noise)].all() == 0:

            self.canvas.create_line(self.car_state['x_real'], self.car_state['y_real'], car_move_x_real, car_move_y_real, width=1, fill='red')
            self.draw_car(car_move_x_real, car_move_y_real, car_turn_c_real, 'red')
            self.draw_car(car_move_x_noise, car_move_y_noise, car_turn_c_noise, 'yellow')

            self.run_slam(
                car_move_x_real, car_move_y_real, car_turn_c_real, car_turn_c_noise,
                car_dx_noise, car_dy_noise, car_dc_noise)
            
            self.car_state['x_real'] = car_move_x_real
            self.car_state['y_real'] = car_move_y_real
            self.car_state['c_real'] = car_turn_c_real
            self.car_state['x_noise'] = car_move_x_noise
            self.car_state['y_noise'] = car_move_y_noise
            self.car_state['c_noise'] = car_turn_c_noise
            self.car_state['dx_noise'] = car_dx_noise
            self.car_state['dy_noise'] = car_dy_noise
            self.car_state['dc_noise'] = car_dc_noise
            self.click_list.append(self.car_state.copy())
        
        else:
            print("重新选点")
    def run_slam(self, car_move_x_real, car_move_y_real, car_turn_c_real, car_turn_c_noise, car_dx_noise, car_dy_noise, car_dc_noise):
        scan_frame = scan.scan_pil_image(
            self.pil_image, 
            car_move_x_real, 
            car_move_y_real, 
            car_turn_c_real, 
            self.n_scan_num
        )
        print(scan_frame)


        print('slam x,y,c : ', self.car_lidar_relate_submap_center_x, self.car_lidar_relate_submap_center_y, self.car_lidar_relate_submap_center_c)
        self.car_lidar_relate_submap_center_x = self.car_lidar_relate_submap_center_x + car_dx_noise
        self.car_lidar_relate_submap_center_y = self.car_lidar_relate_submap_center_y + car_dy_noise
        self.car_lidar_relate_submap_center_c = car_turn_c_noise              # 这个不是相对值
        print('slam x,y,c : ', self.car_lidar_relate_submap_center_x, self.car_lidar_relate_submap_center_y, self.car_lidar_relate_submap_center_c)

        # 给定 noise 的世界坐标、朝向
        # lidar_in_world_x = self.slam.

        is_optimize = False
        select = self.combo.get()
        print(select)
        if select == "优化":
            is_optimize = True
        self.slam.set_point(
            self.car_lidar_relate_submap_center_x, 
            self.car_lidar_relate_submap_center_y, 
            self.car_lidar_relate_submap_center_c, 
            scan_frame,
            is_posture_optimize=is_optimize
        )
    def show_map(self):
        show_map(self.slam.map)

    def draw_car(self, x,y,c, color):
        draw_obj = {
            "oval": None,
            "arrow": None,
        }
        canvas = self.canvas
        radius = 3
        
        draw_obj['oval'] = canvas.create_oval(x-radius,y-radius,x+radius,y+radius,fill=color)

        length = 20
        x2 = x + length * np.cos(np.radians(c))  # 终点 x 坐标
        y2 = y + length * np.sin(np.radians(c))  # 终点 y 坐标（y 坐标方向为向下）

        # 绘制箭头
        draw_obj['arrow'] = canvas.create_line(
            x, y, x2, y2,
            fill=color,
            width=1,
            arrow=tk.LAST  # 在终点添加箭头
        )

        return draw_obj



if __name__ == '__main__':
    envi = Envi()
    envi.run()







