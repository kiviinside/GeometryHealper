
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

FIGURE_TYPES = {
    0: "cube",
    1: "pyramid",
    2: "prism",
    3: "sphere",
    4: "cylinder",
    5: "cone",
}
NUM_FIGURE_TYPES = len(FIGURE_TYPES)  # = 6

class ImprovedSceneRenderer:
    """
    Реалистичный рендерер 3D фигур с использованием matplotlib
    """
    
    def __init__(self, output_size=(256, 256)):
        self.output_size = output_size

    def create_cube(self, size=1.0):
        """Создает вершины куба"""
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]
        ]) * size / 2
        edges = [
            (0,1), (1,2), (2,3), (3,0),
            (4,5), (5,6), (6,7), (7,4),
            (0,4), (1,5), (2,6), (3,7)
        ]
        return vertices, edges
    
    def create_sphere(self, radius=1.0, resolution=10):
        """Создает сферу с правильной размерностью"""
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        phi, theta = np.meshgrid(phi, theta)
        
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        
        return x, y, z

    def create_cylinder(self, radius=1.0, height=2.0, resolution=10):
        """Создает цилиндр с правильной размерностью"""
        z = np.linspace(-height/2, height/2, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        x = radius * np.cos(theta_grid)
        y = radius * np.sin(theta_grid)
        z = z_grid
        
        return x, y, z

    def create_cone(self, radius=1.0, height=2.0, resolution=10):
        """Создает конус с правильной размерностью"""
        z = np.linspace(0, height, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        theta_grid, z_grid = np.meshgrid(theta, z)
        
        r = radius * (1 - z_grid/height)
        x = r * np.cos(theta_grid)
        y = r * np.sin(theta_grid)
        z = z_grid - height/2
        
        return x, y, z
        
    def create_pyramid(self, base_size=1.0, height=2.0, sides=4):
        """Создает пирамиду"""
        angles = np.linspace(0, 2*np.pi, sides+1)[:-1]
        base_vertices = []
        for angle in angles:
            x = base_size * np.cos(angle) / 2
            y = base_size * np.sin(angle) / 2
            base_vertices.append([x, y, -height/2])
        
        base_vertices = np.array(base_vertices)
        apex = np.array([0, 0, height/2])
        
        vertices = np.vstack([base_vertices, apex])
        edges = []
        
        # Боковые ребра
        for i in range(sides):
            edges.append((i, sides))  # от основания к вершине
        
        # Ребра основания
        for i in range(sides):
            edges.append((i, (i+1) % sides))
        
        return vertices, edges
    
    def create_prism(self, base_size=1.0, height=2.0, sides=4):
        """Создает призму"""
        angles = np.linspace(0, 2*np.pi, sides+1)[:-1]
        base_vertices = []
        for angle in angles:
            x = base_size * np.cos(angle) / 2
            y = base_size * np.sin(angle) / 2
            base_vertices.append([x, y, -height/2])
        
        base_vertices = np.array(base_vertices)
        top_vertices = base_vertices + np.array([0, 0, height])
        
        vertices = np.vstack([base_vertices, top_vertices])
        edges = []
        
        # Вертикальные ребра
        for i in range(sides):
            edges.append((i, i + sides))
        
        # Ребра оснований
        for i in range(sides):
            edges.append((i, (i+1) % sides))
            edges.append((i + sides, (i+1) % sides + sides))
        
        return vertices, edges
    
    def apply_rotation(self, vertices, rot_x, rot_y):
        """Применяет вращение к вершинам"""
        rot_x_rad = np.radians(rot_x)
        rot_y_rad = np.radians(rot_y)
        
        # Матрица вращения вокруг X
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)],
            [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]
        ])
        
        # Матрица вращения вокруг Y
        Ry = np.array([
            [np.cos(rot_y_rad), 0, np.sin(rot_y_rad)],
            [0, 1, 0],
            [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]
        ])
        
        R = Ry @ Rx
        return vertices @ R.T

    def _render_surface(self, ax, x, y, z, rot_x, rot_y, line_thickness, max_lines=8):
        points = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)
        points_rot = self.apply_rotation(points, rot_x, rot_y)
        x_rot = points_rot[:, 0].reshape(x.shape)
        y_rot = points_rot[:, 1].reshape(y.shape)
        z_rot = points_rot[:, 2].reshape(z.shape)

        step_u = max(1, x_rot.shape[0] // max_lines)
        step_v = max(1, x_rot.shape[1] // max_lines)

        for j in range(0, x_rot.shape[0], step_u):
            ax.plot(x_rot[j, :], y_rot[j, :], z_rot[j, :], 'black', linewidth=line_thickness)
        for k in range(0, x_rot.shape[1], step_v):
            ax.plot(x_rot[:, k], y_rot[:, k], z_rot[:, k], 'black', linewidth=line_thickness)

    def create_fallback_image(self):
        """Создает простое тестовое изображение с кругом"""
        h, w = self.output_size
        y, x = torch.meshgrid(torch.arange(h, dtype=torch.float32), 
                            torch.arange(w, dtype=torch.float32), indexing='ij')
        center_x, center_y = w // 2, h // 2
        radius = min(w, h) // 4
        
        # Рисуем круг
        distance = torch.sqrt((x - center_x)**2 + (y - center_y)**2)
        circle = (distance <= radius) & (distance >= radius - 3)
        
        # Создаем RGB изображение
        img = torch.zeros(3, h, w)
        img[0] = circle.float()  # Красный канал
        img[1] = circle.float()  # Зеленый канал  
        img[2] = 1.0 - circle.float()  # Синий канал (фон)
        
        return img

    def render_from_params(self, params):
        """
        Рендерит реальные 3D фигуры на основе параметров
        """
        batch_size = params.shape[0]
        images = []

        for idx in range(batch_size):
            fig = plt.figure(figsize=(2.56, 2.56), dpi=100, facecolor='white')
            ax = fig.add_subplot(111, projection='3d')
            
            # Извлекаем параметры
            figure_type = int(params[idx, 0].item())
            scale = params[idx, 1].item()
            radius = params[idx, 2].item()
            height = params[idx, 3].item()
            rot_x = params[idx, 4].item()
            rot_y = params[idx, 5].item()
            cam_dist = params[idx, 6].item()
            line_thickness = params[idx, 7].item()
            is_dashed = params[idx, 8].item() > 0.5
            num_sides = max(3, int(params[idx, 9].item()))
            
            try:
                if figure_type == 0:  # куб
                    vertices, edges = self.create_cube(scale)
                    vertices = self.apply_rotation(vertices, rot_x, rot_y)
                    for edge in edges:
                        points = vertices[list(edge)]
                        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                                'black', linewidth=line_thickness)
                            
                elif figure_type == 1:  # пирамида
                    vertices, edges = self.create_pyramid(scale, height, num_sides)
                    vertices = self.apply_rotation(vertices, rot_x, rot_y)
                    for edge in edges:
                        points = vertices[list(edge)]
                        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                                'black', linewidth=line_thickness)

                elif figure_type == 2:  # призма
                    vertices, edges = self.create_prism(base_size=scale, height=height, sides=num_sides)
                    vertices = self.apply_rotation(vertices, rot_x, rot_y)
                    for edge in edges:
                        points = vertices[list(edge)]
                        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 'black', linewidth=line_thickness)
                            

                elif figure_type == 3:  # шар
                    u = np.linspace(0, 2 * np.pi, 12)
                    v = np.linspace(0, np.pi, 8)
                    x = radius * np.outer(np.cos(u), np.sin(v))
                    y = radius * np.outer(np.sin(u), np.sin(v))
                    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
                    self._render_surface(ax, x, y, z, rot_x, rot_y, line_thickness)

                elif figure_type == 4:  # цилиндр
                    x, y, z = self.create_cylinder(radius=radius, height=height, resolution=20)
                    self._render_surface(ax, x, y, z, rot_x, rot_y, line_thickness, max_lines = 6)

                elif figure_type == 5:  # конус
                    x, y, z = self.create_cone(radius=radius, height=height, resolution=20)
                    self._render_surface(ax, x, y, z, rot_x, rot_y, line_thickness, max_lines = 6)

                else:  # fallback
                    vertices, edges = self.create_cube(scale)
                    vertices = self.apply_rotation(vertices, rot_x, rot_y)
                    for edge in edges:
                        points = vertices[list(edge)]
                        ax.plot3D(points[:, 0], points[:, 1], points[:, 2], 
                                'black', linewidth=line_thickness)
                
                # Настройки камеры
                ax.set_xlim([-cam_dist, cam_dist])
                ax.set_ylim([-cam_dist, cam_dist])
                ax.set_zlim([-cam_dist, cam_dist])
                ax.set_box_aspect([1, 1, 1])
                ax.set_axis_off()
                ax.grid(False)
                ax.set_facecolor('white')
                fig.patch.set_facecolor('white')
                for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
                    pane.set_facecolor('white')
                    pane.set_alpha(1.0)
                    pane.set_edgecolor('white')

                fig.tight_layout(pad=0)
                fig.canvas.draw()
                
                buf = fig.canvas.buffer_rgba()
                ncols, nrows = fig.canvas.get_width_height()
                img_array = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 4)[:, :, :3]
                img_tensor = torch.from_numpy(img_array).float() / 255.0
                images.append(img_tensor.permute(2, 0, 1))
                plt.close(fig)
                
            except Exception as e:
                print(f"Ошибка при рендеринге {idx}: {e}")
                import traceback
                traceback.print_exc()
                images.append(self.create_fallback_image())
                plt.close(fig)
        
        return torch.stack(images) if images else torch.zeros(1, 3, *self.output_size)

class ImprovedParametricGeometryPredictor(nn.Module):
    def __init__(self, text_embed_dim=384, hidden_dim=512, num_figure_types=6, num_cont_params=9):

        super().__init__()
        self.num_figure_types = num_figure_types
        self.num_cont_params = num_cont_params
        
        # Общий скрытый слой
        self.shared = nn.Sequential(
            nn.Linear(text_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        
        # Ветка для классификации типа фигуры
        self.figure_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_figure_types)
        )
        
        # Ветка для непрерывных параметров
        self.cont_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_cont_params),
            nn.Tanh()  # для нормализованных значений [-1, 1]
        )
        
        # Диапазоны для непрерывных параметров (БЕЗ figure_type!)
        self.cont_param_ranges = [
            (0.5, 3.0),   # scale
            (0.3, 2.0),   # radius
            (0.5, 3.0),   # height
            (-60, 60),    # rotation_x
            (-60, 60),    # rotation_y
            (2.0, 8.0),   # camera_distance
            (0.3, 3.0),   # line_thickness
            (0, 1),       # is_dashed (оставим как регрессию, но можно и бинарную)
            (3, 8),       # num_sides
        ]
    def forward(self, text_embeddings):
        shared = self.shared(text_embeddings)
        figure_logits = self.figure_head(shared)      # [B, 6]
        cont_params = self.cont_head(shared)          # [B, 9]
        return figure_logits, cont_params

    def denormalize_params(self, figure_logits, cont_params):
        """
        Возвращает: (figure_type_int, real_cont_params_tensor)
        """
        # 1. Выбираем наиболее вероятный тип фигуры
        figure_type = torch.argmax(figure_logits, dim=1)  # [B]
        
        # 2. Денормализуем непрерывные параметры
        batch_size = cont_params.shape[0]
        real_cont = torch.zeros_like(cont_params)
        
        for i in range(cont_params.shape[1]):
            min_val, max_val = self.cont_param_ranges[i]
            if i == 7:  # is_dashed — бинарный
                real_cont[:, i] = (cont_params[:, i] > 0).float()  # или (cont_params[:, i] + 1)/2 > 0.5
            elif i == 8:  # num_sides — целочисленный
                raw = (cont_params[:, i] + 1) * 0.5 * (max_val - min_val) + min_val
                real_cont[:, i] = torch.round(raw).clamp(min_val, max_val)
            else:  # остальные — непрерывные
                real_cont[:, i] = (cont_params[:, i] + 1) * 0.5 * (max_val - min_val) + min_val
        
        return figure_type, real_cont