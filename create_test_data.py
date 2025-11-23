from text_encoder import TextEncoder
from model import ImprovedSceneRenderer
import torch
import json

def create_test_samples():
    """Создает несколько тестовых образцов для демонстрации"""
    renderer = ImprovedSceneRenderer()
    
    # Тестовые параметры
    test_params = [
        [0, 2.0, 1.0, 2.0, 0, 0, 4.0, 1.5, 0, 4],  # куб
        [2, 1.6, 1.0, 1.8, 15, -25, 2.8, 1.8, 0, 6],  # призма
        [3, 1.5, 1.0, 1.0, 0, 0, 4.0, 1.0, 0, 0],  # шар
        [4, 1.2, 0.9, 2.0, 10, -20, 2.6, 1.7, 0, 0],  # цилиндр
        [1, 1.8, 1.0, 2.5, 0, 0, 4.5, 1.3, 0, 4],  # пирамида
    ]
    
    params_tensor = torch.tensor(test_params, dtype=torch.float32)
    images = renderer.render_from_params(params_tensor)
    
    print("Тестовые изображения сгенерированы!")
    return images

if __name__ == "__main__":
    create_test_samples()