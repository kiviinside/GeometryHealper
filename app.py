import os
import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import base64
import numpy as np

from model import ImprovedParametricGeometryPredictor, ImprovedSceneRenderer
from text_encoder import TextEncoder

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "weights/geometry_generator.pth"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def load_parametric_model():
    try:
        text_encoder = TextEncoder()
        embed_dim = text_encoder.get_embedding_dim()
        
        model = ImprovedParametricGeometryPredictor(text_embed_dim=embed_dim).to(DEVICE)
        
        if os.path.exists(MODEL_PATH):
            checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Параметрическая модель успешно загружена.")
        else:
            print("Веса не найдены. Используется ненаученная модель.")
        
        model.eval()
        
        # Используем ImprovedSceneRenderer для 3D фигур
        renderer = ImprovedSceneRenderer(output_size=(256, 256))
        
        return model, renderer, text_encoder
    
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        # Возвращаем None или создаем заглушки
        return None, None, None

# Инициализация один раз при запуске
predictor_model, renderer, text_encoder = load_parametric_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_drawing():
    if predictor_model is None:
        return jsonify({'success': False, 'error': 'Модель не загружена'}), 500

    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        if not description:
            return jsonify({'success': False, 'error': 'Описание не может быть пустым.'}), 400

        desc_lower = description.lower()

        with torch.no_grad():
            text_emb = text_encoder.encode(description).to(DEVICE)
            if text_emb.dim() == 1:
                text_emb = text_emb.unsqueeze(0)

            figure_logits, cont_params = predictor_model(text_emb)
            figure_type, real_cont = predictor_model.denormalize_params(figure_logits, cont_params)

            # Обнуление поворота при ключевых словах
            if any(word in desc_lower for word in ["прямо", "ровно", "вертикально", "без наклона", "прямая", "прямой", "ровный", "не наклонён"]):
                real_cont[:, 3] = 0.0  # rotation_x
                real_cont[:, 4] = 0.0  # rotation_y

            # Настройка расстояния камеры
            real_cont[:, 5] = torch.clamp(real_cont[:, 5], min=1.5, max=3.0)  # индекс 5 = cam_dist
            if any(word in desc_lower for word in ["крупно", "близко", "вблизи", "большой", "масштаб", "увеличить", "подробно"]):
                real_cont[:, 5] = 1.8

            real_params = torch.cat([figure_type.unsqueeze(1).float(), real_cont], dim=1)

            # Рендер
            image_tensor = renderer.render_from_params(real_params)

            # Конвертация
            if image_tensor.dim() == 4:
                image_tensor = image_tensor[0]
            if image_tensor.shape[0] == 1:
                image_array = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array, mode='L')
            else:
                image_array = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array, mode='RGB')

        # Кодирование в base64
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_data = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        img_url = f"data:image/png;base64,{img_data}"

        return jsonify({
            'success': True,
            'image_url': img_url,
            'description': description
        })

    except Exception as e:
        import traceback
        print("Ошибка:", traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500
    
if __name__ == '__main__':
    try:
        print("Запуск приложения...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
        import traceback
        traceback.print_exc()