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
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        
        if not description:
            return jsonify({'success': False, 'error': 'Описание не может быть пустым.'}), 400

        print(f"Генерация чертежа для: '{description}'")

        with torch.no_grad():
            text_emb = text_encoder.encode(description).to(DEVICE)
            if text_emb.dim() == 1:
                text_emb = text_emb.unsqueeze(0)

            print(f"Размер text_emb: {text_emb.shape}")
            
            figure_logits, cont_params = predictor_model(text_emb)
            figure_type, real_cont = predictor_model.denormalize_params(figure_logits, cont_params)

            description_lower = description.lower()
            if any(word in description_lower for word in ["прямо", "ровно", "вертикально", "без наклона", "прямая", "прямой", "ровный", "не наклонён"]):
                print("Обнаружено ключевое слово для 'прямого' положения — обнуляем поворот.")
                real_cont[:, 3] = 0.0  # rotation_x
                real_cont[:, 4] = 0.0  # rotation_y

            print(f"Предсказанный тип фигуры: {figure_type.item()}")
            
            real_params = torch.cat([figure_type.unsqueeze(1).float(), real_cont], dim=1)
            
            print(f"Параметры модели: {real_params.cpu().numpy()}")

             # Ограничиваем cam_dist, чтобы фигура была ближе
            real_params[:, 6] = torch.clamp(real_params[:, 6], min=1.5, max=3.0)

            # Дополнительно: если в описании есть слова "крупно", "близко" и т.п. — ещё ближе
            description_lower = description.lower()
            if any(word in description_lower for word in ["крупно", "близко", "вблизи", "большой", "масштаб", "увеличить", "подробно"]):
                real_params[:, 6] = 1.8  # ещё ближе
            elif any(word in description_lower for word in ["далеко", "вдали", "маленький", "уменьшить"]):
                real_params[:, 6] = 3.0  # чуть дальше, но не слишком
                
            # Рендерим 3D изображение
            image_tensor = renderer.render_from_params(real_params)
            print(f"Размер image_tensor: {image_tensor.shape}")
            
            # Конвертируем тензор в PIL Image
            if len(image_tensor.shape) == 4:
                image_tensor = image_tensor[0]  # берем первый батч
            
            print(f"Мин/макс значения в тензоре: {image_tensor.min().item():.3f}, {image_tensor.max().item():.3f}")
            
            if image_tensor.shape[0] == 1:  # grayscale
                image_array = (image_tensor.squeeze().cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array, mode='L')
            else:  # RGB
                image_array = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_array, mode='RGB')


        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
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
        return jsonify({'success': False, 'error': f'Ошибка: {str(e)}'}), 500
    
if __name__ == '__main__':
    try:
        print("Запуск приложения...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Ошибка при запуске: {e}")
        import traceback
        traceback.print_exc()