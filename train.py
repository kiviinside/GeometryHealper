import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from model import ImprovedParametricGeometryPredictor
from model import ImprovedSceneRenderer
from text_encoder import TextEncoder
import numpy as np
import json
import os

class GeometryDataset(Dataset):
    def __init__(self, data_file, text_encoder):
        self.text_encoder = text_encoder
        
        # Загружаем или создаем данные
        if os.path.exists(data_file):
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            self.data = self.create_balanced_training_data()
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def create_balanced_training_data(self):
        descriptions_by_type = {
            0: ["куб"],
            1: ["пирамида", "правильная пирамида"],
            2: ["призма", "правильная призма"],
            3: ["шар", "сфера"],
            4: ["цилиндр", "правильный цилиндр"],
            5: ["конус","правильный конус"],
        }

        data = []
        for fig_type, descs in descriptions_by_type.items():
            for desc in descs:
                if fig_type in [1, 2]:  # пирамида или призма
                    for sides in [3, 4, 5, 6, 8]:
                        # Прямая фигура
                        data.append([f"{desc} с {sides} гранями", fig_type, 1.8, 1.0, 2.0, 0, 0, 3.0, 1.5, 0, sides])
                        # Наклонённая
                        data.append([f"наклонённая {desc} с {sides} сторонами", fig_type, 1.8, 1.0, 2.0, 30, -20, 3.0, 1.5, 0, sides])
                        # Большая
                        data.append([f"большая {desc} с {sides} гранями", fig_type, 2.5, 1.5, 2.8, 0, 0, 4.0, 1.8, 0, sides])
                        # Маленькая
                        data.append([f"маленькая {desc} с {sides} сторонами", fig_type, 1.2, 0.7, 1.5, 0, 0, 2.5, 1.2, 0, sides])
                else:
                    # Для остальных фигур (куб, шар, цилиндр, конус) — num_sides не используется (ставим 0)
                    data.append([f"{desc}", fig_type, 1.8, 1.0, 2.0, 0, 0, 3.0, 1.5, 0, 0])
                    data.append([f"{desc} наклонённый", fig_type, 1.8, 1.0, 2.0, 30, -20, 3.0, 1.5, 0, 0])
                    data.append([f"большой {desc}", fig_type, 2.5, 1.5, 2.8, 0, 0, 4.0, 1.8, 0, 0])
                    data.append([f"маленький {desc}", fig_type, 1.2, 0.7, 1.5, 0, 0, 2.5, 1.2, 0, 0])

        # Преобразуем в словари
        result = []
        for item in data:
            result.append({
                "description": item[0],
                "params": item[1:]
            })
        return result
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        desc = item["description"]
        params = torch.tensor(item["params"], dtype=torch.float32)  # [10]
        figure_type = int(params[0].item())  # целое число: 0..5
        cont_params = params[1:]             # [9]
        
        text_emb = self.text_encoder.encode(desc, normalize=True)
        return text_emb, torch.tensor(figure_type, dtype=torch.long), cont_params

def validate_model(model, val_loader, device):
    """Валидация модели"""
    model.eval()
    total_loss = 0
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    with torch.no_grad():
        for text_embs, true_figures, true_cont in val_loader:
            text_embs = text_embs.to(device)
            true_figures = true_figures.to(device)
            true_cont = true_cont.to(device)
            
            pred_fig_logits, pred_cont = model(text_embs)
            loss_cls = criterion_cls(pred_fig_logits, true_figures)
            loss_reg = criterion_reg(pred_cont, true_cont)
            loss = 3.0 * loss_cls + loss_reg
            
            total_loss += loss.item()
    
    model.train()
    return total_loss / len(val_loader)

def train_model():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используется устройство: {DEVICE}")
    
    # Создаем папку для весов если нет
    os.makedirs("weights", exist_ok=True)
    
    text_encoder = TextEncoder()
    
    # Создаем датасет
    dataset = GeometryDataset("training_data.json", text_encoder)
    
    # Разделяем на train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    model = ImprovedParametricGeometryPredictor(
        text_embed_dim=text_encoder.get_embedding_dim()
    ).to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print("Начало обучения...")
    model.train()
    
    for epoch in range(200):
        total_loss = 0
        
        for batch_idx, (text_embs, true_figures, true_cont) in enumerate(train_loader):
            text_embs = text_embs.to(DEVICE)
            true_figures = true_figures.to(DEVICE)      # [batch_size] — long
            true_cont = true_cont.to(DEVICE)            # [batch_size, 9] — float
            
            optimizer.zero_grad()
            
            # Предсказываем
            pred_fig_logits, pred_cont = model(text_embs)
            
            # Считаем два лосса
            loss_cls = criterion_cls(pred_fig_logits, true_figures)   # классификация
            loss_reg = criterion_reg(pred_cont, true_cont)            # регрессия
            
            # Суммируем с весами (можно настроить)
            loss = loss_cls + 5.0 * loss_reg  # штраф за ошибку в параметрах сильнее
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        scheduler.step()
        
        # Валидация
        val_loss = validate_model(model, val_loader, DEVICE)
        
        print(f"Epoch [{epoch+1}/200], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Сохраняем лучшую модель
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, "weights/geometry_generator.pth")
            print(f"  -> Сохранена лучшая модель с val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping на эпохе {epoch+1}")
            break
    
    print("Обучение завершено!")

if __name__ == "__main__":
    train_model()