import torch
from sentence_transformers import SentenceTransformer

class TextEncoder:
    """
    Упрощенный текстовый энкодер на основе sentence-transformers
    """
    
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        print(f"Загрузка текстового энкодера: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Модель загружена. Размерность эмбеддингов: {self.embedding_dim}")
    
    def encode(self, text: str | list[str], normalize: bool = True) -> torch.Tensor:
        """
        Кодирует текст в векторные представления
        
        Args:
            text: Текст или список текстов для кодирования
            normalize: Нормализовать векторы (рекомендуется)
            
        Returns:
            Тензор с эмбеддингами формы [batch_size, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.model.encode(
                text, 
                convert_to_tensor=True, 
                device=self.device,
                normalize_embeddings=normalize
            )
        return embeddings
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
