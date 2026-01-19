import cv2
import numpy as np
from typing import List, Dict, Any, Union, Optional
import torch
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingEngine:
    """Embedding engine for multi-modal embeddings"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.text_model = self._load_text_model()
        
        logger.info(f"✅ EmbeddingEngine initialized on {self.device}")
    
    def _load_text_model(self):
        """Load text embedding model"""
        try:
            from app.core.config import settings
            model_name = getattr(settings, 'TEXT_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
            logger.info(f"📚 Loading text embedding model: {model_name}")
            
            # Import with error handling
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
            model.to(self.device)
            logger.info(f"✅ Text model loaded: {model_name}")
            return model
                
        except ImportError as e:
            logger.error(f"❌ SentenceTransformers not available: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            # Return a dummy model
            class DummyModel:
                def __init__(self, device):
                    self.device = device
                
                def encode(self, texts, **kwargs):
                    if isinstance(texts, list):
                        return np.random.randn(len(texts), 384)
                    else:
                        return np.random.randn(1, 384)
                
                def get_sentence_embedding_dimension(self):
                    return 384
                
                def to(self, device):
                    return self
            
            logger.warning("⚠️ Using dummy embedding model")
            return DummyModel(self.device)
    
    # Rest of the class remains the same...
    
    # Rest of the class remains the same...
    # [Keep all other methods as they were]
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Split text into overlapping chunks for RAG
        """
        if not text or len(text) <= chunk_size:
            return [text] if text else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end < len(text):
                # Try to break at sentence end
                sentence_end = text.rfind('.', start, end)
                paragraph_end = text.rfind('\n\n', start, end)
                
                if paragraph_end > start + chunk_size // 2:
                    end = paragraph_end + 2
                elif sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Break at word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start:
                        end = word_end
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end - overlap > start else end
        
        return chunks
    
    def generate_text_embeddings(self, texts: Union[str, List[str]], 
                                chunk: bool = True) -> np.ndarray:
        """
        Generate text embeddings
        """
        try:
            if isinstance(texts, str):
                if chunk:
                    chunks = self.chunk_text(texts)
                    texts_to_encode = chunks
                else:
                    texts_to_encode = [texts]
            else:
                texts_to_encode = texts
            
            if not texts_to_encode:
                return np.zeros((0, self.text_model.get_sentence_embedding_dimension()))
            
            embeddings = self.text_model.encode(
                texts_to_encode, 
                convert_to_numpy=True,
                show_progress_bar=False,
                device=str(self.device)
            )
            
            return embeddings
            
        except Exception as e:
            logger.error(f"❌ Text embedding generation failed: {e}")
            # Return random embeddings as fallback
            if isinstance(texts, str):
                texts = [texts]
            dim = self.text_model.get_sentence_embedding_dimension()
            return np.random.randn(len(texts), dim)
    
    def generate_visual_embeddings(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Generate visual embeddings
        """
        try:
            embeddings = []
            
            for img in images:
                # Resize for consistency
                img_resized = cv2.resize(img, (224, 224))
                
                # Extract features
                features = []
                
                # Color histogram
                if len(img_resized.shape) == 3:
                    for i in range(3):
                        hist = cv2.calcHist([img_resized], [i], None, [32], [0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        features.append(hist)
                else:
                    hist = cv2.calcHist([img_resized], [0], None, [96], [0, 256])
                    features.append(cv2.normalize(hist, hist).flatten())
                
                # Combine features
                combined = np.concatenate(features)
                embeddings.append(combined)
            
            return np.array(embeddings)
            
        except Exception as e:
            logger.error(f"❌ Visual embedding generation failed: {e}")
            return np.zeros((len(images), 224))
    
    def get_embedding_dimensions(self) -> Dict[str, Any]:
        """Get dimensions of different embedding types"""
        try:
            dimensions = {
                "text": self.text_model.get_sentence_embedding_dimension(),
                "visual_fallback": 224,
                "device": str(self.device)
            }
            return dimensions
        except:
            return {
                "text": 384,
                "visual_fallback": 224,
                "device": str(self.device)
            }