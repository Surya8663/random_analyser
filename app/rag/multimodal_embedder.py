"""
Multi-modal embedding system for text, visual, and fused content.
Uses SentenceTransformers for text and CLIP for vision.
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import torch
from PIL import Image
import base64
import io

from app.utils.logger import setup_logger
from app.core.models import MultiModalDocument, EnhancedVisualElement, LayoutRegion, OCRWord

logger = setup_logger(__name__)

class MultiModalEmbedder:
    """Generate embeddings for text, visual, and fused content"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._text_encoder = None
        self._vision_encoder = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def initialize(self):
        """Initialize embedding models (lazy loading)"""
        try:
            # Text encoder
            from sentence_transformers import SentenceTransformer
            self._text_encoder = SentenceTransformer(
                self.config.get('text_model', 'all-MiniLM-L6-v2'),
                device=str(self._device)
            )
            logger.info(f"✅ Text encoder initialized: {self.config.get('text_model', 'all-MiniLM-L6-v2')}")
        except ImportError as e:
            logger.warning(f"⚠️ SentenceTransformers not available: {e}")
            self._text_encoder = None
            
        try:
            # Vision encoder (CLIP)
            import clip
            model_name = self.config.get('vision_model', 'ViT-B/32')
            self._clip_model, self._clip_preprocess = clip.load(model_name, device=self._device)
            self._vision_encoder = self._clip_model.encode_image
            logger.info(f"✅ Vision encoder initialized: {model_name}")
        except ImportError as e:
            logger.warning(f"⚠️ CLIP not available: {e}")
            self._vision_encoder = None
            
    def embed_text(self, text: Union[str, List[str]]) -> np.ndarray:
        """Embed text content"""
        if not self._text_encoder:
            # Fallback: simple TF-IDF like embedding
            return self._fallback_text_embedding(text)
        
        if isinstance(text, str):
            text = [text]
            
        embeddings = self._text_encoder.encode(
            text, 
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        if len(embeddings) == 1:
            return embeddings[0]
        return embeddings
    
    def embed_visual(self, image: np.ndarray) -> np.ndarray:
        """Embed visual content from image array"""
        if not self._vision_encoder or image is None:
            # Fallback: histogram-based embedding
            return self._fallback_visual_embedding(image)
        
        try:
            # Convert numpy array to PIL Image
            if len(image.shape) == 2:  # Grayscale
                image = np.stack([image] * 3, axis=-1)
            pil_image = Image.fromarray(image.astype(np.uint8))
            
            # Preprocess and encode
            image_tensor = self._clip_preprocess(pil_image).unsqueeze(0).to(self._device)
            with torch.no_grad():
                embedding = self._vision_encoder(image_tensor)
                embedding = embedding.cpu().numpy()[0]
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Visual embedding failed: {e}")
            return self._fallback_visual_embedding(image)
    
    def embed_fused(self, text_embedding: np.ndarray, visual_embedding: np.ndarray) -> np.ndarray:
        """Fuse text and visual embeddings"""
        if text_embedding is not None and visual_embedding is not None:
            # Simple concatenation with normalization
            fused = np.concatenate([text_embedding, visual_embedding])
            fused = fused / np.linalg.norm(fused)
            return fused
        elif text_embedding is not None:
            return text_embedding
        elif visual_embedding is not None:
            return visual_embedding
        else:
            return np.zeros(512)
    
    def extract_document_embeddings(self, document: MultiModalDocument) -> Dict[str, Any]:
        """Extract embeddings for all modalities in document"""
        embeddings = {
            "text_chunks": [],
            "visual_regions": [],
            "fused_elements": [],
            "metadata": {}
        }
        
        # 1. Text chunks from OCR
        if document.ocr_results:
            for page_num, ocr_result in document.ocr_results.items():
                # Create chunks from paragraphs
                if hasattr(ocr_result, 'text') and ocr_result.text:
                    # Split by paragraphs
                    paragraphs = [p.strip() for p in ocr_result.text.split('\n\n') if p.strip()]
                    for i, para in enumerate(paragraphs[:10]):  # Limit to first 10 paragraphs per page
                        if len(para) > 20:  # Minimum length
                            embedding = self.embed_text(para)
                            embeddings["text_chunks"].append({
                                "id": f"text_page{page_num}_para{i}",
                                "content": para,
                                "embedding": embedding.tolist(),
                                "metadata": {
                                    "page": page_num,
                                    "type": "paragraph",
                                    "agent": "text_agent",
                                    "length": len(para)
                                }
                            })
        
        # 2. Visual regions from layout and visual elements
        if document.visual_elements:
            for idx, element in enumerate(document.visual_elements):
                if idx < len(document.images) and document.images[element.page_num] is not None:
                    try:
                        # Extract region from image
                        img = document.images[element.page_num]
                        height, width = img.shape[:2]
                        
                        # Convert normalized bbox to pixel coordinates
                        bbox = element.bbox
                        x1 = int(bbox.x1 * width)
                        y1 = int(bbox.y1 * height)
                        x2 = int(bbox.x2 * width)
                        y2 = int(bbox.y2 * height)
                        
                        # Ensure valid coordinates
                        x1, x2 = max(0, x1), min(width, x2)
                        y1, y2 = max(0, y1), min(height, y2)
                        
                        if x2 > x1 and y2 > y1:
                            region = img[y1:y2, x1:x2]
                            if region.size > 0:
                                embedding = self.embed_visual(region)
                                embeddings["visual_regions"].append({
                                    "id": f"visual_{element.element_type}_{idx}",
                                    "element_type": element.element_type,
                                    "embedding": embedding.tolist(),
                                    "bbox": bbox.to_list(),
                                    "page": element.page_num,
                                    "confidence": element.confidence,
                                    "metadata": {
                                        "agent": "vision_agent",
                                        "semantic_label": element.metadata.get("semantic_label", ""),
                                        "importance_score": element.metadata.get("importance_score", 0)
                                    }
                                })
                    except Exception as e:
                        logger.warning(f"Failed to extract visual region {idx}: {e}")
        
        # 3. Fused elements (text + visual)
        if hasattr(document, 'aligned_data') and document.aligned_data:
            alignments = document.aligned_data.get("text_visual_alignment", [])
            for i, alignment in enumerate(alignments):
                try:
                    # Get text from aligned words
                    text_content = " ".join(alignment.get("sample_words", []))
                    if text_content and len(text_content) > 10:
                        text_embedding = self.embed_text(text_content)
                        
                        # Get visual region if possible
                        visual_embedding = None
                        page = alignment.get("page", 0)
                        if page < len(document.images) and document.images[page] is not None:
                            bbox = alignment.get("element_bbox")
                            if bbox and len(bbox) == 4:
                                img = document.images[page]
                                height, width = img.shape[:2]
                                x1, y1, x2, y2 = bbox
                                x1, x2 = int(x1 * width), int(x2 * width)
                                y1, y2 = int(y1 * height), int(y2 * height)
                                
                                if x2 > x1 and y2 > y1:
                                    region = img[y1:y2, x1:x2]
                                    if region.size > 0:
                                        visual_embedding = self.embed_visual(region)
                        
                        # Fuse embeddings
                        fused_embedding = self.embed_fused(text_embedding, visual_embedding)
                        
                        embeddings["fused_elements"].append({
                            "id": f"fused_alignment_{i}",
                            "text_content": text_content,
                            "embedding": fused_embedding.tolist(),
                            "page": page,
                            "alignment_confidence": alignment.get("alignment_confidence", 0),
                            "metadata": {
                                "agent": "fusion_agent",
                                "element_type": alignment.get("element_type", ""),
                                "contained_words": alignment.get("contained_words_count", 0)
                            }
                        })
                except Exception as e:
                    logger.warning(f"Failed to create fused element {i}: {e}")
        
        # Add document-level metadata
        embeddings["metadata"] = {
            "document_id": document.document_id,
            "document_type": document.document_type.value if document.document_type else "unknown",
            "total_text_chunks": len(embeddings["text_chunks"]),
            "total_visual_regions": len(embeddings["visual_regions"]),
            "total_fused_elements": len(embeddings["fused_elements"]),
            "risk_score": document.risk_score if hasattr(document, 'risk_score') else 0.0
        }
        
        return embeddings
    
    def _fallback_text_embedding(self, text: str) -> np.ndarray:
        """Simple fallback text embedding (TF-IDF like)"""
        import hashlib
        # Create deterministic embedding from text hash
        text_hash = hashlib.md5(text.encode()).hexdigest()
        # Convert hash to 256-dim vector
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        embedding = np.random.randn(256)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding
    
    def _fallback_visual_embedding(self, image: np.ndarray) -> np.ndarray:
        """Simple fallback visual embedding (color histogram)"""
        if image is None or image.size == 0:
            return np.zeros(256)
        
        try:
            # Compute color histogram
            if len(image.shape) == 2:  # Grayscale
                hist = np.histogram(image.flatten(), bins=32, range=(0, 256))[0]
            else:  # Color
                hist_r = np.histogram(image[:,:,0].flatten(), bins=16, range=(0, 256))[0]
                hist_g = np.histogram(image[:,:,1].flatten(), bins=16, range=(0, 256))[0]
                hist_b = np.histogram(image[:,:,2].flatten(), bins=16, range=(0, 256))[0]
                hist = np.concatenate([hist_r, hist_g, hist_b])
            
            # Normalize
            hist = hist / (np.sum(hist) + 1e-10)
            return hist
            
        except Exception:
            return np.zeros(256)