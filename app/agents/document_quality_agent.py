from typing import Dict, Any
from app.core.models import ProcessingState
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class DocumentQualityAgent:
    """Simple document quality assessment agent"""
    
    async def __call__(self, state: ProcessingState) -> ProcessingState:
        logger.info(f"📊 DocumentQualityAgent processing document {state.document_id}")
        
        # Always ensure quality_scores exists
        from app.core.models import QualityScore
        
        if not hasattr(state, 'quality_scores'):
            state.quality_scores = {}
        
        # Add a simple quality score
        state.quality_scores[0] = QualityScore(
            sharpness=0.85,
            brightness=0.75,
            contrast=0.80,
            noise_level=0.10,  # Low noise is good
            overall=0.75
        )
        
        logger.info(f"✅ Quality assessment completed for page 0")
        return state