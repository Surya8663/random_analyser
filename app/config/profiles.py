"""
Configuration profiles for different environments.
"""
import os
from typing import Dict, Any, Optional
from enum import Enum
from pathlib import Path

from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    DEMO = "demo"

class ConfigProfile:
    """Configuration profile for a specific environment"""
    
    def __init__(self, env: Environment = Environment.DEVELOPMENT):
        self.env = env
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration for the environment"""
        base_config = {
            # Application
            "app_name": "Multi-Modal Document Intelligence System",
            "version": "5.0.0",
            "description": "Phase 5: Multi-Modal RAG with Visualization",
            
            # Server
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "workers": int(os.getenv("WORKERS", "1")),
            "debug": self.env == Environment.DEVELOPMENT,
            
            # Paths
            "base_dir": Path(__file__).parent.parent.parent,
            "upload_dir": Path(os.getenv("UPLOAD_DIR", "uploads")),
            "log_dir": Path("logs"),
            "report_dir": Path("reports"),
            
            # Logging
            "log_level": os.getenv("LOG_LEVEL", "INFO"),
            "log_file": f"app_{self.env.value}.log",
            
            # Processing
            "max_file_size": int(os.getenv("MAX_FILE_SIZE", "100000000")),  # 100MB
            "allowed_extensions": [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"],
            "max_pages": int(os.getenv("MAX_PAGES", "50")),
            
            # RAG Configuration
            "rag_index_dir": Path("rag_indices"),
            "rag_default_index": "default",
            "rag_text_model": os.getenv("RAG_TEXT_MODEL", "all-MiniLM-L6-v2"),
            "rag_vision_model": os.getenv("RAG_VISION_MODEL", "ViT-B/32"),
            "rag_search_k": int(os.getenv("RAG_SEARCH_K", "10")),
            "rag_cache_size": int(os.getenv("RAG_CACHE_SIZE", "1000")),
            
            # Visualization
            "visualization_output_dir": Path("visualization_reports"),
            "visualization_max_images": int(os.getenv("VISUALIZATION_MAX_IMAGES", "10")),
            
            # Model Caching
            "model_cache_dir": Path(os.getenv("MODEL_CACHE_DIR", "models")),
            "download_models": self.env != Environment.DEMO,
        }
        
        # Environment-specific overrides
        env_configs = {
            Environment.DEVELOPMENT: {
                "debug": True,
                "workers": 1,
                "log_level": "DEBUG",
                "download_models": True,
                "enable_profiling": True,
            },
            Environment.TESTING: {
                "debug": False,
                "workers": 1,
                "log_level": "INFO",
                "upload_dir": Path("test_uploads"),
                "rag_index_dir": Path("test_rag_indices"),
            },
            Environment.STAGING: {
                "debug": False,
                "workers": 2,
                "log_level": "INFO",
                "max_file_size": 50000000,  # 50MB
                "enable_monitoring": True,
            },
            Environment.PRODUCTION: {
                "debug": False,
                "workers": 4,
                "log_level": "WARNING",
                "max_file_size": 50000000,  # 50MB
                "enable_monitoring": True,
                "enable_metrics": True,
                "enable_tracing": True,
            },
            Environment.DEMO: {
                "debug": False,
                "workers": 1,
                "log_level": "INFO",
                "max_pages": 10,
                "download_models": False,  # Use fallback models for demo
                "enable_demo_mode": True,
            }
        }
        
        # Merge with environment-specific config
        env_config = env_configs.get(self.env, {})
        base_config.update(env_config)
        
        # Ensure directories exist
        self._ensure_directories(base_config)
        
        return base_config
    
    def _ensure_directories(self, config: Dict[str, Any]):
        """Ensure all required directories exist"""
        directories = [
            config["upload_dir"],
            config["log_dir"],
            config["report_dir"],
            config["rag_index_dir"],
            config["visualization_output_dir"],
            config["model_cache_dir"],
        ]
        
        for directory in directories:
            if isinstance(directory, Path):
                directory.mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value using bracket notation"""
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if configuration key exists"""
        return key in self.config
    
    def print_summary(self):
        """Print configuration summary"""
        logger.info(f"📋 Configuration Profile: {self.env.value}")
        logger.info(f"   App: {self.config['app_name']} v{self.config['version']}")
        logger.info(f"   Server: {self.config['host']}:{self.config['port']} ({self.config['workers']} workers)")
        logger.info(f"   Debug: {self.config['debug']}")
        logger.info(f"   Upload Dir: {self.config['upload_dir']}")
        logger.info(f"   RAG Index Dir: {self.config['rag_index_dir']}")

# Global configuration instance
_config_instance: Optional[ConfigProfile] = None

def get_config(env: Optional[str] = None) -> ConfigProfile:
    """Get configuration instance (singleton)"""
    global _config_instance
    
    if _config_instance is None:
        # Determine environment
        env_str = env or os.getenv("ENVIRONMENT", "development")
        try:
            environment = Environment(env_str.lower())
        except ValueError:
            logger.warning(f"Unknown environment: {env_str}, defaulting to development")
            environment = Environment.DEVELOPMENT
        
        # Create config instance
        _config_instance = ConfigProfile(environment)
        _config_instance.print_summary()
    
    return _config_instance

# Convenience function for common config values
def config_value(key: str, default: Any = None) -> Any:
    """Get a configuration value"""
    config = get_config()
    return config.get(key, default)