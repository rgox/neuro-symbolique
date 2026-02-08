"""
Feature Extraction Module.

This module provides visual feature extraction using state-of-the-art models:
- CLIP (Contrastive Language-Image Pre-training) for visual-semantic embeddings
- ResNet for visual features
- DINOv2 for self-supervised features

Features are used for:
- Neural-symbolic grounding (VSA binding)
- Object similarity/clustering
- Semantic search
- Cross-modal reasoning

Example:
    >>> extractor = FeatureExtractor(model="clip", variant="ViT-B/32")
    >>> embedding = extractor.extract(image)
    >>> # Store in UMA
    >>> uma.allocate("object_001_emb", embedding.shape, DataType.FLOAT32, DeviceType.NPU)
"""

from typing import Optional, Dict, Any, Tuple, List
import numpy as np
from dataclasses import dataclass
from enum import Enum

try:
    import torch
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class FeatureBackend(Enum):
    """Supported feature extraction backends."""
    CLIP = "clip"  # OpenAI CLIP
    RESNET = "resnet"  # ResNet features
    DINOV2 = "dinov2"  # Meta DINOv2
    MOCK = "mock"  # Mock extractor for testing


@dataclass
class FeatureVector:
    """
    Extracted feature vector with metadata.
    
    Attributes:
        features: Feature vector as numpy array
        dim: Feature dimensionality
        model: Model name used for extraction
        normalized: Whether features are L2-normalized
    """
    features: np.ndarray
    dim: int
    model: str
    normalized: bool = False
    
    def normalize(self) -> 'FeatureVector':
        """L2-normalize the feature vector."""
        if self.normalized:
            return self
        
        norm = np.linalg.norm(self.features)
        if norm > 0:
            normalized_features = self.features / norm
        else:
            normalized_features = self.features
        
        return FeatureVector(
            features=normalized_features,
            dim=self.dim,
            model=self.model,
            normalized=True,
        )
    
    def cosine_similarity(self, other: 'FeatureVector') -> float:
        """Compute cosine similarity with another feature vector."""
        # Ensure both are normalized
        feat1 = self.normalize().features
        feat2 = other.normalize().features
        
        return float(np.dot(feat1, feat2))


class FeatureExtractor:
    """
    Visual feature extractor with multiple backend support.
    
    This class provides a unified interface for extracting visual features
    using different models (CLIP, ResNet, DINOv2, etc.).
    
    Features can be extracted from:
    - Full images
    - Image crops (bounding boxes)
    - Batches of images
    
    Example:
        >>> extractor = FeatureExtractor(backend="clip", model="ViT-B/32")
        >>> image = np.array(Image.open("scene.jpg"))
        >>> features = extractor.extract(image)
        >>> print(f"Extracted {features.dim}-dim feature vector")
    """
    
    def __init__(
        self,
        backend: str = "mock",
        model: str = "ViT-B/32",
        device: str = "cpu",
        normalize: bool = True,
    ):
        """
        Initialize feature extractor.
        
        Args:
            backend: Feature backend ("clip", "resnet", "dinov2", "mock")
            model: Model variant (depends on backend)
            device: Device to run on ("cpu", "cuda", etc.)
            normalize: Whether to L2-normalize features
        """
        self.backend = FeatureBackend(backend)
        self.model_name = model
        self.device = device
        self.normalize_features = normalize
        
        # Load model
        self.model = self._load_model()
        self.preprocess = self._get_preprocessing()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
    
    def _load_model(self):
        """Load feature extraction model."""
        if self.backend == FeatureBackend.MOCK:
            return MockFeatureModel()
        
        elif self.backend == FeatureBackend.CLIP:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for CLIP backend")
            
            try:
                import clip
                model, _ = clip.load(self.model_name, device=self.device)
                model.eval()
                return model
            except ImportError:
                raise ImportError("clip package required for CLIP backend. Install: pip install git+https://github.com/openai/CLIP.git")
        
        elif self.backend == FeatureBackend.RESNET:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for ResNet backend")
            
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.DEFAULT
            model = resnet50(weights=weights)
            
            # Remove classification head to get features
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            model.to(self.device)
            return model
        
        elif self.backend == FeatureBackend.DINOV2:
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch required for DINOv2 backend")
            
            try:
                model = torch.hub.load('facebookresearch/dinov2', self.model_name)
                model.eval()
                model.to(self.device)
                return model
            except Exception as e:
                raise ImportError(f"Failed to load DINOv2 model: {e}")
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _get_preprocessing(self):
        """Get image preprocessing pipeline."""
        if self.backend == FeatureBackend.MOCK:
            return lambda x: x
        
        elif self.backend == FeatureBackend.CLIP:
            import clip
            _, preprocess = clip.load(self.model_name, device=self.device)
            return preprocess
        
        elif self.backend == FeatureBackend.RESNET:
            # ImageNet normalization
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            return preprocess
        
        elif self.backend == FeatureBackend.DINOV2:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])
            return preprocess
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimensionality."""
        if self.backend == FeatureBackend.MOCK:
            return 512  # Mock dimension
        elif self.backend == FeatureBackend.CLIP:
            if "RN50" in self.model_name:
                return 1024
            elif "RN101" in self.model_name:
                return 512
            elif "ViT-B" in self.model_name:
                return 512
            elif "ViT-L" in self.model_name:
                return 768
            else:
                return 512  # Default
        elif self.backend == FeatureBackend.RESNET:
            return 2048  # ResNet50 feature dim
        elif self.backend == FeatureBackend.DINOV2:
            if "vits14" in self.model_name:
                return 384
            elif "vitb14" in self.model_name:
                return 768
            elif "vitl14" in self.model_name:
                return 1024
            elif "vitg14" in self.model_name:
                return 1536
            else:
                return 768  # Default
        else:
            return 512
    
    def extract(
        self,
        image: np.ndarray,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> FeatureVector:
        """
        Extract features from an image or image crop.
        
        Args:
            image: Input image as numpy array (H, W, 3) in RGB format
            bbox: Optional bounding box (x_min, y_min, x_max, y_max) for crop
        
        Returns:
            FeatureVector object with extracted features
        """
        # Crop if bbox provided
        if bbox is not None:
            x_min, y_min, x_max, y_max = bbox
            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)
            image = image[y_min:y_max, x_min:x_max]
        
        # Extract features based on backend
        if self.backend == FeatureBackend.MOCK:
            features = self._extract_mock(image)
        elif self.backend == FeatureBackend.CLIP:
            features = self._extract_clip(image)
        elif self.backend == FeatureBackend.RESNET:
            features = self._extract_resnet(image)
        elif self.backend == FeatureBackend.DINOV2:
            features = self._extract_dinov2(image)
        else:
            raise ValueError(f"Extraction not implemented for backend: {self.backend}")
        
        # Create feature vector
        feature_vec = FeatureVector(
            features=features,
            dim=len(features),
            model=f"{self.backend.value}:{self.model_name}",
            normalized=False,
        )
        
        # Normalize if requested
        if self.normalize_features:
            feature_vec = feature_vec.normalize()
        
        return feature_vec
    
    def _extract_mock(self, image: np.ndarray) -> np.ndarray:
        """Mock feature extraction - returns random features."""
        return np.random.randn(self.feature_dim).astype(np.float32)
    
    def _extract_clip(self, image: np.ndarray) -> np.ndarray:
        """Extract CLIP features."""
        # Convert to PIL image
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
        else:
            raise ImportError("PIL required for CLIP")
        
        # Preprocess
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model.encode_image(image_input)
            features = features.cpu().numpy().flatten()
        
        return features.astype(np.float32)
    
    def _extract_resnet(self, image: np.ndarray) -> np.ndarray:
        """Extract ResNet features."""
        # Convert to PIL image
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
        else:
            raise ImportError("PIL required for ResNet")
        
        # Preprocess
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_input)
            features = features.cpu().numpy().flatten()
        
        return features.astype(np.float32)
    
    def _extract_dinov2(self, image: np.ndarray) -> np.ndarray:
        """Extract DINOv2 features."""
        # Convert to PIL image
        if PIL_AVAILABLE:
            pil_image = Image.fromarray(image)
        else:
            raise ImportError("PIL required for DINOv2")
        
        # Preprocess
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.model(image_input)
            features = features.cpu().numpy().flatten()
        
        return features.astype(np.float32)
    
    def extract_batch(
        self,
        images: List[np.ndarray],
    ) -> List[FeatureVector]:
        """
        Extract features from a batch of images.
        
        Args:
            images: List of images as numpy arrays
        
        Returns:
            List of FeatureVector objects
        """
        return [self.extract(img) for img in images]


class MockFeatureModel:
    """Mock feature model for testing."""
    pass
