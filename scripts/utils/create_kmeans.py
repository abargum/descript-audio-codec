import logging
import torch
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def cluster_file(features_path, n_clusters):
    
    checkpoint_dir = Path("checkpoints") 
    
    # Load features
    logger.info(f"Loading features from {features_path}")
    features = np.load(features_path)
    logger.info(f"Loaded features with shape: {features.shape}")
    
    # Perform clustering
    logger.info(f"Starting KMeans clustering with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, verbose=0).fit(features)
    
    # Prepare checkpoint path
    checkpoint_path = checkpoint_dir / f"kmeans_{n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the model
    logger.info(f"Saving model to {checkpoint_path}")
    torch.save(
        {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        },
        checkpoint_path
    )
    
    logger.info("Clustering completed successfully")

def cluster(features, features_path, n_clusters):
    
    checkpoint_dir = Path("checkpoints") 
    
    # Perform clustering
    logger.info(f"Starting KMeans clustering with {n_clusters} clusters")
    kmeans = KMeans(n_clusters=n_clusters, verbose=0).fit(features)
    
    # Prepare checkpoint path
    checkpoint_path = checkpoint_dir / f"kmeans_{n_clusters}.pt"
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    
    # Save the model
    logger.info(f"Saving model to {checkpoint_path}")
    torch.save(
        {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        },
        checkpoint_path
    )
    
    logger.info("Clustering completed successfully")

def _kmeans(
    num_clusters: int, pretrained: bool = True, progress: bool = True, checkpoint: str = None) -> KMeans:
    kmeans = KMeans(num_clusters)
    if pretrained:
        if checkpoint:
            # Load from local checkpoint file
            checkpoint_data = torch.load(checkpoint)
        else:
            # Original URL loading behavior
            checkpoint_data = torch.hub.load_state_dict_from_url(
                URLS[f"kmeans{num_clusters}"], progress=progress
            )
        
        kmeans.__dict__["n_features_in_"] = checkpoint_data["n_features_in_"]
        kmeans.__dict__["_n_threads"] = checkpoint_data["_n_threads"]
        kmeans.__dict__["cluster_centers_"] = checkpoint_data["cluster_centers_"]
    return kmeans

def kmeans100(pretrained: bool = True, progress: bool = True, checkpoint: str = None) -> KMeans:
    r"""
    k-means checkpoint for HuBERT-Discrete with 100 clusters.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        checkpoint (str): path to local checkpoint file
    """
    return _kmeans(100, pretrained, progress, checkpoint)

def kmeans(pretrained: bool = True, clusters: int = 100, progress: bool = True, checkpoint: str = None) -> KMeans:
    r"""
    k-means checkpoint for HuBERT-Discrete with N clusters.
    Args:
        pretrained (bool): load pretrained weights into the model
        progress (bool): show progress bar when downloading model
        checkpoint (str): path to local checkpoint file
    """
    return _kmeans(clusters, pretrained, progress, checkpoint)

if __name__ == "__main__":
     
    features_path = "all_features.npy" 
    n_clusters = 100
    
    cluster_file(features_path, n_clusters)