from typing import Dict, List, Optional, Set
import torch
import torch.nn as nn
import os
import numpy as np
import logging
from iopath.common.file_io import PathManager as PathManagerBase
import pickle

PathManager = PathManagerBase()


def normalize_embeddings(embeddings: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Normalize N D-dimensional embedding vectors arranged in a tensor [N, D]

    Args:
        embeddings (tensor [N, D]): N D-dimensional embedding vectors
        epsilon (float): minimum value for a vector norm
    Return:
        Normalized embeddings (tensor [N, D]), such that L2 vector norms are all equal to 1.
    """
    return embeddings / torch.clamp(embeddings.norm(p=None, dim=1, keepdim=True), min=epsilon)

def create_embedder(embedder_spec: Dict, embed_dim: int) -> nn.Module:
    """
    Create an embedder based on the provided configuration

    Args:
        embedder_spec (CfgNode): embedder configuration
        embedder_dim (int): embedding space dimensionality
    Return:
        An embedder instance for the specified configuration
        Raises ValueError, in case of unexpected  embedder type
    """
    emb_type = embedder_spec["TYPE"]
    num_vertices = embedder_spec["NUM_VERTICES"]
    init_file = embedder_spec.get("INIT_FILE", "")
    is_trainable = embedder_spec.get("IS_TRAINABLE", True)

    if emb_type == "vertex_feature":
        embedder = VertexFeatureEmbedder(
            num_vertices=num_vertices,
            feature_dim=embedder_spec["FEATURE_DIM"],
            embed_dim=embed_dim,
            train_features=embedder_spec.get("FEATURES_TRAINABLE", False),
        )
    else:
        raise ValueError(f"Unknown embedder type: {emb_type}")

    if init_file:
        embedder.load(init_file)

    if not is_trainable:
        embedder.requires_grad_(False)

    return embedder

class VertexFeatureEmbedder(nn.Module):
    """
    Class responsible for embedding vertex features. Mapping from
    feature space to the embedding space is a tensor of size [K, D], where
        K = number of dimensions in the feature space
        D = number of dimensions in the embedding space
    Vertex features is a tensor of size [N, K], where
        N = number of vertices
        K = number of dimensions in the feature space
    Vertex embeddings are computed as F * E = tensor of size [N, D]
    """

    def __init__(
        self, num_vertices: int, feature_dim: int, embed_dim: int, train_features: bool = False
    ):
        """
        Initialize embedder, set random embeddings

        Args:
            num_vertices (int): number of vertices to embed
            feature_dim (int): number of dimensions in the feature space
            embed_dim (int): number of dimensions in the embedding space
            train_features (bool): determines whether vertex features should
                be trained (default: False)
        """
        super(VertexFeatureEmbedder, self).__init__()
        if train_features:
            self.features = nn.Parameter(torch.Tensor(num_vertices, feature_dim))
        else:
            self.register_buffer("features", torch.Tensor(num_vertices, feature_dim))
        self.embeddings = nn.Parameter(torch.Tensor(feature_dim, embed_dim))
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        self.features.zero_()
        self.embeddings.zero_()

        # nn.init.orthogonal_(self.embeddings)

    def forward(self) -> torch.Tensor:
        """
        Produce vertex embeddings, a tensor of shape [N, D] where:
            N = number of vertices
            D = number of dimensions in the embedding space

        Return:
           Full vertex embeddings, a tensor of shape [N, D]
        """
        return normalize_embeddings(torch.mm(self.features, self.embeddings))

    @torch.no_grad()
    def load(self, fpath: str):
        """
        Load data from a file

        Args:
            fpath (str): file path to load data from
        """
        with PathManager.open(fpath, "rb") as hFile:
            data = pickle.load(hFile)
            for name in ["features", "embeddings"]:
                if name in data:
                    getattr(self, name).copy_(
                        torch.tensor(data[name]).float().to(device=getattr(self, name).device)
                    )



class Embedder(nn.Module):
    """
    Embedder module that serves as a container for embedders to use with different
    meshes. Extends Module to automatically save / load state dict.
    """

    DEFAULT_MODEL_CHECKPOINT_PREFIX = "roi_heads.embedder."

    def __init__(self, embed_dim, mesh_specs, weight_dict_file=None):
        """
        Initialize mesh embedders. An embedder for mesh `i` is stored in a submodule
        "embedder_{i}".

        Args:
            cfg (CfgNode): configuration options
        """
        super(Embedder, self).__init__()
        self.mesh_names = set()
        logger = logging.getLogger(__name__)
        for mesh_name, embedder_spec in mesh_specs.items():
            logger.info(f"Adding embedder embedder_{mesh_name} with spec {embedder_spec}")
            self.add_module(f"embedder_{mesh_name}", create_embedder(embedder_spec, embed_dim))
            self.mesh_names.add(mesh_name)
        if weight_dict_file != None:
            self.load_from_model_checkpoint(weight_dict_file)

    def load_from_model_checkpoint(self, fpath: str, prefix: Optional[str] = None):
        if prefix is None:
            prefix = Embedder.DEFAULT_MODEL_CHECKPOINT_PREFIX
        state_dict = None
        if fpath.endswith(".pkl"):
            with PathManager.open(fpath, "rb") as hFile:
                state_dict = pickle.load(hFile, encoding="latin1")
        else:
            with PathManager.open(fpath, "rb") as hFile:
                state_dict = torch.load(hFile, map_location=torch.device("cpu"))
        if state_dict is not None and "model" in state_dict:
            state_dict_local = {}
            for key in state_dict["model"]:
                if key.startswith(prefix):
                    print(key)
                    v_key = state_dict["model"][key]
                    if isinstance(v_key, np.ndarray):
                        v_key = torch.from_numpy(v_key)
                    state_dict_local[key[len(prefix) :]] = v_key
            # non-strict loading to finetune on different meshes
            self.load_state_dict(state_dict_local, strict=False)

    def forward(self, mesh_name: str) -> torch.Tensor:
        """
        Produce vertex embeddings for the specific mesh; vertex embeddings are
        a tensor of shape [N, D] where:
            N = number of vertices
            D = number of dimensions in the embedding space
        Args:
            mesh_name (str): name of a mesh for which to obtain vertex embeddings
        Return:
            Vertex embeddings, a tensor of shape [N, D]
        """
        return getattr(self, f"embedder_{mesh_name}")()

    def has_embeddings(self, mesh_name: str) -> bool:
        return hasattr(self, f"embedder_{mesh_name}")