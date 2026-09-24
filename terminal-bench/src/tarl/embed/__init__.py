"""Sentence embeddings (MiniLM via transformers, TF-IDF fallback) and the analyses built on them:
train/eval leakage guard, task map, failure-mode clusters."""

from tarl.embed.model import (  # noqa: F401
    DEFAULT_MODEL,
    EmbedModelUnavailable,
    MiniLMEmbedder,
    TfidfEmbedder,
    embed_texts,
    ensure_model,
    get_embedder,
)
from tarl.embed.analysis import (  # noqa: F401
    ClusterResult,
    cluster_failures,
    cosine_matrix,
    kmeans,
    leakage_report,
    leave_one_out_nn,
    nearest_neighbours,
    pca_2d,
    proxy_train_set,
    task_embeddings,
    trajectory_embeddings,
    trajectory_texts,
)
