import numpy as np
import pytest

from tarl.embed.analysis import cluster_failures, kmeans, leakage_report, leave_one_out_nn, pca_2d
from tarl.embed.model import TfidfEmbedder


def test_kmeans_and_pca():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c, 0.1, (20, 5)) for c in (0.0, 3.0, -3.0)])
    labels, C, inertia = kmeans(X, 3, seed=0)
    purity = max(np.mean(labels[:20] == labels[0]), 0) + max(np.mean(labels[20:40] == labels[20]), 0) + max(np.mean(labels[40:] == labels[40]), 0)
    assert purity / 3 >= 0.9 and len(set(labels)) == 3
    coords, ratio = pca_2d(X)
    assert coords.shape == (60, 2) and ratio[0] >= ratio[1]
    with pytest.raises(ValueError):
        kmeans(X, 100)


def test_leakage_flags_identical():
    E = np.eye(4)
    rep = leakage_report(["a", "b"], ["c", "d", "e", "f"], train_emb=E[:2], eval_emb=E, top_k=2, threshold=0.9, train_labels=["ta", "tb"], eval_labels=list("cdef"))
    top = rep[rep["rank"] == 1]
    assert top["flagged"].all() and top["cosine"].min() == pytest.approx(1.0)
    nn = leave_one_out_nn(list("cdef"), E)
    assert nn["cosine"].max() == pytest.approx(0.0)


def test_tfidf_and_clusters():
    texts = ["git merge conflict resolve branch", "git rebase branch merge", "install package apt-get error", "apt install dependency missing", "regex parse log lines", "parse log with regex"]
    tf = TfidfEmbedder().fit(texts)
    X = tf.encode(texts)
    assert X.shape[0] == 6 and np.allclose(np.linalg.norm(X, axis=1), 1.0)
    res = cluster_failures(X, [0.1, 0.2, 0.5, None, 0.3, 0.3], k=3, texts=texts)
    assert sum(res.sizes.values()) == 6 and res.mean_score[res.labels[0]] is not None
    assert all(isinstance(v, list) for v in res.top_terms.values())


@pytest.mark.embed_model
def test_minilm_encode():
    from tarl.embed.model import MiniLMEmbedder

    emb = MiniLMEmbedder(cache_dir=None, threads=2)
    X = emb.encode(["Fix the git repository merge", "Merge the branch in git", "Bake a chocolate cake"])
    assert X.shape == (3, 384) and np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-4)
    assert X[0] @ X[1] > X[0] @ X[2]
