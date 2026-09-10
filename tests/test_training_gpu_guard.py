import sys

import pytest

from tarl.common.env import GPUUnavailable, has_cuda
from tarl.training.model import LORA_TARGET_REGEX, load_trainable


@pytest.mark.skipif(has_cuda(), reason="only meaningful without a GPU")
def test_load_trainable_guarded():
    with pytest.raises(GPUUnavailable):
        load_trainable()
    assert "bitsandbytes" not in sys.modules
    assert "language_model" in LORA_TARGET_REGEX
