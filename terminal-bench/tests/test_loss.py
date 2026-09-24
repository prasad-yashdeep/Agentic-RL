import pytest
import torch

from tarl.training.loss import clipped_surrogate, grpo_step, token_logprobs, zero_grad_leak_check


def test_ratio_one_reduces_to_reinforce():
    lp = torch.tensor([-1.0, -2.0, -0.5, -3.0], requires_grad=True)
    old = lp.detach().clone()
    mask = torch.tensor([1, 1, 0, 1])
    loss, st = clipped_surrogate(lp, old, 0.25, mask)
    assert float(loss) == pytest.approx(-0.25 * 3)
    assert st["n_tok"] == 3 and st["clip_frac"] == 0.0
    loss.backward()
    assert lp.grad[2] == 0.0  # masked token inert
    assert lp.grad[0] == pytest.approx(-0.25)  # d/dlogp of -ratio*A at ratio=1


def test_clip_sides_zero_gradient():
    old = torch.zeros(3)
    # positive advantage, ratio far above 1+0.28 -> clipped branch is constant -> zero grad
    new = torch.full((3,), 1.0, requires_grad=True)
    loss, st = clipped_surrogate(new, old, 0.5, torch.ones(3))
    loss.backward()
    assert torch.allclose(new.grad, torch.zeros(3)) and st["clip_frac"] == 1.0
    # negative advantage, ratio far below 1-0.2 -> zero grad
    new2 = torch.full((3,), -1.0, requires_grad=True)
    loss2, _ = clipped_surrogate(new2, old, -0.5, torch.ones(3))
    loss2.backward()
    assert torch.allclose(new2.grad, torch.zeros(3))
    # negative advantage with ratio above 1 is NOT clipped (pessimistic min): non-zero grad
    new3 = torch.full((3,), 1.0, requires_grad=True)
    loss3, _ = clipped_surrogate(new3, old, -0.5, torch.ones(3))
    loss3.backward()
    assert float(new3.grad.abs().sum()) > 0


class ToyLM(torch.nn.Module):
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab, dim)
        self.lora_A = torch.nn.Linear(dim, vocab)  # 'lora' in the name -> trainable per leak check

    @property
    def device(self):
        return self.emb.weight.device

    def forward(self, input_ids):
        h = self.emb(input_ids)
        return type("O", (), {"logits": self.lora_A(h)})()


def test_grpo_step_on_toy_model():
    torch.manual_seed(0)
    m = ToyLM()
    for n, p in m.named_parameters():
        p.requires_grad_("lora" in n)
    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad], lr=0.1)
    ids = torch.tensor([1, 2, 3, 4, 5])
    mask = torch.tensor([0, 0, 1, 1, 1])
    before = m.lora_A.weight.detach().clone()
    lp0 = token_logprobs(m, ids).detach()
    stats = grpo_step(m, opt, [(ids, mask, 0.3), (ids, mask, -0.1)])
    zero_grad_leak_check(m)
    assert stats["tokens"] == 6 and torch.isfinite(torch.tensor(stats["loss/tok"]))
    assert not torch.equal(before, m.lora_A.weight.detach())
    assert lp0.shape == (4,)
