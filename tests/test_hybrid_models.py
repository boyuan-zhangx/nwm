import torch

from hybrid_models import HybridCDiT, MemoryBuffer
from models import CDiT


def _inputs(batch: int = 2):
    return {
        "x": torch.randn(batch, 4, 8, 8),
        "t": torch.randint(0, 1000, (batch,)),
        "y": torch.randn(batch, 3),
        "x_cond": torch.randn(batch, 2, 4, 8, 8),
        "rel_t": torch.rand(batch),
    }


def _models():
    kwargs = dict(
        input_size=8,
        context_size=2,
        patch_size=2,
        in_channels=4,
        hidden_size=48,
        depth=2,
        num_heads=6,
    )
    baseline = CDiT(**kwargs)
    hybrid = HybridCDiT(**kwargs, memory_layers=[1])
    return baseline, hybrid


def test_memory_buffer_prefers_matching_turn_and_reports_diagnostics():
    buffer = MemoryBuffer(max_size=4)
    buffer.add_frame(
        torch.zeros(4, 2, 2),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([1.0, 0.0, 0.0]),
        frame_idx=10,
    )
    buffer.add_frame(
        torch.ones(4, 2, 2),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.2]),
        frame_idx=20,
    )

    result = buffer.query(
        torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), k=1
    )

    assert result is not None
    assert result.frames.shape == (1, 4, 2, 2)
    assert result.frame_indices.tolist() == [20]
    assert result.scores.shape == (1,)


def test_memory_buffer_is_bounded_and_clearable():
    buffer = MemoryBuffer(max_size=2)
    for index in range(3):
        buffer.add_frame(
            torch.full((4, 2, 2), float(index)),
            torch.tensor([float(index), 0.0]),
            frame_idx=index,
        )
    assert len(buffer) == 2
    assert buffer.frame_indices == [1, 2]
    buffer.clear()
    assert len(buffer) == 0


def test_hybrid_loads_baseline_checkpoint_without_changing_baseline_output():
    torch.manual_seed(7)
    baseline, hybrid = _models()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(mean=0.0, std=0.02)

    incompatibility = hybrid.load_state_dict(baseline.state_dict(), strict=False)
    assert incompatibility.unexpected_keys == []
    assert incompatibility.missing_keys
    assert all("memory_" in key for key in incompatibility.missing_keys)

    inputs = _inputs()
    memory = torch.randn(2, 3, 4, 8, 8)
    baseline.eval()
    hybrid.eval()
    with torch.no_grad():
        expected = baseline(**inputs)
        without_memory = hybrid(**inputs)
        with_zero_initialized_memory = hybrid(**inputs, memory_latents=memory)

    torch.testing.assert_close(without_memory, expected)
    torch.testing.assert_close(with_zero_initialized_memory, expected)


def test_memory_branch_accepts_latents_and_can_change_output_after_gate_opens():
    torch.manual_seed(11)
    baseline, hybrid = _models()
    with torch.no_grad():
        for parameter in baseline.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    hybrid.load_state_dict(baseline.state_dict(), strict=False)

    hidden = 48
    memory_block = hybrid.blocks[1]
    with torch.no_grad():
        memory_block.memory_adaLN[-1].bias[2 * hidden :].fill_(1.0)

    inputs = _inputs()
    first_memory = torch.randn(2, 3, 4, 8, 8)
    second_memory = first_memory + 2.0
    with torch.no_grad():
        first_output = hybrid(**inputs, memory_latents=first_memory)
        second_output = hybrid(**inputs, memory_latents=second_memory)

    assert first_output.shape == (2, 8, 8, 8)
    assert not torch.allclose(first_output, second_output)


def test_memory_mask_can_disable_memory_per_sample():
    _, hybrid = _models()
    inputs = _inputs()
    memory = torch.randn(2, 3, 4, 8, 8)
    mask = torch.tensor([[True, True, False], [False, False, False]])
    output = hybrid(**inputs, memory_latents=memory, memory_mask=mask)
    assert output.shape == (2, 8, 8, 8)
