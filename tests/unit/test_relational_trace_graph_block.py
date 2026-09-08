"""Explicit-oracle and fusion contracts for a synchronous relation round."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from seis_interp.models.relational_trace_graph import RelationalTraceGraphMessageBlock


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(15)
    latents = torch.randn(5, 8, 3, generator=generator)
    edges = torch.tensor([[0, 1, 2, 0, 3, 1, 2], [3, 3, 3, 3, 4, 4, 0]])
    relations = torch.tensor([0, 0, 0, 2, 1, 3, 0])
    features = torch.randn(7, 15, generator=generator)
    return latents, edges, relations, features


def _oracle(
    block: RelationalTraceGraphMessageBlock,
    latents: torch.Tensor,
    edges: torch.Tensor,
    relations: torch.Tensor,
    features: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    updated = latents + block.temporal_update(block.temporal(F.silu(block.temporal_norm(latents))))
    pooled = updated.mean(dim=-1)
    messages = torch.zeros(len(latents), 4, *latents.shape[1:])
    attention = torch.zeros(len(relations))
    for destination in range(len(latents)):
        for relation in range(4):
            selected = [
                index
                for index in range(len(relations))
                if edges[1, index] == destination and relations[index] == relation
            ]
            logits = []
            values = []
            for index in selected:
                sender = edges[0, index]
                context = torch.cat((features[index], block.relation_embedding.weight[relation]))
                logits.append(
                    block.attention(torch.cat((pooled[destination], pooled[sender], context)))[0]
                )
                gamma = 2 * torch.sigmoid(block.gamma(context))
                values.append(
                    gamma[:, None] * block.value_projection(updated[sender : sender + 1])[0]
                )
            if selected:
                weights = torch.softmax(torch.stack(logits), dim=0)
                assert torch.allclose(weights.sum(), torch.tensor(1.0))
                for index, weight, value in zip(selected, weights, values, strict=True):
                    messages[destination, relation] += weight * value
                    attention[index] = weight
    aggregate = torch.zeros_like(latents)
    for destination in range(len(latents)):
        available = set(relations[edges[1] == destination].tolist())
        if available:
            aggregate[destination] = sum(
                messages[destination, relation] for relation in available
            ) / len(available)
    return updated + block.message_update(aggregate), attention


def test_fixed_mean_matches_loop_attention_gamma_and_residual_oracle() -> None:
    torch.manual_seed(4)
    block = RelationalTraceGraphMessageBlock(8, attention_width=5)
    inputs = _fixture()
    expected, attention = _oracle(block, *inputs)
    actual = block(*inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert attention.shape == (7,)
    assert (attention > 0).all()


def test_mean_weights_relations_equally_despite_different_degrees() -> None:
    block = RelationalTraceGraphMessageBlock(8)
    latents, edges, relations, features = _fixture()
    messages, valid = block._relation_messages(latents, edges, relations, features)
    weights = block._relation_weights(latents, messages, valid, None)
    torch.testing.assert_close(weights[3], torch.tensor([0.5, 0.0, 0.5, 0.0]))
    torch.testing.assert_close(weights[4], torch.tensor([0.0, 0.5, 0.0, 0.5]))
    assert torch.equal(messages[1], torch.zeros_like(messages[1]))
    assert torch.equal(weights[1], torch.zeros(4))


@pytest.mark.parametrize("fusion", ["mean", "learned_gate"])
def test_empty_edges_are_finite_and_have_exact_zero_messages(fusion: str) -> None:
    block = RelationalTraceGraphMessageBlock(8, relation_fusion=fusion)
    latents = torch.randn(1, 8, 1)
    edges = torch.empty(2, 0, dtype=torch.int64)
    relations = torch.empty(0, dtype=torch.int64)
    features = torch.empty(0, 15)
    coverage = torch.zeros(1, 4, 2)
    messages, valid = block._relation_messages(latents, edges, relations, features)
    assert torch.equal(messages, torch.zeros_like(messages))
    assert not valid.any()
    diagnostics = {}
    actual = block(
        latents,
        edges,
        relations,
        features,
        coverage,
        diagnostics=diagnostics,
        diagnostic_query_indices=torch.tensor([0]),
    )
    assert torch.isfinite(actual).all()
    assert torch.equal(diagnostics["gate_sum"], torch.zeros(4))
    assert diagnostics["context_query_count"] == 0
    assert diagnostics["no_context_query_count"] == 1


def test_diagnostics_select_queries_and_exclude_support_from_counts() -> None:
    block = RelationalTraceGraphMessageBlock(8)
    diagnostics = {}
    block(*_fixture(), diagnostics=diagnostics, diagnostic_query_indices=torch.tensor([3, 1]))
    assert torch.equal(diagnostics["gate_sum"], torch.tensor([0.5, 0.0, 0.5, 0.0]))
    assert torch.equal(diagnostics["available_query_count"], torch.tensor([1, 0, 1, 0]))
    assert diagnostics["context_query_count"] == 1
    assert diagnostics["no_context_query_count"] == 1


def test_diagnostics_require_explicit_query_selection() -> None:
    block = RelationalTraceGraphMessageBlock(8)
    with pytest.raises(ValueError, match="diagnostic_query_indices"):
        block(*_fixture(), diagnostics={})


def test_synchronous_update_is_equivariant_to_node_and_edge_order() -> None:
    torch.manual_seed(11)
    block = RelationalTraceGraphMessageBlock(8)
    latents, edges, relations, features = _fixture()
    expected = block(latents, edges, relations, features)
    node_order = torch.tensor([4, 2, 0, 3, 1])
    edge_order = torch.tensor([6, 3, 5, 1, 0, 4, 2])
    inverse = torch.argsort(node_order)
    permuted = block(
        latents[node_order],
        inverse[edges[:, edge_order]],
        relations[edge_order],
        features[edge_order],
    )
    torch.testing.assert_close(permuted, expected[node_order], atol=1e-6, rtol=1e-5)


def test_gradients_flow_through_latents_and_shared_message_parameters() -> None:
    torch.manual_seed(2)
    block = RelationalTraceGraphMessageBlock(8, attention_width=5)
    latents, edges, relations, features = _fixture()
    latents.requires_grad_()
    output = block(latents, edges, relations, features)
    (output * torch.arange(output.numel()).reshape_as(output)).sum().backward()
    assert latents.grad is not None and latents.grad.abs().sum() > 0
    for module in (block.attention, block.gamma, block.value_projection, block.relation_embedding):
        assert sum(parameter.grad.abs().sum() for parameter in module.parameters()) > 0


def test_zero_initialized_gate_matches_mean_and_masks_empty_relations() -> None:
    torch.manual_seed(2)
    mean = RelationalTraceGraphMessageBlock(8)
    learned = RelationalTraceGraphMessageBlock(8, relation_fusion="learned_gate")
    missing, unexpected = learned.load_state_dict(mean.state_dict(), strict=False)
    assert missing and all(name.startswith("relation_gate.") for name in missing)
    assert not unexpected
    inputs = _fixture()
    latents, edges, relations, features = inputs
    coverage = torch.rand(5, 4, 2)
    torch.testing.assert_close(mean(*inputs), learned(*inputs, coverage), atol=1e-6, rtol=1e-5)
    messages, valid = learned._relation_messages(latents, edges, relations, features)
    weights = learned._relation_weights(latents, messages, valid, coverage)
    assert torch.equal(weights[0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.equal(weights[~valid], torch.zeros_like(weights[~valid]))
    assert torch.isfinite(weights).all()


def test_learning_gate_changes_weights_and_output_and_receives_gradients() -> None:
    torch.manual_seed(19)
    block = RelationalTraceGraphMessageBlock(8, relation_fusion="learned_gate")
    latents, edges, relations, features = _fixture()
    coverage = torch.rand(5, 4, 2)
    before = block(latents, edges, relations, features, coverage).detach()
    optimizer = torch.optim.SGD(block.relation_gate.parameters(), lr=0.1)
    for _ in range(3):
        optimizer.zero_grad()
        output = block(latents, edges, relations, features, coverage)
        output[3:].square().sum().backward()
        optimizer.step()
    assert block.relation_gate[0].weight.grad.abs().sum() > 0
    assert block.relation_gate[-1].weight.grad.abs().sum() > 0
    after = block(latents, edges, relations, features, coverage)
    assert not torch.allclose(before, after)
    messages, valid = block._relation_messages(latents, edges, relations, features)
    weights = block._relation_weights(latents, messages, valid, coverage)
    assert not torch.allclose(weights[3, 0], torch.tensor(0.5))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"width": 12}, "divisible"),
        ({"temporal_kernel_size": 2}, "odd"),
        ({"temporal_dilation": 0}, "positive"),
        ({"relation_fusion": "sum"}, "relation_fusion"),
    ],
)
def test_invalid_block_config_is_rejected(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        RelationalTraceGraphMessageBlock(**{"width": 8, **kwargs})
