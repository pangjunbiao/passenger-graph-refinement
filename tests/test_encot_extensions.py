import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from src.v6.encot_bridge import official_affinity_from_logits
from src.v6.encot_extensions import (
    ExtensionWeights,
    V6EnCOT,
    extension_parameter_counts,
    rank_aware_lexical_loss,
)


class _DummyParent(nn.Module):
    def __init__(self, vocabulary=19, topics=4, dimension=7):
        super().__init__()
        self.vocab_size = vocabulary
        self.num_topics = topics
        self.beta_temp = 0.4
        self.alpha_augment = 0.05
        self.weight_ot_doc_cluster = 1.0
        self.weight_ot_topic_cluster = 1.0
        generator = torch.Generator().manual_seed(41)
        self.topic_embeddings = nn.Parameter(
            torch.randn((topics, dimension), generator=generator)
        )
        self.word_embeddings = nn.Parameter(
            F.normalize(torch.randn((vocabulary, dimension), generator=generator), dim=1)
        )
        self.local = nn.Linear(vocabulary, topics)
        self.global_layer = nn.Linear(vocabulary, topics)
        self.decoder_bn = nn.BatchNorm1d(vocabulary)

    def get_beta(self):
        x = self.topic_embeddings
        y = self.word_embeddings
        distance = (x * x).sum(1, keepdim=True) + (y * y).sum(1) - 2 * x @ y.T
        return official_affinity_from_logits(-distance / self.beta_temp)

    def noise_local_encode(self, values):
        return F.softmax(self.local(values), dim=1), values.new_zeros(())

    def global_encode(self, values):
        return F.softmax(self.global_layer(values), dim=1), values.new_zeros(())

    def get_loss_ECR(self):
        return (self.topic_embeddings * 0).sum()

    def compute_ot_loss_doc_cluster(self, theta):
        return (theta * 0).sum()

    def compute_ot_loss_topic_cluster(self):
        return (self.topic_embeddings * 0).sum()

    def forward(self, values, is_ECR=True):
        local = values[:, : self.vocab_size]
        global_values = values[:, self.vocab_size :]
        local_theta, local_kl = self.noise_local_encode(local)
        global_theta, _ = self.global_encode(global_values)
        recon = F.softmax(
            self.decoder_bn((local_theta * global_theta) @ self.get_beta()), dim=-1
        )
        reconstruction = -(
            (local + self.alpha_augment * global_values) * recon.log()
        ).sum(dim=1).mean()
        loss_tm = reconstruction + local_kl
        ecr = self.get_loss_ECR() if is_ECR else reconstruction.new_zeros(())
        doc = self.compute_ot_loss_doc_cluster(local_theta * global_theta)
        topic = self.compute_ot_loss_topic_cluster()
        return {
            "loss": loss_tm + ecr + doc + topic,
            "loss_TM": loss_tm,
            "loss_ECR": ecr,
            "ot_loss_doc_cluster": doc,
            "ot_loss_topic_cluster": topic,
        }


def _wrapper():
    parent = _DummyParent().double()
    generator = torch.Generator().manual_seed(43)
    features = F.normalize(
        torch.randn((parent.vocab_size, 7), generator=generator, dtype=torch.float64),
        dim=1,
    )
    return V6EnCOT(
        parent,
        frozen_vocabulary_features=features,
        city_weights=torch.tensor([0.55, 0.25, 0.20], dtype=torch.float64),
    ).double()


def test_city_residual_has_both_required_identifiability_constraints():
    model = _wrapper()
    with torch.no_grad():
        model.city_contrast_coefficients.normal_(0.0, 0.2)
        model.gate_logits.normal_(-0.4, 0.1)
    state = model.city_residual_state()
    weighted = torch.einsum(
        "c,ckv->kv", model.city_weights, state.effective_logit_residuals
    )
    torch.testing.assert_close(weighted, torch.zeros_like(weighted), atol=2e-15, rtol=0)
    torch.testing.assert_close(
        state.effective_logit_residuals.sum(dim=1),
        torch.zeros_like(state.effective_logit_residuals[:, 0]),
        atol=2e-15,
        rtol=0,
    )


def test_alpha_zero_delegates_to_exact_parent_forward_and_decoder():
    model = _wrapper().eval()
    generator = torch.Generator().manual_seed(47)
    with torch.no_grad():
        model.city_contrast_coefficients.normal_(0.0, 0.3, generator=generator)
        model.gate_logits.normal_(0.2, 0.1, generator=generator)
    values = torch.rand((8, 2 * model.vocabulary_size), generator=generator, dtype=torch.float64)
    direct = model.parent(values)
    wrapped = model(
        values,
        None,
        residual_scale=0.0,
        weights=ExtensionWeights(),
    )
    for key in direct:
        torch.testing.assert_close(wrapped[key], direct[key], atol=0.0, rtol=0.0)
    with torch.no_grad():
        local, _ = model.parent.noise_local_encode(values[:, : model.vocabulary_size])
        global_theta, _ = model.parent.global_encode(values[:, model.vocabulary_size :])
        direct_probability = F.softmax(
            model.parent.decoder_bn((local * global_theta) @ model.parent.get_beta()),
            dim=-1,
        )
        wrapped_probability = model.infer_probabilities(
            values, None, residual_scale=0.0
        )
    torch.testing.assert_close(wrapped_probability, direct_probability, atol=0, rtol=0)


def test_city_parameters_receive_reconstruction_gradient():
    model = _wrapper().eval()
    generator = torch.Generator().manual_seed(53)
    values = torch.rand((11, 2 * model.vocabulary_size), generator=generator, dtype=torch.float64)
    cities = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1])
    with torch.no_grad():
        model.city_contrast_coefficients.normal_(0.0, 0.05)
    result = model.reconstruction_objective(
        values,
        cities,
        residual_scale=1.0,
        weights=ExtensionWeights(pooling=0.01, gate=0.001),
    )
    gradients = torch.autograd.grad(
        result["loss"],
        (model.city_contrast_coefficients, model.gate_logits),
    )
    assert all(torch.isfinite(value).all() for value in gradients)
    assert all(float(value.norm()) > 0 for value in gradients)


def test_rank_aware_loss_has_real_tail_gradients_and_exact_soft_mass():
    generator = torch.Generator().manual_seed(59)
    logits = torch.randn((3, 23), generator=generator, dtype=torch.float64, requires_grad=True)
    affinity = torch.softmax(logits, dim=0)
    raw = torch.rand((23, 23), generator=generator, dtype=torch.float64)
    graph = 0.5 * (raw + raw.T)
    graph.fill_diagonal_(0.0)
    loss, _, membership, scores = rank_aware_lexical_loss(
        affinity, graph, top_k=10, temperature=0.2, epsilon=1e-12
    )
    gradient = torch.autograd.grad(loss, scores)[0].abs()
    ranking = torch.argsort(affinity.detach(), dim=1, descending=True, stable=True)
    tail = torch.gather(gradient, 1, ranking[:, 5:10])
    torch.testing.assert_close(
        membership.sum(dim=1),
        torch.full((3,), 10.0, dtype=torch.float64),
        atol=1e-10,
        rtol=0,
    )
    assert bool((tail > 1e-12).all())


def test_training_only_initializer_preserves_constraints_and_is_active():
    model = _wrapper().eval()
    rng = np.random.default_rng(61)
    counts = rng.poisson(1.2, size=(45, model.vocabulary_size)).astype(np.float64)
    cities = np.repeat(np.arange(3), 15)
    counts[cities == 1, :4] += 3
    counts[cities == 2, 4:8] += 3
    report = model.initialize_city_residual_from_counts(
        counts,
        cities,
        shrinkage=0.5,
        pseudocount=0.5,
        strength=0.35,
        projection_ridge=0.05,
        log_odds_clip=1.5,
    )
    assert report["effective_residual_rms"] > 0
    assert report["weighted_city_center_error"] < 1e-12
    assert report["topic_null_direction_error"] < 1e-12


def test_low_rank_city_extension_has_less_capacity_than_independent_decoders():
    counts = extension_parameter_counts(
        cities=3, topics=10, vocabulary=1148, embedding_dimension=768
    )
    assert counts["v6_city_extension"] == 15_390
    assert counts["independent_city_affinities"] == 34_440
    assert counts["saving"] == 19_050
