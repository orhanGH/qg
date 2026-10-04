"""MOEFN: M1..M7 with all three AEFN attention positions.

Place this file in qg/models/efn/moefn.py on branch pbpb_vs_pp.
It implements the same four hooks as the existing Keras models. Because
main.py does not yet register moefn, it can also be run directly:

    python models/efn/moefn.py --self-test
    python models/efn/moefn.py --data-root /path/to/ptmin50 --fold 1

Input convention (unchanged): X[..., :] = (z_i, delta_eta_i, delta_phi_i).
Weights are used ONCE per moment, not raised to the moment order:

    M_k[a1,...,ak] = sum_i z_i * psi_i[a1] * ... * psi_i[ak].

These are raw multivariate moments, including all mixed components in the
learned D-dimensional basis. Only a1 <= ... <= ak are stored. Seven orders
are not seven scalars: D=4, K=7 gives C(11,7)-1 = 329 features. D=1 gives 7.

Architecture:
    p -> attention before Phi -> Phi -> attention before pooling
      -> [original z-weighted Phi sum, compressed M1..M7]
      -> attention over four learned global tokens -> F -> softmax.

Compared with the branch's earlier moefn.py, the defaults explicitly change
K from 3 to 7, D from 16 to 4, and the basis to a bounded tanh projection.
The full Phi sum is retained as a parallel path (keep_efn_sum=True), so the
small moment basis does not replace the entire original jet representation.
Set moment_dim=16, max_moment_order=3, moment_basis_activation='linear',
moment_basis_layer_norm=True and keep_efn_sum=False for the earlier basis
and representation choices. Existing trained checkpoints are not compatible
with the new default architecture; train a new model.

Optimization: monomials are built only through ceil(K/2). Batched matrix
multiplications pool products before selecting unique symmetric components.
No B x P x D**7 tensor, no per-component TensorFlow loops, no tiled P x P
padding mask. Dense acts directly on particle tensors without TimeDistributed.
Float32 moment accumulation and output support optional mixed precision in
the surrounding network. XLA and mixed precision are opt-in, hardware-specific.

Model basis: project report, Appendix A, and Gambhir et al., arXiv:2403.08854.
Repository reference: orhanGH/qg, pbpb_vs_pp,
3f7b16eab7f93ca9c09ac85dc3acb13eb1f9f8db (2026-10-04).
Requires the project's TensorFlow + tf_keras environment, NumPy and sklearn.
The standalone entry uses the project's utils.py and runners/keras_runner.py;
it preserves their file-level split and evaluation procedures.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path

import numpy as np
import tensorflow as tf
import tf_keras as keras
from tf_keras import Model
from tf_keras.layers import (
    Add, Concatenate, Dense, Dropout, GlobalAveragePooling1D, Input, Layer,
    LayerNormalization, MultiHeadAttention, Reshape,
)
from tf_keras.optimizers import AdamW
from tf_keras.utils import register_keras_serializable


MODEL_NAME = "moefn"
USE_ATTENTION_BEFORE_PHI = True
USE_ATTENTION_BEFORE_SUM = True
USE_ATTENTION_AFTER_SUM = True


def get_default_config() -> dict:
    """Report-style training settings; they are configurable, not HPO results."""
    return {
        "model_name": MODEL_NAME,
        "results_dir_name": "moefn_results",
        "input_dim": 2,
        "output_dim": 2,
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),
        "activation": "silu",
        "latent_dropout": 0.15,
        "F_dropouts": 0.15,
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.15,
        "global_tokens": 4,
        "max_moment_order": 7,
        "moment_dim": 4,
        "moment_basis_activation": "tanh",
        "moment_basis_layer_norm": False,
        "moment_compression_dim": 128,
        "keep_efn_sum": True,
        "batch_size": 512,
        "epochs": 500,
        "learning_rate": 1e-5,
        "weight_decay": 1e-4,
        "clipnorm": 1.0,
        "use_early_stopping": True,
        "patience": 25,
        "early_stopping_threshold": 1e-4,
        "steps_per_execution": 10,
        "jit_compile": False,
        "seed": 42,
        "max_particles": 128,
        "num_folds": 4,
        "final_test_ratio": 0.20,
    }


def moment_feature_count(dim: int, max_order: int) -> int:
    """Number of unique nonconstant symmetric components, orders 1..K."""
    return comb(dim + max_order, max_order) - 1


@lru_cache(maxsize=16)
def _moment_indices(dim: int, max_order: int):
    """Build integer gather indices once; never enumerate terms in call()."""
    combos = {
        k: tuple(combinations_with_replacement(range(dim), k))
        for k in range(1, max_order + 1)
    }
    positions = {k: {c: i for i, c in enumerate(cs)} for k, cs in combos.items()}
    recursive = {}
    for k in range(2, (max_order + 1) // 2 + 1):
        recursive[k] = (
            np.asarray([positions[k - 1][c[:-1]] for c in combos[k]], np.int32),
            np.asarray([c[-1] for c in combos[k]], np.int32),
        )
    selections = {}
    for k in range(2, max_order + 1):
        left, right = k // 2, k - k // 2
        right_width = len(combos[right])
        selections[k] = np.asarray([
            positions[left][c[:left]] * right_width + positions[right][c[left:]]
            for c in combos[k]
        ], np.int32)
    return recursive, selections


@register_keras_serializable(package="MOEFN")
class ParticleAttentionMask(Layer):
    """Real keys only; singleton query axis broadcasts inside MHA."""

    def call(self, z):
        return tf.expand_dims(z > 0, axis=1)

    def compute_output_shape(self, shape):
        return (shape[0], 1, shape[1])


@register_keras_serializable(package="MOEFN")
class MaskParticles(Layer):
    """Remove padding before nonlinear operations, including NaN pad values."""

    def call(self, inputs):
        z, features = inputs
        return tf.where((z > 0)[..., None], features, tf.zeros_like(features))

    def compute_output_shape(self, shapes):
        return shapes[1]


@register_keras_serializable(package="MOEFN")
class WeightedSum(Layer):
    """Original EFN sum, with float32 accumulation and zero padding."""

    def __init__(self, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)

    def call(self, inputs):
        z, features = inputs
        z = tf.cast(z, self.compute_dtype)
        features = tf.cast(features, self.compute_dtype)
        features = tf.where((z > 0)[..., None], features, tf.zeros_like(features))
        return tf.squeeze(tf.matmul(z[:, None, :], features), axis=1)

    def compute_output_shape(self, shapes):
        return (shapes[0][0], shapes[1][-1])


@register_keras_serializable(package="MOEFN")
class SymmetricMomentPooling(Layer):
    """Exact mixed moments M1..MK, with one factor z per constituent.

    Inputs: [z (B,P), psi (B,P,D)]. Output: (B,C(D+K,K)-1).
    Orders are concatenated increasingly; components within an order use
    lexicographic combinations_with_replacement, matching mefn.py.

    For k=7, split every monomial into an order-3 and an order-4 factor:
        pooled[u,v] = sum_i z_i * terms3[i,u] * terms4[i,v].
    Batched matmul performs the reduction over particles first. Gathering
    afterwards keeps the canonical symmetric components, including mixed
    terms. Redundant contractions are small at the default D=4, and avoid
    storing the much larger per-particle seventh-order feature tensor.

    The maximum per-particle monomial width for D=4,K=7 is 35; the largest
    contraction width is 20*35=700. At D=8 these become 330 and 39,600.
    Increasing D remains combinatorially expensive; this is not an
    approximation that secretly discards cross-moments.
    """

    def __init__(self, max_order=7, **kwargs):
        kwargs.setdefault("dtype", "float32")
        super().__init__(**kwargs)
        if isinstance(max_order, bool) or int(max_order) != max_order:
            raise ValueError("max_order must be an integer in 1..7.")
        self.max_order = int(max_order)
        if not 1 <= self.max_order <= 7:
            raise ValueError("max_order must be in 1..7.")
        self.feature_dim = None
        self.output_dim = None

    def build(self, input_shape):
        z_shape, psi_shape = map(tf.TensorShape, input_shape)
        if z_shape.rank != 2 or psi_shape.rank != 3:
            raise ValueError("Expected z (B,P) and psi (B,P,D).")
        if psi_shape[-1] is None or int(psi_shape[-1]) < 1:
            raise ValueError("psi needs a positive static feature dimension.")
        if z_shape[1] is not None and psi_shape[1] is not None:
            if z_shape[1] != psi_shape[1]:
                raise ValueError("z and psi must have equal particle counts.")
        self.feature_dim = int(psi_shape[-1])
        self.output_dim = moment_feature_count(self.feature_dim, self.max_order)
        # Fail before allocating millions of high-order components by mistake.
        if self.output_dim > 300_000:
            raise ValueError(
                f"D={self.feature_dim}, K={self.max_order} gives {self.output_dim:,} "
                "moment features. Use a smaller moment_dim (default: 4)."
            )
        self._recursive, self._selections = _moment_indices(
            self.feature_dim, self.max_order
        )
        super().build(input_shape)

    def call(self, inputs):
        z, psi = inputs
        z = tf.cast(z, self.compute_dtype)
        psi = tf.cast(psi, self.compute_dtype)
        # Zero before forming powers: multiplication by 0 cannot remove NaN.
        psi = tf.where((z > 0)[..., None], psi, tf.zeros_like(psi))
        terms = {1: psi}
        for k, (prefix, last) in self._recursive.items():
            terms[k] = (
                tf.gather(terms[k - 1], prefix, axis=2)
                * tf.gather(psi, last, axis=2)
            )

        weighted = {
            k: value * z[..., None]
            for k, value in terms.items()
            if k <= self.max_order // 2
        }
        outputs = [tf.squeeze(tf.matmul(z[:, None, :], psi), axis=1)]
        for k in range(2, self.max_order + 1):
            left, right = k // 2, k - k // 2
            pooled = tf.matmul(weighted[left], terms[right], transpose_a=True)
            flat = tf.reshape(pooled, [tf.shape(z)[0], -1])
            outputs.append(tf.gather(flat, self._selections[k], axis=1))

        result = tf.concat(outputs, axis=-1)
        result.set_shape((z.shape[0], self.output_dim))
        return result

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], moment_feature_count(
            int(input_shape[1][-1]), self.max_order
        ))

    def get_config(self):
        return {**super().get_config(), "max_order": self.max_order}


def _validated_config(config):
    cfg = {**get_default_config(), **config}
    for key in (
        "input_dim", "output_dim", "attention_dim", "num_heads", "global_tokens",
        "max_moment_order", "moment_dim", "steps_per_execution", "batch_size",
    ):
        value = cfg[key]
        if isinstance(value, bool) or int(value) != value or int(value) < 1:
            raise ValueError(f"{key} must be a positive integer; got {value!r}.")
        cfg[key] = int(value)
    if cfg["input_dim"] != 2 or cfg["output_dim"] != 2:
        raise ValueError("This pp/PbPb model uses input_dim=2 and output_dim=2.")
    if cfg["max_moment_order"] > 7:
        raise ValueError("max_moment_order must be in 1..7.")
    if cfg["global_tokens"] < 2:
        raise ValueError("global_tokens must be >=2 for nontrivial global attention.")
    for key in ("Phi_sizes", "F_sizes"):
        if not cfg[key] or any(
            isinstance(v, bool) or int(v) != v or int(v) < 1 for v in cfg[key]
        ):
            raise ValueError(f"{key} must contain positive integer widths.")
        cfg[key] = tuple(map(int, cfg[key]))
    for width in (cfg["attention_dim"], cfg["Phi_sizes"][-1]):
        if width % cfg["num_heads"]:
            raise ValueError("attention_dim and final Phi width must divide by num_heads.")
    for key in ("latent_dropout", "F_dropouts", "attention_dropout"):
        if not 0 <= float(cfg[key]) < 1:
            raise ValueError(f"{key} must be in [0,1).")
    if (not np.isfinite(float(cfg["learning_rate"]))
            or not np.isfinite(float(cfg["weight_decay"]))
            or float(cfg["learning_rate"]) <= 0 or float(cfg["weight_decay"]) < 0):
        raise ValueError("learning_rate must be >0 and weight_decay >=0.")
    compression = cfg["moment_compression_dim"]
    if isinstance(compression, bool) or int(compression) != compression or compression < 0:
        raise ValueError("moment_compression_dim must be an integer >=0.")
    cfg["moment_compression_dim"] = int(compression)
    if cfg.get("clipnorm") is not None and float(cfg["clipnorm"]) <= 0:
        raise ValueError("clipnorm must be positive or None.")
    return cfg


def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """Keep the existing runner interface and file-level split untouched.

    Contiguous float32 arrays avoid repeated dtype conversions at training.
    Accept either the particle tensor or the shared loader's {'parts': ...}.
    No transformations are fitted to validation/test data.
    """
    parts = X["parts"] if isinstance(X, dict) else X
    if parts.ndim != 3 or parts.shape[2] < 3 or parts.shape[1] < 1:
        raise ValueError("Expected X with shape (jets,particles,3).")
    prepared = []
    for indices in (train_idx, val_idx, test_idx):
        z = np.ascontiguousarray(parts[indices, :, 0], dtype=np.float32)
        p = np.ascontiguousarray(parts[indices, :, 1:3], dtype=np.float32)
        if not np.isfinite(z).all() or np.any(z < 0):
            raise ValueError("z must contain finite, nonnegative momentum fractions.")
        # Padding may contain arbitrary values; real coordinates must be finite.
        if np.any((~np.isfinite(p)).any(axis=-1) & (z > 0)):
            raise ValueError("Real particle coordinates must be finite.")
        prepared.append([z, p])
    fold_match = re.fullmatch(r"fold_(\d+)", Path(fold_dir).name)
    fold_number = int(fold_match.group(1)) if fold_match else 0
    extra = {
        "num_particles": int(parts.shape[1]),
        "fold_seed": int(config.get("seed", 42)) + fold_number,
    }
    return (*prepared, extra)


def feed_forward_block(x, sizes, activation, dropout_rate, name_prefix):
    for i, width in enumerate(sizes, 1):
        x = Dense(width, activation=activation, name=f"{name_prefix}_dense_{i}")(x)
        if dropout_rate:
            x = Dropout(dropout_rate, name=f"{name_prefix}_dropout_{i}")(x)
    return x


def transformer_style_attention_block(
    x, num_heads, key_dim, activation, dropout_rate, name_prefix, attention_mask=None
):
    """Same post-norm attention/FFN/residual structure as aefn_all_three.py."""
    attended = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate,
        name=f"{name_prefix}_mha",
    )(query=x, key=x, value=x, attention_mask=attention_mask)
    x = Add(name=f"{name_prefix}_attention_add")([x, attended])
    x = LayerNormalization(name=f"{name_prefix}_attention_norm")(x)
    ff = Dense(int(x.shape[-1]), activation=activation,
               name=f"{name_prefix}_ff_dense")(x)
    if dropout_rate:
        ff = Dropout(dropout_rate, name=f"{name_prefix}_ff_dropout")(ff)
    x = Add(name=f"{name_prefix}_ff_add")([x, ff])
    return LayerNormalization(name=f"{name_prefix}_ff_norm")(x)


def build_model(config: dict, extra_info: dict | None = None):
    cfg = _validated_config(config)
    extra = extra_info or {}
    keras.utils.set_random_seed(int(extra.get("fold_seed", cfg["seed"])))
    particles = extra.get("num_particles")
    if particles is not None:
        if isinstance(particles, bool) or int(particles) != particles or particles < 1:
            raise ValueError("num_particles must be positive or None.")
        particles = int(particles)
    z = Input((particles,), dtype="float32", name="z_input")
    p = Input((particles, 2), dtype="float32", name="p_input")
    mask = ParticleAttentionMask(dtype="float32", name="particle_attention_mask")(z)
    p_clean = MaskParticles(dtype="float32", name="clean_particle_padding")([z, p])

    # 1. Attention before Phi.
    x = Dense(cfg["attention_dim"], activation=cfg["activation"],
              name="particle_embedding_before_phi")(p_clean)
    x = transformer_style_attention_block(
        x, cfg["num_heads"], cfg["attention_dim"] // cfg["num_heads"],
        cfg["activation"], cfg["attention_dropout"], "attention_before_phi", mask,
    )
    phi = feed_forward_block(x, cfg["Phi_sizes"], cfg["activation"],
                             cfg["latent_dropout"], "Phi")

    # 2. Attention after Phi, before the weighted aggregations.
    phi = transformer_style_attention_block(
        phi, cfg["num_heads"], cfg["Phi_sizes"][-1] // cfg["num_heads"],
        cfg["activation"], cfg["attention_dropout"], "attention_before_sum", mask,
    )
    basis = Dense(cfg["moment_dim"], use_bias=False, activation=None,
                  dtype="float32", name="moment_basis_projection")(phi)
    if cfg["moment_basis_layer_norm"]:
        basis = LayerNormalization(dtype="float32", name="moment_basis_norm")(basis)
    basis = keras.layers.Activation(cfg["moment_basis_activation"], dtype="float32",
                                    name="moment_basis_activation")(basis)
    moments = SymmetricMomentPooling(cfg["max_moment_order"],
                                     name="symmetric_weighted_moments")([z, basis])
    if cfg["moment_compression_dim"]:
        moments = Dense(cfg["moment_compression_dim"], activation=cfg["activation"],
                        name="moment_compression")(moments)
        moments = LayerNormalization(name="moment_compression_norm")(moments)
    if cfg["keep_efn_sum"]:
        original_sum = WeightedSum(name="efn_weighted_sum")([z, phi])
        event = Concatenate(name="efn_sum_plus_moments")([original_sum, moments])
    else:
        event = moments
    if cfg["latent_dropout"]:
        event = Dropout(cfg["latent_dropout"], name="moment_dropout")(event)

    # 3. Attention after pooling: four learned tokens, then average pooling.
    event = Dense(cfg["global_tokens"] * cfg["attention_dim"],
                  activation=cfg["activation"], name="after_sum_token_projection")(event)
    tokens = Reshape((cfg["global_tokens"], cfg["attention_dim"]),
                     name="after_sum_tokens")(event)
    tokens = transformer_style_attention_block(
        tokens, cfg["num_heads"], cfg["attention_dim"] // cfg["num_heads"],
        cfg["activation"], cfg["attention_dropout"], "attention_after_sum",
    )
    event = GlobalAveragePooling1D(name="after_sum_token_pooling")(tokens)
    event = feed_forward_block(event, cfg["F_sizes"], cfg["activation"],
                               cfg["F_dropouts"], "F")
    output = Dense(2, activation="softmax", dtype="float32", name="output")(event)
    model = Model([z, p], output, name=MODEL_NAME)
    model.compile(
        optimizer=AdamW(learning_rate=float(cfg["learning_rate"]),
                        weight_decay=float(cfg["weight_decay"]),
                        clipnorm=cfg.get("clipnorm")),
        loss="categorical_crossentropy", metrics=["accuracy"],
        steps_per_execution=cfg["steps_per_execution"], jit_compile=cfg["jit_compile"],
    )
    return model


def get_model_summary_fields(config: dict) -> dict:
    cfg = _validated_config(config)
    fields = {
        key: cfg[key] for key in (
            "input_dim", "output_dim", "activation", "latent_dropout", "F_dropouts",
            "attention_dim", "num_heads", "attention_dropout", "global_tokens",
            "max_moment_order", "moment_dim", "moment_basis_activation",
            "moment_basis_layer_norm", "moment_compression_dim", "keep_efn_sum",
            "weight_decay", "clipnorm", "steps_per_execution", "jit_compile",
        )
    }
    fields.update({
        "model_name": MODEL_NAME,
        "Phi_sizes": str(cfg["Phi_sizes"]), "F_sizes": str(cfg["F_sizes"]),
        "attention_before_phi": True, "attention_before_sum": True,
        "attention_after_sum": True,
        "moment_feature_count_before_compression": moment_feature_count(
            cfg["moment_dim"], cfg["max_moment_order"]
        ),
        "moment_accumulation_dtype": "float32",
        "precision_policy": keras.mixed_precision.global_policy().name,
    })
    return fields


def run_self_test():
    """Independent formula check, derivatives, invariance and model round-trip."""
    import tempfile

    rng = np.random.default_rng(17)
    z = np.array([[0.6, 0.4, 0], [0, 0, 0]], dtype=np.float32)
    psi = rng.uniform(-0.8, 0.8, (2, 3, 4)).astype(np.float32)
    expected = []
    for order in range(1, 8):
        for channels in combinations_with_replacement(range(4), order):
            expected.append(np.sum(z * np.prod(psi[..., list(channels)], axis=-1), axis=1))
    expected = np.stack(expected, axis=1)
    layer = SymmetricMomentPooling(7)
    actual = layer([z, psi]).numpy()
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    assert actual.shape == (2, 329)
    np.testing.assert_array_equal(actual[1], np.zeros(329, dtype=np.float32))
    jacobian_layer = SymmetricMomentPooling(7, dtype="float64")
    small = rng.uniform(-0.5, 0.5, (1, 2, 2))
    analytical, numerical = tf.test.compute_gradient(
        lambda value: jacobian_layer([tf.constant([[0.7, 0.3]], tf.float64), value]),
        [small], delta=1e-5,
    )
    np.testing.assert_allclose(analytical[0], numerical[0], atol=1e-7, rtol=1e-6)

    cfg = get_default_config()
    cfg.update(attention_dim=16, num_heads=2, Phi_sizes=(16, 16), F_sizes=(16,),
               moment_compression_dim=16, latent_dropout=0., F_dropouts=0.,
               attention_dropout=0., steps_per_execution=1)
    model = build_model(cfg)  # Variable particle count for the padding check.
    p = rng.normal(size=(2, 3, 2)).astype(np.float32)
    probabilities = model([z, p], training=False).numpy()
    perm = [2, 0, 1]
    np.testing.assert_allclose(
        model([z[:, perm], p[:, perm]], training=False), probabilities, atol=2e-6
    )
    z_padded = np.pad(z, ((0, 0), (0, 2)))
    p_padded = np.pad(p, ((0, 0), (0, 2), (0, 0)), constant_values=np.nan)
    np.testing.assert_allclose(
        model([z_padded, p_padded], training=False), probabilities, atol=2e-6
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1., atol=2e-7)
    assert np.isfinite(probabilities).all()
    assert sum(isinstance(x, MultiHeadAttention) for x in model.layers) == 3
    with tf.GradientTape() as tape:
        loss = keras.losses.categorical_crossentropy(np.eye(2), model([z, p], training=True))
        loss = tf.reduce_mean(loss)
    gradients = tape.gradient(loss, model.trainable_weights)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in gradients)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "moefn.keras"
        model.save(path)
        loaded = keras.models.load_model(path, compile=False)
        np.testing.assert_allclose(loaded([z, p]), probabilities, atol=2e-6)
    print("PASS: M1..M7 (all mixed terms), numerical gradients, three attentions,")
    print("      permutation/padding invariance, empty jets, finite gradients, save/load.")


def main():
    """Run using the existing qg loader, saved split and Keras experiment runner."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-root", default="/lustre/scratch/data/jdearrud_hpc-jewel/phase4/ptmin50")
    parser.add_argument("--class-0", default="vac", choices=("vac", "rec"))
    parser.add_argument("--class-1", default="rec", choices=("vac", "rec"))
    parser.add_argument("--num-data", type=int, default=-1)
    parser.add_argument("--max-files-per-class", type=int)
    parser.add_argument("--max-particles", type=int, default=128)
    parser.add_argument("--num-folds", type=int, default=4)
    parser.add_argument("--final-test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--moment-dim", type=int)
    parser.add_argument("--moment-order", type=int, dest="max_moment_order")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--optimized-config", help="JSON object, or {'moefn': {...}}.")
    parser.add_argument("--precision", choices=("float32", "mixed_float16", "mixed_bfloat16"),
                        default="float32")
    parser.add_argument("--jit-compile", action="store_true", default=None)
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.class_0 == args.class_1:
        parser.error("--class-0 and --class-1 must be different.")
    if args.fold is not None and not 1 <= args.fold <= args.num_folds:
        parser.error("--fold must be between 1 and --num-folds.")

    cfg = get_default_config()
    if args.optimized_config:
        raw = json.loads(Path(args.optimized_config).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            parser.error("--optimized-config must contain a JSON object.")
        if "moefn" in raw:
            overrides = raw["moefn"]
        elif "moefn_all_three" in raw:
            overrides = raw["moefn_all_three"]
        elif any(key in cfg for key in raw):
            overrides = raw
        else:
            parser.error("Config needs a 'moefn' section or direct MOEFN settings.")
        if not isinstance(overrides, dict):
            parser.error("The MOEFN config section must be a JSON object.")
        cfg.update(overrides)
    for name in ("epochs", "batch_size", "moment_dim", "max_moment_order",
                 "learning_rate", "weight_decay", "jit_compile"):
        if getattr(args, name) is not None:
            cfg[name] = getattr(args, name)
    cfg.update({key: getattr(args, key) for key in (
        "seed", "max_particles", "num_folds", "final_test_ratio",
        "num_data", "class_0", "class_1", "data_root", "max_files_per_class",
    )})
    cfg["model_name"] = MODEL_NAME
    cfg = _validated_config(cfg)
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    keras.mixed_precision.set_global_policy(args.precision)

    # Find the existing project when run from either models/efn or its root.
    candidates = (Path(__file__).resolve().parent, *Path(__file__).resolve().parents,
                  Path.cwd())
    root = next((p for p in candidates if (p / "utils.py").is_file()
                 and (p / "runners" / "keras_runner.py").is_file()), None)
    if root is None:
        parser.error("Place moefn.py inside the qg project; utils.py and runners are required.")
    sys.path.insert(0, str(root))
    from utils import load_marvin_parts_dataset, get_or_create_file_level_test_cv_splits
    from runners.keras_runner import run_keras_experiment

    parts, y, file_ids, file_labels, file_paths = load_marvin_parts_dataset(
        data_root=Path(args.data_root), class_0=args.class_0, class_1=args.class_1,
        max_particles=args.max_particles,
        max_jets=None if args.num_data <= 0 else args.num_data,
        max_files_per_class=args.max_files_per_class, sort_by_pt=True,
        seed=args.seed, return_file_paths=True,
    )
    y = y.astype(np.int64, copy=False)
    _, _, folds = get_or_create_file_level_test_cv_splits(
        project_root=root, y=y, file_ids=file_ids, file_labels=file_labels,
        shared_config=cfg, file_paths=file_paths,
    )
    if args.fold is not None:
        folds = [f for f in folds if f["fold"] == args.fold]
        if len(folds) != 1:
            raise ValueError(f"Requested fold {args.fold} not found in the saved split.")
    print(json.dumps({**cfg, **get_model_summary_fields(cfg)}, indent=2))
    run_keras_experiment(
        X=parts, y=y, folds=folds, shared_config=cfg, model_config=cfg,
        build_model_fn=build_model, prepare_fold_inputs_fn=prepare_fold_inputs,
        get_model_summary_fields_fn=get_model_summary_fields,
    )


if __name__ == "__main__":
    main()
