"""Moment-augmented AEFN (all three attention positions), optimized.

This file is based on the supplied aefn_all_three implementation.
Main changes:
  1) Adds MEFN-style weighted moments up to configurable order K.
  2) Uses a small learned moment basis (moment_dim) so K=3 remains practical.
  3) Keeps only unique symmetric moment components (no duplicate ab/ba terms).
  4) Replaces the tiled NxN padding mask with a broadcastable (B, 1, N) mask.
  5) Uses static particle shapes when extra_info['num_particles'] is available.
  6) Uses Keras Reshape/GlobalAveragePooling1D instead of Lambda where possible.
  7) Supports steps_per_execution for lower Python overhead during training.

Architecture:
  (deta, dphi)
      -> attention before Phi
      -> Phi
      -> attention before sum
      -> learned moment basis psi_i
      -> weighted symmetric moments M1..MK
      -> compression
      -> optional global-token attention after pooling
      -> F
      -> softmax

For K=3:
  M1_a     = sum_i z_i psi_i^a
  M2_ab    = sum_i z_i psi_i^a psi_i^b
  M3_abc   = sum_i z_i psi_i^a psi_i^b psi_i^c

Only a<=b<=c components are retained because the moment tensors are symmetric.
"""

from itertools import combinations_with_replacement
from math import comb

import tensorflow as tf
from tf_keras import Model
from tf_keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    Lambda,
    MultiHeadAttention,
    Reshape,
)
from tf_keras.optimizers import AdamW


# ============================================================
# Fixed architecture switches for this file
# ============================================================
MODEL_NAME = "moefn_all_three"
USE_ATTENTION_BEFORE_PHI = True
USE_ATTENTION_BEFORE_SUM = True
USE_ATTENTION_AFTER_SUM = True


# ============================================================
# Configuration
# ============================================================
def get_default_config() -> dict:
    """Default settings for the moment-augmented AEFN."""
    return {
        "model_name": MODEL_NAME,
        "results_dir_name": f"{MODEL_NAME}_results",
        "input_dim": 2,
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),
        "output_dim": 2,
        "latent_dropout": 0.1,
        "F_dropouts": 0.1,
        "activation": "gelu",
        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,  # preserves the supplied code's effective default
        "use_early_stopping": True,
        # Attention
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,
        # Moments
        # K=3 -> first, second and third weighted moments.
        "max_moment_order": 3,
        # Full K=3 moments at Phi_dim=128 are unnecessarily expensive.
        # We first learn a compact moment basis psi_i in R^moment_dim.
        "moment_dim": 16,
        # Compress concatenated moments before the jet-level network.
        "moment_compression_dim": 128,
        # Used only for attention after moment pooling.
        "global_tokens": 4,
        # Reduces Python overhead in model.fit without changing the model.
        "steps_per_execution": 10,
        # Keep False initially; enable only after a successful Marvin sanity run.
        "jit_compile": False,
    }


# ============================================================
# Data preparation
# ============================================================
def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """Prepare z and particle-coordinate inputs.

    Tensor layout:
        X[..., 0] = z_i
        X[..., 1] = Delta eta_i
        X[..., 2] = Delta phi_i
    """
    z_train = X[train_idx, :, 0]
    p_train = X[train_idx, :, 1:3]
    z_val = X[val_idx, :, 0]
    p_val = X[val_idx, :, 1:3]
    z_test = X[test_idx, :, 0]
    p_test = X[test_idx, :, 1:3]

    return (
        [z_train, p_train],
        [z_val, p_val],
        [z_test, p_test],
        {"num_particles": X.shape[1]},
    )


# ============================================================
# Reusable blocks
# ============================================================
def feed_forward_block(x, sizes, activation, dropout_rate, name_prefix):
    """Feed-forward network used for Phi and F."""
    for i, size in enumerate(sizes, start=1):
        x = Dense(size, activation=activation, name=f"{name_prefix}_dense_{i}")(x)
        if dropout_rate > 0.0:
            x = Dropout(dropout_rate, name=f"{name_prefix}_dropout_{i}")(x)
    return x


def transformer_style_attention_block(
    x,
    num_heads,
    key_dim,
    activation,
    dropout_rate,
    name_prefix,
    attention_mask=None,
):
    """Compact Transformer-style self-attention block.

    MHA -> residual + LayerNorm -> FFN -> residual + LayerNorm.
    """
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=f"{name_prefix}_mha",
    )(
        query=x,
        key=x,
        value=x,
        attention_mask=attention_mask,
    )
    x = Add(name=f"{name_prefix}_attention_add")([x, attn_out])
    x = LayerNormalization(name=f"{name_prefix}_attention_norm")(x)

    # Compact FFN: same hidden width as the token width for speed.
    width = int(x.shape[-1])
    ff_out = Dense(
        width,
        activation=activation,
        name=f"{name_prefix}_ff_dense",
    )(x)
    if dropout_rate > 0.0:
        ff_out = Dropout(
            dropout_rate,
            name=f"{name_prefix}_ff_dropout",
        )(ff_out)

    x = Add(name=f"{name_prefix}_ff_add")([x, ff_out])
    x = LayerNormalization(name=f"{name_prefix}_ff_norm")(x)
    return x


def build_particle_attention_mask(z_input):
    """Broadcastable padding mask for particle self-attention.

    Output shape is (batch, 1, particles), not (batch, particles, particles).
    MultiHeadAttention broadcasts the query dimension. This avoids explicitly
    materializing an NxN mask for every jet.
    """
    return Lambda(
        lambda z: tf.expand_dims(tf.greater(z, 0.0), axis=1),
        name="particle_attention_mask",
    )(z_input)


# ============================================================
# Moment pooling
# ============================================================
class SymmetricMomentPooling(Layer):
    """Energy-weighted symmetric moments up to order K.

    Inputs:
        z:   (B, P)
        psi: (B, P, D)

    Output:
        concatenated unique components of M1, M2, ..., MK.

    For D=16 and K=3 the output dimension is:
        C(16,1) + C(17,2) + C(18,3)
        = 16 + 136 + 816 = 968.
    """

    def __init__(self, max_order=3, **kwargs):
        super().__init__(**kwargs)
        if max_order < 1 or max_order > 3:
            raise ValueError("This optimized implementation supports max_order 1, 2, or 3.")
        self.max_order = int(max_order)
        self.feature_dim = None
        self.output_dim = None
        self._idx2 = None
        self._idx3 = None

    def build(self, input_shape):
        # input_shape = [z_shape, psi_shape]
        feature_dim = input_shape[1][-1]
        if feature_dim is None:
            raise ValueError("Moment feature dimension must be statically known.")
        self.feature_dim = int(feature_dim)

        total = self.feature_dim

        if self.max_order >= 2:
            pairs = list(combinations_with_replacement(range(self.feature_dim), 2))
            flat2 = [a * self.feature_dim + b for a, b in pairs]
            self._idx2 = tf.constant(flat2, dtype=tf.int32)
            total += comb(self.feature_dim + 1, 2)

        if self.max_order >= 3:
            triples = list(combinations_with_replacement(range(self.feature_dim), 3))
            d = self.feature_dim
            flat3 = [a * d * d + b * d + c for a, b, c in triples]
            self._idx3 = tf.constant(flat3, dtype=tf.int32)
            total += comb(self.feature_dim + 2, 3)

        self.output_dim = int(total)
        super().build(input_shape)

    def call(self, inputs):
        z, psi = inputs
        z = tf.cast(z, psi.dtype)

        outputs = []

        # M1_a = sum_i z_i psi_i^a
        m1 = tf.einsum("bp,bpd->bd", z, psi)
        outputs.append(m1)

        if self.max_order >= 2:
            # M2_ab = sum_i z_i psi_i^a psi_i^b
            m2 = tf.einsum("bp,bpd,bpe->bde", z, psi, psi)
            m2 = tf.reshape(m2, [tf.shape(m2)[0], -1])
            m2 = tf.gather(m2, self._idx2, axis=1)
            outputs.append(m2)

        if self.max_order >= 3:
            # M3_abc = sum_i z_i psi_i^a psi_i^b psi_i^c
            m3 = tf.einsum("bp,bpd,bpe,bpf->bdef", z, psi, psi, psi)
            m3 = tf.reshape(m3, [tf.shape(m3)[0], -1])
            m3 = tf.gather(m3, self._idx3, axis=1)
            outputs.append(m3)

        result = tf.concat(outputs, axis=-1)
        result.set_shape([None, self.output_dim])
        return result

    def compute_output_shape(self, input_shape):
        if self.output_dim is not None:
            return (input_shape[0][0], self.output_dim)

        d = int(input_shape[1][-1])
        total = d
        if self.max_order >= 2:
            total += comb(d + 1, 2)
        if self.max_order >= 3:
            total += comb(d + 2, 3)
        return (input_shape[0][0], total)

    def get_config(self):
        config = super().get_config()
        config.update({"max_order": self.max_order})
        return config


def build_moment_representation(phi_output, z_input, config, activation):
    """Project particle features to a compact basis and calculate M1..MK."""
    moment_dim = int(config.get("moment_dim", 16))
    max_order = int(config.get("max_moment_order", 3))

    # Learned compact basis psi_i. This makes full cross-moments practical.
    psi = Dense(
        moment_dim,
        activation=None,
        use_bias=False,
        name="moment_basis_projection",
    )(phi_output)
    psi = LayerNormalization(name="moment_basis_norm")(psi)

    moments = SymmetricMomentPooling(
        max_order=max_order,
        name="symmetric_weighted_moments",
    )([z_input, psi])

    compression_dim = int(config.get("moment_compression_dim", 128))
    if compression_dim > 0:
        moments = Dense(
            compression_dim,
            activation=activation,
            name="moment_compression",
        )(moments)
        moments = LayerNormalization(name="moment_compression_norm")(moments)

    latent_dropout = float(config.get("latent_dropout", 0.0))
    if latent_dropout > 0.0:
        moments = Dropout(latent_dropout, name="moment_dropout")(moments)

    return moments


# ============================================================
# Jet-level attention after moment pooling
# ============================================================
def apply_attention_after_sum(
    event_representation,
    config,
    activation,
    num_heads,
    attention_dropout,
):
    """Refine one jet vector through several learned global tokens."""
    attention_dim = int(config.get("attention_dim", 128))
    global_tokens = int(config.get("global_tokens", 4))

    projected = Dense(
        global_tokens * attention_dim,
        activation=activation,
        name="after_sum_token_projection",
    )(event_representation)

    tokens = Reshape(
        (global_tokens, attention_dim),
        name="after_sum_tokens",
    )(projected)

    tokens = transformer_style_attention_block(
        x=tokens,
        num_heads=num_heads,
        key_dim=attention_dim // num_heads,
        activation=activation,
        dropout_rate=attention_dropout,
        name_prefix="attention_after_sum",
        attention_mask=None,
    )

    return GlobalAveragePooling1D(name="after_sum_token_pooling")(tokens)


# ============================================================
# Model
# ============================================================
def build_model(config: dict, extra_info: dict | None = None):
    """Build the moment-augmented AEFN."""
    activation = config.get("activation", "gelu")
    attention_dim = int(config.get("attention_dim", 128))
    num_heads = int(config.get("num_heads", 4))
    attention_dropout = float(config.get("attention_dropout", 0.0))
    latent_dropout = float(config.get("latent_dropout", 0.0))
    F_dropouts = float(config.get("F_dropouts", 0.0))

    if attention_dim % num_heads != 0:
        raise ValueError(
            "attention_dim must be divisible by num_heads, "
            f"got {attention_dim} and {num_heads}."
        )

    phi_dim = int(config["Phi_sizes"][-1])
    if USE_ATTENTION_BEFORE_SUM and phi_dim % num_heads != 0:
        raise ValueError(
            "The final Phi dimension must be divisible by num_heads when "
            f"attention_before_sum=True; got Phi_dim={phi_dim}, heads={num_heads}."
        )

    num_particles = None
    if extra_info is not None:
        num_particles = extra_info.get("num_particles")
        if num_particles is not None:
            num_particles = int(num_particles)

    # Static P when known improves graph specialization on GPU.
    z_input = Input(shape=(num_particles,), name="z_input")
    p_input = Input(
        shape=(num_particles, int(config["input_dim"])),
        name="p_input",
    )

    particle_attention_mask = build_particle_attention_mask(z_input)

    # --------------------------------------------------------
    # 1) Attention before Phi
    # --------------------------------------------------------
    x = p_input
    if USE_ATTENTION_BEFORE_PHI:
        x = Dense(
            attention_dim,
            activation=activation,
            name="particle_embedding_before_phi",
        )(x)
        x = transformer_style_attention_block(
            x=x,
            num_heads=num_heads,
            key_dim=attention_dim // num_heads,
            activation=activation,
            dropout_rate=attention_dropout,
            name_prefix="attention_before_phi",
            attention_mask=particle_attention_mask,
        )

    # --------------------------------------------------------
    # 2) Phi
    # --------------------------------------------------------
    phi_output = feed_forward_block(
        x=x,
        sizes=config["Phi_sizes"],
        activation=activation,
        dropout_rate=latent_dropout,
        name_prefix="Phi",
    )

    # --------------------------------------------------------
    # 3) Attention after Phi / before moment pooling
    # --------------------------------------------------------
    if USE_ATTENTION_BEFORE_SUM:
        phi_output = transformer_style_attention_block(
            x=phi_output,
            num_heads=num_heads,
            key_dim=phi_dim // num_heads,
            activation=activation,
            dropout_rate=attention_dropout,
            name_prefix="attention_before_sum",
            attention_mask=particle_attention_mask,
        )

    # --------------------------------------------------------
    # 4) MEFN-style weighted moments M1..MK
    # --------------------------------------------------------
    # This replaces the single EFN sum with a richer moment representation.
    event_representation = build_moment_representation(
        phi_output=phi_output,
        z_input=z_input,
        config=config,
        activation=activation,
    )

    # --------------------------------------------------------
    # 5) Optional attention after moment pooling / before F
    # --------------------------------------------------------
    if USE_ATTENTION_AFTER_SUM:
        event_representation = apply_attention_after_sum(
            event_representation=event_representation,
            config=config,
            activation=activation,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )

    # --------------------------------------------------------
    # 6) F
    # --------------------------------------------------------
    f_output = feed_forward_block(
        x=event_representation,
        sizes=config["F_sizes"],
        activation=activation,
        dropout_rate=F_dropouts,
        name_prefix="F",
    )

    # float32 output is numerically safer if mixed precision is enabled outside.
    output = Dense(
        int(config["output_dim"]),
        activation="softmax",
        dtype="float32",
        name="output",
    )(f_output)

    model = Model(
        inputs=[z_input, p_input],
        outputs=output,
        name=MODEL_NAME,
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=AdamW(
            learning_rate=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        ),
        metrics=["accuracy"],
        steps_per_execution=int(config.get("steps_per_execution", 1)),
        jit_compile=bool(config.get("jit_compile", False)),
    )
    return model


# ============================================================
# Logging
# ============================================================
def get_model_summary_fields(config: dict) -> dict:
    """Return important settings for result logging."""
    d = int(config.get("moment_dim", 16))
    k = int(config.get("max_moment_order", 3))
    moment_features = d
    if k >= 2:
        moment_features += comb(d + 1, 2)
    if k >= 3:
        moment_features += comb(d + 2, 3)

    return {
        "model_name": MODEL_NAME,
        "attention_before_phi": USE_ATTENTION_BEFORE_PHI,
        "attention_before_sum": USE_ATTENTION_BEFORE_SUM,
        "attention_after_sum": USE_ATTENTION_AFTER_SUM,
        "input_dim": config["input_dim"],
        "Phi_sizes": str(config["Phi_sizes"]),
        "F_sizes": str(config["F_sizes"]),
        "activation": config.get("activation", "gelu"),
        "latent_dropout": config.get("latent_dropout", 0.0),
        "F_dropouts": config.get("F_dropouts", 0.0),
        "output_dim": config["output_dim"],
        "attention_dim": config.get("attention_dim", 128),
        "num_heads": config.get("num_heads", 4),
        "attention_dropout": config.get("attention_dropout", 0.0),
        "max_moment_order": k,
        "moment_dim": d,
        "moment_feature_count_before_compression": moment_features,
        "moment_compression_dim": config.get("moment_compression_dim", 128),
        "global_tokens": config.get("global_tokens", 4),
        "weight_decay": config.get("weight_decay", 0.0),
        "steps_per_execution": config.get("steps_per_execution", 1),
        "jit_compile": config.get("jit_compile", False),
    }
