from tf_keras.models import Model

from tf_keras.layers import (
    Input,
    Dense,
    TimeDistributed,
    Lambda,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Dropout,
)

from tf_keras.optimizers import Adam
from tf_keras import backend as K

import tensorflow as tf

def get_default_config() -> dict:
    return {
        "model_name": "aefn_after_phi",
        "results_dir_name": "aefn_after_phi_results",

        "input_dim": 2,
        "output_dim": 2,

        # 4 Phi layers + 3 F layers = 7 main dense layers
        "Phi_sizes": (100, 100, 128, 128),
        "activation": "gelu",
        "phi_dropout": 0.1,

        # Attention settings
        "attention_position": "after_phi",
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,

        # Number of attention blocks at each selected attention position
        "num_attention_blocks": 2,

        "latent_dropout": 0.1,

        "F_sizes": (100, 100, 100),
        "F_dropout": 0.1,

        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "use_early_stopping": True,
    }

def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """
    AEFN input:

    z: (batch, max_particles)
    p: (batch, max_particles, 2)

    Shared X:
    X[..., 0] = z
    X[..., 1] = centered_eta / delta_eta
    X[..., 2] = centered_phi / delta_phi
    """

    z_train = X[train_idx, :, 0]
    p_train = X[train_idx, :, 1:3]

    z_val = X[val_idx, :, 0]
    p_val = X[val_idx, :, 1:3]

    z_test = X[test_idx, :, 0]
    p_test = X[test_idx, :, 1:3]

    train_inputs = [z_train, p_train]
    val_inputs = [z_val, p_val]
    test_inputs = [z_test, p_test]

    extra_info = {
        "num_particles": X.shape[1],
    }

    return train_inputs, val_inputs, test_inputs, extra_info

def transformer_attention_block(
    x,
    attention_mask,
    attention_dim,
    num_heads,
    attention_dropout,
    activation,
    block_name,
):
    """
    One transformer-style attention block.

    It contains:
    1. MultiHeadAttention
    2. residual connection
    3. LayerNormalization
    4. small feed-forward layer
    5. residual connection
    6. LayerNormalization
    """

    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=attention_dim // num_heads,
        dropout=attention_dropout,
        name=f"{block_name}_mha",
    )(
        query=x,
        value=x,
        key=x,
        attention_mask=attention_mask,
    )

    x = Add(name=f"{block_name}_attn_residual")([x, attn_out])
    x = LayerNormalization(name=f"{block_name}_attn_norm")(x)

    ff = TimeDistributed(
        Dense(attention_dim, activation=activation),
        name=f"{block_name}_ff",
    )(x)

    ff = Dropout(
        attention_dropout,
        name=f"{block_name}_ff_dropout",
    )(ff)

    x = Add(name=f"{block_name}_ff_residual")([x, ff])
    x = LayerNormalization(name=f"{block_name}_ff_norm")(x)

    return x


def apply_attention_blocks(
    x,
    attention_mask,
    attention_dim,
    num_heads,
    num_attention_blocks,
    attention_dropout,
    activation,
    block_prefix,
):
    """
    Applies several attention blocks at one position.
    """

    for block_idx in range(num_attention_blocks):
        x = transformer_attention_block(
            x=x,
            attention_mask=attention_mask,
            attention_dim=attention_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            activation=activation,
            block_name=f"{block_prefix}_block_{block_idx + 1}",
        )

    return x

def build_model(config: dict, extra_info: dict | None = None):
    """
    AEFN variant: after_phi

    Structure:
        input_p -> Phi -> Attention -> z-weighted sum -> F -> output
    """

    num_particles = (
        extra_info["num_particles"]
        if extra_info is not None
        else config["max_particles"]
    )

    input_dim = config.get("input_dim", 2)
    output_dim = config.get("output_dim", 2)

    Phi_sizes = config["Phi_sizes"]
    F_sizes = config["F_sizes"]

    attention_dim = config.get("attention_dim", 128)
    num_heads = config.get("num_heads", 4)
    num_attention_blocks = config.get("num_attention_blocks", 2)

    phi_dropout = config.get("phi_dropout", 0.0)
    attention_dropout = config.get("attention_dropout", 0.0)
    latent_dropout = config.get("latent_dropout", 0.0)
    F_dropout = config.get("F_dropout", 0.0)

    learning_rate = config.get("learning_rate", 1e-3)
    activation = config.get("activation", "relu")

    if attention_dim % num_heads != 0:
        raise ValueError(
            f"attention_dim must be divisible by num_heads, "
            f"but got attention_dim={attention_dim}, num_heads={num_heads}."
        )

    input_z = Input(shape=(num_particles,), name="input_z")
    input_p = Input(shape=(num_particles, input_dim), name="input_p")

    # ------------------------------------------------------------------
    # Padding mask
    # ------------------------------------------------------------------
    # z > 0  -> real particle
    # z = 0  -> padded particle
    particle_mask = Lambda(
        lambda z: tf.cast(tf.greater(z, 0.0), tf.bool),
        name="particle_mask",
    )(input_z)

    # Attention mask for particle-level attention:
    # shape: (batch, query_particles, key_particles)
    particle_attention_mask = Lambda(
        lambda m: tf.tile(tf.expand_dims(m, axis=1), [1, tf.shape(m)[1], 1]),
        name="particle_attention_mask",
    )(particle_mask)

    # ------------------------------------------------------------------
    # Start with particle coordinates
    # ------------------------------------------------------------------
    x = input_p

    # ------------------------------------------------------------------
    # Phi network
    # ------------------------------------------------------------------
    for i, units in enumerate(Phi_sizes):
        x = TimeDistributed(
            Dense(units, activation=activation),
            name=f"phi_dense_{i + 1}",
        )(x)

        x = Dropout(
            phi_dropout,
            name=f"phi_dropout_{i + 1}",
        )(x)

    # ------------------------------------------------------------------
    # Attention AFTER Phi
    # ------------------------------------------------------------------
    x = TimeDistributed(
        Dense(attention_dim, activation=activation),
        name="post_phi_attention_projection",
    )(x)

    x = Dropout(
        phi_dropout,
        name="post_phi_attention_projection_dropout",
    )(x)

    x = apply_attention_blocks(
        x=x,
        attention_mask=particle_attention_mask,
        attention_dim=attention_dim,
        num_heads=num_heads,
        num_attention_blocks=num_attention_blocks,
        attention_dropout=attention_dropout,
        activation=activation,
        block_prefix="post_phi_attention",
    )

    # ------------------------------------------------------------------
    # EFN-style z-weighted sum
    # latent = sum_i z_i * x_i
    # ------------------------------------------------------------------
    z_expanded = Lambda(
        lambda z: tf.expand_dims(z, axis=-1),
        name="expand_z",
    )(input_z)

    weighted_x = Lambda(
        lambda tensors: tensors[0] * tensors[1],
        name="energy_weighted_particles",
    )([z_expanded, x])

    latent = Lambda(
        lambda t: K.sum(t, axis=1),
        name="energy_weighted_sum",
    )(weighted_x)

    latent = Dropout(
        latent_dropout,
        name="latent_dropout",
    )(latent)

    # ------------------------------------------------------------------
    # F network
    # ------------------------------------------------------------------
    y = latent

    for i, units in enumerate(F_sizes):
        y = Dense(
            units,
            activation=activation,
            name=f"F_dense_{i + 1}",
        )(y)

        y = Dropout(
            F_dropout,
            name=f"F_dropout_{i + 1}",
        )(y)

    output = Dense(
        output_dim,
        activation="softmax",
        name="output",
    )(y)

    model = Model(
        inputs=[input_z, input_p],
        outputs=output,
        name="aefn_after_phi",
    )

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model

def get_model_summary_fields(config: dict) -> dict:
    return {
        "input_dim": config["input_dim"],
        "Phi_sizes": str(config["Phi_sizes"]),
        "activation": config.get("activation", "relu"),
        "phi_dropout": config.get("phi_dropout", 0.0),

        "attention_position": config.get("attention_position", "after_phi"),
        "attention_dim": config["attention_dim"],
        "num_heads": config["num_heads"],
        "attention_dropout": config.get("attention_dropout", 0.0),
        "num_attention_blocks": config["num_attention_blocks"],

        "latent_dropout": config.get("latent_dropout", 0.0),
        "F_sizes": str(config["F_sizes"]),
        "F_dropout": config.get("F_dropout", 0.0),
        "output_dim": config["output_dim"],
    }
