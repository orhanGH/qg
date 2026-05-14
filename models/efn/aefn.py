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


def get_default_config() -> dict:
    return {
        "model_name": "aefn",
        "results_dir_name": "aefn_results",

        "input_dim": 2,
        "output_dim": 2,

        # EFN-style frontend Phi
        "Phi_sizes": (100, 100, 128),

        # Dropout in Phi network
        "phi_dropout": 0.1,

        # Attention block
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,
        "num_attention_blocks": 1,

        # Dropout after energy-weighted latent sum
        "latent_dropout": 0.1,

        # Backend F
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
    aEFN input:

    z: (batch, max_particles)
    p: (batch, max_particles, 2)

    Shared X:
    X[..., 0] = z
    X[..., 1] = centered_y
    X[..., 2] = centered_phi
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


def build_model(config: dict, extra_info: dict | None = None):
    """
    Attention-based EFN.

    Standard EFN structure:
    Phi(p_i)
    sum_i z_i Phi(p_i)
    F(...)

    aEFN structure:
    Phi(p_i)
    Attention(Phi(p_1), ..., Phi(p_M))
    sum_i z_i AttentionPhi_i
    F(...)

    This keeps the same input/output interface as the EnergyFlow EFN:
    inputs = [z, p]
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

    attention_dim = config.get("attention_dim", Phi_sizes[-1])
    num_heads = config.get("num_heads", 4)
    num_attention_blocks = config.get("num_attention_blocks", 1)

    phi_dropout = config.get("phi_dropout", 0.0)
    attention_dropout = config.get("attention_dropout", 0.0)
    latent_dropout = config.get("latent_dropout", 0.0)
    F_dropout = config.get("F_dropout", 0.0)

    learning_rate = config.get("learning_rate", 1e-3)

    input_z = Input(shape=(num_particles,), name="input_z")
    input_p = Input(shape=(num_particles, input_dim), name="input_p")

    # Mask real particles. Padding particles should have z = 0.
    # Shape: (batch, particles)
    particle_mask = Lambda(
        lambda z: K.cast(K.greater(z, 0.0), "bool"),
        name="particle_mask",
    )(input_z)

    # Phi network: applied independently to each particle coordinate p_i.
    x = input_p

    for i, units in enumerate(Phi_sizes):
        x = TimeDistributed(
            Dense(units, activation="relu"),
            name=f"phi_dense_{i + 1}",
        )(x)

        x = Dropout(
            phi_dropout,
            name=f"phi_dropout_{i + 1}",
        )(x)

    # Project Phi output to attention dimension.
    x = TimeDistributed(
        Dense(attention_dim, activation="relu"),
        name="phi_attention_projection",
    )(x)

    x = Dropout(
        phi_dropout,
        name="phi_attention_projection_dropout",
    )(x)

    # Attention blocks over particles.
    for block_idx in range(num_attention_blocks):
        attn_out = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=attention_dim // num_heads,
            dropout=attention_dropout,
            name=f"particle_attention_{block_idx + 1}",
        )(
            query=x,
            value=x,
            key=x,
            attention_mask=particle_mask,
        )

        x = Add(name=f"attention_residual_{block_idx + 1}")([x, attn_out])
        x = LayerNormalization(name=f"attention_norm_{block_idx + 1}")(x)

        # Small feed-forward block after attention, transformer-style.
        ff = TimeDistributed(
            Dense(attention_dim, activation="relu"),
            name=f"attention_ff_{block_idx + 1}",
        )(x)

        ff = Dropout(
            attention_dropout,
            name=f"attention_ff_dropout_{block_idx + 1}",
        )(ff)

        x = Add(name=f"ff_residual_{block_idx + 1}")([x, ff])
        x = LayerNormalization(name=f"ff_norm_{block_idx + 1}")(x)

    # EFN-style energy-weighted sum:
    # latent = sum_i z_i * x_i
    z_expanded = Lambda(
        lambda z: K.expand_dims(z, axis=-1),
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

    # Backend F network.
    y = latent

    for i, units in enumerate(F_sizes):
        y = Dense(
            units,
            activation="relu",
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
        name="aefn",
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
        "phi_dropout": config.get("phi_dropout", 0.0),
        "attention_dim": config["attention_dim"],
        "num_heads": config["num_heads"],
        "attention_dropout": config.get("attention_dropout", 0.0),
        "num_attention_blocks": config["num_attention_blocks"],
        "latent_dropout": config.get("latent_dropout", 0.0),
        "F_sizes": str(config["F_sizes"]),
        "F_dropout": config.get("F_dropout", 0.0),
        "output_dim": config["output_dim"],
    }
