import tensorflow as tf
from tf_keras import Model
from tf_keras.layers import (
    Input,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Lambda,
)
from tf_keras.optimizers import Adam


# ============================================================
# Fixed architecture switches for this file
# ============================================================
MODEL_NAME = "aefn_before_sum_after_sum"
USE_ATTENTION_BEFORE_PHI = False
USE_ATTENTION_BEFORE_SUM = True
USE_ATTENTION_AFTER_SUM = True


def get_default_config() -> dict:
    """Default settings for this AEFN variant.

    The EFN backbone is:
        particle coordinates -> Phi -> z-weighted sum -> F -> output

    Transformer-style attention blocks are inserted only at the
    positions enabled by the three constants above.
    """
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
        "use_early_stopping": True,
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,
        # Used only for attention after the EFN sum.
        # After the sum there is one jet vector, so we project it
        # into several global tokens before self-attention.
        "global_tokens": 4,
    }


def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """Prepare the two AEFN inputs.

    Shared tensor layout:
        X[..., 0] = z_i
        X[..., 1] = Delta eta_i
        X[..., 2] = Delta phi_i

    Returned input shapes:
        z: (batch, particles)
        p: (batch, particles, 2)
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
    extra_info = {"num_particles": X.shape[1]}
    return train_inputs, val_inputs, test_inputs, extra_info


def feed_forward_block(x, sizes, activation, dropout_rate, name_prefix):
    """Simple feed-forward network used for Phi and F."""
    for i, size in enumerate(sizes):
        x = Dense(
            size,
            activation=activation,
            name=f"{name_prefix}_dense_{i + 1}",
        )(x)
        if dropout_rate > 0:
            x = Dropout(
                dropout_rate,
                name=f"{name_prefix}_dropout_{i + 1}",
            )(x)
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
    """Transformer-style block.

    Structure:
        Multi-Head Attention
        -> residual Add + LayerNorm
        -> Feed Forward
        -> residual Add + LayerNorm

    This is Transformer-like, but the complete model is still an EFN
    because we keep Phi, the z-weighted sum, and F.
    """
    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=f"{name_prefix}_mha",
    )(
        query=x,
        value=x,
        key=x,
        attention_mask=attention_mask,
    )
    x = Add(name=f"{name_prefix}_attention_add")([x, attn_out])
    x = LayerNormalization(name=f"{name_prefix}_attention_norm")(x)

    ff_out = Dense(
        int(x.shape[-1]),
        activation=activation,
        name=f"{name_prefix}_ff_dense",
    )(x)
    ff_out = Dropout(
        dropout_rate,
        name=f"{name_prefix}_ff_dropout",
    )(ff_out)
    x = Add(name=f"{name_prefix}_ff_add")([x, ff_out])
    x = LayerNormalization(name=f"{name_prefix}_ff_norm")(x)
    return x


def build_particle_attention_mask(z_input):
    """Mask padded particles in particle-level attention.

    Real particle:   z_i > 0
    Padded particle: z_i = 0
    """
    particle_mask = Lambda(
        lambda z: tf.cast(tf.greater(z, 0.0), tf.bool),
        name="particle_mask",
    )(z_input)
    attention_mask = Lambda(
        lambda m: tf.tile(
            tf.expand_dims(m, axis=1),
            [1, tf.shape(m)[1], 1],
        ),
        name="particle_attention_mask",
    )(particle_mask)
    return attention_mask


def apply_attention_after_sum(
    event_representation,
    config,
    activation,
    num_heads,
    attention_dropout,
):
    """Apply attention after the EFN sum and before F.

    After sum_i z_i h_i, only one jet vector remains.
    Self-attention over one token would be trivial. Therefore, this
    version projects the jet vector into several learned global tokens,
    applies Transformer-style self-attention, and pools the tokens back
    into one refined jet vector.
    """
    attention_dim = config.get("attention_dim", 128)
    global_tokens = config.get("global_tokens", 4)

    projected = Dense(
        global_tokens * attention_dim,
        activation=activation,
        name="after_sum_token_projection",
    )(event_representation)

    tokens = Lambda(
        lambda x: tf.reshape(
            x,
            (-1, global_tokens, attention_dim),
        ),
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

    refined_jet = Lambda(
        lambda x: tf.reduce_mean(x, axis=1),
        name="after_sum_token_pooling",
    )(tokens)
    return refined_jet


def build_model(config: dict, extra_info: dict | None = None):
    """Build this AEFN variant.

    Possible attention positions:
        1) before Phi
        2) after Phi, before the EFN z-weighted sum
        3) after the EFN z-weighted sum, before F
    """
    activation = config.get("activation", "gelu")
    attention_dim = config.get("attention_dim", 128)
    num_heads = config.get("num_heads", 4)
    attention_dropout = config.get("attention_dropout", 0.0)
    latent_dropout = config.get("latent_dropout", 0.0)
    F_dropouts = config.get("F_dropouts", 0.0)

    if attention_dim % num_heads != 0:
        raise ValueError(
            "attention_dim must be divisible by num_heads, "
            f"got {attention_dim} and {num_heads}."
        )

    # z_i: momentum fraction, shape (batch, particles)
    z_input = Input(shape=(None,), name="z_input")

    # p_i = (Delta eta_i, Delta phi_i), shape (batch, particles, 2)
    p_input = Input(
        shape=(None, config["input_dim"]),
        name="p_input",
    )

    particle_attention_mask = build_particle_attention_mask(z_input)

    # ========================================================
    # 1. Optional attention before Phi
    # ========================================================
    x = p_input
    if USE_ATTENTION_BEFORE_PHI:
        # Raw particle coordinates are embedded first because attention
        # works in a higher-dimensional feature space.
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

    # ========================================================
    # 2. Phi: per-particle feature learning
    # ========================================================
    phi_output = feed_forward_block(
        x=x,
        sizes=config["Phi_sizes"],
        activation=activation,
        dropout_rate=latent_dropout,
        name_prefix="Phi",
    )

    # ========================================================
    # 3. Optional attention after Phi / before sum
    # ========================================================
    if USE_ATTENTION_BEFORE_SUM:
        # Attention now acts on learned particle representations.
        # Each particle can use information from the other particles
        # before the global EFN aggregation is performed.
        phi_output = transformer_style_attention_block(
            x=phi_output,
            num_heads=num_heads,
            key_dim=config["Phi_sizes"][-1] // num_heads,
            activation=activation,
            dropout_rate=attention_dropout,
            name_prefix="attention_before_sum",
            attention_mask=particle_attention_mask,
        )

    # ========================================================
    # 4. EFN-style z-weighted sum
    # ========================================================
    # h_jet = sum_i z_i * h_i
    z_expanded = Lambda(
        lambda z: tf.expand_dims(z, axis=-1),
        name="expand_z",
    )(z_input)

    weighted_particles = Lambda(
        lambda inputs: inputs[0] * inputs[1],
        name="z_times_particle_representation",
    )([z_expanded, phi_output])

    event_representation = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name="efn_weighted_sum",
    )(weighted_particles)

    event_representation = Dropout(
        latent_dropout,
        name="latent_dropout",
    )(event_representation)

    # ========================================================
    # 5. Optional attention after sum / before F
    # ========================================================
    if USE_ATTENTION_AFTER_SUM:
        event_representation = apply_attention_after_sum(
            event_representation=event_representation,
            config=config,
            activation=activation,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )

    # ========================================================
    # 6. F: jet-level classifier
    # ========================================================
    f_output = feed_forward_block(
        x=event_representation,
        sizes=config["F_sizes"],
        activation=activation,
        dropout_rate=F_dropouts,
        name_prefix="F",
    )

    output = Dense(
        config["output_dim"],
        activation="softmax",
        name="output",
    )(f_output)

    model = Model(
        inputs=[z_input, p_input],
        outputs=output,
        name=MODEL_NAME,
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=config["learning_rate"]),
        metrics=["accuracy"],
    )
    return model


def get_model_summary_fields(config: dict) -> dict:
    """Return the important settings for result logging."""
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
        "global_tokens": config.get("global_tokens", 4),
    }
