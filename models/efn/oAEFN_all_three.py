import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tf_keras import Model
from tf_keras.layers import (
    Input,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    Add,
    Lambda,
    Concatenate,
)
from tf_keras.optimizers import Adam


MODEL_NAME = "oaefn_all_three"

USE_ATTENTION_BEFORE_PHI = True
USE_ATTENTION_BEFORE_SUM = True
USE_ATTENTION_AFTER_SUM = True


def get_default_config() -> dict:
    return {
        "model_name": MODEL_NAME,
        "results_dir_name": f"{MODEL_NAME}_results",

        # AEFN
        "input_dim": 2,
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),
        "output_dim": 2,

        "latent_dropout": 0.1,
        "F_dropouts": 0.1,
        "activation": "gelu",

        # Attention
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,
        "global_tokens": 4,

        "remove_eecs": True,
        "nsubs_dim": 60,
        "eecs_dim": 23,
        "n_pca_components": 13,

        # Training
        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "use_early_stopping": True,
    }


def remove_eecs_from_observables(
    X_obs: np.ndarray,
    config: dict,
) -> np.ndarray:

    if not config.get("remove_eecs", False):
        return X_obs

    nsubs_dim = config.get("nsubs_dim", 60)
    eecs_dim = config.get("eecs_dim", 23)

    eecs_start = nsubs_dim
    eecs_end = nsubs_dim + eecs_dim

    if eecs_end > X_obs.shape[1]:
        raise ValueError(
            f"EEC slice outside observable matrix. "
            f"nsubs_dim={nsubs_dim}, "
            f"eecs_dim={eecs_dim}, "
            f"X_obs shape={X_obs.shape}"
        )

    return np.concatenate(
        [
            X_obs[:, :eecs_start],
            X_obs[:, eecs_end:],
        ],
        axis=1,
    )


# ============================================================
# Prepare fold inputs
# ============================================================

def prepare_fold_inputs(
    X,
    train_idx,
    val_idx,
    test_idx,
    config,
    fold_dir,
    context,
):

    if not isinstance(X, dict):
        raise ValueError(
            "oAEFN requires Marvin data with "
            "X['parts'] and X['obsvs']."
        )

    if "parts" not in X:
        raise KeyError(
            "oAEFN expected X['parts']."
        )

    if "obsvs" not in X:
        raise KeyError(
            "oAEFN expected X['obsvs']."
        )

    # ========================================================
    # Read directly from Marvin
    # ========================================================

    X_parts = X["parts"]
    X_obs = X["obsvs"]

    print(
        "Using precomputed Marvin observables "
        "from obsvs/x for oAEFN."
    )

    print(
        "Particle tensor shape:",
        X_parts.shape,
    )

    print(
        "Observable matrix shape:",
        X_obs.shape,
    )

    # ========================================================
    # Remove EECs
    # ========================================================

    X_obs = remove_eecs_from_observables(
        X_obs,
        config,
    )

    # ========================================================
    # Particle inputs
    # ========================================================

    z_train = X_parts[train_idx, :, 0]
    p_train = X_parts[train_idx, :, 1:3]

    z_val = X_parts[val_idx, :, 0]
    p_val = X_parts[val_idx, :, 1:3]

    z_test = X_parts[test_idx, :, 0]
    p_test = X_parts[test_idx, :, 1:3]

    # ========================================================
    # Observable inputs
    # ========================================================

    obs_train_raw = X_obs[train_idx]
    obs_val_raw = X_obs[val_idx]
    obs_test_raw = X_obs[test_idx]

    # ========================================================
    # Scale observables
    #
    # Fit ONLY on training fold
    # ========================================================

    scaler = StandardScaler()

    obs_train_scaled = scaler.fit_transform(
        obs_train_raw
    )

    obs_val_scaled = scaler.transform(
        obs_val_raw
    )

    obs_test_scaled = scaler.transform(
        obs_test_raw
    )

    # ========================================================
    # PCA
    #
    # Fit ONLY on training fold
    # ========================================================

    max_components = min(
        config["n_pca_components"],
        obs_train_scaled.shape[0],
        obs_train_scaled.shape[1],
    )

    pca = PCA(
        n_components=max_components,
        random_state=config["seed"],
    )

    obs_train = pca.fit_transform(
        obs_train_scaled
    )

    obs_val = pca.transform(
        obs_val_scaled
    )

    obs_test = pca.transform(
        obs_test_scaled
    )

    # ========================================================
    # Three inputs
    # ========================================================

    train_inputs = [
        z_train,
        p_train,
        obs_train,
    ]

    val_inputs = [
        z_val,
        p_val,
        obs_val,
    ]

    test_inputs = [
        z_test,
        p_test,
        obs_test,
    ]

    extra_info = {
        "num_particles":
            X_parts.shape[1],

        "raw_num_observables":
            X_obs.shape[1],

        "num_observables":
            obs_train.shape[1],

        "observable_source":
            "marvin_obsvs_x_without_eecs",

        "n_pca_components":
            config["n_pca_components"],

        "remove_eecs":
            config.get(
                "remove_eecs",
                False,
            ),

        "nsubs_dim":
            config.get(
                "nsubs_dim",
                60,
            ),

        "eecs_dim":
            config.get(
                "eecs_dim",
                23,
            ),
    }

    return (
        train_inputs,
        val_inputs,
        test_inputs,
        extra_info,
    )


# ============================================================
# Feed-forward block
# ============================================================

def feed_forward_block(
    x,
    sizes,
    activation,
    dropout_rate,
    name_prefix,
):

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


# ============================================================
# Transformer-style attention block
# ============================================================

def transformer_style_attention_block(
    x,
    num_heads,
    key_dim,
    activation,
    dropout_rate,
    name_prefix,
    attention_mask=None,
):

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

    x = Add(
        name=f"{name_prefix}_attention_add"
    )(
        [
            x,
            attn_out,
        ]
    )

    x = LayerNormalization(
        name=f"{name_prefix}_attention_norm"
    )(x)

    ff_out = Dense(
        int(x.shape[-1]),
        activation=activation,
        name=f"{name_prefix}_ff_dense",
    )(x)

    ff_out = Dropout(
        dropout_rate,
        name=f"{name_prefix}_ff_dropout",
    )(ff_out)

    x = Add(
        name=f"{name_prefix}_ff_add"
    )(
        [
            x,
            ff_out,
        ]
    )

    x = LayerNormalization(
        name=f"{name_prefix}_ff_norm"
    )(x)

    return x


# ============================================================
# Particle attention mask
# ============================================================

def build_particle_attention_mask(
    z_input,
):

    particle_mask = Lambda(
        lambda z: tf.cast(
            tf.greater(z, 0.0),
            tf.bool,
        ),
        name="particle_mask",
    )(z_input)

    attention_mask = Lambda(
        lambda m: tf.tile(
            tf.expand_dims(
                m,
                axis=1,
            ),
            [
                1,
                tf.shape(m)[1],
                1,
            ],
        ),
        name="particle_attention_mask",
    )(particle_mask)

    return attention_mask


# ============================================================
# Attention after EFN sum
# ============================================================

def apply_attention_after_sum(
    event_representation,
    config,
    activation,
    num_heads,
    attention_dropout,
):

    attention_dim = config.get(
        "attention_dim",
        128,
    )

    global_tokens = config.get(
        "global_tokens",
        4,
    )

    projected = Dense(
        global_tokens * attention_dim,
        activation=activation,
        name="after_sum_token_projection",
    )(event_representation)

    tokens = Lambda(
        lambda x: tf.reshape(
            x,
            (
                -1,
                global_tokens,
                attention_dim,
            ),
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
        lambda x: tf.reduce_mean(
            x,
            axis=1,
        ),
        name="after_sum_token_pooling",
    )(tokens)

    return refined_jet


# ============================================================
# Build oAEFN
# ============================================================

def build_model(
    config: dict,
    extra_info: dict | None = None,
):

    if extra_info is None:
        raise ValueError(
            "oAEFN requires extra_info "
            "from prepare_fold_inputs."
        )

    activation = config.get(
        "activation",
        "gelu",
    )

    attention_dim = config.get(
        "attention_dim",
        128,
    )

    num_heads = config.get(
        "num_heads",
        4,
    )

    attention_dropout = config.get(
        "attention_dropout",
        0.0,
    )

    latent_dropout = config.get(
        "latent_dropout",
        0.0,
    )

    F_dropouts = config.get(
        "F_dropouts",
        0.0,
    )

    num_particles = extra_info[
        "num_particles"
    ]

    num_observables = extra_info[
        "num_observables"
    ]

    if attention_dim % num_heads != 0:
        raise ValueError(
            "attention_dim must be "
            "divisible by num_heads."
        )

    # ========================================================
    # Inputs
    # ========================================================

    z_input = Input(
        shape=(num_particles,),
        name="z_input",
    )

    p_input = Input(
        shape=(
            num_particles,
            config["input_dim"],
        ),
        name="p_input",
    )

    obs_input = Input(
        shape=(num_observables,),
        name="jet_observables",
    )

    particle_attention_mask = (
        build_particle_attention_mask(
            z_input
        )
    )

    # ========================================================
    # 1. Attention before Phi
    # ========================================================

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

    # ========================================================
    # 2. Phi
    # ========================================================

    phi_output = feed_forward_block(
        x=x,
        sizes=config["Phi_sizes"],
        activation=activation,
        dropout_rate=latent_dropout,
        name_prefix="Phi",
    )

    # ========================================================
    # 3. Attention before weighted sum
    # ========================================================

    if USE_ATTENTION_BEFORE_SUM:

        if (
            config["Phi_sizes"][-1]
            % num_heads
            != 0
        ):
            raise ValueError(
                "Last Phi dimension must "
                "be divisible by num_heads."
            )

        phi_output = (
            transformer_style_attention_block(
                x=phi_output,
                num_heads=num_heads,
                key_dim=(
                    config["Phi_sizes"][-1]
                    // num_heads
                ),
                activation=activation,
                dropout_rate=attention_dropout,
                name_prefix="attention_before_sum",
                attention_mask=particle_attention_mask,
            )
        )

    # ========================================================
    # 4. EFN z-weighted sum
    # ========================================================

    z_expanded = Lambda(
        lambda z: tf.expand_dims(
            z,
            axis=-1,
        ),
        name="expand_z",
    )(z_input)

    weighted_particles = Lambda(
        lambda inputs:
            inputs[0] * inputs[1],
        name="z_times_particle_representation",
    )(
        [
            z_expanded,
            phi_output,
        ]
    )

    event_representation = Lambda(
        lambda x: tf.reduce_sum(
            x,
            axis=1,
        ),
        name="efn_weighted_sum",
    )(weighted_particles)

    event_representation = Dropout(
        latent_dropout,
        name="latent_dropout",
    )(event_representation)

    # ========================================================
    # 5. Attention after sum
    # ========================================================

    if USE_ATTENTION_AFTER_SUM:

        event_representation = (
            apply_attention_after_sum(
                event_representation=event_representation,
                config=config,
                activation=activation,
                num_heads=num_heads,
                attention_dropout=attention_dropout,
            )
        )

    # ========================================================
    # 6. CONCATENATE Marvin observables before F
    # ========================================================

    f_input = Concatenate(
        name="event_plus_jet_observables",
    )(
        [
            event_representation,
            obs_input,
        ]
    )

    # ========================================================
    # 7. F network
    # ========================================================

    f_output = feed_forward_block(
        x=f_input,
        sizes=config["F_sizes"],
        activation=activation,
        dropout_rate=F_dropouts,
        name_prefix="F",
    )

    # ========================================================
    # Output
    # ========================================================

    output = Dense(
        config["output_dim"],
        activation="softmax",
        name="output",
    )(f_output)

    model = Model(
        inputs=[
            z_input,
            p_input,
            obs_input,
        ],
        outputs=output,
        name=MODEL_NAME,
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(
            learning_rate=config[
                "learning_rate"
            ]
        ),
        metrics=["accuracy"],
    )

    return model


# ============================================================
# Summary fields
# ============================================================

def get_model_summary_fields(
    config: dict,
) -> dict:

    return {
        "model_name":
            MODEL_NAME,

        "attention_before_phi":
            USE_ATTENTION_BEFORE_PHI,

        "attention_before_sum":
            USE_ATTENTION_BEFORE_SUM,

        "attention_after_sum":
            USE_ATTENTION_AFTER_SUM,

        "input_dim":
            config["input_dim"],

        "Phi_sizes":
            str(config["Phi_sizes"]),

        "F_sizes":
            str(config["F_sizes"]),

        "activation":
            config.get(
                "activation",
                "gelu",
            ),

        "latent_dropout":
            config.get(
                "latent_dropout",
                0.0,
            ),

        "F_dropouts":
            config.get(
                "F_dropouts",
                0.0,
            ),

        "output_dim":
            config["output_dim"],

        "attention_dim":
            config.get(
                "attention_dim",
                128,
            ),

        "num_heads":
            config.get(
                "num_heads",
                4,
            ),

        "attention_dropout":
            config.get(
                "attention_dropout",
                0.0,
            ),

        "global_tokens":
            config.get(
                "global_tokens",
                4,
            ),

        "n_pca_components":
            config.get(
                "n_pca_components",
                13,
            ),

        "remove_eecs":
            config.get(
                "remove_eecs",
                False,
            ),

        "nsubs_dim":
            config.get(
                "nsubs_dim",
                60,
            ),

        "eecs_dim":
            config.get(
                "eecs_dim",
                23,
            ),
    }
