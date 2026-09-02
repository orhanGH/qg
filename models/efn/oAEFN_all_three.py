import numpy as np
import tensorflow as tf
import energyflow as ef

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


# ============================================================
# Fixed architecture switches
# ============================================================

MODEL_NAME = "oaefn_all_three"

USE_ATTENTION_BEFORE_PHI = True
USE_ATTENTION_BEFORE_SUM = True
USE_ATTENTION_AFTER_SUM = True


# ============================================================
# Default configuration
# ============================================================

def get_default_config() -> dict:
    return {
        "model_name": MODEL_NAME,
        "results_dir_name": f"{MODEL_NAME}_results",

        # ----------------------------------------------------
        # Particle / AEFN branch
        # ----------------------------------------------------
        "input_dim": 2,

        "Phi_sizes": (
            100,
            100,
            128,
        ),

        "F_sizes": (
            100,
            100,
            100,
        ),

        "output_dim": 2,

        "activation": "gelu",

        "latent_dropout": 0.1,
        "F_dropouts": 0.1,

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,

        # Used only for attention after the EFN sum
        "global_tokens": 4,

        # ----------------------------------------------------
        # Observable branch
        # ----------------------------------------------------

        # Fallback only
        "efp_degree": 3,

        # Marvin observables:
        #
        # nsubs = 60
        # eecs  = 23
        # efps  = 3389
        #
        # Total:
        # 3472
        #
        # We remove EECs:
        # 60 + 3389 = 3449
        "remove_eecs": True,
        "nsubs_dim": 60,
        "eecs_dim": 23,

        # PCA compression before observables enter F
        "n_pca_components": 13,

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------
        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "use_early_stopping": True,
    }


# ============================================================
# Observable utilities
# ============================================================

def compute_efp_observables(
    X: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Fallback mode only.

    On Marvin, observables should normally come from X["obsvs"].
    """

    efpset = ef.EFPSet(
        f"d<={config['efp_degree']}",
        measure="hadr",
        beta=1,
        kappa=1,
        normed=True,
        coords="ptyphim",
    )

    return efpset.batch_compute(X)


def remove_eecs_from_observables(
    X_obs: np.ndarray,
    config: dict,
) -> np.ndarray:
    """
    Marvin observable layout:

        concat(
            nsubs,
            eecs,
            efps
        )

    We remove the EEC block.

    Result:

        concat(
            nsubs,
            efps
        )
    """

    if not config.get(
        "remove_eecs",
        False,
    ):
        return X_obs

    nsubs_dim = config.get(
        "nsubs_dim",
        60,
    )

    eecs_dim = config.get(
        "eecs_dim",
        23,
    )

    eecs_start = nsubs_dim

    eecs_end = (
        nsubs_dim
        + eecs_dim
    )

    if eecs_end > X_obs.shape[1]:
        raise ValueError(
            "EEC slice is outside observable matrix. "
            f"nsubs_dim={nsubs_dim}, "
            f"eecs_dim={eecs_dim}, "
            f"X_obs.shape={X_obs.shape}"
        )

    X_obs_without_eecs = np.concatenate(
        [
            X_obs[:, :eecs_start],
            X_obs[:, eecs_end:],
        ],
        axis=1,
    )

    print(
        "Removed EECs from oAEFN observables."
    )

    print(
        "Observable matrix before:",
        X_obs.shape,
    )

    print(
        "Observable matrix after:",
        X_obs_without_eecs.shape,
    )

    return X_obs_without_eecs


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
    """
    Marvin mode:

        X["parts"]
            shape:
            (N_jets, max_particles, 3)

            [..., 0] = z_i
            [..., 1] = Delta eta_i
            [..., 2] = Delta phi_i

        X["obsvs"]
            shape:
            (N_jets, 3472)

    Model inputs:

        [
            z,
            particle_coordinates,
            jet_observables
        ]
    """

    # ========================================================
    # Load particle + observable data
    # ========================================================

    if isinstance(X, dict):

        if "parts" not in X:
            raise KeyError(
                "oAEFN expected X['parts'], "
                "but the key is missing."
            )

        if "obsvs" not in X:
            raise KeyError(
                "oAEFN expected X['obsvs'], "
                "but the key is missing."
            )

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

        X_obs = (
            remove_eecs_from_observables(
                X_obs,
                config,
            )
        )

        observable_source = (
            "marvin_obsvs_x_without_eecs"
        )

    else:

        # ----------------------------------------------------
        # Fallback mode
        # ----------------------------------------------------

        X_parts = X

        if "X_obs" not in context:

            print(
                "Computing EFP observables "
                "for oAEFN fallback mode..."
            )

            context["X_obs"] = (
                compute_efp_observables(
                    X_parts,
                    config,
                )
            )

        X_obs = context["X_obs"]

        observable_source = (
            "computed_efps"
        )

    # ========================================================
    # Shape checks
    # ========================================================

    if X_parts.ndim != 3:
        raise ValueError(
            "Expected X_parts with shape "
            "(N, particles, features), "
            f"got {X_parts.shape}"
        )

    if X_parts.shape[-1] < 3:
        raise ValueError(
            "Expected particle features "
            "[z, Delta eta, Delta phi], "
            f"got {X_parts.shape}"
        )

    if X_obs.ndim != 2:
        raise ValueError(
            "Expected X_obs with shape "
            "(N, observables), "
            f"got {X_obs.shape}"
        )

    if (
        X_parts.shape[0]
        != X_obs.shape[0]
    ):
        raise ValueError(
            "Particles and observables have "
            "different number of jets: "
            f"{X_parts.shape[0]} "
            f"vs {X_obs.shape[0]}"
        )

    # ========================================================
    # Particle inputs
    # ========================================================

    z_train = (
        X_parts[
            train_idx,
            :,
            0,
        ]
    )

    p_train = (
        X_parts[
            train_idx,
            :,
            1:3,
        ]
    )

    z_val = (
        X_parts[
            val_idx,
            :,
            0,
        ]
    )

    p_val = (
        X_parts[
            val_idx,
            :,
            1:3,
        ]
    )

    z_test = (
        X_parts[
            test_idx,
            :,
            0,
        ]
    )

    p_test = (
        X_parts[
            test_idx,
            :,
            1:3,
        ]
    )

    # ========================================================
    # Observable inputs
    # ========================================================

    obs_train_raw = (
        X_obs[
            train_idx
        ]
    )

    obs_val_raw = (
        X_obs[
            val_idx
        ]
    )

    obs_test_raw = (
        X_obs[
            test_idx
        ]
    )

    # ========================================================
    # StandardScaler
    #
    # Fit ONLY on training fold
    # ========================================================

    scaler = StandardScaler()

    obs_train_scaled = (
        scaler.fit_transform(
            obs_train_raw
        )
    )

    obs_val_scaled = (
        scaler.transform(
            obs_val_raw
        )
    )

    obs_test_scaled = (
        scaler.transform(
            obs_test_raw
        )
    )

    # ========================================================
    # PCA
    #
    # Fit ONLY on training fold
    # ========================================================

    max_components = min(
        config[
            "n_pca_components"
        ],
        obs_train_scaled.shape[0],
        obs_train_scaled.shape[1],
    )

    pca = PCA(
        n_components=max_components,
        random_state=config[
            "seed"
        ],
    )

    obs_train = (
        pca.fit_transform(
            obs_train_scaled
        )
    )

    obs_val = (
        pca.transform(
            obs_val_scaled
        )
    )

    obs_test = (
        pca.transform(
            obs_test_scaled
        )
    )

    # ========================================================
    # Model inputs
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
            observable_source,

        "n_pca_components":
            config[
                "n_pca_components"
            ],

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
# Feed-forward helper
# ============================================================

def feed_forward_block(
    x,
    sizes,
    activation,
    dropout_rate,
    name_prefix,
):
    """
    Dense feed-forward block used for Phi and F.
    """

    for i, size in enumerate(
        sizes
    ):

        x = Dense(
            size,
            activation=activation,
            name=(
                f"{name_prefix}"
                f"_dense_{i + 1}"
            ),
        )(x)

        if dropout_rate > 0:

            x = Dropout(
                dropout_rate,
                name=(
                    f"{name_prefix}"
                    f"_dropout_{i + 1}"
                ),
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
    """
    Transformer-style block:

        MultiHeadAttention
            ↓
        Residual + LayerNorm
            ↓
        Feed Forward
            ↓
        Residual + LayerNorm
    """

    # ========================================================
    # Multi-Head Attention
    # ========================================================

    attn_out = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate,
        name=(
            f"{name_prefix}_mha"
        ),
    )(
        query=x,
        value=x,
        key=x,
        attention_mask=attention_mask,
    )

    # ========================================================
    # Residual connection + normalization
    # ========================================================

    x = Add(
        name=(
            f"{name_prefix}"
            "_attention_add"
        )
    )(
        [
            x,
            attn_out,
        ]
    )

    x = LayerNormalization(
        name=(
            f"{name_prefix}"
            "_attention_norm"
        )
    )(x)

    # ========================================================
    # Feed-forward layer
    # ========================================================

    ff_out = Dense(
        int(
            x.shape[-1]
        ),
        activation=activation,
        name=(
            f"{name_prefix}"
            "_ff_dense"
        ),
    )(x)

    ff_out = Dropout(
        dropout_rate,
        name=(
            f"{name_prefix}"
            "_ff_dropout"
        ),
    )(ff_out)

    # ========================================================
    # Residual + normalization
    # ========================================================

    x = Add(
        name=(
            f"{name_prefix}"
            "_ff_add"
        )
    )(
        [
            x,
            ff_out,
        ]
    )

    x = LayerNormalization(
        name=(
            f"{name_prefix}"
            "_ff_norm"
        )
    )(x)

    return x


# ============================================================
# Particle padding mask
# ============================================================

def build_particle_attention_mask(
    z_input,
):
    """
    Real particle:
        z_i > 0

    Padding:
        z_i = 0
    """

    particle_mask = Lambda(
        lambda z:
            tf.cast(
                tf.greater(
                    z,
                    0.0,
                ),
                tf.bool,
            ),
        name="particle_mask",
    )(z_input)

    attention_mask = Lambda(
        lambda m:
            tf.tile(
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
        name=(
            "particle_attention_mask"
        ),
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
    """
    After the z-weighted sum only one event vector remains.

    Attention on a single token would be trivial.

    Therefore:
        event vector
            ↓
        project to multiple learned global tokens
            ↓
        self-attention
            ↓
        mean pooling
            ↓
        refined event representation
    """

    attention_dim = config.get(
        "attention_dim",
        128,
    )

    global_tokens = config.get(
        "global_tokens",
        4,
    )

    # ========================================================
    # Event vector -> global tokens
    # ========================================================

    projected = Dense(
        global_tokens
        * attention_dim,
        activation=activation,
        name=(
            "after_sum_"
            "token_projection"
        ),
    )(
        event_representation
    )

    tokens = Lambda(
        lambda x:
            tf.reshape(
                x,
                (
                    -1,
                    global_tokens,
                    attention_dim,
                ),
            ),
        name="after_sum_tokens",
    )(projected)

    # ========================================================
    # Third attention layer
    # ========================================================

    tokens = (
        transformer_style_attention_block(
            x=tokens,

            num_heads=num_heads,

            key_dim=(
                attention_dim
                // num_heads
            ),

            activation=activation,

            dropout_rate=(
                attention_dropout
            ),

            name_prefix=(
                "attention_after_sum"
            ),

            attention_mask=None,
        )
    )

    # ========================================================
    # Tokens -> one event representation
    # ========================================================

    refined_event = Lambda(
        lambda x:
            tf.reduce_mean(
                x,
                axis=1,
            ),
        name=(
            "after_sum_token_pooling"
        ),
    )(tokens)

    return refined_event


# ============================================================
# Build oAEFN
# ============================================================

def build_model(
    config: dict,
    extra_info: dict | None = None,
):
    """
    oAEFN architecture:

        particle coordinates
            ↓
        Attention 1
            ↓
        Phi
            ↓
        Attention 2
            ↓
        z-weighted sum
            ↓
        Attention 3
            ↓
        event representation
            │
            │
            ├─────────────── jet observables
            │
            ▼
        Concatenate
            ↓
        F
            ↓
        Softmax
    """

    if extra_info is None:
        raise ValueError(
            "oAEFN build_model requires "
            "extra_info from "
            "prepare_fold_inputs."
        )

    # ========================================================
    # Config
    # ========================================================

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

    attention_dropout = (
        config.get(
            "attention_dropout",
            0.0,
        )
    )

    latent_dropout = (
        config.get(
            "latent_dropout",
            0.0,
        )
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

    # ========================================================
    # Safety checks
    # ========================================================

    if (
        attention_dim
        % num_heads
        != 0
    ):
        raise ValueError(
            "attention_dim must be "
            "divisible by num_heads. "
            f"Got attention_dim="
            f"{attention_dim}, "
            f"num_heads="
            f"{num_heads}"
        )

    if (
        config[
            "Phi_sizes"
        ][-1]
        % num_heads
        != 0
    ):
        raise ValueError(
            "The last Phi dimension "
            "must be divisible by "
            "num_heads. "
            f"Got Phi output="
            f"{config['Phi_sizes'][-1]}, "
            f"num_heads="
            f"{num_heads}"
        )

    # ========================================================
    # Inputs
    # ========================================================

    # z_i
    z_input = Input(
        shape=(
            num_particles,
        ),
        name="z_input",
    )

    # p_i = (Delta eta, Delta phi)
    p_input = Input(
        shape=(
            num_particles,
            config[
                "input_dim"
            ],
        ),
        name="p_input",
    )

    # PCA-compressed jet observables
    obs_input = Input(
        shape=(
            num_observables,
        ),
        name="jet_observables",
    )

    # ========================================================
    # Padding mask
    # ========================================================

    particle_attention_mask = (
        build_particle_attention_mask(
            z_input
        )
    )

    # ========================================================
    # PARTICLE / AEFN BRANCH
    # ========================================================

    x = p_input

    # ========================================================
    # 1. ATTENTION BEFORE PHI
    # ========================================================

    if USE_ATTENTION_BEFORE_PHI:

        # Raw coordinates have dimension 2.
        # First embed them into attention_dim.
        x = Dense(
            attention_dim,
            activation=activation,
            name=(
                "particle_embedding_"
                "before_phi"
            ),
        )(x)

        x = (
            transformer_style_attention_block(
                x=x,

                num_heads=num_heads,

                key_dim=(
                    attention_dim
                    // num_heads
                ),

                activation=activation,

                dropout_rate=(
                    attention_dropout
                ),

                name_prefix=(
                    "attention_before_phi"
                ),

                attention_mask=(
                    particle_attention_mask
                ),
            )
        )

    # ========================================================
    # 2. PHI
    # ========================================================

    phi_output = (
        feed_forward_block(
            x=x,

            sizes=config[
                "Phi_sizes"
            ],

            activation=activation,

            dropout_rate=(
                latent_dropout
            ),

            name_prefix="Phi",
        )
    )

    # ========================================================
    # 3. ATTENTION BEFORE SUM
    # ========================================================

    if USE_ATTENTION_BEFORE_SUM:

        phi_output = (
            transformer_style_attention_block(
                x=phi_output,

                num_heads=num_heads,

                key_dim=(
                    config[
                        "Phi_sizes"
                    ][-1]
                    // num_heads
                ),

                activation=activation,

                dropout_rate=(
                    attention_dropout
                ),

                name_prefix=(
                    "attention_before_sum"
                ),

                attention_mask=(
                    particle_attention_mask
                ),
            )
        )

    # ========================================================
    # 4. EFN z-WEIGHTED SUM
    # ========================================================

    z_expanded = Lambda(
        lambda z:
            tf.expand_dims(
                z,
                axis=-1,
            ),
        name="expand_z",
    )(z_input)

    weighted_particles = Lambda(
        lambda inputs:
            (
                inputs[0]
                * inputs[1]
            ),
        name=(
            "z_times_particle_"
            "representation"
        ),
    )(
        [
            z_expanded,
            phi_output,
        ]
    )

    event_representation = Lambda(
        lambda x:
            tf.reduce_sum(
                x,
                axis=1,
            ),
        name="efn_weighted_sum",
    )(
        weighted_particles
    )

    event_representation = Dropout(
        latent_dropout,
        name="latent_dropout",
    )(
        event_representation
    )

    # ========================================================
    # 5. ATTENTION AFTER SUM
    # ========================================================

    if USE_ATTENTION_AFTER_SUM:

        event_representation = (
            apply_attention_after_sum(
                event_representation=(
                    event_representation
                ),

                config=config,

                activation=activation,

                num_heads=num_heads,

                attention_dropout=(
                    attention_dropout
                ),
            )
        )

    # ========================================================
    # 6. CONCATENATE JET OBSERVABLES
    #
    # IMPORTANT:
    #
    # Observables enter only AFTER all three attention
    # stages and directly BEFORE the F network.
    # ========================================================

    f_input = Concatenate(
        name=(
            "event_plus_"
            "jet_observables"
        ),
    )(
        [
            event_representation,
            obs_input,
        ]
    )

    # ========================================================
    # 7. F NETWORK
    # ========================================================

    f_output = (
        feed_forward_block(
            x=f_input,

            sizes=config[
                "F_sizes"
            ],

            activation=activation,

            dropout_rate=(
                F_dropouts
            ),

            name_prefix="F",
        )
    )

    # ========================================================
    # 8. OUTPUT
    # ========================================================

    output = Dense(
        config[
            "output_dim"
        ],
        activation="softmax",
        name="output",
    )(
        f_output
    )

    # ========================================================
    # Complete model
    # ========================================================

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
        loss=(
            "categorical_crossentropy"
        ),

        optimizer=Adam(
            learning_rate=(
                config[
                    "learning_rate"
                ]
            )
        ),

        metrics=[
            "accuracy"
        ],
    )

    return model


# ============================================================
# Model summary fields
# ============================================================

def get_model_summary_fields(
    config: dict,
) -> dict:

    return {
        "model_name":
            MODEL_NAME,

        # ----------------------------------------------------
        # Attention positions
        # ----------------------------------------------------

        "attention_before_phi":
            USE_ATTENTION_BEFORE_PHI,

        "attention_before_sum":
            USE_ATTENTION_BEFORE_SUM,

        "attention_after_sum":
            USE_ATTENTION_AFTER_SUM,

        # ----------------------------------------------------
        # AEFN architecture
        # ----------------------------------------------------

        "input_dim":
            config[
                "input_dim"
            ],

        "Phi_sizes":
            str(
                config[
                    "Phi_sizes"
                ]
            ),

        "F_sizes":
            str(
                config[
                    "F_sizes"
                ]
            ),

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
            config[
                "output_dim"
            ],

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Observables
        # ----------------------------------------------------

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
