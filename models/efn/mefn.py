from math import comb

import tensorflow as tf

from tf_keras.models import Model
from tf_keras.layers import Input, Dense, TimeDistributed, Layer, Dropout
from tf_keras.optimizers import Adam

try:
    from tf_keras.optimizers import AdamW
except Exception:
    AdamW = None


def get_default_config() -> dict:
    return {
        "model_name": "mefn",
        "results_dir_name": "mefn_results",

        # Input:
        # z_i = momentum fraction
        # p_i = (Delta eta_i, Delta phi_i)
        "input_dim": 2,
        "output_dim": 2,

        # MEFN-specific
        "latent_dim": 16,
        "moment_order": 3,

        # Networks
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),
        "activation": "gelu",

        # Dropout
        "phi_dropout": 0.1,
        "moment_dropout": 0.1,
        "F_dropout": 0.1,

        # Training
        "batch_size": 512,
        "epochs": 200,
        "patience": 30,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "clipnorm": 1.0,
        "use_early_stopping": True,
        "early_stopping_threshold": 1e-4,
    }


def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """
    MEFN input:

    z: (batch, max_particles)
    p: (batch, max_particles, 2)

    Shared X convention:
    X[..., 0] = z = pT_i / sum_j pT_j
    X[..., 1] = delta_eta
    X[..., 2] = delta_phi
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


class RecursiveMomentPooling(Layer):
    """
    Recursive Moment Pooling for MEFN.

    Computes moments up to order K:

        M_{a1...ak} =
            sum_i z_i * Phi_{a1}(p_i) * ... * Phi_{ak}(p_i)

    The mathematical formula is direct.
    This implementation computes the products recursively:
    order k is built from order k-1.
    """

    def __init__(self, latent_dim: int, moment_order: int, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.moment_order = int(moment_order)

    def call(self, inputs):
        phi, z = inputs

        # z: (batch, particles) -> (batch, particles, 1)
        if len(z.shape) == 2:
            z = tf.expand_dims(z, axis=-1)

        moment_features = []

        # ---------------------------------------------------
        # Order 1:
        # M_a = sum_i z_i * Phi_a(p_i)
        # ---------------------------------------------------
        previous_terms = phi
        previous_combos = [(a,) for a in range(self.latent_dim)]

        moment_1 = tf.reduce_sum(previous_terms * z, axis=1)
        moment_features.append(moment_1)

        # ---------------------------------------------------
        # Higher orders:
        # Build order k products from order k-1 products.
        # ---------------------------------------------------
        for order in range(2, self.moment_order + 1):
            current_terms_parts = []
            current_combos = []

            for combo_idx, combo in enumerate(previous_combos):
                # To avoid duplicate symmetric products, only append
                # channels >= last channel.
                start_channel = combo[-1]

                # previous product term:
                # shape = (batch, particles, 1)
                base = previous_terms[:, :, combo_idx:combo_idx + 1]

                # allowed next Phi channels:
                # shape = (batch, particles, latent_dim - start_channel)
                phi_tail = phi[:, :, start_channel:]

                # recursive product:
                # Phi_a1 * ... * Phi_a{k-1} * Phi_b
                new_terms = base * phi_tail
                current_terms_parts.append(new_terms)

                for new_channel in range(start_channel, self.latent_dim):
                    current_combos.append(combo + (new_channel,))

            # shape:
            # (batch, particles, number_of_order_k_combinations)
            current_terms = tf.concat(current_terms_parts, axis=-1)

            # M_{a1...ak} =
            # sum_i z_i * product_j Phi_aj(p_i)
            current_moment = tf.reduce_sum(current_terms * z, axis=1)
            moment_features.append(current_moment)

            previous_terms = current_terms
            previous_combos = current_combos

        return tf.concat(moment_features, axis=-1)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0][0]

        feature_dim = sum(
            comb(self.latent_dim + k - 1, k)
            for k in range(1, self.moment_order + 1)
        )

        return (batch_size, feature_dim)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "latent_dim": self.latent_dim,
                "moment_order": self.moment_order,
            }
        )
        return config


def build_optimizer(config: dict):
    learning_rate = config.get("learning_rate", 3e-4)
    weight_decay = config.get("weight_decay", 0.0)
    clipnorm = config.get("clipnorm", None)

    if weight_decay > 0:
        if AdamW is None:
            print(
                "WARNING: AdamW is not available. "
                "Falling back to Adam without decoupled weight decay."
            )
            return Adam(
                learning_rate=learning_rate,
                clipnorm=clipnorm,
            )

        print(f"Using AdamW with weight_decay={weight_decay}")

        return AdamW(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            clipnorm=clipnorm,
        )

    print("Using Adam without weight decay")

    return Adam(
        learning_rate=learning_rate,
        clipnorm=clipnorm,
    )


def build_model(config: dict, extra_info: dict | None = None):
    """
    Moment Energy Flow Network.

    Standard EFN:
        F(sum_i z_i Phi(p_i))

    MEFN:
        F(<Phi>, <Phi Phi>, ..., <Phi ... Phi>)

    where <...> is a z-weighted sum over particles.
    """

    num_particles = (
        extra_info["num_particles"]
        if extra_info is not None
        else config["max_particles"]
    )

    input_dim = config.get("input_dim", 2)
    output_dim = config.get("output_dim", 2)

    latent_dim = config.get("latent_dim", 16)
    moment_order = config.get("moment_order", 3)

    Phi_sizes = config.get("Phi_sizes", (100, 100, 128))
    F_sizes = config.get("F_sizes", (100, 100, 100))

    activation = config.get("activation", "gelu")

    phi_dropout = config.get("phi_dropout", 0.1)
    moment_dropout = config.get("moment_dropout", 0.1)

    # Support both naming styles.
    F_dropout = config.get("F_dropout", config.get("F_dropouts", 0.1))

    # -------------------------------------------------------
    # Inputs
    # -------------------------------------------------------
    input_z = Input(
        shape=(num_particles,),
        name="input_z",
    )

    input_p = Input(
        shape=(num_particles, input_dim),
        name="input_p",
    )

    # -------------------------------------------------------
    # Phi network
    # -------------------------------------------------------
    phi = input_p

    for i, units in enumerate(Phi_sizes):
        phi = TimeDistributed(
            Dense(
                units,
                activation=activation,
            ),
            name=f"phi_dense_{i + 1}",
        )(phi)

        if phi_dropout > 0:
            phi = Dropout(
                phi_dropout,
                name=f"phi_dropout_{i + 1}",
            )(phi)

    # -------------------------------------------------------
    # Project Phi to latent_dim channels
    # -------------------------------------------------------
    # Important:
    # activation=None keeps the final latent moment channels
    # less constrained before products are computed.
    phi = TimeDistributed(
        Dense(
            latent_dim,
            activation=None,
        ),
        name="phi_output",
    )(phi)

    if phi_dropout > 0:
        phi = Dropout(
            phi_dropout,
            name="phi_output_dropout",
        )(phi)

    # -------------------------------------------------------
    # Recursive Moment Pooling
    # -------------------------------------------------------
    x = RecursiveMomentPooling(
        latent_dim=latent_dim,
        moment_order=moment_order,
        name="recursive_moment_pooling",
    )([phi, input_z])

    if moment_dropout > 0:
        x = Dropout(
            moment_dropout,
            name="moment_dropout",
        )(x)

    # -------------------------------------------------------
    # F classifier network
    # -------------------------------------------------------
    for i, units in enumerate(F_sizes):
        x = Dense(
            units,
            activation=activation,
            name=f"F_dense_{i + 1}",
        )(x)

        if F_dropout > 0:
            x = Dropout(
                F_dropout,
                name=f"F_dropout_{i + 1}",
            )(x)

    output = Dense(
        output_dim,
        activation="softmax",
        name="output",
    )(x)

    model = Model(
        inputs=[input_z, input_p],
        outputs=output,
        name="mefn",
    )

    optimizer = build_optimizer(config)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary_fields(config: dict) -> dict:
    return {
        "model_name": config.get("model_name", "mefn"),
        "results_dir_name": config.get("results_dir_name", "mefn_results"),

        "input_dim": config.get("input_dim", 2),
        "output_dim": config.get("output_dim", 2),

        "latent_dim": config.get("latent_dim", 16),
        "moment_order": config.get("moment_order", 3),

        "Phi_sizes": str(config.get("Phi_sizes", (100, 100, 128))),
        "F_sizes": str(config.get("F_sizes", (100, 100, 100))),
        "activation": config.get("activation", "gelu"),

        "phi_dropout": config.get("phi_dropout", 0.1),
        "moment_dropout": config.get("moment_dropout", 0.1),
        "F_dropout": config.get("F_dropout", config.get("F_dropouts", 0.1)),

        "batch_size": config.get("batch_size", 512),
        "epochs": config.get("epochs", 200),
        "patience": config.get("patience", 30),
        "learning_rate": config.get("learning_rate", 3e-4),
        "weight_decay": config.get("weight_decay", 0.0),
        "clipnorm": config.get("clipnorm", None),
        "use_early_stopping": config.get("use_early_stopping", True),
        "early_stopping_threshold": config.get(
            "early_stopping_threshold",
            1e-4,
        ),
    }
