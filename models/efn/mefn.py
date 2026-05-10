from itertools import combinations_with_replacement
import tensorflow as tf 
from tf_keras.models import Model
from tf_keras.layers import Input, Dense, TimeDistributed, Layer
from tf_keras.optimizers import Adam
from tf_keras import backend as K

from math import comb



def get_default_config() -> dict:
    return {
        "model_name": "mefn",
        "results_dir_name": "mefn_results",

        "input_dim": 2,
        "output_dim": 2,

        # MEFN-specific
        "latent_dim": 16,
        "moment_order": 3,

        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),

        # These can be overwritten by shared_config in main.py
        "batch_size": 256,
        "epochs": 5,
        "patience": 2,
        "learning_rate": 3e-4,
        "use_early_stopping": False,
    }


def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """
    MEFN input:
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

class RecursiveMomentPooling(Layer):
    """
    Recursive Moment Pooling for MEFN.

    Computes, for orders 1,...,K:

        M_{a1...ak} = sum_i z_i * Phi_{a1}(p_i) * ... * Phi_{ak}(p_i)

    using combinations with replacement a1 <= ... <= ak.

    Input:
        phi: shape (batch, particles, latent_dim)
        z:   shape (batch, particles) or (batch, particles, 1)

    Output:
        concatenated moment features:
        shape (batch, sum_{k=1}^{K} C(latent_dim + k - 1, k))
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

        # Order 1 per-particle monomials:
        # terms_a(i) = Phi_a(p_i)
        previous_terms = phi
        previous_combos = [(a,) for a in range(self.latent_dim)]

        # M_a = sum_i z_i * Phi_a(p_i)
        moment_1 = tf.reduce_sum(previous_terms * z, axis=1)
        moment_features.append(moment_1)

        # Higher orders:
        # Build order k terms recursively from order k-1 terms.
        for order in range(2, self.moment_order + 1):
            current_terms_parts = []
            current_combos = []

            for combo_idx, combo in enumerate(previous_combos):
                # To avoid duplicates, only append channels >= last channel.
                # Example: from (2, 5), append 5,6,...,L-1.
                start_channel = combo[-1]

                # previous_terms[:, :, combo_idx] has shape (batch, particles)
                base = previous_terms[:, :, combo_idx:combo_idx + 1]

                # phi_tail shape: (batch, particles, latent_dim - start_channel)
                phi_tail = phi[:, :, start_channel:]

                # Recursive products:
                # Phi_{a1}...Phi_{ak-1} * Phi_b
                # for b >= a_{k-1}
                new_terms = base * phi_tail

                current_terms_parts.append(new_terms)

                for new_channel in range(start_channel, self.latent_dim):
                    current_combos.append(combo + (new_channel,))

            # Shape: (batch, particles, number_of_order_k_combinations)
            current_terms = tf.concat(current_terms_parts, axis=-1)

            # M_{a1...ak} = sum_i z_i * product_j Phi_aj(p_i)
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

def build_model(config: dict, extra_info: dict | None = None):
    """
    Recursive channel-style MEFN.

    Formula:
        M_{a1...ak} = sum_i z_i * Phi_{a1}(p_i) * ... * Phi_{ak}(p_i)

    The formula is the same as before, but the moment products are built
    recursively to avoid recomputing each product from scratch.
    """

    num_particles = (
        extra_info["num_particles"]
        if extra_info is not None
        else config["max_particles"]
    )

    latent_dim = config["latent_dim"]
    moment_order = config["moment_order"]
    Phi_sizes = config["Phi_sizes"]
    F_sizes = config["F_sizes"]
    output_dim = config.get("output_dim", 2)

    input_z = Input(shape=(num_particles,), name="input_z")
    input_p = Input(shape=(num_particles, 2), name="input_p")

    # Phi network on particle coordinates p = (y, phi)
    # Output shape: (batch, num_particles, latent_dim)
    phi = input_p

    for i, units in enumerate(Phi_sizes):
        phi = TimeDistributed(
            Dense(units, activation="relu"),
            name=f"phi_dense_{i + 1}",
        )(phi)

    phi = TimeDistributed(
        Dense(latent_dim, activation="relu"),
        name="phi_output",
    )(phi)

    # Recursive Moment Pooling
    x = RecursiveMomentPooling(
        latent_dim=latent_dim,
        moment_order=moment_order,
        name="recursive_moment_pooling",
    )([phi, input_z])

    # F classifier network
    for i, units in enumerate(F_sizes):
        x = Dense(units, activation="relu", name=f"F_dense_{i + 1}")(x)

    output = Dense(output_dim, activation="softmax", name="output")(x)

    model = Model(inputs=[input_z, input_p], outputs=output)

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary_fields(config: dict) -> dict:
    return {
        "latent_dim": config["latent_dim"],
        "moment_order": config["moment_order"],
        "Phi_sizes": str(config["Phi_sizes"]),
        "F_sizes": str(config["F_sizes"]),
    }
