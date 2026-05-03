from itertools import combinations_with_replacement

from tf_keras.models import Model
from tf_keras.layers import Input, Dense, TimeDistributed, Lambda
from tf_keras.optimizers import Adam
from tf_keras import backend as K


def get_default_config() -> dict:
    return {
        "model_name": "mefn",
        "results_dir_name": "mefn_results",

        "input_dim": 2,
        "output_dim": 2,

        # MEFN-specific
        # Keep these small first, because moment features grow very fast.
        "latent_dim": 8,
        "moment_order": 2,

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

def build_model(config: dict, extra_info: dict | None = None):
    """
    Channel-style MEFN:
        1. Phi maps particle coordinates p=(y, phi) to latent channels.
        2. Moments are built from Phi channels.
        3. Each moment is weighted once by z.
        4. The concatenated moment features are classified by F.

    Moment definition:
        M_{a1...ak} = sum_i z_i * Phi_{a1}(p_i) * ... * Phi_{ak}(p_i)
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

    # Phi network on particle coordinates
    # output shape: (batch, num_particles, latent_dim)
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

    # Expand z from (batch, particles) to (batch, particles, 1)
    z_expanded = Lambda(
        lambda x: K.expand_dims(x, axis=-1),
        name="expand_z",
    )(input_z)

    # z-weighted latent particle channels
    # Used for the first moment:
    # M_a = sum_i z_i * Phi_a(p_i)
    weighted_phi = Lambda(
        lambda tensors: tensors[0] * tensors[1],
        name="weighted_phi",
    )([z_expanded, phi])

    pooled_features = []

    # First-order channel moments:
    # M_a = sum_i z_i * Phi_a(p_i)
    first_moment = Lambda(
        lambda x: K.sum(x, axis=1),
        name="moment_1",
    )(weighted_phi)

    pooled_features.append(first_moment)

    # Higher-order channel moments:
    # M_{a1...ak} = sum_i z_i * Phi_{a1}(p_i) * ... * Phi_{ak}(p_i)
    if moment_order >= 2:
        for order in range(2, moment_order + 1):
            channel_combos = list(
                combinations_with_replacement(range(latent_dim), order)
            )

            def make_channel_moment_layer(combos_for_order):
                def channel_moment_fn(tensors):
                    phi_tensor, z_tensor = tensors
                    terms = []

                    for combo in combos_for_order:
                        # Start with Phi_{a1}(p_i)
                        term = phi_tensor[:, :, combo[0]]

                        # Multiply by Phi_{a2}(p_i), ..., Phi_{ak}(p_i)
                        for ch in combo[1:]:
                            term = term * phi_tensor[:, :, ch]

                        # Weight once by z_i
                        term = term * z_tensor[:, :, 0]

                        # Sum over particles i
                        term = K.sum(term, axis=1, keepdims=True)

                        terms.append(term)

                    return K.concatenate(terms, axis=1)

                return channel_moment_fn

            order_moment = Lambda(
                make_channel_moment_layer(channel_combos),
                name=f"moment_{order}",
            )([phi, z_expanded])

            pooled_features.append(order_moment)

    # Concatenate all moment orders
    if len(pooled_features) == 1:
        x = pooled_features[0]
    else:
        x = Lambda(
            lambda tensors: K.concatenate(tensors, axis=1),
            name="moment_concat",
        )(pooled_features)

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
