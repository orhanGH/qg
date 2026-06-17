import numpy as np
import energyflow as ef

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tf_keras.models import Model
from tf_keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate, Dropout
from tf_keras.optimizers import Adam
from tf_keras import backend as K


def get_default_config() -> dict:
    return {
        "model_name": "oefn",
        "results_dir_name": "oefn_results",

        # Used only in fallback mode when no Marvin obsvs/x is passed.
        "efp_degree": 3,

        # Marvin obsvs/x has 3472 features.
        # We scale on train fold only and then apply PCA.
        "n_pca_components": 13,

        # oEFN architecture
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),
        "activation": "relu",

        # Observable branch projection
        "obs_latent_dim": 64,
        "obs_dropout": 0.1,

        # Dropout
        "phi_dropout": 0.1,
        "latent_dropout": 0.1,
        "F_dropout": 0.1,

        # Training
        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "use_early_stopping": True,
    }


def compute_efp_observables(X: np.ndarray, config: dict) -> np.ndarray:
    """
    Fallback only.

    This is kept for old experiments where X is passed as a particle array.
    For the Marvin dataset, do not use this. Marvin observables are already
    stored in obsvs/x and are passed through X["obsvs"].
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


def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
    """
    Prepare oEFN fold inputs.

    Marvin mode:
        X is a dict:
            X["parts"] -> shape (N_jets, max_particles, 3)
            X["obsvs"] -> shape (N_jets, 3472)

        X["parts"][..., 0] = z
        X["parts"][..., 1] = delta_eta
        X["parts"][..., 2] = delta_phi

        X["obsvs"] = concat(nsubs, eecs, efps)

    Fallback mode:
        X is a normal particle array.
        Then EFP observables are computed on the fly.
    """

    if isinstance(X, dict):
        if "parts" not in X:
            raise KeyError("oEFN expected X['parts'], but key 'parts' is missing.")

        if "obsvs" not in X:
            raise KeyError("oEFN expected X['obsvs'], but key 'obsvs' is missing.")

        X_parts = X["parts"]
        X_obs = X["obsvs"]

        print("Using precomputed Marvin observables from obsvs/x for oEFN.")
        print("Particle tensor shape:", X_parts.shape)
        print("Observable matrix shape:", X_obs.shape)

    else:
        X_parts = X

        if "X_obs" not in context:
            print("Computing EFP observables for oEFN fallback mode...")
            context["X_obs"] = compute_efp_observables(X_parts, config)
            print("Observable matrix shape:", context["X_obs"].shape)

        X_obs = context["X_obs"]

    if X_parts.ndim != 3:
        raise ValueError(
            f"Expected X_parts to have shape (N, max_particles, features), "
            f"got shape {X_parts.shape}."
        )

    if X_parts.shape[-1] < 3:
        raise ValueError(
            f"Expected X_parts last dimension to contain at least "
            f"[z, delta_eta, delta_phi], got shape {X_parts.shape}."
        )

    if X_obs.ndim != 2:
        raise ValueError(
            f"Expected X_obs to have shape (N, num_observables), "
            f"got shape {X_obs.shape}."
        )

    if X_parts.shape[0] != X_obs.shape[0]:
        raise ValueError(
            f"X_parts and X_obs have different number of jets: "
            f"{X_parts.shape[0]} vs {X_obs.shape[0]}."
        )

    z_train = X_parts[train_idx, :, 0]
    p_train = X_parts[train_idx, :, 1:3]
    obs_train_raw = X_obs[train_idx]

    z_val = X_parts[val_idx, :, 0]
    p_val = X_parts[val_idx, :, 1:3]
    obs_val_raw = X_obs[val_idx]

    z_test = X_parts[test_idx, :, 0]
    p_test = X_parts[test_idx, :, 1:3]
    obs_test_raw = X_obs[test_idx]

    # Fit scaler only on the training fold.
    scaler = StandardScaler()
    obs_train_scaled = scaler.fit_transform(obs_train_raw)
    obs_val_scaled = scaler.transform(obs_val_raw)
    obs_test_scaled = scaler.transform(obs_test_raw)

    # Fit PCA only on the training fold.
    max_components = min(
        config["n_pca_components"],
        obs_train_scaled.shape[0],
        obs_train_scaled.shape[1],
    )

    pca = PCA(
        n_components=max_components,
        random_state=config["seed"],
    )

    obs_train = pca.fit_transform(obs_train_scaled)
    obs_val = pca.transform(obs_val_scaled)
    obs_test = pca.transform(obs_test_scaled)

    train_inputs = [z_train, p_train, obs_train]
    val_inputs = [z_val, p_val, obs_val]
    test_inputs = [z_test, p_test, obs_test]

    observable_source = "marvin_obsvs_x" if isinstance(X, dict) else "computed_efps"

    extra_info = {
        "num_particles": X_parts.shape[1],
        "raw_num_observables": X_obs.shape[1],
        "num_observables": obs_train.shape[1],
        "observable_source": observable_source,
        "efp_degree": config.get("efp_degree"),
        "n_pca_components": config["n_pca_components"],
        "obs_latent_dim": config.get("obs_latent_dim", 64),
    }

    return train_inputs, val_inputs, test_inputs, extra_info


def build_model(config: dict, extra_info: dict | None = None):
    """
    oEFN architecture:

        particle branch:
            z, p -> Phi(p) -> sum_i z_i Phi(p_i) -> latent_summary

        observable branch:
            obsvs/x -> StandardScaler -> PCA -> Dense projection -> obs_latent

        classifier:
            concat(latent_summary, obs_latent) -> F network -> softmax
    """

    if extra_info is None:
        raise ValueError("oEFN build_model requires extra_info from prepare_fold_inputs.")

    num_particles = extra_info["num_particles"]
    num_observables = extra_info["num_observables"]

    activation = config.get("activation", "relu")

    input_z = Input(shape=(num_particles,), name="input_z")
    input_p = Input(shape=(num_particles, 2), name="input_p")
    input_obs = Input(shape=(num_observables,), name="input_obs")

    # -------------------------------------------------------------------------
    # Particle / EFN branch
    # -------------------------------------------------------------------------
    phi = input_p

    for i, units in enumerate(config["Phi_sizes"]):
        phi = TimeDistributed(
            Dense(units, activation=activation),
            name=f"phi_dense_{i + 1}",
        )(phi)

        phi = Dropout(
            config.get("phi_dropout", 0.0),
            name=f"phi_dropout_{i + 1}",
        )(phi)

    z_expanded = Lambda(
        lambda x: K.expand_dims(x, axis=-1),
        name="expand_z",
    )(input_z)

    weighted_phi = Lambda(
        lambda tensors: tensors[0] * tensors[1],
        name="energy_weighted_phi",
    )([z_expanded, phi])

    latent_summary = Lambda(
        lambda x: K.sum(x, axis=1),
        name="latent_summary",
    )(weighted_phi)

    latent_summary = Dropout(
        config.get("latent_dropout", 0.0),
        name="latent_dropout",
    )(latent_summary)

    # -------------------------------------------------------------------------
    # Observable branch
    # -------------------------------------------------------------------------
    obs_latent = Dense(
        config.get("obs_latent_dim", 64),
        activation=activation,
        name="observable_projection",
    )(input_obs)

    obs_latent = Dropout(
        config.get("obs_dropout", 0.0),
        name="observable_dropout",
    )(obs_latent)

    # -------------------------------------------------------------------------
    # Merge branches
    # -------------------------------------------------------------------------
    x = Concatenate(name="latent_plus_observables")(
        [latent_summary, obs_latent]
    )

    # -------------------------------------------------------------------------
    # Classifier F
    # -------------------------------------------------------------------------
    for i, units in enumerate(config["F_sizes"]):
        x = Dense(
            units,
            activation=activation,
            name=f"F_dense_{i + 1}",
        )(x)

        x = Dropout(
            config.get("F_dropout", 0.0),
            name=f"F_dropout_{i + 1}",
        )(x)

    output = Dense(2, activation="softmax", name="output")(x)

    model = Model(
        inputs=[input_z, input_p, input_obs],
        outputs=output,
        name="oefn",
    )

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def get_model_summary_fields(config: dict) -> dict:
    return {
        "efp_degree": config.get("efp_degree"),
        "n_pca_components": config["n_pca_components"],
        "obs_latent_dim": config.get("obs_latent_dim", 64),
        "obs_dropout": config.get("obs_dropout", 0.0),
        "Phi_sizes": str(config["Phi_sizes"]),
        "F_sizes": str(config["F_sizes"]),
        "activation": config.get("activation", "relu"),
        "phi_dropout": config.get("phi_dropout", 0.0),
        "latent_dropout": config.get("latent_dropout", 0.0),
        "F_dropout": config.get("F_dropout", 0.0),
    }
