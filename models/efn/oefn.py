import numpy as np
import energyflow as ef

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from tf_keras.models import Model
from tf_keras.layers import Input, Dense, TimeDistributed, Lambda, Concatenate
from tf_keras.optimizers import Adam
from tf_keras import backend as K


def get_default_config() -> dict:
    return {
        "model_name": "oefn",
        "results_dir_name": "oefn_results",

        # EFP observable settings from the notebook
        "efp_degree": 3,
        "n_pca_components": 13,

        # oEFN architecture from the notebook
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),

        # Keras training defaults from the notebook
        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "use_early_stopping": True,
    }


def compute_efp_observables(X: np.ndarray, config: dict) -> np.ndarray:
    """
    Compute EFP observables as in the oEFN notebook.

    The notebook used:
        ef.EFPSet(
            f'd<={efp_degree}',
            measure='hadr',
            beta=1,
            kappa=1,
            normed=True,
            coords='ptyphim'
        )
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
    oEFN input:
        z:   (batch, max_particles)
        p:   (batch, max_particles, 2)
        obs: (batch, n_pca_components)

    Important:
        scaler and PCA are fitted only on the training fold.
    """

    if "X_obs" not in context:
        print("Computing EFP observables for oEFN...")
        context["X_obs"] = compute_efp_observables(X, config)
        print("Observable matrix shape:", context["X_obs"].shape)

    X_obs = context["X_obs"]

    z_train = X[train_idx, :, 0]
    p_train = X[train_idx, :, 1:3]
    obs_train_raw = X_obs[train_idx]

    z_val = X[val_idx, :, 0]
    p_val = X[val_idx, :, 1:3]
    obs_val_raw = X_obs[val_idx]

    z_test = X[test_idx, :, 0]
    p_test = X[test_idx, :, 1:3]
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

    extra_info = {
        "num_particles": X.shape[1],
        "raw_num_observables": X_obs.shape[1],
        "num_observables": obs_train.shape[1],
        "efp_degree": config["efp_degree"],
        "n_pca_components": config["n_pca_components"],
    }

    return train_inputs, val_inputs, test_inputs, extra_info


def build_model(config: dict, extra_info: dict | None = None):
    """
    Paper-style oEFN from the notebook:
        EFN branch on particles
        observable branch after latent sum pooling
        dense classifier F on concatenated features
    """

    num_particles = extra_info["num_particles"] if extra_info is not None else config["max_particles"]
    num_observables = extra_info["num_observables"]

    input_z = Input(shape=(num_particles,), name="input_z")
    input_p = Input(shape=(num_particles, 2), name="input_p")
    input_obs = Input(shape=(num_observables,), name="input_obs")

    phi = input_p

    for units in config["Phi_sizes"]:
        phi = TimeDistributed(Dense(units, activation="relu"))(phi)

    z_expanded = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_z)

    weighted_phi = Lambda(
        lambda tensors: tensors[0] * tensors[1],
    )([z_expanded, phi])

    latent_summary = Lambda(
        lambda x: K.sum(x, axis=1),
        name="latent_summary",
    )(weighted_phi)

    x = Concatenate(name="latent_plus_observables")(
        [latent_summary, input_obs]
    )

    for units in config["F_sizes"]:
        x = Dense(units, activation="relu")(x)

    output = Dense(2, activation="softmax", name="output")(x)

    model = Model(
        inputs=[input_z, input_p, input_obs],
        outputs=output,
    )

    model.compile(
        optimizer=Adam(learning_rate=config["learning_rate"]),
        loss="categorical_crossentropy",
    )

    return model


def get_model_summary_fields(config: dict) -> dict:
    return {
        "efp_degree": config["efp_degree"],
        "n_pca_components": config["n_pca_components"],
        "Phi_sizes": str(config["Phi_sizes"]),
        "F_sizes": str(config["F_sizes"]),
    }