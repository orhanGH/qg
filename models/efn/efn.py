from energyflow.archs import EFN
from tf_keras.optimizers import Adam

def get_default_config() -> dict:
    return {
        "model_name": "efn",
        "results_dir_name": "efn_results",
        "input_dim": 2,
        "Phi_sizes": (100, 100, 128),
        "F_sizes": (100, 100, 100),
        "output_dim": 2,
        "latent_dropout": 0.1,
        "F_dropouts": 0.1,
        "activation": "relu",
        "batch_size": 500,
        "epochs": 50,
        "patience": 2,
        "learning_rate": 1e-3,
        "use_early_stopping": True,
    }


def prepare_fold_inputs(X, train_idx, val_idx, test_idx, config, fold_dir, context):
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
    model = EFN(
        input_dim=config["input_dim"],
        Phi_sizes=config["Phi_sizes"],
        F_sizes=config["F_sizes"],
        Phi_acts=config.get("activation", "relu"),
        F_acts=config.get("activation", "relu"),
        output_dim=config["output_dim"],
        latent_dropout=config.get("latent_dropout", 0.0),
        F_dropouts=config.get("F_dropouts", 0.0),
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=config["learning_rate"]),
        metrics=["accuracy"],
        summary=False,
    )

    return model

def get_model_summary_fields(config: dict) -> dict:
    return {
        "input_dim": config["input_dim"],
        "Phi_sizes": str(config["Phi_sizes"]),
        "F_sizes": str(config["F_sizes"]),
        "activation": config.get("activation", "relu"),
        "latent_dropout": config.get("latent_dropout", 0.0),
        "F_dropouts": config.get("F_dropouts", 0.0),
        "output_dim": config["output_dim"],
    }
