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


def get_default_config() -> dict:
    return {
        "model_name": "aefn_before_F",
        "results_dir_name": "aefn_before_F_results",

        # EFN input:
        # z_i = momentum fraction
        # p_i = (Delta eta_i, Delta phi_i)
        "input_dim": 2,

        # Phi network
        "Phi_sizes": (100, 100, 128),

        # F network
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

        # Attention settings
        "attention_dim": 128,
        "num_heads": 4,
        "attention_dropout": 0.1,
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


def feed_forward_block(x, sizes, activation, dropout_rate, name_prefix):
    for i, size in enumerate(sizes):
        x = Dense(
            size,
            activation=activation,
            name=f"{name_prefix}_dense_{i+1}",
        )(x)

        if dropout_rate > 0:
            x = Dropout(
                dropout_rate,
                name=f"{name_prefix}_dropout_{i+1}",
            )(x)

    return x


def build_model(config: dict, extra_info: dict | None = None):
    activation = config.get("activation", "relu")
    num_heads = config.get("num_heads", 4)
    attention_dropout = config.get("attention_dropout", 0.0)

    latent_dropout = config.get("latent_dropout", 0.0)
    F_dropouts = config.get("F_dropouts", 0.0)

    # -------------------------------------------------------
    # Inputs
    # -------------------------------------------------------
    # z has shape: (batch_size, num_particles)
    z_input = Input(
        shape=(None,),
        name="z_input",
    )

    # p has shape: (batch_size, num_particles, 2)
    # where 2 = (Delta eta, Delta phi)
    p_input = Input(
        shape=(None, config["input_dim"]),
        name="p_input",
    )

    # -------------------------------------------------------
    # 1. Phi Feed-Forward Network
    # -------------------------------------------------------
    # Here there is no attention before Phi.
    # Phi processes raw particle coordinates independently.
    phi_output = feed_forward_block(
        x=p_input,
        sizes=config["Phi_sizes"],
        activation=activation,
        dropout_rate=latent_dropout,
        name_prefix="Phi",
    )

    # -------------------------------------------------------
    # 2. Attention before F
    # -------------------------------------------------------
    # Attention is applied in the latent particle space.
    # It lets Phi(p_i) representations interact before the EFN sum.
    latent_dim = config["Phi_sizes"][-1]

    attention_f = MultiHeadAttention(
        num_heads=num_heads,
        key_dim=latent_dim // num_heads,
        dropout=attention_dropout,
        name="attention_before_F",
    )(phi_output, phi_output)

    # Residual connection + normalization.
    phi_attention_output = Add(name="add_attention_before_F")(
        [phi_output, attention_f]
    )

    phi_attention_output = LayerNormalization(
        name="norm_attention_before_F"
    )(phi_attention_output)

    # -------------------------------------------------------
    # 3. EFN weighted sum
    # -------------------------------------------------------
    # EFN formula:
    # sum_i z_i * Phi_attention(p_i)
    z_expanded = Lambda(
        lambda z: tf.expand_dims(z, axis=-1),
        name="expand_z",
    )(z_input)

    weighted_particles = Lambda(
        lambda inputs: inputs[0] * inputs[1],
        name="z_times_particle_representation",
    )([z_expanded, phi_attention_output])

    event_representation = Lambda(
        lambda x: tf.reduce_sum(x, axis=1),
        name="efn_weighted_sum",
    )(weighted_particles)

    # -------------------------------------------------------
    # 4. F Feed-Forward Network
    # -------------------------------------------------------
    f_output = feed_forward_block(
        x=event_representation,
        sizes=config["F_sizes"],
        activation=activation,
        dropout_rate=F_dropouts,
        name_prefix="F",
    )

    # Final classification output.
    output = Dense(
        config["output_dim"],
        activation="softmax",
        name="output",
    )(f_output)

    model = Model(
        inputs=[z_input, p_input],
        outputs=output,
        name="AEFN_BEFORE_F",
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=config["learning_rate"]),
        metrics=["accuracy"],
    )

    return model


def get_model_summary_fields(config: dict) -> dict:
    return {
        "model_name": config.get("model_name", "aefn_before_F"),
        "input_dim": config["input_dim"],
        "Phi_sizes": str(config["Phi_sizes"]),
        "F_sizes": str(config["F_sizes"]),
        "activation": config.get("activation", "relu"),
        "latent_dropout": config.get("latent_dropout", 0.0),
        "F_dropouts": config.get("F_dropouts", 0.0),
        "output_dim": config["output_dim"],
        "attention_position": "before_F",
        "attention_dim": config.get("attention_dim", 128),
        "num_heads": config.get("num_heads", 4),
        "attention_dropout": config.get("attention_dropout", 0.0),
    }
