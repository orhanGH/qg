import torch
import torch.nn as nn

from transformers import MambaModel, MambaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

class HFMambaJetClassifier(nn.Module):
    """
    Hugging Face Mamba encoder adapted to qg_jets.

    Input:
        particles: (batch_size, max_particles, 3)

    Particle features:
        particles[..., 0] = z = normalized pT fraction
        particles[..., 1] = centered rapidity/y
        particles[..., 2] = centered phi
    """

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 2,
        state_size: int = 16,
        conv_kernel: int = 4,
        expand: int = 2,
        num_classes: int = 2,
        max_particles: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        mamba_config = MambaConfig(
            vocab_size=4,  # unused because we pass inputs_embeds
            hidden_size=hidden_dim,
            state_size=state_size,
            num_hidden_layers=num_layers,
            conv_kernel=conv_kernel,
            expand=expand,
            hidden_act="silu",
            pad_token_id=0,
            bos_token_id=0,
            eos_token_id=0,
            use_cache=False,
        )

        self.mamba = MambaModel(mamba_config)

        self.final_norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, particles=None, labels=None):
        valid_mask = (particles.abs().sum(dim=-1) > 0).long()

        inputs_embeds = self.input_projection(particles)

        outputs = self.mamba(
            inputs_embeds=inputs_embeds,
            attention_mask=valid_mask,
            use_cache=False,
        )

        encoded = outputs.last_hidden_state
        encoded = self.final_norm(encoded)

        # Safety: remove padded positions from representation
        encoded = encoded * valid_mask.unsqueeze(-1).float()

        # z-weighted pooling
        weights = particles[..., 0].clamp_min(0.0) * valid_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        jet_repr = (encoded * weights.unsqueeze(-1)).sum(dim=1)

        logits = self.classifier(jet_repr)

        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )
    
def get_default_config() -> dict:
    return {
        "model_name": "mamba",
        "results_dir_name": "mamba_hf_trainer_results",

        "hidden_dim": 64,
        "num_layers": 2,
        "state_size": 16,
        "conv_kernel": 4,
        "expand": 2,
        "dropout": 0.1,

        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
    }


def build_model(config: dict):
    return HFMambaJetClassifier(
        input_dim=3,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        state_size=config["state_size"],
        conv_kernel=config["conv_kernel"],
        expand=config["expand"],
        num_classes=2,
        max_particles=config["max_particles"],
        dropout=config["dropout"],
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "state_size": config["state_size"],
        "conv_kernel": config["conv_kernel"],
        "expand": config["expand"],
        "dropout": config["dropout"],
    }