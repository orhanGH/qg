import torch
import torch.nn as nn

from transformers import RobertaModel, RobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput


# Hugging Face RoBERTa-based jet classifier
class HFRobertaJetClassifier(nn.Module):
    """
    Hugging Face RoBERTa encoder adapted to qg_jets.

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
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        num_classes: int = 2,
        max_particles: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()

        ff_dim = hidden_dim * 4

        self.input_projection = nn.Linear(input_dim, hidden_dim)

        roberta_config = RobertaConfig(
            vocab_size=4,  # unused because we pass inputs_embeds
            hidden_size=hidden_dim,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=ff_dim,

            # Important for RoBERTa:
            # RoBERTa position ids are shifted because pad_token_id=1.
            # Therefore max_particles alone is too small.
            max_position_embeddings=max_particles + 2,

            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            type_vocab_size=1,
            num_labels=num_classes,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        self.roberta = RobertaModel(
            roberta_config,
            add_pooling_layer=False,
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, particles=None, labels=None):
        attention_mask = (particles.abs().sum(dim=-1) > 0).long()

        inputs_embeds = self.input_projection(particles)

        token_type_ids = torch.zeros_like(attention_mask)

        outputs = self.roberta(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        encoded = outputs.last_hidden_state

        valid_mask = attention_mask.float()
        weights = particles[..., 0].clamp_min(0.0) * valid_mask
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
        "model_name": "roberta",
        "results_dir_name": "roberta_hf_trainer_results",

        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,

        "batch_size": 256,
        "epochs": 10,
        "learning_rate": 3e-4,
        "weight_decay": 1e-5,
    }


def build_model(config: dict):
    return HFRobertaJetClassifier(
        input_dim=3,
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        num_classes=2,
        max_particles=config["max_particles"],
        dropout=config["dropout"],
    )


def get_model_summary_fields(config: dict) -> dict:
    return {
        "hidden_dim": config["hidden_dim"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "dropout": config["dropout"],
    }
