"""Fine-tuned Wav2Vec2 / IndicWav2Vec neural architecture with contextual feature pooling."""

import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Model


class Wav2VecDetector(nn.Module):
    """End-to-end speech deepfake detector leveraging self-supervised Wav2Vec2 embeddings."""

    def __init__(
        self,
        pretrained_model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        hidden_dim: int = 128,
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.pretrained_model_name = pretrained_model_name

        # Load Wav2Vec2 backbone
        try:
            self.wav2vec = Wav2Vec2Model.from_pretrained(pretrained_model_name)
        except Exception:
            # Fallback for offline / isolated environments: initialize from config
            config = AutoConfig.from_pretrained(
                pretrained_model_name,
                conv_dim=[512, 512, 512, 512, 512, 512, 512],
                hidden_size=768,
            )
            self.wav2vec = Wav2Vec2Model(config)

        # Freeze the 7-layer temporal convolutional encoder
        if freeze_feature_extractor:
            self.wav2vec.feature_extractor._freeze_parameters()

        # Contextual pooling and forensic classification head
        embedding_dim = self.wav2vec.config.hidden_size  # Typically 768
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        return_probs: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Input:
            waveform: Raw 1D audio tensor of shape (Batch, num_samples), e.g. (B, 48000)
        Output:
            logits: Tensor of shape (Batch,) or probabilities if return_probs=True
        """
        # Ensure 2D tensor (Batch, num_samples)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 3:
            waveform = waveform.squeeze(1)

        # Extract contextual transformer representations
        outputs = self.wav2vec(waveform)
        hidden_states = outputs.last_hidden_state  # (Batch, time_frames, 768)

        # Temporal Average Pooling across speech frames
        pooled = torch.mean(hidden_states, dim=1)  # (Batch, 768)

        # Project through classification head
        logits = self.classifier(pooled).squeeze(-1)

        if return_probs:
            return torch.sigmoid(logits)
        return logits
