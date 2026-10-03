import torch.nn as nn


class LanguageModelLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, logits, targets):

        batch_size = logits.size(0)
        sequence_length = logits.size(1)
        vocab_size = logits.size(2)

        logits = logits.view(
            batch_size * sequence_length,
            vocab_size
        )

        targets = targets.view(
            batch_size * sequence_length
        )

        loss = self.loss_function(
            logits,
            targets
        )

        return loss