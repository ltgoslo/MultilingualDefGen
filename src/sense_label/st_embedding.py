import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

class STEmbedding:
    def __init__(self, model, bsize):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval().to(self.device)
        self.bsize = bsize

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.hidden_states[-1]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9,
        )

    def encode(self, representations, **kwargs):
        inputs = self.tokenizer(
            representations,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        embeddings = np.zeros((len(representations), self.model.config.hidden_size))
        token_dataset = torch.utils.data.TensorDataset(
            inputs.input_ids,
            inputs.attention_mask,
            torch.tensor(np.arange(len(representations))),
        )
        dataloader = torch.utils.data.DataLoader(
            token_dataset, batch_size=self.bsize, shuffle=False
        )
        for _inputs, att_masks, target_indices in dataloader:
            with torch.no_grad():
                model_output = self.model(
                    input_ids=_inputs.to(self.device),
                    attention_mask=att_masks.to(self.device),
                    output_hidden_states=True,
                )

            definition_embeddings = self._mean_pooling(
                model_output, att_masks.to(self.device)
            )
            definition_embeddings = torch.nn.functional.normalize(
                definition_embeddings, dim=1
            ).to("cpu")
            embeddings[
                    target_indices[0]: target_indices[-1] + 1, :
                ] = definition_embeddings
        return embeddings