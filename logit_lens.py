import torch
from transformer_lens import HookedTransformer
from IPython.display import display, Markdown, HTML
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from tqdm import tqdm


class Lens:
    """
    Represents a logit lens (or tuned lens) that inspects top tokens by log odds at each layer for a specific query.
    """

    def __init__(
        self,
        model: HookedTransformer = None,
        query: str = "The sky is blue and the grass is",
        translator = None,
        top_k: int = 5,
    ):
        """
        Return Lens object.
        Optionally takes:
        model:       HookedTransformer pretrained model. If blank, uses default pythia-14m.
        query:       Query to generate tokens from model with. If blank, uses sample.
        translator:  A trained Translator object. If provided, uses tuned lens logic.
        top_k:       The top k tokens by logit at each layer. Default is 5.
        """
        self.model = model if model is not None else Lens.get_model()
        self.query = query
        self.translator = translator
        self.top_k = top_k

        self.lens, self.cache = self.model.run_with_cache(self.query)
        self.n_layers = self.model.cfg.n_layers

        print("SUCCESS: Lens object created.")

    # helpers

    def get_layer_data(self) -> tuple[list, list]:
        """
        Shared data-extraction logic for both logit lens and tuned lens.
        Returns (layer_data, all_logits).
        layer_data : list[list[tuple[str, float]]]  — [layer][k] = (token, logit)
        all_logits : flat list of every logit value (used for colour normalisation)
        """
        layer_data = []
        all_logits = []
        for layer in range(self.n_layers):
            resid = self.cache[f"blocks.{layer}.hook_resid_post"]

            if self.translator is not None:
                resid = self.translator.translate(resid, layer)

            scaled       = self.model.ln_final(resid)
            layer_logits = self.model.unembed(scaled)
            last_logits  = layer_logits[0, -1, :]

            top_values, top_indices = torch.topk(last_logits, self.top_k)
            top_tokens = [self.model.tokenizer.decode([idx]) for idx in top_indices]
            pairs = list(zip(top_tokens, top_values.tolist()))

            layer_data.append(pairs)
            all_logits.extend(v for _, v in pairs)

        return layer_data, all_logits

    # methods of printing

    def __str__(self):
        layer_data, _ = self.get_layer_data()
        lines = []

        for layer, pairs in enumerate(layer_data):
            lines.append(f"Layer {layer}:")
            for token, value in pairs:
                lines.append(f"  {repr(token):<15}  {value:.1f}")
            lines.append("")

        return "\n".join(lines)

    def to_table(self, html: bool = False):
        """
        Renders the top-k tokens per layer as a table.
        html=False  plain IPython Markdown table
        html=True   heatmap-coloured HTML table
        """
        layer_data, all_logits = self.get_layer_data()

        if not html:
            header = "| Rank | " + " | ".join(f"Layer {l}" for l in range(self.n_layers)) + " |"
            sep    = "| --- |" + "|".join(" --- " for _ in range(self.n_layers)) + "|"
            rows   = []
            for k in range(self.top_k):
                cells = [f"**{k+1}**"]
                for layer in range(self.n_layers):
                    token, logit = layer_data[layer][k]
                    cells.append(f"`{repr(token)}` {logit:.1f}")
                rows.append("| " + " | ".join(cells) + " |")
            display(Markdown("\n".join([header, sep] + rows)))

        else:
            min_l, max_l = min(all_logits), max(all_logits)

            def logit_to_color(val):
                t   = (val - min_l) / (max_l - min_l) if max_l > min_l else 0.5
                r   = int(70  + t * (178 - 70))
                g   = int(130 + t * (34  - 130))
                b   = int(180 + t * (34  - 180))
                lum = 0.299*r + 0.587*g + 0.114*b
                return f"rgb({r},{g},{b})", ("#000" if lum > 140 else "#fff")

            header_cells = "<th style='padding:6px 10px;'>Rank</th>" + "".join(
                f"<th style='padding:6px 10px;'>Layer {l}</th>" for l in range(self.n_layers)
            )
            rows = [f"<tr>{header_cells}</tr>"]

            for k in range(self.top_k):
                cells = f"<td style='padding:6px 10px; font-weight:bold; text-align:center;'>{k+1}</td>"
                for layer in range(self.n_layers):
                    token, logit = layer_data[layer][k]
                    bg, fg = logit_to_color(logit)
                    cells += (
                        f"<td style='background:{bg}; color:{fg}; padding:6px 10px;"
                        f" text-align:center; font-family:monospace; white-space:nowrap;'>"
                        f"{repr(token)}<br>"
                        f"<span style='font-size:0.8em; opacity:0.85;'>{logit:.1f}</span></td>"
                    )
                rows.append(f"<tr>{cells}</tr>")

            display(HTML(
                "<div style='overflow-x:auto;'>"
                "<table style='border-collapse:collapse; font-size:0.9em;'>"
                + "".join(rows)
                + "</table></div>"
            ))

    # static methods

    @staticmethod
    def get_model(model_name: str = "EleutherAI/pythia-14m") -> HookedTransformer:
        """
        Utility function for retrieving a pretrained model.
        Optionally takes model name.
        Returns the model of HookedTransformer type.
        """
        return HookedTransformer.from_pretrained(model_name)


class Translator:
    """
    A trained affine translator for use with a tuned lens.
    Maps each layer's residual stream into the final layer's residual space,
    allowing the logit lens to make better predictions at intermediate layers.

    Train once against a model, then pass to any number of Lens objects.
    """

    class AffineMap(nn.Module):
        """Affine map: resid_l to resid_final space. Initialised as identity + zero bias."""
        def __init__(self, d_model: int):
            super().__init__()
            self.W = nn.Parameter(torch.eye(d_model))
            self.b = nn.Parameter(torch.zeros(d_model))

        def forward(self, resid: torch.Tensor) -> torch.Tensor:
            return resid @ self.W.T + self.b

    def __init__(self, model: HookedTransformer):
        """
        Initialise a Translator for a given model.
        Call .train() to learn the affine maps, or .load() to restore saved weights.
        """
        self.model = model
        self.n_layers = model.cfg.n_layers
        self.d_model = model.cfg.d_model
        self.maps = None # nn.ModuleList, populated by .train() or .load()

        print("Translator initialised. Call .train() or .load() before use.")

    def translate(self, resid: torch.Tensor, layer: int) -> torch.Tensor:
        """
        Apply the trained affine map for the given layer.
        Raises RuntimeError if called before training or loading.
        """
        if self.maps is None:
            raise RuntimeError("Translator has no weights. Call .train() or .load() first.")
        device = next(self.model.parameters()).device
        return self.maps[layer](resid.to(device))

    def train(
        self,
        n_epochs:    int   = 1,
        batch_size:  int   = 4,
        lr:          float = 1e-3,
        max_seq_len: int   = 64,
        max_samples: int   = 1000,
        device = None,
        save_path:   str   = "translator.pt",
    ):
        """
        Train one affine map per transformer layer.

        Loss: KL( softmax(final_logits) ‖ softmax(translated_logits) ) per layer,
        encouraging each map to predict the same distribution as the full model.
        Saves weights to save_path on completion.
        """
        if device is None:
            device = next(self.model.parameters()).device

        self.maps = nn.ModuleList(
            [Translator.AffineMap(self.d_model).to(device) for _ in range(self.n_layers)]
        )
        optimizer = optim.Adam(self.maps.parameters(), lr=lr)

        ds    = load_dataset("NeelNanda/pile-10k", split="train")
        texts = [row["text"] for row in ds][:max_samples]

        def tokenize_batch(batch_texts):
            return self.model.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )["input_ids"].to(device)

        self.model.eval()
        self.maps.train()
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=False)

        for epoch in range(n_epochs):
            total_loss, n_batches = 0.0, 0

            for start in tqdm(range(0, len(texts), batch_size),
                              desc=f"Epoch {epoch+1}/{n_epochs}"):
                tokens = tokenize_batch(texts[start : start + batch_size])

                with torch.no_grad():
                    _, cache = self.model.run_with_cache(tokens)
                    final_resid  = cache[f"blocks.{self.n_layers-1}.hook_resid_post"]
                    final_logits = self.model.unembed(self.model.ln_final(final_resid))
                    target_probs = torch.softmax(final_logits, dim=-1)

                optimizer.zero_grad()
                batch_loss = 0.0

                for layer in range(self.n_layers):
                    resid = cache[f"blocks.{layer}.hook_resid_post"]
                    resid_mapped = self.maps[layer](resid)
                    layer_logits = self.model.unembed(self.model.ln_final(resid_mapped))
                    log_probs = torch.log_softmax(layer_logits, dim=-1)

                    B, T, V = log_probs.shape
                    batch_loss = batch_loss + kl_loss(
                        log_probs.reshape(B * T, V),
                        target_probs.reshape(B * T, V),
                    )

                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches  += 1

            print(f"Epoch {epoch+1} — avg loss: {total_loss / max(n_batches, 1):.4f}")

        self.save(save_path)
        print("SUCCESS: Translator trained and ready.")

    def save(self, path: str = "translator.pt"):
        """Save all affine map weights to a .pt file."""
        if self.maps is None:
            raise RuntimeError("Nothing to save — Translator has not been trained.")
        torch.save(
            {f"layer_{i}": self.maps[i].state_dict() for i in range(self.n_layers)},
            path,
        )
        print(f"Translator saved to {path}")

    def load(self, path: str = "translator.pt"):
        """
        Load affine map weights from a .pt file.
        Initialises self.maps if not already done.
        """
        device      = next(self.model.parameters()).device
        state_dicts = torch.load(path, map_location=device)

        self.maps = nn.ModuleList(
            [Translator.AffineMap(self.d_model).to(device) for _ in range(self.n_layers)]
        )
        for i, m in enumerate(self.maps):
            m.load_state_dict(state_dicts[f"layer_{i}"])
            m.eval()

        print(f"Translator loaded from {path}")

    @staticmethod
    def from_file(path: str, model: HookedTransformer) -> "Translator":
        """
        Convenience constructor. Equivalent to Translator(model).load(path).
        Returns a ready-to-use Translator.
        """
        t = Translator(model)
        t.load(path)
        return t


if __name__ == "__main__":
    model = Lens.get_model()

    # plain logit lens
    lens = Lens(model=model)
    print(lens)

    # tuned lens: train once, reuse on many queries
    translator = Translator(model)
    translator.train(save_path="translator.pt")
    # or load an existing one:
    # translator = Translator.from_file("translator.pt", model)
    lens2 = Lens(model=model, query="The sky is blue and the grass is", translator=translator)
    # lens3 = Lens(model=model, query="The capital of France is", translator=translator)
    print(lens2)