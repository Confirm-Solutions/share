from dataclasses import dataclass

import pandas as pd
import torch

import epo as epo
import llama
from util import mellowmax


@dataclass
class Prompt:
    begin_ids: torch.Tensor
    prompt_ids: torch.Tensor
    force_ids: torch.Tensor
    full_ids: torch.Tensor

    def to(self, device):
        return Prompt(
            self.begin_ids.to(device),
            self.modify_indices.to(device),
            self.prompt_ids.to(device),
            self.shared_force_ids.to(device),
            self.force_ids.to(device),
            self.full_ids.to(device),
        )


def generate_prompts(tokenizer, prompt_template, prefix, suffix, force):
    # General tokenization principles:
    # - Tokenize left to right. Avoid tokenizing a subset of the prompt without its leftward prefix.
    # - Always split tokenization at the end of the prompt. If you tokenize the forcing together
    #   with the prompt, the result will be different than the concatenated prompt + forcing.
    #   Tokenizing the prompt alone matches the process during generation.

    marker = "<<<<"
    begin, end = prompt_template.format(instruction=marker).split(marker)
    prompt = begin + prefix + suffix + end
    assert prompt == prompt_template.format(instruction=prefix + suffix)

    begin_ids = tokenizer.encode(begin + prefix, return_tensors="pt")[0]
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt")[0]
    force_ids = tokenizer.encode(force, return_tensors="pt", add_special_tokens=False)[
        0
    ]
    full_ids = torch.cat([prompt_ids, force_ids])
    return Prompt(begin_ids, prompt_ids, force_ids, full_ids)


class PromptManager:
    def __init__(
        self, model, tokenizer, prompt_template="", prefix="", suffix="", batch_size=32
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.prompt_template = prompt_template
        self.prefix = prefix
        self.suffix = suffix
        self.batch_size = batch_size
        self.embeddings = self.model.get_input_embeddings()

        # Extract the constant prefix from the prompt and store KV cache:
        prompt_ids = tokenizer.encode(
            prompt_template.format(instruction=prefix + suffix),
            return_tensors="pt",
        )[0].to(model.device)

        # TODO: This is not correct! Need to match the full subsequence to be
        # robust.
        suffix_ids = tokenizer.encode(
            suffix, return_tensors="pt", add_special_tokens=False
        )[0].to(model.device)
        if suffix_ids[0] == 29871:
            suffix_ids = suffix_ids[1:]
        suffix_begin_idx = None
        for i in range(
            prompt_ids.shape[0] - 10 - suffix_ids.shape[0],
            prompt_ids.shape[0] + 1 - suffix_ids.shape[0],
        ):
            if (prompt_ids[i : i + suffix_ids.shape[0]] == suffix_ids).all():
                suffix_begin_idx = i
                break
        assert suffix_begin_idx is not None

        self.start_ids = prompt_ids[:suffix_begin_idx]
        self.start_str = tokenizer.decode(self.start_ids)
        self.end_ids = prompt_ids[suffix_begin_idx:]
        self.end_str = tokenizer.decode(self.end_ids)

        # Check prompt construction correctness.
        assert self.start_str.strip().endswith(prefix.strip())
        assert self.end_str.strip().startswith(suffix.strip())

        self.n_start = self.start_ids.shape[0]
        self.start_emb = self.embeddings(self.start_ids).detach()
        if self.start_ids.shape[0] > 0:
            # Repeat the KV cache entries so that they can be subsetted and
            # used with any batch size <= batch_size without copying.
            self.kv_cache = [
                (
                    k.detach().repeat_interleave(batch_size, dim=0),
                    v.detach().repeat_interleave(batch_size, dim=0),
                )
                for k, v in model(self.start_ids.unsqueeze(0)).past_key_values
            ]
        else:
            self.kv_cache = None

        self.n_end = self.end_ids.shape[0]
        self.end_emb = self.embeddings(self.end_ids).detach()

    def __call__(self, force_ids=None, input_ids=None, inputs_embeds=None):
        if input_ids is not None:
            cat_ids = cat_broadcast(
                input_ids,
                [input_ids, self.end_ids, force_ids],
            )
            r = self.model(
                cat_ids,
                use_cache=True,
                past_key_values=subset_kv_cache(self.kv_cache, input_ids.shape[0]),
            )
        else:
            if force_ids is not None:
                response_emb = self.embeddings(force_ids).detach()
            else:
                response_emb = None
            cat_embeds = cat_broadcast(
                inputs_embeds, [inputs_embeds, self.end_emb, response_emb]
            )
            r = self.model(
                inputs_embeds=cat_embeds,
                use_cache=True,
                past_key_values=subset_kv_cache(self.kv_cache, cat_embeds.shape[0]),
            )
        return r

    def retokenize(self, input_ids):
        n = input_ids.shape[0]
        full_input_ids = cat_broadcast(input_ids, [self.start_ids, input_ids])
        full_retok_ids = retokenize(full_input_ids, self.tokenizer)
        good = torch.empty(n, dtype=torch.bool, device=input_ids.device)
        retok_ids = torch.empty_like(input_ids)
        for i in range(n):
            if full_retok_ids[i].shape[0] >= full_input_ids.shape[1]:
                retok_ids[i] = full_retok_ids[i][
                    self.n_start : self.n_start + input_ids.shape[1]
                ]
                good[i] = True
            else:
                retok_ids[i] = input_ids[i]
                good[i] = False
        return retok_ids, good

    def check_invertibility(self, input_ids):
        cat_ids = cat_broadcast(
            input_ids,
            # the [1:] is important for removing the <s> token
            [self.start_ids[1:], input_ids, self.end_ids],
        )
        retok_ids = retokenize(
            cat_ids,
            self.tokenizer,
            add_special_tokens=False,
            skip_special_tokens=False,
        )
        invertible = []
        for i in range(cat_ids.shape[0]):
            invertible.append(
                (cat_ids.shape[1] == retok_ids[i].shape[0])
                and (cat_ids[i] == retok_ids[i]).all()
            )
        return torch.tensor(invertible, dtype=torch.bool, device=cat_ids.device)

    def generate(self, trigger_ids, force_ids=None, **kwargs):
        prompt_force_ids = cat_broadcast(
            trigger_ids,
            [self.start_ids, trigger_ids, self.end_ids, force_ids],
        )
        invertible = self.check_invertibility(trigger_ids)
        all_ids = llama.hf_generate(
            self.model, self.tokenizer, prompt_force_ids, **kwargs
        )
        assert (all_ids[:, : prompt_force_ids.shape[1]] == prompt_force_ids).all()
        generation_ids = all_ids[:, prompt_force_ids.shape[1] :]
        return all_ids, generation_ids, invertible


def cat_broadcast(base, order):
    reshaped = []
    for entry in order:
        if entry is base:
            reshaped.append(entry)
            continue
        if entry is None:
            continue
        tile_shape = [1] * len(base.shape)
        tile_shape[0] = base.shape[0]
        reshaped.append(torch.tile(entry[None], tuple(tile_shape)))
    return torch.cat(reshaped, dim=1)


def retokenize(ids, tokenizer, skip_special_tokens=True, add_special_tokens=True):
    text = tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens)
    return [
        tokenizer.encode(t, add_special_tokens=add_special_tokens, return_tensors="pt")[
            0
        ].to(ids.device)
        for t in text
    ]


def subset_kv_cache(kv_cache, n):
    if kv_cache is None:
        return None
    return tuple([(k[:n], v[:n]) for k, v in kv_cache])


class Force:
    def __init__(
        self,
        pm,
        tokenizer,
        force_string,
        verbose,
        x_penalty_min,
        x_penalty_max,
        terminate=True,
    ):
        self.pm = pm
        self.tokenizer = tokenizer
        self.force_ids = tokenizer.encode(
            force_string, return_tensors="pt", add_special_tokens=False
        ).to(pm.model.device)[0]

        self.force_tokens = tokenizer.batch_decode(self.force_ids)
        self.n_force = self.force_ids.shape[0]
        self.verbose = verbose
        self.terminate = terminate
        self.x_penalty_min = x_penalty_min
        self.x_penalty_max = x_penalty_max

    def __call__(self, input_ids, inputs_embeds=None):
        out = self.pm(
            self.force_ids,
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
        )
        invertible = self.pm.check_invertibility(input_ids)

        last_prompt_idx = input_ids.shape[1] + self.pm.n_end - 1
        force_logits = out.logits[:, last_prompt_idx:-1]
        force_xentropy = torch.nn.functional.cross_entropy(
            force_logits.reshape((-1, force_logits.shape[-1])),
            torch.tile(self.force_ids[None], (force_logits.shape[0], 1)).ravel(),
            reduction="none",
        ).reshape(force_logits.shape[:-1])
        force_xentropy[~invertible] = float("inf")

        trigger_logits = out.logits[:, : input_ids.shape[1]]

        return dict(
            target=-mellowmax(force_xentropy),
            logits=trigger_logits,
            force_logits=force_logits,
        )

    def callback(self, i, state, last_runtime, history, final=False):
        epo.pareto_callback(
            self, self.pm.model, self.tokenizer, self.x_penalty_min, self.x_penalty_max
        )(i, state, last_runtime, history, final)

        terminate, df = self.evaluate_convergence(
            state.extra["force_logits"][state.target.argmax()]
        )
        if self.verbose:
            print(df[df["flagged"]])
        return terminate

    def evaluate_convergence(self, logits, prob_margin=0.05):
        p = torch.softmax(logits, dim=-1)

        # Termination criterion: Is the force target the most likely token in
        # each position by a specified probability margin.
        n_force = self.force_ids.shape[0]
        p_top2 = p[:n_force].topk(k=2)
        p_0 = p_top2.values[:, 0]
        p_1 = p_top2.values[:, 1]
        ids_0 = p_top2.indices[:, 0]
        terminate = (
            False  # (self.force_ids == ids_0).all() and (p_0 - p_1 > prob_margin).all()
        )

        # Report on the most likely tokens
        p_force = p[torch.arange(n_force), self.force_ids]
        tok_0 = self.tokenizer.batch_decode(ids_0)
        ids_1 = p_top2.indices[:, 1]
        tok_1 = self.tokenizer.batch_decode(ids_1)
        df = pd.DataFrame(
            dict(
                tok_force=self.tokenizer.batch_decode(self.force_ids),
                p_force=p_force.cpu().numpy(),
                tok_0=tok_0,
                p_0=p_0.cpu().numpy(),
                tok_1=tok_1,
                p_1=p_1.cpu().numpy(),
                flagged=(self.force_ids != ids_0).cpu().numpy(),
            )
        )
        return terminate, df
