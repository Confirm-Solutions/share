import pandas as pd
import torch

import epo
import llama


def construct_prompt(prompt_template, prefix, trigger, suffix):
    # This function is factored out to highlight the construction of the
    # prompt. The spaces in between the substrings are important and prevent a
    # lot of tokenization issues.
    if trigger != "":
        trigger = " " + trigger
    return prompt_template.format(instruction=prefix + trigger + " " + suffix)


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
            construct_prompt(prompt_template, prefix, "", suffix),
            return_tensors="pt",
        )[0].to(model.device)

        # TODO: This is not correct! Need to match the full subsequence to be
        # robust.
        if prefix == "":
            if suffix == "":
                raise ValueError("Suffix cannot be empty if prefix is not empty.")
            suffix_ids = tokenizer.encode(
                suffix, return_tensors="pt", add_special_tokens=False
            )[0].to(model.device)
            last_match = torch.where(prompt_ids == suffix_ids[0])[0][-1] - 1
        else:
            prefix_ids = tokenizer.encode(
                prefix, return_tensors="pt", add_special_tokens=False
            )[0].to(model.device)
            last_match = torch.where(prompt_ids == prefix_ids[-1])[-1]
        self.start_ids = prompt_ids[: last_match + 1]
        self.start_str = tokenizer.decode(self.start_ids)
        self.end_ids = prompt_ids[last_match + 1 :]
        self.end_str = tokenizer.decode(self.end_ids)

        # Check prompt construction correctness.
        assert self.start_str.endswith(prefix)
        assert self.end_str.startswith(suffix)

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

    def __call__(
        self, force_ids=None, input_ids=None, inputs_embeds=None, correct_seq_len=None
    ):
        if input_ids is not None:
            # Retokenize and check for invertibility:
            prompt_force_ids, invertible = self.build_prompt_ids(
                input_ids, force_ids, correct_seq_len=correct_seq_len
            )
            r = self.model(
                prompt_force_ids[:, self.start_ids.shape[0] :],
                use_cache=True,
                past_key_values=subset_kv_cache(self.kv_cache, input_ids.shape[0]),
            )
        else:
            # Any prompt that has become a "parent" has already been checked
            # for invertibility so we can just concatenate ids here:
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
            invertible = torch.ones(
                r.logits.shape[0], dtype=torch.bool, device=r.logits.device
            )
        return r, invertible

    def build_prompt_ids(self, trigger_ids, force_ids=None, correct_seq_len=None):
        if force_ids is not None and force_ids.shape[0] == 0:
            force_ids = None

        # Prompts need to be invertible, so we need to decode and then
        # re-encode instead of just concatenating id tensors:
        triggers = self.tokenizer.batch_decode(trigger_ids, skip_special_tokens=True)
        prompts = [
            construct_prompt(self.prompt_template, self.prefix, t, self.suffix)
            for t in triggers
        ]
        ps = [
            self.tokenizer.encode(p, return_tensors="pt").to(self.model.device)
            for p in prompts
        ]

        if correct_seq_len is not None:
            correct_prompt_len = correct_seq_len - force_ids.shape[0]
            ps = [
                torch.cat(
                    (
                        p[:, :correct_prompt_len],
                        torch.zeros(
                            p.shape[0],
                            max(correct_prompt_len - p.shape[1], 0),
                            dtype=torch.long,
                            device=self.model.device,
                        ),
                    ),
                    dim=1,
                )
                for p in ps
            ]
        prompt_ids = torch.cat(ps, dim=0)

        # While we "retokenize" the entire prompt to ensure
        # realistic/invertible tokens, the forcing ids *should* be concatenated
        # because in a generation situation, the forced tokens will be
        # generated one at a time.
        prompt_force_ids = cat_broadcast(prompt_ids, [prompt_ids, force_ids])

        # Check that the start_ids are correctly decoded and thus we can safely
        # use the kv_cache
        same_as_start = prompt_force_ids[:, : self.start_ids.shape[0]] == self.start_ids
        invertible = same_as_start.all(dim=-1)

        # Check that the end_ids are correctly decoded.
        trigger_start = self.start_ids.shape[0]
        trigger_end = self.start_ids.shape[0] + trigger_ids.shape[1]
        prompt_end = trigger_end + self.end_ids.shape[0]
        actual_end = prompt_force_ids[:, trigger_end:prompt_end]
        invertible &= actual_end.shape[1] == self.end_ids.shape[0]
        invertible &= (actual_end == self.end_ids[: actual_end.shape[1]]).all(dim=-1)

        # This code is commented out but shows how we might subset the kv cache
        # if only some of the start_ids are the same
        #
        # if same_as_start.all():
        #     n_kv = rm.start_ids.shape[0]
        # else:
        #     n_kv = same_as_start.to(torch.int).argmin().item()

        # Check that triggers are correctly decoded:
        decoded_triggers = self.tokenizer.batch_decode(
            prompt_force_ids[:, trigger_start:trigger_end]
        )
        invertible &= torch.tensor(
            [t0 == t1 for t0, t1 in zip(decoded_triggers, triggers)],
            device=self.model.device,
        )

        return prompt_force_ids, invertible

    def generate(self, trigger_ids, force_ids=None, **kwargs):
        prompt_force_ids, invertible = self.build_prompt_ids(trigger_ids, force_ids)
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


def subset_kv_cache(kv_cache, n):
    if kv_cache is None:
        return None
    return tuple([(k[:n], v[:n]) for k, v in kv_cache])


def mellowmax(t: torch.Tensor, alpha=1.0, dim=-1):
    return (
        1.0
        / alpha
        * (
            torch.logsumexp(alpha * t, dim=dim)
            - torch.log(torch.tensor(t.shape[-1], dtype=t.dtype, device=t.device))
        )
    )


class Force:
    def __init__(self, runner, tokenizer, force_string, verbose, terminate=True):
        self.runner = runner
        self.tokenizer = tokenizer
        self.force_ids = tokenizer.encode(
            force_string, return_tensors="pt", add_special_tokens=False
        ).to(runner.model.device)[0]

        temp_ids = tokenizer.encode(
            "a", return_tensors="pt", add_special_tokens=False
        ).to(runner.model.device)
        temp_all_ids, invertible = self.runner.build_prompt_ids(
            temp_ids, self.force_ids
        )
        assert invertible.all()
        self.n_total_ids_minus_trigger = temp_all_ids.shape[1] - temp_ids.shape[0]

        self.force_tokens = tokenizer.batch_decode(self.force_ids)
        self.n_force = self.force_ids.shape[0]
        self.verbose = verbose
        self.terminate = terminate

    def __call__(self, input_ids, inputs_embeds=None):
        out, invertible = self.runner(
            self.force_ids,
            input_ids=input_ids if inputs_embeds is None else None,
            inputs_embeds=inputs_embeds,
            correct_seq_len=self.n_total_ids_minus_trigger + input_ids.shape[1],
        )

        last_prompt_idx = input_ids.shape[1] + self.runner.n_end - 1
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
        epo.pareto_callback(self, self.runner.model, self.tokenizer)(
            i, state, last_runtime, history, final
        )

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
