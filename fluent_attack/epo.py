"""
This file implements the EPO algorithm. See the `epo` function for the main entrypoint.
"""

import dataclasses
import time
from typing import Callable, Dict, List, Union

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import transformers


@dataclasses.dataclass
class History:
    """
    The `epo` function returns a History object that contains the full history
    of the population members at each iteration.
    """

    # Cross-entropy penalties for each population member.
    Xvs: np.ndarray
    # The token ids for each population member at each iteration.
    ids: List = dataclasses.field(default_factory=lambda: [])
    # The main objective for each population member at each iteration.
    target: List = dataclasses.field(default_factory=lambda: [])
    # The cross-entropy loss for each population member at each iteration.
    xentropy: List = dataclasses.field(default_factory=lambda: [])
    # The token gradients for each population member at each iteration.
    token_grads: List = dataclasses.field(default_factory=lambda: [])
    # The indices of the population members that were retained at each iteration.
    keep: List = dataclasses.field(default_factory=lambda: [])
    # The parent indices for each population member at each iteration.
    parents: List = dataclasses.field(default_factory=lambda: [])
    # The runtime for each iteration.
    runtime: List = dataclasses.field(default_factory=lambda: [])

    def subset(self, slc):
        """
        Return a History object sliced along the iterations dimension.
        """
        return History(
            Xvs=self.Xvs,
            ids=self.ids[slc],
            target=self.target[slc],
            xentropy=self.xentropy[slc],
            token_grads=self.token_grads[slc],
            keep=self.keep[slc],
            parents=self.parents[slc],
            runtime=self.runtime[slc],
        )

    def get_state(self, i, device=None):
        return State(
            ids=torch.tensor(self.ids[i]).to(device),
            target=torch.tensor(self.target[i]).to(device),
            xentropy=torch.tensor(self.xentropy[i]).to(device),
            token_grads=torch.tensor(self.token_grads[i]).to(device),
            extra=dict(),
        )

    def _insert(self, ids, target, xentropy, token_grads, keep, parents, runtime):
        self.ids.append(ids.cpu().numpy())
        self.target.append(target.cpu().numpy())
        self.xentropy.append(xentropy.cpu().numpy())
        self.token_grads.append(
            token_grads.cpu().numpy() if token_grads is not None else None
        )
        self.keep.append(keep.cpu().numpy())
        self.parents.append(parents.cpu().numpy())
        self.runtime.append(runtime)


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    target: torch.Tensor
    xentropy: torch.Tensor
    token_grads: torch.Tensor
    extra: Dict[str, torch.Tensor]

    def cat(self, state2):
        return State(
            ids=torch.cat((self.ids, state2.ids), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            token_grads=cat_if_not_none(self.token_grads, state2.token_grads),
            extra={
                k: cat_if_not_none(self.extra[k], state2.extra[k]) for k in self.extra
            },
        )

    def subset(self, keep):
        return State(
            ids=self.ids[keep],
            target=self.target[keep],
            xentropy=self.xentropy[keep],
            token_grads=(
                self.token_grads[keep.to("cpu")]
                if self.token_grads is not None
                else None
            ),
            extra={k: self.extra[k][keep] for k in self.extra},
        )

    def interleave(self, state2, which):
        ids = []
        target = []
        xentropy = []
        token_grads = []
        extra = dict()
        for k in self.extra:
            extra[k] = []
        left_idx = 0
        right_idx = 0
        for i in range(len(which)):
            if which[i]:
                ids.append(self.ids[left_idx])
                target.append(self.target[left_idx])
                xentropy.append(self.xentropy[left_idx])
                if self.token_grads is not None:
                    token_grads.append(self.token_grads[left_idx])
                for k in self.extra:
                    extra[k].append(self.extra[k][left_idx])
                left_idx += 1
            else:
                ids.append(state2.ids[right_idx])
                target.append(state2.target[right_idx])
                xentropy.append(state2.xentropy[right_idx])
                if state2.token_grads is not None:
                    token_grads.append(state2.token_grads[right_idx])
                for k in self.extra:
                    extra[k].append(state2.extra[k][right_idx])
                right_idx += 1
        return State(
            ids=torch.stack(ids, dim=0),
            target=torch.stack(target, dim=0),
            xentropy=torch.stack(xentropy, dim=0),
            token_grads=(
                torch.stack(token_grads, dim=0)
                if self.token_grads is not None
                else None
            ),
            extra={k: torch.stack(extra[k], dim=0) for k in self.extra},
        )


@torch.no_grad()
def epo(
    objective: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    initial_seq_len: int = 12,
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 256,
    topk: int = 512,
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    elongate_freq: int = 10,
    elongate_factor: int = 2,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    retain_parents: bool = False,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
) -> History:
    """
    Run the EPO algorithm. See the paper for details.

    Parameters
    ----------
    objective
        A callable that accepts either input_ids or inputs_embeds and returns a
        dictionary containing the `target` and the logits for each token
        position.
    model
    tokenizer
    initial_seq_len, optional
        The number of tokens in the optimized prompt, by default 12
    population_size, optional
        The population to keep at each iteration, by default 8
    iters, optional
        Number of iterations to run EPO, by default 300
    explore_per_pop, optional
        Number of children per population member per iteration, by default 32
    batch_size, optional
        GPU batch size, by default 256
    topk, optional
        When selecting token replacements, we select the `topk` tokens by
        gradient magnitude and choose uniformly at random between those, by
        default 512.
    x_penalty_min, optional
        The minimum cross-entropy penalty, by default 1.0 / 10.0
    x_penalty_max, optional
        The maximum cross-entropy penalty, by default 10.0
    seed, optional
        Random seed used for initialization, by default 0
    initial_ids, optional
        The initial token ids to begin optimizing from. If None, the initial
        token ids will be selected randomly, by default None
    history, optional
        The history of an EPO run that we want to continue, by default None
    catch_keyboard_interrupt, optional
        Should we catch keyboard interrupts and end the EPO loop?, by default False
    callback, optional
        A function called at the beginning of each iteration, by default None
    always_recompute_gradients, optional
        If a population member is retained across an iteration, we default to
        not recomputing that population member's token gradients. If your
        objective stores internal state that changes, you may want to override
        this behavior and recompute gradients every iteration.

    Returns
    -------
        A History object containing the full history of the EPO run.
    """
    start = time.time()
    iter_start = start
    explore_size = population_size * explore_per_pop
    device = model.device

    if seed is not None:
        torch.manual_seed(seed)

    if x_penalty_min is None or x_penalty_max is None:
        X = torch.zeros(population_size, device=device)
    else:
        X = torch.cat(
            (
                torch.tensor([0.0]),
                torch.exp(
                    torch.linspace(
                        np.log(x_penalty_min),
                        np.log(x_penalty_max),
                        population_size - 1,
                    )
                ),
            )
        ).to(device)

    if callback is None:
        callback = pareto_callback(
            objective, model, tokenizer, x_penalty_min, x_penalty_max
        )
    elif callback is False:
        callback = lambda *x: True

    #### history and initial_ids ####
    if history is not None:
        history = history._unfinalize()
        history.Xvs = X.cpu().numpy()
        if initial_ids is not None:
            raise ValueError("Cannot specify both history and initial_ids.")
        input_ids = torch.tensor(
            history.ids[-1][history.keep[-1]], dtype=torch.long
        ).to(device)
    elif initial_ids is not None:
        history = History(X.cpu().numpy())
        input_ids = initial_ids.to(device)
        if initial_ids.shape[0] != population_size:
            raise ValueError(f"initial_ids must have shape ({population_size}, *)")
    else:
        history = History(X.cpu().numpy())
        input_ids = torch.randint(
            0,
            model.get_input_embeddings().num_embeddings,
            (population_size, initial_seq_len),
            device=device,
        )

    assert input_ids.shape[0] == population_size

    if hasattr(objective, "setup"):
        objective.setup(input_ids)

    effort = torch.full((input_ids.shape[1],), 1.0, device=device, dtype=torch.float)
    # We use a try/except block so that we can catch keyboard interrupts and
    # still return results. This is useful for interactive use when it's nice
    # to launch with a large `iters` parameter and then just stop the run when
    # the results look good enough.
    try:
        #### Run the EPO loop: ####
        for i in range(iters):
            if i % elongate_freq == 0 and i > 0 and elongate_factor > 0:
                for e in range(elongate_factor):
                    replace_ids = torch.empty(
                        (input_ids.shape[0], explore_per_pop, input_ids.shape[1] + 1),
                        dtype=torch.long,
                        device=device,
                    )
                    insert_idx = input_ids.shape[1] // 2
                    replace_ids[:, :, :insert_idx] = input_ids[:, None, :insert_idx]
                    replace_ids[:, :, insert_idx] = (
                        model(input_ids).logits[0, -1].topk(explore_per_pop).indices
                    )
                    replace_ids[:, :, insert_idx + 1 :] = input_ids[
                        :, None, insert_idx:
                    ]
                    elongate_state = evaluate_fitness(
                        objective,
                        replace_ids.reshape((-1, replace_ids.shape[-1])),
                        batch_size,
                    )
                    elongate_loss = -elongate_state.target.reshape(
                        replace_ids.shape[:2]
                    ) + X[:, None] * elongate_state.xentropy.reshape(
                        replace_ids.shape[:2]
                    )
                    best_idx = elongate_loss.argmin(dim=-1)
                    input_ids = replace_ids[torch.arange(population_size), best_idx]
                effort = torch.cat(
                    (torch.full((elongate_factor,), 1.0, device=device), effort)
                )

            state = token_grads(
                objective, model, input_ids, x_penalty=X, batch_size=batch_size
            )

            ########################################
            # 1) Report!
            ########################################
            terminate_flag = callback(i, state, time.time() - iter_start, history)
            if (
                (isinstance(terminate_flag, str) and terminate_flag == "terminate")
                or (isinstance(terminate_flag, torch.Tensor) and terminate_flag.item())
                or (isinstance(terminate_flag, bool) and terminate_flag)
            ):
                if i == 0:
                    history._insert(
                        ids=state.ids,
                        target=state.target,
                        xentropy=state.xentropy,
                        token_grads=None,  # state.token_grads,
                        keep=torch.arange(population_size),
                        parents=torch.full(population_size, -1, dtype=torch.int),
                        runtime=time.time() - iter_start,
                    )
                break
            else:
                iter_start = time.time()

            ########################################
            # 2) Birth children from parents
            # copy inputs to expand out to explore_size new candidates.
            ########################################
            source_idx = (
                torch.tile(torch.arange(population_size)[:, None], (1, explore_per_pop))
                .flatten()
                .to(device)
            )
            assert source_idx.shape[0] == explore_size
            assert (source_idx < population_size).all()
            new_ids = state.ids[source_idx, :].clone()

            ########################################
            # 3) GCG-style mutation
            ########################################
            ranking = (-state.token_grads).topk(k=topk, dim=-1)
            pos_scores = (1.0 / effort) ** 2
            pos_scores /= pos_scores.sum()
            pos = torch.multinomial(pos_scores, new_ids.shape[0], replacement=True)
            effort += pos_scores
            # pos = torch.randint(
            #     low=0,
            #     high=input_ids.shape[1],
            #     size=(new_ids.shape[0],),
            #     device=device,
            # )
            token_idx = torch.randint(
                low=0,
                high=topk,
                size=(new_ids.shape[0],),
                device=device,
            )
            new_ids[torch.arange(new_ids.shape[0]), pos] = ranking.indices.to(device)[
                source_idx, pos, token_idx
            ]

            ########################################
            # 5) Evaluate fitness
            ########################################
            child_state = evaluate_fitness(objective, new_ids, batch_size=batch_size)
            if retain_parents:
                all_state = state.cat(child_state)
            else:
                all_state = child_state

            # note that all_loss is a matrix with a row for each population
            # member because each population member slot uses a different
            # xentropy penalty.
            all_loss = (
                -all_state.target[None, :] + X[:, None] * all_state.xentropy[None, :]
            )
            keep = (-all_loss).argmax(dim=1).to(torch.int)

            parents = torch.empty_like(keep)
            if retain_parents:
                is_fresh = keep >= population_size
                parents[is_fresh] = source_idx[keep[is_fresh] - population_size].to(
                    parents.dtype
                )
                parents[~is_fresh] = keep[~is_fresh]
            else:
                parents = source_idx[keep].to(parents.dtype)

            history_token_grads = state.token_grads
            history._insert(
                ids=all_state.ids,
                target=all_state.target,
                xentropy=all_state.xentropy,
                token_grads=history_token_grads,
                keep=keep,
                parents=parents,
                runtime=time.time() - iter_start,
            )

            input_ids = all_state.ids[keep]

    # it's handy to sometimes be able to interrupt the loop and still get
    # results!
    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            pass
        else:
            raise

    terminate_flag = callback(i, state, time.time() - iter_start, history, final=True)

    return history


@dataclasses.dataclass
class ParetoFrontier:
    # the range of cross-entropy penalties used
    Xvs: np.ndarray
    # the target and xentropy values for each penalty level
    full_target: np.ndarray
    full_xentropy: np.ndarray
    # the unique indices in full_target/full_xentropy that make up the pareto frontier.
    unique: np.ndarray
    # the target and xentropy values for the unique entries
    target: np.ndarray
    xentropy: np.ndarray
    # the token ids for each unique point on the frontier.
    ids: np.ndarray
    # the detokenized text for each unique point on the frontier.
    text: List[str]


def build_pareto_frontier(tokenizer, histories, Xvs=None):
    """
    Construct a pareto frontier from the history of several EPO runs. We allow
    multiple histories to be passed so that we can construct the Pareto
    frontier across several different runs of EPO with different random
    initializations.

    Parameters
    ----------
    tokenizer
    histories
        A list of History objects returned by the EPO algorithm. We allow
        multiple independent histories to be combined
    Xvs, optional
        The range of cross-entropy penalties to use.
        By default Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    Returns
    -------
        A ParetoFrontier object.
    """

    if Xvs is None:
        Xvs = 1.0 / np.linspace(0, 50, 1000)[1:]

    if not isinstance(histories, list):
        histories = [histories]
    x = []
    t = []
    ids = []
    for h in histories:
        x.append(h.xentropy.flatten())
        t.append(h.target.flatten())
        ids.append(h.ids.reshape((-1, h.ids.shape[-1])))

    history_x = np.concatenate(x)
    history_t = np.concatenate(t)
    history_ids = np.concatenate(ids, axis=0)
    pareto_t = np.empty(Xvs.shape[0])
    pareto_x = np.empty(Xvs.shape[0])
    pareto_idxs = []
    for i, Xv in enumerate(Xvs):
        loss = -history_t + Xv * history_x
        idx = loss.argmin()
        pareto_idxs.append(idx)
        pareto_t[i] = history_t[idx]
        pareto_x[i] = history_x[idx]
    pareto_unique = np.unique(pareto_idxs, return_index=True)[1]
    pareto_ids = [history_ids[pareto_idxs[i]] for i in pareto_unique]
    pareto_text = [tokenizer.decode(ids) for ids in pareto_ids]
    return ParetoFrontier(
        np.array(Xvs),
        pareto_t,
        pareto_x,
        pareto_unique,
        pareto_t[pareto_unique],
        pareto_x[pareto_unique],
        pareto_ids,
        pareto_text,
    )


def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat((a, b), dim=0)


# based on https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
def token_grads(
    objective: Callable,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    batch_size: int = 1,
):
    """
    Compute gradients with respect to one-hot encoded input tokens. This is a
    infinitesimal approximation to the token influence on the loss so it's a
    very noisy indicator of which tokens might reduce loss.
    """
    embed = model.get_input_embeddings()

    token_grads = torch.empty(
        (input_ids.shape[0], input_ids.shape[1], embed.num_embeddings),
        dtype=torch.float,
    )
    loss = torch.empty(input_ids.shape[0], device=model.device)
    xentropy = torch.empty(input_ids.shape[0], device=model.device)
    target = torch.empty(input_ids.shape[0], device=model.device)
    extra = dict()

    with torch.enable_grad():
        model.zero_grad()

        for i in range(0, input_ids.shape[0], batch_size):
            imax = min(i + batch_size, input_ids.shape[0])

            # using a one hot matrix as input to the model gives us gradients with
            # respect to potential input tokens.
            one_hot = F.one_hot(input_ids[i:imax], num_classes=embed.num_embeddings).to(
                embed.weight.dtype
            )
            one_hot.requires_grad = True

            inputs_embeds = torch.matmul(one_hot, embed.weight)

            out = objective(input_ids=input_ids, inputs_embeds=inputs_embeds)

            logits_offset = out["logits"][:, :-1]
            this_xentropy = (
                -(torch.log_softmax(logits_offset, dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            this_loss = -out["target"] + this_xentropy * x_penalty[i:imax]
            this_loss.sum().backward()

            loss[i:imax] = this_loss
            target[i:imax] = out["target"]
            xentropy[i:imax] = this_xentropy
            token_grads[i:imax] = one_hot.grad

            for k in out:
                if k not in ["target"]:
                    e = out[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e

            # important to zero out gradients here to release memory
            model.zero_grad()

    return State(input_ids, target, xentropy, token_grads, extra)


def calc_xentropy(logits, input_ids):
    logits_offset = logits[:, :-1]
    return (
        torch.nn.CrossEntropyLoss(reduction="none")(
            logits_offset.reshape(-1, logits_offset.shape[-1]),
            input_ids[:, 1:].reshape(-1),
        )
        .view(*logits_offset.shape[:2])
        .mean(dim=-1)
    )


def evaluate_fitness(
    objective: Callable,
    input_ids: torch.Tensor,
    batch_size: int,
):
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=input_ids.device)
    xentropy = torch.empty(
        input_ids.shape[0], dtype=torch.float, device=input_ids.device
    )
    retok_ids = torch.empty_like(input_ids)
    extra = dict()
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        out = objective(input_ids=input_ids[i:imax])
        if "retok_ids" in out:
            retok_ids[i:imax] = out["retok_ids"]
        else:
            retok_ids[i:imax] = input_ids[i:imax]
        target[i:imax] = out["target"]
        xentropy[i:imax] = calc_xentropy(out["logits"], input_ids[i:imax])

        for k in out:
            if k not in ["target"]:
                e = out[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e

    return State(retok_ids, target, xentropy, None, extra)


def pareto_callback(objective, model, tokenizer, x_penalty_min, x_penalty_max):
    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))
        print(f"\nbeginning step {i}, current pareto frontier prompts:")
        last_idx = None

        if x_penalty_min is None or x_penalty_max is None:
            Xvs = torch.tensor([0], device=model.device)
        else:
            Xvs = torch.cat(
                (
                    torch.exp(
                        torch.linspace(
                            np.log(x_penalty_min / 10), np.log(x_penalty_max * 10), 400
                        )
                    ),
                )
            ).to(model.device)
        loss = -state.target[None] + Xvs[:, None] * state.xentropy[None]
        idxs = loss.argmin(dim=1)
        for i in range(len(Xvs)):
            idx = idxs[i]
            if idx == last_idx:
                continue
            text = tokenizer.decode(state.ids[idx])
            print(
                f"penalty={Xvs[i]:.2f} xentropy={state.xentropy[idx]:.2f} target={state.target[idx]:.2f} {repr(text)}"
            )
            last_idx = idx

    return f
