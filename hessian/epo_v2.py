"""
This file implements the EPO algorithm. See the `epo` function for the main entrypoint.
"""

import contextlib
import dataclasses
import time
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributions
import torch.nn.functional as F
import transformers


@contextlib.contextmanager
def add_fwd_hooks(module_hooks: List[Tuple[torch.nn.Module, Callable]]):
    """
    Context manager for temporarily adding forward hooks to a model.

    Parameters
    ----------
    module_hooks
        A list of pairs: (module, fnc) The function will be registered as a
            forward hook on the module
    """
    try:
        handles = []
        for mod, hk in module_hooks:
            handles.append(mod.register_forward_hook(hk))
        yield
    finally:
        for h in handles:
            h.remove()


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
    # The target objective for each population member at each iteration.
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
        self.token_grads.append(token_grads.cpu().numpy())
        self.keep.append(keep.cpu().numpy())
        self.parents.append(parents.cpu().numpy())
        self.runtime.append(runtime)

    def _finalize(self):
        return History(
            Xvs=self.Xvs,
            ids=np.stack(self.ids, axis=0),
            target=np.stack(self.target, axis=0),
            xentropy=np.stack(self.xentropy, axis=0),
            token_grads=np.stack(self.token_grads, axis=0),
            keep=np.stack(self.keep, axis=0),
            parents=np.stack(self.parents, axis=0),
            runtime=np.stack(self.runtime, axis=0),
        )

    def _unfinalize(self):
        return History(
            Xvs=self.Xvs,
            ids=[x for x in self.ids],
            target=[x for x in self.target],
            xentropy=[x for x in self.xentropy],
            token_grads=[x for x in self.token_grads],
            keep=[x for x in self.keep],
            parents=[x for x in self.parents],
            runtime=[x for x in self.runtime],
        )


@dataclasses.dataclass
class State:
    ids: torch.Tensor
    target: torch.Tensor
    xentropy: torch.Tensor
    token_grads: torch.Tensor
    token_dhess: torch.Tensor
    extra: Dict[str, torch.Tensor]

    def cat(self, state2):
        return State(
            ids=torch.cat((self.ids, state2.ids), dim=0),
            target=torch.cat((self.target, state2.target), dim=0),
            xentropy=torch.cat((self.xentropy, state2.xentropy), dim=0),
            token_grads=cat_if_not_none(self.token_grads, state2.token_grads),
            token_dhess=cat_if_not_none(self.token_dhess, state2.token_dhess),
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
            token_dhess=(
                self.token_dhess[keep.to("cpu")]
                if self.token_dhess is not None
                else None
            ),
            extra={k: self.extra[k][keep] for k in self.extra},
        )

    def interleave(self, state2, which):
        ids = []
        target = []
        xentropy = []
        token_grads = []
        token_dhess = []
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
                if self.token_dhess is not None:
                    token_dhess.append(self.token_dhess[left_idx])
                for k in self.extra:
                    extra[k].append(self.extra[k][left_idx])
                left_idx += 1
            else:
                ids.append(state2.ids[right_idx])
                target.append(state2.target[right_idx])
                xentropy.append(state2.xentropy[right_idx])
                if state2.token_grads is not None:
                    token_grads.append(state2.token_grads[right_idx])
                if state2.token_dhess is not None:
                    token_dhess.append(state2.token_dhess[right_idx])
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
            token_dhess=(
                torch.stack(token_dhess, dim=0)
                if self.token_dhess is not None
                else None
            ),
            extra={k: torch.stack(extra[k], dim=0) for k in self.extra},
        )


@torch.no_grad()
def epo_v2(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 12,
    population_size: int = 8,
    iters: int = 300,
    explore_per_pop: int = 32,
    batch_size: int = 256,
    topk: int = 512,
    x_penalty_min: float = 1.0 / 10.0,
    x_penalty_max: float = 10.0,
    drop_prob: float = 0.0,
    restart_frequency: int = None,
    restart_xentropy: float = 2.0,
    restart_xentropy_max_mult: float = 3.0,
    gradient_method="gcg",
    embedding_noise=None,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
) -> History:
    """
    Run the EPO algorithm. See the paper for details.

    Parameters
    ----------
    cache_run
        A callable that accepts either input_ids or inputs_embeds and returns a
        dictionary containing the `target` and the logits for each token
        position.
    model
    tokenizer
    seq_len, optional
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
    drop_probability: the probability that we drop a given population member at
                      each iteration
    restart_frequency, optional
        How often do we reset the Pareto frontier, by default 50
    restart_xentropy, optional
        When we reset the Pareto frontier, we select a population member that
        is optimal according to a cross-entropy penalty that is selected
        uniformly at random in the domain
        [restart_xentropy / restart_xentropy_max_mult,
         restart_xentropy * restart_xentropy_max_mult],
        restart_xentropy is by default 2.0
    restart_xentropy_max_mult, optional
        See the explanation for restart_xentropy, by default 3.0
    gradient_method, optional
        The method used to calculate gradients. Currently, the only supported
        methods are "gcg" and "multi", by default "gcg".
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
        cache_run stores internal state that changes, you may want to override
        this behavior and recompute gradients every iteration.

    Returns
    -------
        A History object containing the full history of the EPO run.
    """
    start = time.time()
    explore_size = population_size * explore_per_pop
    device = model.device

    if seed is not None:
        torch.manual_seed(seed)

    if x_penalty_min is None or x_penalty_max is None:
        X = torch.zeros(population_size, device=model.device)
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
        ).to(model.device)

    if callback is None:
        callback = pareto_callback(cache_run, model, tokenizer)
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
        ).to(model.device)
    elif initial_ids is not None:
        history = History(X.cpu().numpy())
        input_ids = initial_ids.to(model.device)
        if initial_ids.shape[0] != population_size or initial_ids.shape[1] != seq_len:
            raise ValueError(
                f"initial_ids must have shape ({population_size}, {seq_len})"
            )
    else:
        history = History(X.cpu().numpy())
        input_ids = torch.randint(
            0,
            model.get_input_embeddings().num_embeddings,
            (population_size, seq_len),
            device=model.device,
        )

    assert input_ids.shape[0] == population_size

    if hasattr(cache_run, "setup"):
        cache_run.setup(input_ids)

    if embedding_noise is not None:
        always_recompute_gradients = True

    state = token_grads(
        cache_run,
        model,
        input_ids,
        x_penalty=X,
        batch_size=batch_size,
        embedding_noise=embedding_noise,
        include_hess=3 if gradient_method == "dhess" else 0,
    )

    # We use a try/except block so that we can catch keyboard interrupts and
    # still return results. This is useful for interactive use when it's nice
    # to launch with a large `iters` parameter and then just stop the run when
    # the results look good enough.
    try:
        #### Run the EPO loop: ####
        for i in range(iters):
            ########################################
            # 1) Report!
            ########################################
            terminate_flag = callback(i, state, time.time() - start, history)
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
                        token_grads=state.token_grads,
                        keep=torch.arange(population_size),
                        parents=torch.full(population_size, -1, dtype=torch.int),
                        runtime=time.time() - start,
                    )
                break
            else:
                start = time.time()
            recompute_gradients = always_recompute_gradients or (
                terminate_flag == "recompute_gradients"
            )

            ########################################
            # 2) Birth children from parents
            # copy inputs to expand out to explore_size new candidates.
            ########################################
            source_idx = (
                torch.tile(torch.arange(population_size)[:, None], (1, explore_per_pop))
                .flatten()
                .to(model.device)
            )
            assert source_idx.shape[0] == explore_size
            assert (source_idx < population_size).all()
            new_ids = state.ids[source_idx, :].clone()

            ########################################
            # Mike's gradient averaging
            ########################################
            if gradient_method == "dhess":
                predicted_loss = dhess_predict(state, X, device=model.device)
            elif len(history.token_grads) == 0 or gradient_method == "gcg":
                predicted_loss = quick_gcg_predict(state)
            elif gradient_method == "multi":
                predicted_loss = multi_predict(
                    state, history, X, device=model.device, beta=0.2
                )
            elif gradient_method == "child_multi":
                predicted_loss = child_multi_predict(
                    state, history, X, device=model.device, beta=0.2, max_swaps=2
                )
            else:
                raise ValueError(f"Unknown gradient method: {gradient_method}")

            ########################################
            # 3) GCG-style mutation
            ########################################
            ranking = (-predicted_loss).topk(k=topk, dim=-1)
            pos = torch.randint(
                low=0,
                high=input_ids.shape[1],
                size=(new_ids.shape[0],),
                device=input_ids.device,
            )
            token_idx = torch.randint(
                low=0,
                high=topk,
                size=(new_ids.shape[0],),
                device=input_ids.device,
            )
            new_ids[torch.arange(new_ids.shape[0]), pos] = ranking.indices.to(
                input_ids.device
            )[source_idx, pos, token_idx]

            ########################################
            # 5) Evaluate fitness
            ########################################
            if gradient_method == "child_multi":
                child_X = X[source_idx]
                child_state = token_grads(
                    cache_run,
                    model,
                    new_ids,
                    x_penalty=child_X,
                    batch_size=batch_size,
                )
            else:
                child_state = evaluate_fitness(
                    cache_run, new_ids, batch_size=batch_size
                )
            if embedding_noise is None:
                parent_state = state
            else:
                parent_state = evaluate_fitness(
                    cache_run, state.ids, batch_size=batch_size
                )
            all_state = parent_state.cat(child_state)

            # note that all_loss is a matrix with a row for each population
            # member because each population member slot uses a different
            # xentropy penalty.
            all_loss = (
                -all_state.target[None, :] + X[:, None] * all_state.xentropy[None, :]
            )
            keep = (-all_loss).argmax(dim=1).to(torch.int)

            if drop_prob > 0.0:
                drop_replace = torch.randint(
                    0, keep.shape[0], (keep.shape[0],), device=device, dtype=keep.dtype
                )
                which_drop = torch.rand(keep.shape[0], device=device) < drop_prob
                keep = torch.where(which_drop, drop_replace, keep)

            if (
                restart_frequency is not None
                and (i > 0)
                and (i % restart_frequency == 0)
            ):
                min_mult = 1.0 / restart_xentropy_max_mult
                max_mult = restart_xentropy_max_mult
                mult = min_mult + (max_mult - min_mult) * torch.rand(1).item()
                restart_X = restart_xentropy * mult
                restart_loss = -all_state.target + restart_xentropy * all_state.xentropy
                print(f"restarting with xentropy penalty of {restart_X:.2f}")
                keep[:] = restart_loss.argmin()

            parents = torch.empty_like(keep)
            is_fresh = keep >= population_size
            parents[is_fresh] = source_idx[keep[is_fresh] - population_size].to(
                parents.dtype
            )
            parents[~is_fresh] = keep[~is_fresh]

            history_token_grads = state.token_grads
            if gradient_method == "child_multi":
                history_token_grads = all_state.token_grads
            history._insert(
                ids=all_state.ids,
                target=all_state.target,
                xentropy=all_state.xentropy,
                token_grads=history_token_grads,
                keep=keep,
                parents=parents,
                runtime=time.time() - start,
            )

            ########################################
            # 6) Calculate gradients for the next iteration.
            ########################################
            if i != iters - 1:
                if recompute_gradients:
                    survived = torch.tensor([])
                    new = keep
                else:
                    survived = keep[~is_fresh]
                    new = keep[is_fresh]
                if new.shape[0] > 0:
                    new_ids = all_state.ids[new]
                    new_state = token_grads(
                        cache_run,
                        model,
                        new_ids,
                        x_penalty=X,
                        batch_size=batch_size,
                        embedding_noise=embedding_noise,
                        include_hess=3 if gradient_method == "dhess" else 0,
                    )
                if survived.shape[0] > 0:
                    state_survived = state.subset(survived)
                    if new.shape[0] > 0:
                        state = state_survived.interleave(new_state, ~is_fresh)
                        # state = state_survived.cat(state_new)
                    else:
                        state = state_survived
                else:
                    state = new_state

    # it's handy to sometimes be able to interrupt the loop and still get
    # results!
    except KeyboardInterrupt:
        if catch_keyboard_interrupt:
            pass
        else:
            raise

    terminate_flag = callback(i, state, time.time() - start, history, final=True)

    return history._finalize()


def dhess_predict(state, X, device=None):
    pred_first_order = true_gcg_predict(state, X, device=device)

    population_size, seq_len = state.ids.shape
    parent_ids = state.ids
    parent_dhess = state.token_dhess[
        torch.arange(population_size)[:, None],
        torch.arange(seq_len)[None, :],
        parent_ids.to(state.token_grads.device),
    ].to(device)
    pred_second_order = pred_first_order + 0.5 * (
        parent_dhess[..., None] + state.token_dhess.to(device)
    )
    return pred_second_order


def quick_gcg_predict(state):
    return state.token_grads


def true_gcg_predict(state, X, device=None):
    population_size, seq_len = state.ids.shape
    parent_f0 = -state.target + X * state.xentropy
    parent_ids = state.ids
    parent_token_grads = state.token_grads[
        torch.arange(population_size)[:, None],
        torch.arange(seq_len)[None, :],
        parent_ids.to(state.token_grads.device),
    ].to(device)
    f_child = parent_f0[:, None, None] + (
        state.token_grads.to(device) - parent_token_grads[:, :, None]
    )
    return f_child


def child_multi_predict(
    state, history, X, beta=0.0, device=None, max_swaps=None, n_source=3
):
    f_child = true_gcg_predict(state, X, device=device)
    parent_ids = state.ids

    population_size, seq_len = parent_ids.shape
    vocab_size = state.token_grads.shape[-1]

    # TODO: the combo numpy/torch thing here is a bit ugly. switch to using
    # torch for history until finalization?
    source_ids = torch.tensor(
        np.concatenate(
            [
                history.ids[i][population_size:].reshape((population_size, -1, seq_len))
                for i in range(-n_source, 0)
            ],
            axis=1,
        ),
        device=device,
    )
    source_token_grads = torch.tensor(
        np.concatenate(
            [
                history.token_grads[i][population_size:].reshape(
                    (population_size, -1, seq_len, vocab_size)
                )
                for i in range(-n_source, 0)
            ],
            axis=1,
        ),
        device=device,
    )
    source_target = torch.tensor(
        np.concatenate(
            [
                history.target[i][population_size:].reshape((population_size, -1))
                for i in range(-n_source, 0)
            ],
            axis=1,
        ),
        device=device,
    )
    source_xentropy = torch.tensor(
        np.concatenate(
            [
                history.xentropy[i][population_size:].reshape((population_size, -1))
                for i in range(-n_source, 0)
            ],
            axis=1,
        ),
        device=device,
    )

    source_f0 = -source_target + X[:, None] * source_xentropy
    parent_from_source_grads = torch.swapaxes(
        source_token_grads[
            torch.arange(population_size)[:, None],
            :,
            torch.arange(seq_len)[None, :],
            parent_ids,
        ],
        1,
        2,
    )
    source_grads = source_token_grads[
        torch.arange(population_size)[:, None, None],
        torch.arange(source_token_grads.shape[1])[None, :, None],
        torch.arange(seq_len)[None, None],
        source_ids,
    ]
    parent_f0_appx = source_f0 + (parent_from_source_grads - source_grads).sum(axis=-1)
    f_child_from_source = parent_f0_appx[..., None, None] + (
        source_token_grads - parent_from_source_grads[..., None]
    )

    n_swaps = (parent_ids[:, None] != source_ids).sum(axis=-1)
    include = (
        (n_swaps <= max_swaps)
        if max_swaps is not None
        else torch.ones(f_child_from_source.shape[:2], device=device)
    )
    beta0_prediction = (f_child_from_source * include[..., None, None]).sum(dim=1) / (
        include.sum(dim=1)[:, None, None]
    )
    loss_prediction = beta * f_child + (1 - beta) * beta0_prediction

    # in typical GCG:
    # loss_prediction = state.token_grads
    return loss_prediction


def multi_predict(
    state, history, X, beta=0.0, device=None, n_source=10, max_swaps=None
):
    # The goal here is to rank potential token swaps by the predicted loss.
    #
    # In typical GCG, we use a first-order extrapolation to predict the child
    # loss from the parent loss with token gradients. If we isolate our one-hot
    # token coordinates to just the current token and the potential swap token,
    # we aim to predict f(1, 0) from f(0, 1). The first-order prediction will
    # be: f(0, 1) + (df/dx - df/dy).
    # But, the parent_f0 and -parent_token_grads terms drop out because they
    # are constant for a single token position. So, we can just
    # use -state.token_grads to rank potential swaps.
    #
    # To explain the multipoint procedure here, consider:
    # - a parent prompt on which we will perform a token swap.
    # - a source prompt which we will use for a second prediction of loss.
    #
    # The source and parent prompts may differ in several token positions. But,
    # to explain the math, imagine the source and parent prompts differ only in
    # one token position and that token position is not the position being
    # swapped:
    # - the parent loss (parent_f0) is: f(t00 = 0, t01 = 1, t10 = 0, t11 = 1)
    # - the source loss (source_f0) is: f(t00 = 1, t01 = 0, t10 = 0, t11 = 1)
    # - the loss we'd like to predict (f_child) is: f(t00 = 0, t01 = 1, t10 = 1, t11 = 0)
    #
    # First, we compute:
    # f_child = parent_f0 + (df(t_parent)/dt10 - df(t_parent)/dt11) / sqrt(2)
    # Then, we compute:
    # parent_f0_appx = source_f0 + (df(t_source)/dt01 - df(t_source)/dt00) / sqrt(2)
    # Then, we compute:
    # f_child_from_source = parent_f0_appx + (df(t_source)/dt10 - df(t_source)/dt11) / sqrt(2)
    #
    # The code below adapts these expressions to the setting where many token
    # positions differ between the source and parent prompts.
    #
    # Finally, we average f_child and f_child_from_source together to get an
    # improved prediction.
    #
    f_child = true_gcg_predict(state, X, device=device)
    parent_ids = state.ids

    population_size, seq_len = parent_ids.shape
    # TODO: the combo numpy/torch thing here is a bit ugly. switch to using
    # torch for history until finalization?
    source_ids = torch.tensor(
        np.transpose(
            np.stack(
                [ids[:population_size] for ids in history.ids[-n_source:]],
                axis=0,
            ),
            (1, 0, 2),
        ),
        device=device,
    )
    source_token_grads = torch.tensor(
        np.transpose(
            np.stack(
                [t[:population_size] for t in history.token_grads[-n_source:]], axis=0
            ),
            (1, 0, 2, 3),
        ),
        device=device,
    )
    source_target = torch.tensor(
        np.transpose(
            np.stack(
                [t[:population_size] for t in history.target[-n_source:]],
                axis=0,
            ),
            (1, 0),
        ),
        device=device,
    )
    source_xentropy = torch.tensor(
        np.transpose(
            np.stack(
                [x[:population_size] for x in history.xentropy[-n_source:]],
                axis=0,
            ),
            (1, 0),
        ),
        device=device,
    )

    source_f0 = -source_target + X[:, None] * source_xentropy
    parent_from_source_grads = torch.swapaxes(
        source_token_grads[
            torch.arange(population_size)[:, None],
            :,
            torch.arange(seq_len)[None, :],
            parent_ids,
        ],
        1,
        2,
    )
    source_grads = source_token_grads[
        torch.arange(population_size)[:, None, None],
        torch.arange(source_token_grads.shape[1])[None, :, None],
        torch.arange(seq_len)[None, None],
        source_ids,
    ]
    parent_f0_appx = source_f0 + (parent_from_source_grads - source_grads).sum(axis=-1)
    f_child_from_source = parent_f0_appx[..., None, None] + (
        source_token_grads - parent_from_source_grads[..., None]
    )

    n_swaps = (parent_ids[:, None] != source_ids).sum(axis=-1)
    include = (
        (n_swaps <= max_swaps)
        if max_swaps is not None
        else torch.ones(f_child_from_source.shape[:2], device=device)
    )
    beta0_prediction = (f_child_from_source * include[..., None, None]).sum(dim=1) / (
        include.sum(dim=1)[:, None, None]
    )
    loss_prediction = beta * f_child + (1 - beta) * beta0_prediction

    # in typical GCG:
    # loss_prediction = state.token_grads
    return loss_prediction


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


def gcg(
    cache_run: Callable,
    model: torch.nn.Module,
    tokenizer: transformers.PreTrainedTokenizer,
    seq_len: int = 12,
    iters: int = 300,
    batch_size: int = 256,
    topk: int = 512,
    x_penalty_min: float = None,
    x_penalty_max: float = None,
    seed: int = 0,
    initial_ids: torch.Tensor = None,
    history: History = None,
    catch_keyboard_interrupt: bool = False,
    callback: Union[Callable, bool] = None,
    always_recompute_gradients: bool = False,
):
    """GCG is a special case of EPO where the population size is 1."""
    # TODO: check that this is up to date with latest EPO interface.
    return epo_v2(
        cache_run,
        model,
        tokenizer,
        seq_len=seq_len,
        population_size=1,
        iters=iters,
        explore_per_pop=batch_size,
        batch_size=batch_size,
        topk=topk,
        mutation_method="gradient",
        x_penalty_min=x_penalty_min,
        x_penalty_max=x_penalty_max,
        seed=seed,
        initial_ids=initial_ids,
        history=history,
        catch_keyboard_interrupt=catch_keyboard_interrupt,
        callback=callback,
        always_recompute_gradients=always_recompute_gradients,
    )


########################################
# Private implementation details below here.
########################################


def cat_if_not_none(a, b):
    if a is None or b is None:
        return None
    else:
        return torch.cat((a, b), dim=0)


# based on https://github.com/llm-attacks/llm-attacks/blob/main/llm_attacks/gcg/gcg_attack.py
def token_grads(
    cache_run: Callable,
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    x_penalty: torch.Tensor,
    batch_size: int = 1,
    embedding_noise: float = None,
    include_hess: bool = 0,
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
    token_dhess = torch.empty(
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
            if embedding_noise is not None:
                inputs_embeds = inputs_embeds + torch.normal(
                    0.0,
                    embedding_noise,
                    inputs_embeds.shape,
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                )

            cache = cache_run(inputs_embeds=inputs_embeds)

            logits_offset = cache["logits"][:, :-1]
            this_xentropy = (
                -(torch.log_softmax(logits_offset, dim=-1) * one_hot[:, 1:])
                .sum(dim=-1)
                .mean(dim=-1)
            )

            this_loss = -cache["target"] + this_xentropy * x_penalty[i:imax]
            this_loss.sum().backward(create_graph=True)

            loss[i:imax] = this_loss
            target[i:imax] = cache["target"]
            xentropy[i:imax] = this_xentropy
            token_grads[i:imax] = one_hot.grad

            if include_hess > 0:
                g = one_hot.grad
                for i in range(include_hess):
                    z = torch.randint(0, 2, g.shape, device=model.device) * 2.0 - 1.0
                    Hz = torch.autograd.grad(
                        g, one_hot, grad_outputs=z, retain_graph=True
                    )[0]
                    token_dhess[i:imax] += (Hz * z / include_hess).cpu()
                token_dhess[i:imax] = torch.where(token_dhess > 0, token_dhess, 0)

            for k in cache:
                if k not in ["target", "logits"]:
                    e = cache[k]
                    if k not in extra:
                        extra[k] = torch.empty(
                            (input_ids.shape[0], *e.shape[1:]),
                            dtype=e.dtype,
                            device=e.device,
                        )
                    extra[k][i:imax] = e

            # important to zero out gradients here to release memory
            model.zero_grad()

    return State(input_ids, target, xentropy, token_grads, token_dhess, extra)


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
    cache_run: Callable,
    input_ids: torch.Tensor,
    batch_size: int,
):
    target = torch.empty(input_ids.shape[0], dtype=torch.float, device=input_ids.device)
    xentropy = torch.empty(
        input_ids.shape[0], dtype=torch.float, device=input_ids.device
    )
    extra = dict()
    for i in range(0, input_ids.shape[0], batch_size):
        imax = min(i + batch_size, input_ids.shape[0])
        mini_batch = cache_run(input_ids=input_ids[i:imax])
        target[i:imax] = mini_batch["target"]
        xentropy[i:imax] = calc_xentropy(mini_batch["logits"], input_ids[i:imax])

        for k in mini_batch:
            if k not in ["target", "logits"]:
                e = mini_batch[k]
                if k not in extra:
                    extra[k] = torch.empty(
                        (input_ids.shape[0], *e.shape[1:]),
                        dtype=e.dtype,
                        device=e.device,
                    )
                extra[k][i:imax] = e

    return State(input_ids, target, xentropy, None, None, extra)


def pareto_callback(cache_run, model, tokenizer):
    def f(i, state, last_runtime, history, final=False):
        if last_runtime is not None:
            print("runtime: {:.2f} seconds".format(last_runtime))
        print(f"\nbeginning step {i}, current pareto frontier prompts:")
        last_idx = None

        Xvs = torch.exp(torch.linspace(np.log(0.001), np.log(100), 400)).to(
            model.device
        )
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
