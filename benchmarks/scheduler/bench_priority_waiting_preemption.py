# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only scheduler simulation of high-priority queueing delay at max_num_seqs.

Fills the running queue with low-priority long requests, injects high-priority
requests, then measures scheduling steps until each high-priority request is
scheduled, preemption counts, and the decode-token throughput of the
low-priority workload.

This does not run a model. It drives Scheduler.schedule() and
update_from_output() the same way tests/v1/core does.

Usage:
    VLLM_TARGET_DEVICE=cpu PYTHONPATH=. python \\
        benchmarks/scheduler/bench_priority_waiting_preemption.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from tests.v1.core.test_scheduler import create_requests_with_priority
from tests.v1.core.utils import create_scheduler
from vllm.v1.outputs import ModelRunnerOutput


def _apply_decode(scheduler, output, token_id: int = 99) -> None:
    req_ids = list(output.num_scheduled_tokens)
    if not req_ids:
        return
    model_output = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )
    scheduler.update_from_output(output, model_output)


def _total_preemptions(scheduler) -> int:
    return sum(req.num_preemptions for req in scheduler.requests.values())


@dataclass
class ScenarioResult:
    name: str
    max_num_seqs: int
    n_low: int
    n_high: int
    low_priority: int
    high_priority: int
    low_max_tokens: int
    steps_until_each_high: list[int | None]
    all_high_scheduled_step: int | None
    preemptions_at_injection: int
    preemptions_total: int
    low_tokens_after_injection: int
    high_tokens_after_injection: int
    window_steps: int
    elapsed_s: float
    policy: str


def run_scenario(
    *,
    name: str,
    max_num_seqs: int,
    n_low: int,
    n_high: int,
    low_priority: int,
    high_priority: int,
    low_max_tokens: int,
    high_max_tokens: int,
    window_steps: int,
    policy: str = "priority",
    prompt_tokens: int = 8,
) -> ScenarioResult:
    scheduler = create_scheduler(
        scheduling_policy=policy,  # type: ignore[arg-type]
        skip_tokenizer_init=True,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=8192,
        num_blocks=10000,
        block_size=16,
    )

    low_reqs = create_requests_with_priority(
        num_requests=n_low,
        priorities=[low_priority] * n_low,
        arrival_times=[float(i) for i in range(n_low)],
        num_tokens=prompt_tokens,
        max_tokens=low_max_tokens,
        req_ids=[f"low{i}" for i in range(n_low)],
    )
    for req in low_reqs:
        scheduler.add_request(req)

    # Admit the low-priority fill and take one decode step so they occupy
    # running slots the way a live server would before interactive traffic.
    output = scheduler.schedule()
    _apply_decode(scheduler, output)

    high_reqs = create_requests_with_priority(
        num_requests=n_high,
        priorities=[high_priority] * n_high,
        arrival_times=[float(n_low + i) for i in range(n_high)],
        num_tokens=prompt_tokens,
        max_tokens=high_max_tokens,
        req_ids=[f"high{i}" for i in range(n_high)],
    )
    for req in high_reqs:
        scheduler.add_request(req)

    high_ids = [req.request_id for req in high_reqs]
    steps_until: list[int | None] = [None] * n_high
    preemptions_at_injection = 0
    low_tokens = 0
    high_tokens = 0
    all_high_step: int | None = None

    t0 = time.perf_counter()
    for step in range(1, window_steps + 1):
        output = scheduler.schedule()
        if step == 1:
            preemptions_at_injection = _total_preemptions(scheduler)

        scheduled = output.num_scheduled_tokens
        for i, hid in enumerate(high_ids):
            if steps_until[i] is None and hid in scheduled:
                steps_until[i] = step
        if all_high_step is None and all(s is not None for s in steps_until):
            all_high_step = step

        for rid, ntok in scheduled.items():
            if rid.startswith("high"):
                high_tokens += ntok
            elif rid.startswith("low"):
                low_tokens += ntok

        _apply_decode(scheduler, output)

        # Stop early once every high-priority request has been scheduled at
        # least once and we have counted the injection-step preemptions.
        # Keep running the full window so throughput is comparable.
    elapsed_s = time.perf_counter() - t0

    return ScenarioResult(
        name=name,
        max_num_seqs=max_num_seqs,
        n_low=n_low,
        n_high=n_high,
        low_priority=low_priority,
        high_priority=high_priority,
        low_max_tokens=low_max_tokens,
        steps_until_each_high=steps_until,
        all_high_scheduled_step=all_high_step,
        preemptions_at_injection=preemptions_at_injection,
        preemptions_total=_total_preemptions(scheduler),
        low_tokens_after_injection=low_tokens,
        high_tokens_after_injection=high_tokens,
        window_steps=window_steps,
        elapsed_s=elapsed_s,
        policy=policy,
    )


def result_to_markdown(result: ScenarioResult) -> str:
    steps = ", ".join(
        str(s) if s is not None else "never" for s in result.steps_until_each_high
    )
    all_high = (
        str(result.all_high_scheduled_step)
        if result.all_high_scheduled_step is not None
        else "never"
    )
    return (
        f"### {result.name}\n"
        f"\n"
        f"- policy: `{result.policy}`\n"
        f"- max_num_seqs: {result.max_num_seqs}\n"
        f"- low-priority fill: {result.n_low} requests "
        f"(priority {result.low_priority}, max_tokens {result.low_max_tokens})\n"
        f"- high-priority inject: {result.n_high} requests "
        f"(priority {result.high_priority})\n"
        f"- window after inject: {result.window_steps} schedule() steps\n"
        f"- steps until each high-priority request is first scheduled: {steps}\n"
        f"- step when all high-priority requests have been scheduled: {all_high}\n"
        f"- preemptions on the injection step: {result.preemptions_at_injection}\n"
        f"- preemptions by end of window: {result.preemptions_total}\n"
        f"- low-priority tokens scheduled in window: "
        f"{result.low_tokens_after_injection}\n"
        f"- high-priority tokens scheduled in window: "
        f"{result.high_tokens_after_injection}\n"
        f"- wall time (CPU schedule+update loop): {result.elapsed_s:.4f}s\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--label",
        default="current",
        help="Label for this binary (e.g. main or pr-54530).",
    )
    args = parser.parse_args()

    scenarios = [
        dict(
            name="minimal max_num_seqs=2 (one high-priority waiter)",
            max_num_seqs=2,
            n_low=2,
            n_high=1,
            low_priority=10,
            high_priority=0,
            low_max_tokens=40,
            high_max_tokens=8,
            window_steps=40,
        ),
        dict(
            name="production-like max_num_seqs=8 (two interactive waiters)",
            max_num_seqs=8,
            n_low=8,
            n_high=2,
            low_priority=10,
            high_priority=0,
            low_max_tokens=40,
            high_max_tokens=8,
            window_steps=40,
        ),
        dict(
            name="fcfs control max_num_seqs=8 (priority field ignored)",
            max_num_seqs=8,
            n_low=8,
            n_high=2,
            low_priority=10,
            high_priority=0,
            low_max_tokens=40,
            high_max_tokens=8,
            window_steps=40,
            policy="fcfs",
        ),
    ]

    print(f"# Priority waiting-preemption scheduler benchmark ({args.label})")
    print()
    print(
        "CPU-only Scheduler.schedule() simulation. No GPU engine run. "
        "Numbers are scheduling steps, not wall-clock serving latency."
    )
    print()

    for spec in scenarios:
        result = run_scenario(**spec)
        print(result_to_markdown(result))


if __name__ == "__main__":
    main()
