# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Edge-case regression tests for waiting-queue preemption at max_num_seqs.

Pins the #40004 behavior: under scheduling_policy="priority", a strictly
higher-priority WAITING request must be able to displace a lower-priority
RUNNING request when the running queue is full. Main currently breaks the
waiting loop unconditionally at max_num_active_reqs, so the core tests fail
there and should pass with the #54530-style fix.

Each test states the behavior it pins. Tests that encode review findings
(desired behavior the current fix may miss) are marked with a comment
starting with REVIEW.
"""

from __future__ import annotations

import pytest

from vllm.config import LoRAConfig
from vllm.lora.request import LoRARequest
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

from .test_scheduler import create_requests_with_priority
from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


def _priority_scheduler(**kwargs):
    kwargs.setdefault("scheduling_policy", "priority")
    kwargs.setdefault("skip_tokenizer_init", True)
    kwargs.setdefault("max_num_batched_tokens", 200)
    kwargs.setdefault("num_blocks", 100)
    kwargs.setdefault("block_size", 16)
    return create_scheduler(**kwargs)


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


def _fill_running(scheduler, requests):
    for req in requests:
        scheduler.add_request(req)
    output = scheduler.schedule()
    assert {req.request_id for req in scheduler.running} == {
        req.request_id for req in requests
    }
    _apply_decode(scheduler, output)
    return output


def _attach_lora(requests, lora_int_ids) -> None:
    for request, lora_int_id in zip(requests, lora_int_ids, strict=True):
        request.lora_request = LoRARequest(
            lora_name=f"lora{lora_int_id}",
            lora_int_id=lora_int_id,
            lora_path=f"/tmp/lora{lora_int_id}",
        )


def _running_ids(scheduler) -> set[str]:
    return {req.request_id for req in scheduler.running}


@pytest.mark.parametrize("use_v2_model_runner", [False, True])
def test_priority_waiting_preempts_at_max_num_seqs(use_v2_model_runner):
    """A strictly higher-priority WAITING request preempts the lowest-priority
    RUNNING request when the running queue is full (issue #40004).

    max_num_seqs=2, no KV pressure: two priority-5 runners fill the queue,
    then a priority-0 waiter arrives. The later-arrived runner is the victim,
    the waiter is scheduled this step, and the victim is rolled back out of
    this step's scheduled tokens.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        use_v2_model_runner=use_v2_model_runner,
    )
    lo_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=10,
        req_ids=["lo1", "lo2"],
    )
    _fill_running(scheduler, lo_requests)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=10,
        req_ids=["hi"],
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    scheduled_ids = set(output.num_scheduled_tokens)
    assert "hi" in scheduled_ids
    assert "lo1" in scheduled_ids
    assert "lo2" not in scheduled_ids
    assert scheduler.requests["lo2"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["lo2"].num_preemptions == 1
    assert _running_ids(scheduler) == {"lo1", "hi"}


def test_priority_waiting_no_preempt_when_lower_priority():
    """A lower-priority waiter must not displace a higher-priority runner
    just because the running queue is full.
    """
    scheduler = _priority_scheduler(max_num_seqs=2)
    hi_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 1],
        arrival_times=[1.0, 2.0],
        num_tokens=10,
        req_ids=["hi1", "hi2"],
    )
    _fill_running(scheduler, hi_requests)

    lo_req = create_requests_with_priority(
        num_requests=1,
        priorities=[5],
        arrival_times=[3.0],
        num_tokens=10,
        req_ids=["lo"],
    )[0]
    scheduler.add_request(lo_req)

    output = scheduler.schedule()
    assert "lo" not in output.num_scheduled_tokens
    assert scheduler.requests["lo"].status == RequestStatus.WAITING
    assert _running_ids(scheduler) == {"hi1", "hi2"}
    assert scheduler.requests["hi1"].num_preemptions == 0
    assert scheduler.requests["hi2"].num_preemptions == 0


def test_priority_waiting_equal_priority_no_ping_pong():
    """Equal-priority waiters must not preempt, even with an earlier
    arrival_time.

    Request.__lt__ falls back to arrival_time, and a preempted request keeps
    its original arrival_time, so a non-strict comparison would ping-pong
    across steps. Only a numerically lower priority may preempt.
    """
    scheduler = _priority_scheduler(max_num_seqs=2)
    running_reqs = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[2.0, 3.0],
        num_tokens=10,
        req_ids=["r1", "r2"],
    )
    _fill_running(scheduler, running_reqs)

    waiting_req = create_requests_with_priority(
        num_requests=1,
        priorities=[5],
        arrival_times=[1.0],
        num_tokens=10,
        req_ids=["w1"],
    )[0]
    scheduler.add_request(waiting_req)

    for _ in range(3):
        output = scheduler.schedule()
        _apply_decode(scheduler, output)
        assert "w1" not in output.num_scheduled_tokens
        assert scheduler.requests["r1"].status == RequestStatus.RUNNING
        assert scheduler.requests["r2"].status == RequestStatus.RUNNING
        assert scheduler.requests["w1"].status == RequestStatus.WAITING
        assert len(scheduler.running) == 2


def test_fcfs_no_preempt_at_max_num_seqs():
    """Queue-full waiting-queue preemption is exclusive to the priority
    policy. FCFS never evicts a runner for a waiter, even if the waiter
    has a smaller priority value.
    """
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=200,
        num_blocks=100,
        skip_tokenizer_init=True,
        scheduling_policy="fcfs",
    )
    fcfs_requests = create_requests(num_requests=2, num_tokens=10, req_ids=["f1", "f2"])
    _fill_running(scheduler, fcfs_requests)

    waiting_request = create_requests(num_requests=1, num_tokens=10, req_ids=["f3"])[0]
    waiting_request.priority = 0
    scheduler.add_request(waiting_request)

    output = scheduler.schedule()
    assert waiting_request.request_id not in output.num_scheduled_tokens
    assert waiting_request.status == RequestStatus.WAITING
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 1


def test_priority_waiting_preempts_lowest_priority_latest_arrival_victim():
    """When several runners could be evicted, the victim is the lowest
    priority (largest value), breaking ties with the latest arrival_time.
    """
    scheduler = _priority_scheduler(max_num_seqs=3)
    runners = create_requests_with_priority(
        num_requests=3,
        priorities=[4, 6, 6],
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=10,
        req_ids=["mid", "lo_early", "lo_late"],
    )
    _fill_running(scheduler, runners)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[4.0],
        num_tokens=10,
        req_ids=["hi"],
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    assert "hi" in output.num_scheduled_tokens
    assert "lo_late" not in output.num_scheduled_tokens
    assert scheduler.requests["lo_late"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["mid"].status == RequestStatus.RUNNING
    assert scheduler.requests["lo_early"].status == RequestStatus.RUNNING
    assert _running_ids(scheduler) == {"mid", "lo_early", "hi"}


def test_priority_waiting_multiple_high_priority_waiters():
    """Several strictly higher-priority waiters each preempt one runner in
    the same step, matching the production report (10 requests, max_num_seqs=8,
    interactive priority 0 vs background priority 10).
    """
    scheduler = _priority_scheduler(
        max_num_seqs=8,
        max_num_batched_tokens=8192,
    )
    background = create_requests_with_priority(
        num_requests=8,
        priorities=[10] * 8,
        arrival_times=[float(i) for i in range(8)],
        num_tokens=10,
        req_ids=[f"bg{i}" for i in range(8)],
    )
    _fill_running(scheduler, background)

    interactive = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 0],
        arrival_times=[8.0, 9.0],
        num_tokens=10,
        req_ids=["int0", "int1"],
    )
    for req in interactive:
        scheduler.add_request(req)

    output = scheduler.schedule()
    scheduled_ids = set(output.num_scheduled_tokens)
    assert {"int0", "int1"} <= scheduled_ids
    # Latest-arrived background requests are the victims.
    assert "bg6" not in scheduled_ids
    assert "bg7" not in scheduled_ids
    assert scheduler.requests["bg6"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["bg7"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["bg6"].num_preemptions == 1
    assert scheduler.requests["bg7"].num_preemptions == 1
    assert _running_ids(scheduler) == {
        "bg0",
        "bg1",
        "bg2",
        "bg3",
        "bg4",
        "bg5",
        "int0",
        "int1",
    }


def test_priority_waiting_preempted_request_resumes_after_high_priority_finishes():
    """A request preempted for a higher-priority waiter is re-queued and
    scheduled again once that waiter finishes, without needing KV pressure.
    """
    scheduler = _priority_scheduler(max_num_seqs=2)
    lo_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=10,
        req_ids=["lo1", "lo2"],
    )
    _fill_running(scheduler, lo_requests)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=10,
        req_ids=["hi"],
        max_tokens=4,
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    assert scheduler.requests["lo2"].status == RequestStatus.PREEMPTED
    _apply_decode(scheduler, output)
    scheduler.finish_requests("hi", RequestStatus.FINISHED_STOPPED)

    output = scheduler.schedule()
    assert len(scheduler.waiting) == 0
    assert scheduler.requests["lo2"].status == RequestStatus.RUNNING
    assert _running_ids(scheduler) == {"lo1", "lo2"}
    assert "lo2" in output.num_scheduled_tokens


def test_priority_waiting_preempted_request_waits_behind_higher_priority():
    """A preempted runner re-enters the waiting heap ordered by
    (priority, original arrival_time). A still-waiting higher-priority
    request is admitted first when a slot frees; the victim does not jump
    ahead of it.
    """
    scheduler = _priority_scheduler(max_num_seqs=2)
    runners = create_requests_with_priority(
        num_requests=2,
        priorities=[8, 9],
        arrival_times=[1.0, 2.0],
        num_tokens=10,
        req_ids=["mid", "lo"],
    )
    _fill_running(scheduler, runners)

    waiters = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 1],
        arrival_times=[3.0, 4.0],
        num_tokens=10,
        req_ids=["hi", "next_hi"],
    )
    for req in waiters:
        scheduler.add_request(req)

    output = scheduler.schedule()
    # hi takes the free comparison against lo; next_hi then preempts mid.
    assert {"hi", "next_hi"} <= set(output.num_scheduled_tokens)
    assert scheduler.requests["lo"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["mid"].status == RequestStatus.PREEMPTED

    _apply_decode(scheduler, output)
    scheduler.finish_requests("hi", RequestStatus.FINISHED_STOPPED)

    output = scheduler.schedule()
    # The remaining running request is next_hi (priority 1). The freed slot
    # must go to mid (priority 8) before lo (priority 9).
    assert "mid" in output.num_scheduled_tokens
    assert "lo" not in output.num_scheduled_tokens
    assert scheduler.requests["lo"].status == RequestStatus.PREEMPTED
    assert _running_ids(scheduler) == {"next_hi", "mid"}


def test_priority_waiting_same_step_rollback_excludes_victim():
    """The victim's running-pass scheduling is rolled back in the same step:
    it must not appear in num_scheduled_tokens, and its token budget must be
    available for the waiter.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=30,
    )
    lo_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=10,
        req_ids=["lo1", "lo2"],
    )
    _fill_running(scheduler, lo_requests)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=10,
        req_ids=["hi"],
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    assert "lo2" not in output.num_scheduled_tokens
    assert output.num_scheduled_tokens.get("hi", 0) > 0
    assert "lo2" not in output.scheduled_cached_reqs.req_ids


def test_priority_waiting_preemption_keeps_shared_lora():
    """Preempting a runner must not drop its LoRA ID from this step's
    accounting while another request scheduled this step still uses it.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        lora_config=LoRAConfig(max_loras=1, max_lora_rank=8),
    )
    req_y = create_requests_with_priority(
        num_requests=1,
        priorities=[7],
        arrival_times=[0.0],
        num_tokens=10,
        req_ids=["y"],
    )
    _attach_lora(req_y, [1])
    _fill_running(scheduler, req_y)

    waiters = create_requests_with_priority(
        num_requests=3,
        priorities=[0, 1, 6],
        arrival_times=[1.0, 2.0, 3.0],
        num_tokens=10,
        req_ids=["hi", "mid", "x"],
    )
    _attach_lora(waiters, [1, 2, 1])
    for req in waiters:
        scheduler.add_request(req)

    output = scheduler.schedule()
    scheduled_ids = set(output.num_scheduled_tokens)
    assert {"hi", "x"} <= scheduled_ids
    assert "mid" not in scheduled_ids
    assert "y" not in scheduled_ids
    assert scheduler.requests["y"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["mid"].status == RequestStatus.WAITING
    assert _running_ids(scheduler) == {"hi", "x"}


def test_priority_waiting_preemption_discards_unused_lora():
    """Preempting a runner frees its LoRA slot when no still-scheduled
    request uses it, so a later waiter with a new LoRA can be admitted.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        lora_config=LoRAConfig(max_loras=2, max_lora_rank=8),
    )
    runners = create_requests_with_priority(
        num_requests=2,
        priorities=[7, 8],
        arrival_times=[0.0, 1.0],
        num_tokens=10,
        req_ids=["a", "b"],
    )
    _attach_lora(runners, [1, 2])
    _fill_running(scheduler, runners)

    waiters = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 1],
        arrival_times=[2.0, 3.0],
        num_tokens=10,
        req_ids=["hi", "mid"],
    )
    _attach_lora(waiters, [1, 3])
    for req in waiters:
        scheduler.add_request(req)

    output = scheduler.schedule()
    scheduled_ids = set(output.num_scheduled_tokens)
    assert {"hi", "mid"} <= scheduled_ids
    assert "a" not in scheduled_ids
    assert "b" not in scheduled_ids
    assert scheduler.requests["a"].status == RequestStatus.PREEMPTED
    assert scheduler.requests["b"].status == RequestStatus.PREEMPTED
    assert _running_ids(scheduler) == {"hi", "mid"}


def test_priority_waiting_new_lora_can_preempt_unique_lora_runner():
    """REVIEW: a higher-priority waiter with a new LoRA must still be able
    to preempt a unique-LoRA runner at max_loras. Checking max_loras before
    considering preemption skips the waiter even though evicting the victim
    would free a LoRA slot.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=1,
        lora_config=LoRAConfig(max_loras=1, max_lora_rank=8),
    )
    lo = create_requests_with_priority(
        num_requests=1,
        priorities=[10],
        arrival_times=[1.0],
        num_tokens=10,
        req_ids=["lo"],
    )
    _attach_lora(lo, [1])
    _fill_running(scheduler, lo)

    hi = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[2.0],
        num_tokens=10,
        req_ids=["hi"],
    )
    _attach_lora(hi, [2])
    scheduler.add_request(hi[0])

    output = scheduler.schedule()
    assert "hi" in output.num_scheduled_tokens
    assert "lo" not in output.num_scheduled_tokens
    assert scheduler.requests["lo"].status == RequestStatus.PREEMPTED
    assert _running_ids(scheduler) == {"hi"}


def test_priority_waiting_skipped_new_lora_does_not_admit_lower_priority():
    """REVIEW: if a high-priority new-LoRA waiter is skipped by the
    max_loras check, a lower-priority waiter that shares the runner's LoRA
    must not then preempt and take the slot.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=1,
        lora_config=LoRAConfig(max_loras=1, max_lora_rank=8),
    )
    lo = create_requests_with_priority(
        num_requests=1,
        priorities=[10],
        arrival_times=[1.0],
        num_tokens=10,
        req_ids=["lo"],
    )
    _attach_lora(lo, [1])
    _fill_running(scheduler, lo)

    waiters = create_requests_with_priority(
        num_requests=2,
        priorities=[0, 5],
        arrival_times=[2.0, 3.0],
        num_tokens=10,
        req_ids=["hi", "mid"],
    )
    _attach_lora(waiters, [2, 1])
    for req in waiters:
        scheduler.add_request(req)

    output = scheduler.schedule()
    # hi (priority 0, LoRA 2) should take the slot. mid must not sneak in
    # because hi was skipped by a pre-preemption max_loras check.
    assert "hi" in output.num_scheduled_tokens
    assert "mid" not in output.num_scheduled_tokens
    assert scheduler.requests["lo"].status == RequestStatus.PREEMPTED


def test_priority_waiting_no_preempt_when_waiter_cannot_fit_token_budget():
    """REVIEW: do not evict a runner for a waiter that then cannot be
    admitted because chunked prefill is disabled and the remaining token
    budget is too small. Eviction would waste the victim's decode slot
    without scheduling the waiter.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=12,
        enable_chunked_prefill=False,
    )
    lo_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=4,
        req_ids=["lo1", "lo2"],
    )
    _fill_running(scheduler, lo_requests)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        # Larger than remaining budget even after the victim's 1 decode
        # token is restored, so the waiter still cannot be admitted.
        num_tokens=20,
        req_ids=["hi"],
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    assert "hi" not in output.num_scheduled_tokens
    assert scheduler.requests["hi"].status == RequestStatus.WAITING
    assert scheduler.requests["lo1"].status == RequestStatus.RUNNING
    assert scheduler.requests["lo2"].status == RequestStatus.RUNNING
    assert scheduler.requests["lo1"].num_preemptions == 0
    assert scheduler.requests["lo2"].num_preemptions == 0
    assert len(scheduler.running) == 2


def test_priority_waiting_no_preempt_when_kv_alloc_still_fails():
    """REVIEW: do not evict a runner for a waiter whose KV allocation still
    fails after a single victim is freed. The waiting path does not loop
    KV-pressure preemption; a one-shot eviction would leave an empty slot
    without admitting the waiter.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        num_blocks=4,
        block_size=16,
        max_num_batched_tokens=200,
    )
    lo_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=8,
        req_ids=["lo1", "lo2"],
        block_size=16,
    )
    _fill_running(scheduler, lo_requests)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=40,
        req_ids=["hi"],
        block_size=16,
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    assert "hi" not in output.num_scheduled_tokens
    assert scheduler.requests["hi"].status == RequestStatus.WAITING
    assert scheduler.requests["lo1"].num_preemptions == 0
    assert scheduler.requests["lo2"].num_preemptions == 0
    assert len(scheduler.running) == 2


def test_priority_waiting_async_scheduler_preempts_at_max_num_seqs():
    """Async scheduling uses the same waiting-loop admission cap. A
    higher-priority waiter must still preempt at max_num_seqs.
    """
    scheduler = _priority_scheduler(
        max_num_seqs=2,
        async_scheduling=True,
    )
    lo_requests = create_requests_with_priority(
        num_requests=2,
        priorities=[5, 5],
        arrival_times=[1.0, 2.0],
        num_tokens=10,
        req_ids=["lo1", "lo2"],
    )
    _fill_running(scheduler, lo_requests)

    hi_req = create_requests_with_priority(
        num_requests=1,
        priorities=[0],
        arrival_times=[3.0],
        num_tokens=10,
        req_ids=["hi"],
    )[0]
    scheduler.add_request(hi_req)

    output = scheduler.schedule()
    assert "hi" in output.num_scheduled_tokens
    assert scheduler.requests["lo2"].status == RequestStatus.PREEMPTED
    assert _running_ids(scheduler) == {"lo1", "hi"}
