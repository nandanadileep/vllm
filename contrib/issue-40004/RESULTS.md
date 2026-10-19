# Priority waiting-queue preemption: tests and benchmark

Measured against current upstream `main` (`c1dc2e9cf9`) and a port of
[#54530](https://github.com/vllm-project/vllm/pull/54530) onto that main.
No GPU was available, so there is no real-engine run (e.g. facebook/opt-125m).

CPU-only command used for tests:

```bash
VLLM_TARGET_DEVICE=cpu PYTHONPATH=. python -m pytest \
  tests/v1/core/test_priority_waiting_preemption.py -v --noconftest
```

CPU-only command used for the scheduler simulation:

```bash
VLLM_TARGET_DEVICE=cpu PYTHONPATH=. python \
  benchmarks/scheduler/bench_priority_waiting_preemption.py --label <label>
```

## Rebase notes for #54530

A direct `git rebase` of `refs/pull/54530/head` onto current main does not
apply cleanly. The waiting loop has moved on: admission is gated by
`max_num_active_reqs` (not `max_num_running_reqs`), skipped waiters go through
`skip_request` / `kv_holding_waiting`, and the pause check is
`PauseState.PAUSED_ALL`.

The port kept the #54530 design and resolved those conflicts by:

1. Using `queue_full = num_running >= self.max_num_active_reqs`.
2. Leaving FCFS on the original unconditional break.
3. Inserting the preemption attempt after the blocked-status, stale-output,
   and `max_loras` checks, then `continue` to reschedule the same waiter.
4. Rolling back the victim from `scheduled_running_reqs` /
   `scheduled_new_reqs` / `scheduled_resumed_reqs`, restoring token/input
   budget, spec-decode tokens, encoder budget, and LoRA accounting.

#45558 was not used. #54530 was portable once the waiting-loop symbols were
updated.

## Test matrix

| Test | Behavior pinned | main | PR port |
| --- | --- | --- | --- |
| `test_priority_waiting_preempts_at_max_num_seqs` (v1 and v2) | Core #40004: priority-0 waiter preempts a priority-5 runner at `max_num_seqs=2` | FAIL | PASS |
| `test_priority_waiting_no_preempt_when_lower_priority` | Lower-priority waiter does not evict a higher-priority runner | PASS | PASS |
| `test_priority_waiting_equal_priority_no_ping_pong` | Equal priority plus earlier arrival does not preempt (no ping-pong) | PASS | PASS |
| `test_fcfs_no_preempt_at_max_num_seqs` | FCFS never preempts at a full queue | PASS | PASS |
| `test_priority_waiting_preempts_lowest_priority_latest_arrival_victim` | Victim is lowest priority, latest arrival | FAIL | PASS |
| `test_priority_waiting_multiple_high_priority_waiters` | Two interactive waiters preempt two background runners in one step (`max_num_seqs=8`) | FAIL | PASS |
| `test_priority_waiting_preempted_request_resumes_after_high_priority_finishes` | Preempted runner resumes after the waiter finishes | FAIL | PASS |
| `test_priority_waiting_preempted_request_waits_behind_higher_priority` | Preempted runner stays behind remaining higher-priority waiters | FAIL | PASS |
| `test_priority_waiting_same_step_rollback_excludes_victim` | Victim is dropped from this step's scheduled tokens | FAIL | PASS |
| `test_priority_waiting_preemption_keeps_shared_lora` | Shared LoRA id is kept when another scheduled request still uses it | FAIL | PASS |
| `test_priority_waiting_preemption_discards_unused_lora` | Unused victim LoRA id is dropped so a later waiter can use a new LoRA | FAIL | PASS |
| `test_priority_waiting_new_lora_can_preempt_unique_lora_runner` | New-LoRA waiter should still preempt a unique-LoRA runner at `max_loras` | FAIL | FAIL |
| `test_priority_waiting_skipped_new_lora_does_not_admit_lower_priority` | Skipping a new-LoRA high-priority waiter must not let a lower-priority same-LoRA waiter take the slot | FAIL | FAIL |
| `test_priority_waiting_no_preempt_when_waiter_cannot_fit_token_budget` | Do not evict if the waiter still cannot fit after restore (chunked prefill off) | PASS | FAIL |
| `test_priority_waiting_no_preempt_when_kv_alloc_still_fails` | Do not evict if KV allocation still fails after one victim | PASS | FAIL |
| `test_priority_waiting_async_scheduler_preempts_at_max_num_seqs` | AsyncScheduler uses the same waiting-loop cap | FAIL | PASS |

Core #40004 tests fail on main and pass on the PR port, as expected.

The four FAIL-on-PR rows are review findings, not test bugs.

## Review findings on the #54530 design

1. `max_loras` is checked before preemption. A higher-priority waiter with a
   new LoRA is skipped even when evicting the unique-LoRA runner would free a
   slot. In a mixed waiting queue, that skip then lets a lower-priority waiter
   that shares the runner's LoRA preempt and take the slot.
2. Preemption runs before token-budget / KV allocation. If the waiter still
   cannot be admitted, the victim is already gone and the step leaves an empty
   running slot. Main does not have this problem because it never preempts at
   `max_num_seqs`.

## Benchmark (CPU scheduler simulation)

Window is 40 `schedule()` steps after inject. Low-priority requests use
`max_tokens=40`; high-priority requests use `max_tokens=8`. Prompt length is 8
tokens. Steps are scheduler steps, not serving wall-clock.

### minimal `max_num_seqs=2` (one high-priority waiter)

| metric | main | PR port |
| --- | --- | --- |
| steps until high-priority first scheduled | 40 | 1 |
| preemptions on injection step | 0 | 1 |
| preemptions by end of window | 0 | 1 |
| low-priority tokens in window | 78 | 79 |
| high-priority tokens in window | 8 | 15 |
| wall time | 0.0022s | 0.0022s |

### production-like `max_num_seqs=8` (two interactive waiters)

| metric | main | PR port |
| --- | --- | --- |
| steps until each high-priority first scheduled | 40, 40 | 1, 1 |
| step when all high-priority have been scheduled | 40 | 1 |
| preemptions on injection step | 0 | 2 |
| preemptions by end of window | 0 | 2 |
| low-priority tokens in window | 312 | 314 |
| high-priority tokens in window | 16 | 30 |
| wall time | 0.0042s | 0.0042s |

On main this matches the #40004 report: `num_preemptions_total` stays 0 and
the interactive requests wait until a background request finishes
(`max_tokens=40`, so first schedule is at step 40 of the window). With the
fix they are scheduled on the injection step.

Low-priority token count in the 40-step window is essentially unchanged
(312 vs 314). High-priority work roughly doubles (16 vs 30) because those
requests start immediately instead of at the end of the window.

### FCFS control `max_num_seqs=8` (priority field ignored)

| metric | main | PR port |
| --- | --- | --- |
| steps until each high-priority first scheduled | 40, 40 | 40, 40 |
| preemptions | 0 | 0 |
| low-priority tokens in window | 312 | 312 |
| high-priority tokens in window | 16 | 16 |

FCFS is unchanged by the port.

## GPU engine run

Not run. This environment has no GPU.
