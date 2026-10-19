# ruff: noqa: E402
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise #61905 with real inference and controlled connector load failure.

The worker runs real model forwards. Only the example connector's failure is
injected. Cancellation and interleaving use the in-process engine's public
request methods and its normal batch queue. No scheduler state is forced.
"""

import os

os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

import vllm
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.inputs import TokensPrompt

SOURCE = Path(vllm.__file__).resolve().parents[1]
sys.path.insert(
    0, str(SOURCE / "examples/disaggregated/kv_load_failure_recovery_offline")
)
import rg_reuse_connector as probe

OWNER_TOKENS = [1000 + i % 31 for i in range(114)]
OTHER_TOKENS = [2000 + i % 37 for i in range(96)]


def snapshot(request):
    return {"status": request.status.name, "computed": request.num_computed_tokens}


def output_record(output):
    if output is None:
        return None
    completion = output.outputs[0]
    chosen_logprobs = []
    for token, logprobs in zip(completion.token_ids, completion.logprobs or []):
        chosen_logprobs.append(logprobs[token].logprob if token in logprobs else None)
    return {
        "tokens": list(completion.token_ids),
        "chosen_logprobs": chosen_logprobs,
        "finish_reason": completion.finish_reason,
        "num_cached_tokens": output.num_cached_tokens,
        "text": completion.text,
    }


def create_llm(args, connector=None):
    return LLM(
        model=args.model,
        enforce_eager=True,
        seed=0,
        dtype="float16",
        enable_prefix_caching=True,
        async_scheduling=args.mode != "store",
        max_model_len=128,
        max_num_batched_tokens=256,
        max_num_seqs=2,
        gpu_memory_utilization=0.15,
        block_size=16,
        num_gpu_blocks_override=9,
        kv_transfer_config=connector,
    )


def run_trial(llm, args, trial):
    engine = llm.llm_engine
    core = engine.engine_core.engine_core
    scheduler = core.scheduler
    assert core.batch_queue_size == 2
    assert type(core.model_executor).__name__ == "UniProcExecutor"
    assert len(scheduler.kv_cache_config.kv_cache_groups) == 1
    probe.STATE.clear()
    probe.FAIL_LOAD = args.load_fails
    llm.reset_prefix_cache()

    owner_id = engine.add_request(
        f"owner-{trial}",
        TokensPrompt(prompt_token_ids=OWNER_TOKENS),
        SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
    )
    probe.OWNER = owner_id
    owner = scheduler.requests[owner_id]
    first_outputs = engine.step()
    assert not first_outputs, "The first result must remain queued"
    assert len(core.batch_queue) == 1
    assert probe.STATE["external_tokens"] == 112
    assert probe.STATE["load_steps"] == 1
    owner_blocks = list(probe.STATE["owner_blocks"])
    assert len(owner_blocks) == 8

    # The result remains in the normal batch queue while this client cancels A.
    engine.abort_request([owner_id], internal=True)
    assert owner.is_finished()
    other_id = engine.add_request(
        f"unrelated-{trial}",
        TokensPrompt(prompt_token_ids=OTHER_TOKENS),
        SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=20),
    )
    other = scheduler.requests[other_id]
    events = []
    scheduled_other_blocks = []
    original_update = scheduler.update_from_output
    original_schedule = scheduler.schedule

    def observe_schedule(*positional, **keywords):
        output = original_schedule(*positional, **keywords)
        if other_id in output.num_scheduled_tokens:
            blocks = list(scheduler.kv_cache_manager.get_block_ids(other_id)[0])
            scheduled_other_blocks.append(blocks)
        return output

    def observe_update(scheduler_output, model_output):
        event = None
        connector_output = model_output.kv_connector_output
        if owner_id in scheduler_output.num_scheduled_tokens:
            event = {
                "before": snapshot(other),
                "invalid_blocks": sorted(
                    connector_output.invalid_block_ids if connector_output else ()
                ),
                "other_blocks_before_report": list(
                    scheduler.kv_cache_manager.get_block_ids(other_id)[0]
                ),
            }
        result = original_update(scheduler_output, model_output)
        if event is not None:
            event["after"] = snapshot(other)
            events.append(event)
        return result

    scheduler.schedule = observe_schedule
    scheduler.update_from_output = observe_update
    final = None
    first_token_s = None
    stopped_after_report = False
    started = time.perf_counter()
    steps = 0
    try:
        while engine.has_unfinished_requests() and steps < 64:
            for output in engine.step():
                if output.request_id == f"unrelated-{trial}":
                    if first_token_s is None and output.outputs[0].token_ids:
                        first_token_s = time.perf_counter() - started
                    if output.finished:
                        final = output
            steps += 1
            if args.stop_after_report and events and not stopped_after_report:
                assert events[0]["before"] != events[0]["after"]
                engine.abort_request([other_id], internal=True)
                stopped_after_report = True
        if engine.has_unfinished_requests():
            raise RuntimeError("Unrelated request did not finish after 64 engine steps")
        # Retire any extra asynchronously scheduled output before the next trial.
        for _ in range(16):
            if not core.batch_queue and not scheduler.has_requests():
                break
            engine.step()
    finally:
        scheduler.schedule = original_schedule
        scheduler.update_from_output = original_update

    assert len(events) == 1
    event = events[0]
    affected = event["before"] != event["after"]
    reused_before_report = bool(
        set(event["invalid_blocks"]) & set(event["other_blocks_before_report"])
    )
    eventually_reused = any(
        set(owner_blocks) & set(ids) for ids in scheduled_other_blocks
    )
    result = {
        "trial": trial,
        "defer_block_free": scheduler.defer_block_free,
        "load_fails": args.load_fails,
        "policy": args.policy,
        "external_tokens": probe.STATE["external_tokens"],
        "load_steps": probe.STATE["load_steps"],
        "owner_blocks": owner_blocks,
        "report": event,
        "unrelated_affected_by_old_report": affected,
        "reused_before_failure_report": reused_before_report,
        "eventually_reused": eventually_reused,
        "unrelated_output": output_record(final),
        "ttft_s": first_token_s,
        "elapsed_s": time.perf_counter() - started,
        "steps": steps,
        "stopped_after_wrong_attribution": stopped_after_report,
        "free_blocks_after_drain": (
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        ),
        "deferred_frees_after_drain": len(scheduler.deferred_frees),
    }
    if args.expect_fixed or not args.load_fails:
        assert not affected, result
        assert final is not None, result
        assert final.outputs[0].finish_reason == "length", result
    if args.expect_fixed:
        assert not reused_before_report, result
        assert eventually_reused, result
        assert result["free_blocks_after_drain"] == 8, result
        assert result["deferred_frees_after_drain"] == 0, result
    elif args.load_fails:
        assert affected and reused_before_report, result
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["store", "oracle", "trial"])
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=["fail", "recompute"], default="fail")
    parser.add_argument("--load-fails", action="store_true")
    parser.add_argument("--expect-fixed", action="store_true")
    parser.add_argument("--stop-after-report", action="store_true")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    if args.stop_after_report:
        assert args.mode == "trial" and args.load_fails and not args.expect_fixed
    args.storage.mkdir(parents=True, exist_ok=True)
    record = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SOURCE, text=True
        ).strip(),
        "source_diff": subprocess.check_output(
            ["git", "diff", "--stat"], cwd=SOURCE, text=True
        ).strip(),
        "source_path": str(SOURCE),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "mode": args.mode,
        "model_runner_v2_env": os.environ.get("VLLM_USE_V2_MODEL_RUNNER"),
    }
    if args.mode == "store":
        connector = KVTransferConfig(
            kv_connector="ExampleConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"shared_storage_path": str(args.storage)},
        )
        llm = create_llm(args, connector)
        output = llm.generate(
            [TokensPrompt(prompt_token_ids=OWNER_TOKENS)],
            SamplingParams(temperature=0, max_tokens=1, ignore_eos=True),
        )[0]
        record["output"] = output_record(output)
    elif args.mode == "oracle":
        llm = create_llm(args)
        output = llm.generate(
            [TokensPrompt(prompt_token_ids=OTHER_TOKENS)],
            SamplingParams(temperature=0, max_tokens=8, ignore_eos=True, logprobs=20),
        )[0]
        assert output.num_cached_tokens == 0
        record["output"] = output_record(output)
    else:
        connector = KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_producer",
            kv_load_failure_policy=args.policy,
            kv_connector_extra_config={
                "connectors": [
                    {
                        "kv_connector": "ReuseProbeConnector",
                        "kv_connector_module_path": "rg_reuse_connector",
                        "kv_role": "kv_consumer",
                        "kv_connector_extra_config": {
                            "shared_storage_path": str(args.storage),
                            "async_load": False,
                        },
                    }
                ]
            },
        )
        llm = create_llm(args, connector)
        record["trials"] = [run_trial(llm, args, i) for i in range(args.trials)]
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print("RESULT", args.output, flush=True)


if __name__ == "__main__":
    main()
