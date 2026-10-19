# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fault injection using vLLM's upstream synchronous-load example connector."""

from load_recovery_example_connector import LoadRecoveryExampleConnector

from vllm.distributed.kv_transfer.kv_connector.v1.example_connector import (
    ExampleConnector,
)

OWNER = None
FAIL_LOAD = True
STATE = {}


class ReuseProbeConnector(LoadRecoveryExampleConnector):
    def get_block_ids_with_load_errors(self):
        # MultiConnector requires a set even when this example has no error.
        return super().get_block_ids_with_load_errors() or set()

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if request.request_id != OWNER:
            return 0, False
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        STATE["external_tokens"] = result[0]
        return result

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        super().update_state_after_alloc(request, blocks, num_external_tokens)
        if num_external_tokens:
            STATE["owner_blocks"] = list(blocks.get_block_ids()[0])

    def bind_connector_metadata(self, connector_metadata):
        has_load = any(not req.is_store for req in connector_metadata.requests)
        if FAIL_LOAD:
            super().bind_connector_metadata(connector_metadata)
        else:
            ExampleConnector.bind_connector_metadata(self, connector_metadata)
        if has_load:
            STATE["load_steps"] = STATE.get("load_steps", 0) + 1
            STATE["invalid_blocks"] = sorted(self._invalid_block_ids or ())
