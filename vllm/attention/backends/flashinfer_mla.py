# SPDX-License-Identifier: Apache-2.0

import math
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

from vllm.multimodal import MultiModalPlaceholderMap

try:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024
except ImportError:
    BatchMLAPagedAttentionWrapper = None
    FLASHINFER_WORKSPACE_BUFFER_SIZE = 0

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionState, AttentionType)
from vllm.attention.backends.mla.utils import MLACommonImpl, MLACommonMetadata
from vllm.attention.backends.utils import (PAD_SLOT_ID, compute_slot_mapping,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.utils import (async_tensor_h2d, get_kv_cache_torch_dtype,
                        make_tensor_with_pad)

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)


class FlashInferMLABackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_impl_cls() -> Type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashInferMLAMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashInferMLAState"]:
        return FlashInferMLAState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        ops.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        ops.copy_blocks_mla(kv_caches, src_to_dists)

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [576]

    @staticmethod
    def get_fp8_dtype_for_flashinfer(kv_cache_dtype: str) -> torch.dtype:
        if kv_cache_dtype in ("fp8", "fp8_e4m3"):
            return torch.float8_e4m3fn
        elif kv_cache_dtype == "fp8_e5m2":
            return torch.float8_e5m2
        else:
            raise ValueError(f"Unrecognized FP8 dtype: {kv_cache_dtype}")


class FlashInferMLAState(AttentionState):

    def __init__(self, runner):
        self.runner = runner
        self._is_graph_capturing = False
        self._workspace_buffer = None
        self._decode_wrapper = None
        self._prefill_wrapper = None

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.empty(
                FLASHINFER_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.runner.device)
        return self._workspace_buffer

    def _get_prefill_wrapper(self):
        if self._prefill_wrapper is None:
            self._prefill_wrapper = BatchMLAPagedAttentionWrapper(
                self._get_workspace_buffer(), backend="fa2")
        return self._prefill_wrapper

    def _get_decode_wrapper(self):
        if self._decode_wrapper is None:
            self._decode_wrapper = BatchMLAPagedAttentionWrapper(
                self._get_workspace_buffer(), backend="fa2")
        return self._decode_wrapper

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._is_graph_capturing = True
        self._graph_decode_wrapper = None
        self._graph_slot_mapping = torch.full((max_batch_size, ),
                                              PAD_SLOT_ID,
                                              dtype=torch.long,
                                              device=self.runner.device)
        self._graph_block_tables = torch.from_numpy(
            self.runner.graph_block_tables).to(device=self.runner.device)
        self._graph_decode_workspace_buffer = self._get_workspace_buffer()
        self._graph_query_start_loc_buffer = torch.empty(
            max_batch_size + 1, dtype=torch.int32, device=self.runner.device)
        self._graph_indices_buffer = torch.empty(
            max_batch_size * self.runner.cache_config.num_gpu_blocks,
            dtype=torch.int32,
            device=self.runner.device)
        self._graph_indptr_buffer = torch.empty(max_batch_size + 1,
                                                dtype=torch.int32,
                                                device=self.runner.device)
        self._graph_last_page_len_buffer = torch.empty(
            max_batch_size, dtype=torch.int32, device=self.runner.device)
        self._graph_seq_lens_buffer = torch.empty(max_batch_size,
                                                  dtype=torch.int32,
                                                  device=self.runner.device)
        self._graph_input_positions = torch.zeros((max_batch_size, ),
                                                  dtype=torch.long,
                                                  device=self.runner.device)
        yield
        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_block_tables
        del self._graph_decode_workspace_buffer
        del self._graph_query_start_loc_buffer
        del self._graph_indices_buffer
        del self._graph_indptr_buffer
        del self._graph_last_page_len_buffer
        del self._graph_seq_lens_buffer
        del self._graph_decode_wrapper
        del self._graph_input_positions

    def graph_clone(self, batch_size: int):
        assert self._is_graph_capturing
        state = self.__class__(self.runner)
        state._workspace_buffer = self._graph_decode_workspace_buffer
        state._decode_wrapper = self._graph_decode_wrapper
        state._prefill_wrapper = self._get_prefill_wrapper()
        return state

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        assert self._is_graph_capturing
        _query_start_loc_indptr = self._graph_query_start_loc_buffer[:
                                                                     batch_size
                                                                     + 1]
        _indptr_buffer = self._graph_indptr_buffer[:batch_size + 1]
        _last_page_len_buffer = self._graph_last_page_len_buffer[:batch_size]
        _seq_lens_buffer = self._graph_seq_lens_buffer[:batch_size]

        self._graph_decode_wrapper = BatchMLAPagedAttentionWrapper(
            self._graph_decode_workspace_buffer,
            use_cuda_graph=True,
            qo_indptr=_query_start_loc_indptr,
            kv_indptr=_indptr_buffer,
            kv_indices=self._graph_indices_buffer,
            kv_len_arr=_seq_lens_buffer,
            backend="fa2",
        )
        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferMLABackend.get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        paged_kv_indptr_tensor_host = torch.arange(0,
                                                   batch_size + 1,
                                                   dtype=torch.int32)
        paged_kv_indices_tensor_host = torch.arange(0,
                                                    batch_size,
                                                    dtype=torch.int32)
        paged_kv_last_page_len_tensor_host = torch.full((batch_size, ),
                                                        self.runner.block_size,
                                                        dtype=torch.int32)
        seq_lens_tensor_host = torch.full((batch_size, ),
                                          self.runner.block_size,
                                          dtype=torch.int32)
        query_start_loc_host = torch.arange(0,
                                            batch_size + 1,
                                            dtype=torch.int32)

        # TODO remove hard code
        assert self.runner.model_config.is_deepseek_mla
        num_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        qk_nope_head_dim = self.runner.model_config.hf_text_config.qk_nope_head_dim
        qk_rope_head_dim = self.runner.model_config.hf_text_config.qk_rope_head_dim
        head_dim_ckv = self.runner.model_config.hf_text_config.kv_lora_rank
        head_dim_kpe = qk_rope_head_dim
        v_head_dim = self.runner.model_config.hf_text_config.v_head_dim
        sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

        attn_metadata = self.runner.attn_backend.make_metadata(
            # AttentionMetadata
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=self._graph_slot_mapping[:batch_size],
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            input_positions=self._graph_input_positions[:batch_size],
            # FlashInferMLAMetadata
            max_prefill_seq_len=0,
            max_decode_seq_len=self.runner.max_seq_len_to_capture,
            use_cuda_graph=True,
            decode_wrapper=self._graph_decode_wrapper,
            seq_start_loc=None,
            query_start_loc=query_start_loc_host,
            block_tables=self._graph_block_tables[:batch_size],
            seq_lens_tensor=seq_lens_tensor_host,
            paged_kv_indptr=paged_kv_indptr_tensor_host,
            paged_kv_indices=paged_kv_indices_tensor_host,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor_host,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            page_size=self.runner.block_size,
            data_type=kv_cache_dtype,
            q_data_type=self.runner.model_config.dtype,
            device=self.runner.device,
            sm_scale=sm_scale,
        )
        attn_metadata.begin_forward()
        return attn_metadata

    def get_graph_input_buffers(self,
                                attn_metadata,
                                is_encoder_decoder_model: bool = False):
        return {
            "slot_mapping": attn_metadata.slot_mapping,
            "input_positions": attn_metadata.decode_metadata.input_positions,
        }

    def prepare_graph_input_buffers(self,
                                    input_buffers,
                                    attn_metadata,
                                    is_encoder_decoder_model: bool = False):
        # CUDA graph buffer is padded so only perform a partial copy based on
        # num_positions
        input_positions = attn_metadata.input_positions
        num_positions = input_positions.shape[0]
        input_buffers["input_positions"][:num_positions].copy_(
            input_positions, non_blocking=True)

    def begin_forward(self, model_input):
        assert not self._is_graph_capturing
        state = self
        use_cuda_graph = model_input.attn_metadata.use_cuda_graph
        is_decode = model_input.attn_metadata.num_prefills == 0
        # In case of multistep chunked-prefill, there might be prefill requests
        # scheduled while CUDA graph mode is enabled. We don't run graph in that
        # case.
        if use_cuda_graph and is_decode:
            batch_size = model_input.input_tokens.shape[0]
            state = (self.runner.graph_runners[model_input.virtual_engine]
                     [batch_size].attn_state)
        model_input.attn_metadata.prefill_wrapper = state._get_prefill_wrapper(
        )
        model_input.attn_metadata.decode_wrapper = state._get_decode_wrapper()
        model_input.attn_metadata.begin_forward()


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata):
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # Number of query tokens for each request in the batch.
    # Currently, we require that all requests have the same number of query
    # tokens during the decoding phase. When speculavie decoding is enabled,
    # decode_query_len might be greater than 1. In all other cases, it is 1.
    decode_query_len: Optional[int] = 1

    use_cuda_graph: bool = True

    # TODO: Use flash attention in prefill now
    prefill_wrapper: Optional[BatchMLAPagedAttentionWrapper] = None
    decode_wrapper: Optional[BatchMLAPagedAttentionWrapper] = None

    # Metadata for the prefill stage
    seq_start_loc: Optional[torch.Tensor] = None
    query_start_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None

    # used for GPU in-place advance_step
    seq_lens_tensor: Optional[torch.Tensor] = None
    block_table_bound: Optional[torch.Tensor] = None

    # request 1, page indices [0, 5, 8]
    # request 2, page indices [1, 6, 7]
    # request 3, page indices [3, 4]
    # paged_kv_indices is a concatenation of page indices of all requests:
    # [0, 5, 8, 1, 6, 7, 3, 4]
    # paged_kv_indptr is used to index into paged_kv_indices:
    # [0, 3, 6, 8]
    # The indptr of the paged kv cache, shape: [batch_size + 1]
    paged_kv_indptr: Optional[torch.Tensor] = None
    # The page indices of the paged kv cache
    paged_kv_indices: Optional[torch.Tensor] = None
    # The number of entries in the last page of each request in
    # the paged kv cache, shape: [batch_size]
    paged_kv_last_page_len: Optional[torch.Tensor] = None

    # The number of query/output heads
    num_heads: Optional[int] = None
    # The dimension of compressed-kv
    head_dim_ckv: Optional[int] = None
    # The dimension of rope k-cache
    head_dim_kpe: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None
    # Block size of vllm
    page_size: Optional[int] = None
    # The data type of the paged kv cache
    data_type: torch.dtype = None
    # The data type of the query
    q_data_type: torch.dtype = None
    # FlashInfer 0.2 encourages passing host tensors
    device: torch.device = torch.device("cpu")
    is_profile_run: bool = False

    # The scale used in softmax, if not provided, will be set to
    # `1.0 / sqrt(head_dim)`.
    sm_scale: Optional[float] = None

    def begin_forward(self):
        if self.num_prefill_tokens > 0:
            if self.paged_kv_indices is None:
                return

            assert self.prefill_wrapper is not None
            assert self.query_start_loc is not None
            self.query_start_loc = self.query_start_loc[:self.num_prefills + 1]
            batch_size = self.query_start_loc.shape[0] - 1
            assert batch_size >= 0
            # Use flash attention for profiling and prefill
        if self.num_decode_tokens > 0:
            assert self.query_start_loc is not None
            assert self.paged_kv_indices is not None
            assert self.paged_kv_indptr is not None
            assert self.seq_lens_tensor is not None
            assert self.paged_kv_last_page_len is not None
            self.query_start_loc = self.query_start_loc.to(self.device)
            self.paged_kv_indices = self.paged_kv_indices.to(self.device)
            self.paged_kv_indptr = self.paged_kv_indptr.to(self.device)
            self.paged_kv_last_page_len = self.paged_kv_last_page_len.to(
                self.device)
            # handle model warmup path
            if self.block_table_bound is not None:
                self.block_table_bound = self.block_table_bound.to(self.device)
            self.seq_lens_tensor = self.seq_lens_tensor.to(self.device)

            assert self.decode_wrapper is not None
            self.decode_wrapper.plan(
                qo_indptr=self.query_start_loc,
                kv_indptr=self.paged_kv_indptr,
                kv_indices=self.paged_kv_indices,
                kv_len_arr=self.seq_lens_tensor,
                num_heads=self.num_heads,
                head_dim_ckv=self.head_dim_ckv,
                head_dim_kpe=self.head_dim_kpe,
                page_size=self.page_size,
                causal=True,
                sm_scale=self.sm_scale,
                q_data_type=self.q_data_type,
                kv_data_type=self.data_type,
            )

    def asdict_zerocopy(self,
                        skip_fields: Optional[Set[str]] = None
                        ) -> Dict[str, Any]:
        if skip_fields is None:
            skip_fields = set()
        # We need to skip the prefill/decode_wrapper field since it cannot be
        # broadcasted with nccl when TP is enabled.
        skip_fields.add('prefill_wrapper')
        skip_fields.add('decode_wrapper')
        return super().asdict_zerocopy(skip_fields)

    @property
    def prefill_metadata(self) -> Optional["FlashInferMLAMetadata"]:
        if self.num_prefills == 0:
            return None
        return self

    @property
    def decode_metadata(self) -> Optional["FlashInferMLAMetadata"]:
        if self.num_decode_tokens == 0:
            return None
        return self

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Mutli-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class FlashInferMLAMetadataBuilder(
        AttentionMetadataBuilder[FlashInferMLAMetadata]):

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.input_positions: List[int] = []
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        # Please follow https://docs.flashinfer.ai/tutorials/kv_layout.html#page-layout
        # for the precise definition of the following fields.
        # An example:
        # request 1, page indices [0, 5, 8]
        # request 2, page indices [1, 6, 7]
        # request 3, page indices [3, 4]
        # paged_kv_indices is a concatenation of page indices of all requests:
        # [0, 5, 8, 1, 6, 7, 3, 4]
        # paged_kv_indptr is used to index into paged_kv_indices:
        # [0, 3, 6, 8]
        self.paged_kv_indices: List[int] = []
        # 0 at the beginning of paged_kv_indptr indicates the start of the
        # first request’s page indices in the paged_kv_indices list.
        self.paged_kv_indptr: List[int] = [0]
        # paged_kv_last_page_len is the length of the last page of each request
        self.paged_kv_last_page_len: List[int] = []
        self.total_blocks = 0
        self.is_profile_run: bool = False

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block, input_positions) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks,
                 inter_data.input_positions):
            self.input_positions.extend(input_positions)
            self.context_lens.append(context_len)
            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

            # It is not necessary to add paged_kv_indices, paged_kv_indptr,
            # and paged_kv_last_page_len for profile run because we will
            # create dummy inputs.
            if is_profile_run:
                self.is_profile_run = is_profile_run
                return
            self._update_paged_kv_tensors(block_table, seq_len)

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def _update_paged_kv_tensors(self, block_table: List[int], seq_len: int):
        # Get the number of valid blocks based on sequence length.
        # If seq_len = 16, block_size = 16,
        # block_table_bound is 1 with 1 valid block.
        # If seq_len = 15, block_size = 16,
        # block_table_bound is 0 + 1 with 1 valid block.
        self.total_blocks += len(block_table)
        block_table_bound = seq_len // self.block_size + 1 \
                            if seq_len % self.block_size != 0 \
                            else seq_len // self.block_size
        self.paged_kv_indices.extend(block_table[:block_table_bound])
        self.paged_kv_indptr.append(self.paged_kv_indptr[-1] +
                                    block_table_bound)

        last_page_len = seq_len % self.block_size
        if last_page_len == 0:
            last_page_len = self.block_size
        self.paged_kv_last_page_len.append(last_page_len)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        decode_query_len = max(query_lens[self.num_prefills:], default=1)

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
            last_paged_kv_indptr = self.paged_kv_indptr[-1]
            self.paged_kv_indptr.extend([last_paged_kv_indptr] *
                                        cuda_graph_pad_size)
            self.paged_kv_last_page_len.extend([0] * cuda_graph_pad_size)
            # query_lens needs padding to create correct qo_indptr for flashinfer
            query_lens_pad = query_lens.copy()
            query_lens_pad.extend([0] * cuda_graph_pad_size)
        else:
            query_lens_pad = query_lens
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )

        assert device is not None
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        input_positions = async_tensor_h2d(self.input_positions, torch.long,
                                           device, self.runner.pin_memory)

        query_lens_tensor = async_tensor_h2d(query_lens_pad, torch.long,
                                             device, self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        if len(self.paged_kv_indptr) > 0:
            # extend to the maximum number of blocks as returned by the
            # scheduler
            self.paged_kv_indices.extend(
                [0] * (self.total_blocks - len(self.paged_kv_indices)))
            paged_kv_indices_tensor = torch.tensor(self.paged_kv_indices,
                                                   device="cpu",
                                                   dtype=torch.int)
            paged_kv_indptr_tensor = torch.tensor(self.paged_kv_indptr,
                                                  device="cpu",
                                                  dtype=torch.int)
            paged_kv_last_page_len_tensor = torch.tensor(
                self.paged_kv_last_page_len, device="cpu", dtype=torch.int)
            block_table_bound_tensor = torch.zeros(len(self.paged_kv_indptr) -
                                                   1,
                                                   device="cpu",
                                                   dtype=torch.int)
        else:
            paged_kv_indices_tensor = None
            paged_kv_indptr_tensor = None
            paged_kv_last_page_len_tensor = None
            block_table_bound_tensor = None

        if self.runner.kv_cache_dtype.startswith("fp8"):
            kv_cache_dtype = FlashInferMLABackend.get_fp8_dtype_for_flashinfer(
                self.runner.kv_cache_dtype)
        else:
            kv_cache_dtype = get_kv_cache_torch_dtype(
                self.runner.kv_cache_dtype, self.runner.model_config.dtype)

        # TODO remove hard code
        assert self.runner.model_config.is_deepseek_mla
        num_heads = self.runner.model_config.get_num_attention_heads(
            self.runner.parallel_config)
        qk_nope_head_dim = self.runner.model_config.hf_text_config.qk_nope_head_dim
        qk_rope_head_dim = self.runner.model_config.hf_text_config.qk_rope_head_dim
        head_dim_ckv = self.runner.model_config.hf_text_config.kv_lora_rank
        head_dim_kpe = qk_rope_head_dim
        v_head_dim = self.runner.model_config.hf_text_config.v_head_dim

        sm_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)

        return FlashInferMLAMetadata(
            # AttentionMetadata
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            slot_mapping=slot_mapping_tensor,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            input_positions=input_positions,
            # FlashInferMLAMetadata
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            decode_query_len=decode_query_len,
            use_cuda_graph=use_captured_graph,
            seq_start_loc=seq_start_loc,
            query_start_loc=query_start_loc,
            block_tables=block_tables,
            seq_lens_tensor=seq_lens_tensor,
            block_table_bound=block_table_bound_tensor,
            paged_kv_indptr=paged_kv_indptr_tensor,
            paged_kv_indices=paged_kv_indices_tensor,
            paged_kv_last_page_len=paged_kv_last_page_len_tensor,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            page_size=self.block_size,
            data_type=kv_cache_dtype,
            q_data_type=self.runner.model_config.dtype,
            device=device,
            is_profile_run=self.is_profile_run,
            sm_scale=sm_scale,
        )


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **kwargs) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **kwargs)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, FlashInferMLAMetadata)
        return self._forward_prefill_flash(q, kv_c_normed, k_pe,
                                           attn_metadata.seq_start_loc,
                                           attn_metadata.max_prefill_seq_len)

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
        assert decode_meta.decode_wrapper is not None
        decode_output = decode_meta.decode_wrapper.run(
            q_nope,
            q_pe,
            kv_c_and_k_pe_cache[:, :, :decode_meta.head_dim_ckv],
            kv_c_and_k_pe_cache[:, :, decode_meta.head_dim_ckv:],
            return_lse=False,
        )
        return self._v_up_proj_and_o_proj(decode_output)
