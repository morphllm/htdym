import type { ModelSpec } from '../../model/models';
import type { Deployment, Diagnostic } from './deploy';

import type { Segment } from '../sim/ir/ops';
import type { CostBackend, TracePriceOf } from '../sim/cost/types';
import type { HbmTraffic } from '../sim/run/traffic';

export type { HbmTraffic };

export interface SimInput {
  model: ModelSpec;
  deployment: Deployment;
  workload: {
    // prefill (prompt) length T, tokens
    prefillLen: number;
    // decode (generation) length S, tokens
    generateLen: number;
  };
}

// How a serving stack provisions the recurrent state of linear-attention
// blocks. The simulator's default charges one state per resident sequence,
// which is the pure-physics floor; real hybrid KV managers keep several
// per request (SGLang: 3 with the radix cache off, 4 with extra_buffer_lazy,
// 5 with extra_buffer) plus one per speculative draft token, and on a model
// like Kimi K3 that pool, not the paged cache, caps concurrency.
export interface StatePoolOptions {
  // state slots reserved per resident sequence (1 = one working state)
  slotsPerSeq: number;
  // extra slots per sequence for speculative decoding intermediates
  // (DSPARK: draft block + 1)
  specSlots: number;
  // bytes per state element for capacity, when the stack stores the state
  // narrower than the model's stateBytes (2 = bf16). Capacity only: the
  // HBM traffic the step streams keeps the model's dtype.
  stateDtypeBytes?: number;
}

export const DEFAULT_STATE_POOL: StatePoolOptions = { slotsPerSeq: 1, specSlots: 0 };

export interface EvalOptions<TBackend extends CostBackend> {
  // Skip the KV-residency feasibility gate: evaluate the step as if the
  // batch fit. Only for B_inf saturation diagnostics (batchSaturation's own
  // ceiling), never for reported operating points.
  ignoreKvCapacity?: boolean;
  // reserved recurrent-state slots for linear-attention blocks; unset means
  // DEFAULT_STATE_POOL (one slot, nothing speculative, the model's dtype)
  statePool?: Partial<StatePoolOptions>;
  // bound per evaluation, then prices traces and candidate collectives
  costBackend: TBackend;
}

export type HardwareResource = 'compute' | 'memory' | 'comms';

export interface EvalFailure {
  ok: false;
  diags: Diagnostic[];
}

export interface BaseEvaluation<TBackend extends CostBackend> {
  ok: true;
  diags: Diagnostic[];
  // steady-state time of one stage step (slowest stage)
  stepTime: number;
  // pipeline stage index that set stepTime / cost
  criticalStage: number;
  // per-chip token throughput at this batch
  tokPerSecPerChip: number;
  // full backend-specific result for the stage that set stepTime
  cost: TracePriceOf<TBackend>;
  // one per-chip expanded trace per pipeline stage (index = stage),
  // segment-structured: repeated layers stay one segment
  perStageTrace: Segment[][];
}

export interface MemoryFootprint {
  // resident weight bytes on the heaviest chip
  weightBytesPerChip: number;
  // bytes one full-length sequence costs its group's chips (worst stage):
  // paged cache plus every reserved state slot, what actually divides HBM
  kvBytesPerSeqPerChip: number;
  // the paged (growing) part of that: MLA latents, GQA heads, windows
  pagedKvBytesPerSeqPerChip: number;
  // the recurrent-state part, all reserved slots included
  stateBytesPerSeqPerChip: number;
  // slots that state part reserves per sequence (slotsPerSeq + specSlots)
  stateSlotsPerSeq: number;
  // sequences one chip's free HBM holds KV for (worst stage). Each DPA
  // group holds its own sequences, so the machine total is dpa times this.
  maxResidentSeqsPerChip: number;
}

export type DecodeEvaluation<TBackend extends CostBackend> =
  | EvalFailure
  | (BaseEvaluation<TBackend> & {
      memory: MemoryFootprint;
      // time between successive tokens of one sequence (= PP * stepTime)
      tpot: number;
      // weight and KV bytes one chip streams per step, averaged over the
      // pipeline's stages (every stage holds the same chip count, so this
      // is the machine mean). Over stepTime it is the achieved bandwidth
      // MBU quotes against the chip's peak.
      traffic: HbmTraffic;
    });

export type PrefillEvaluation<TBackend extends CostBackend> =
  | EvalFailure
  | (BaseEvaluation<TBackend> & {
      // single-pass latency (TTFT when evaluating one sequence)
      latency: number;
    });
