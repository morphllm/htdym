import { ModelSpec } from '../../../model/models';
import {
  blockAttnParams,
  blockDenseMlpParams,
  blockKvBytes,
  blockKvHeads,
  blockLatentProjParams,
  blockRouterParams,
  blockRoutedExpertParams,
  blockSharedExpertParams,
  blockStateBytes,
} from '../../../model/block';
import { chipsPerStage, dcpSize, kvFraction, Deployment, roleSize } from '../../surface/deploy';
import type { Stage } from '../lowering/stages';
import { DEFAULT_STATE_POOL, type MemoryFootprint, type StatePoolOptions } from '../../surface/api';
import { DTYPE_BYTES } from '../../../model/dtype';

// Exact per-chip memory footprint from the ordered stage map. Weights are
// counted at the stored format on every chip: they stay packed, with a
// widening kernel assumed where none ships (validate warns there).
//
// A sequence's bytes split into the paged cache and the recurrent state of
// linear-attention blocks. The state is reserved statePool slots at a time:
// one pool, so the resident count is where free HBM runs out under
// paged + slots * state per sequence (which is also the fixed point a
// two-pool manager sized "so both fill together" lands on).
export function memoryFootprint(
  m: ModelSpec,
  d: Deployment,
  stages: Stage[],
  fullLen: number,
  statePool: Partial<StatePoolOptions> = {},
): MemoryFootprint {
  const pool = { ...DEFAULT_STATE_POOL, ...statePool };
  const slots = pool.slotsPerSeq + pool.specSlots;
  const tp = roleSize(d.mesh, 'TP');
  const B = (c: keyof typeof m.precision.weights) => DTYPE_BYTES[m.precision.weights[c]];

  let worstWeights = 0;
  let worstKv = 0;
  let worstPaged = 0;
  let worstState = 0;
  let residents = Infinity;
  for (const s of stages) {
    let weights = 0;
    let pagedPerSeq = 0;
    let statePerSlot = 0;
    for (const g of s.groups)
      for (const { block: b, count } of g.pattern) {
        const n = g.repeat * count;
        weights +=
          (n *
            (blockAttnParams(m, b).total * B('attention') +
              (blockDenseMlpParams(m, b) + blockLatentProjParams(m, b)) * B('denseMlp') +
              blockSharedExpertParams(m, b) * B('sharedExperts'))) /
          tp;

        // router is replicated, not sharded
        weights += n * blockRouterParams(m, b) * B('router');

        weights +=
          (n * blockRoutedExpertParams(m, b).total * B('routedExperts')) / chipsPerStage(d);

        const share = kvFraction(blockKvHeads(b), tp, dcpSize(d));
        if (b.attn.kind === 'linear')
          statePerSlot += n * blockStateBytes(b, pool.stateDtypeBytes) * share;
        else
          pagedPerSeq += n * blockKvBytes(b, DTYPE_BYTES[m.precision.kv], fullLen, 'store') * share;
      }

    const emb = (m.vocab * m.modelDim * B('embeddings')) / tp;
    if (s.hasEmbedding) weights += emb;
    if (s.hasUnembedding && !(m.tiedEmbeddings && s.hasEmbedding)) weights += emb;

    const statePerSeq = slots * statePerSlot;
    const kvPerSeq = pagedPerSeq + statePerSeq;
    worstWeights = Math.max(worstWeights, weights);
    if (kvPerSeq > worstKv) {
      worstKv = kvPerSeq;
      worstPaged = pagedPerSeq;
      worstState = statePerSeq;
    }

    const free = d.chip.hbmCapacity - weights;
    residents = Math.min(
      residents,
      kvPerSeq > 0 ? Math.max(0, Math.floor(free / kvPerSeq)) : Infinity,
    );
  }
  return {
    weightBytesPerChip: worstWeights,
    kvBytesPerSeqPerChip: worstKv,
    pagedKvBytesPerSeqPerChip: worstPaged,
    stateBytesPerSeqPerChip: worstState,
    stateSlotsPerSeq: slots,
    maxResidentSeqsPerChip: residents,
  };
}
