import { matmulSeconds, roofline } from '../roofline';
import { evaluateDecodeAtBatch } from '../sim/run/decode';
import { evaluatePrefill } from '../sim/run/prefill';
import { runnableOn } from '../sim/run/validate';
import { ROLES, ShardingRole } from '../sim/ir/sharding/roles';
import { Diagnostic, Mesh, MoeDispatch, roleSize } from '../surface/deploy';
import type { ResourceCostBackend } from '../sim/cost/types';
import type { HardwareResource } from '../surface/api';
import { flopsPerDecodeToken, flopsPerPrefillToken, hasMoeLayers } from '../../model/utils';
import { ChipSpec } from '../../hardware/chips';
import { ModelSpec } from '../../model/models';
import type { Candidate } from './search';

export type ComponentTimes = Record<HardwareResource, number>;
export type Boundedness = HardwareResource;

// One searched configuration, evaluated for both phases: display-ready
// scalars plus the winning mesh, so a trace can be re-lowered from it.
export interface ConfigResult {
  id: string;
  chipId: string;
  sizes: Partial<Record<ShardingRole, number>>;
  placement: string;
  // how many of the TP ranks hold a sequence slice instead of a head slice
  decodeContextParallel: number;
  dispatch: MoeDispatch;
  // the placement's resolved mesh
  mesh: Mesh;
  nChips: number;
  workload: { prefillLen: number; generateLen: number };
  diagnostics: Diagnostic[];
  memory?: {
    weightBytesPerChip: number;
    // HBM left for KV after the weights land: what actually caps batch
    kvSpaceBytesPerChip: number;
    kvBytesPerSeqPerChip: number;
    // the paged and reserved-state parts of kvBytesPerSeqPerChip, and how
    // many state slots the latter reserves per sequence
    pagedKvBytesPerSeqPerChip: number;
    stateBytesPerSeqPerChip: number;
    stateSlotsPerSeq: number;
    // machine total (per-chip residency x DPA groups)
    maxResidentSeqs: number;
  };
  prefill?: {
    tokPerSecPerChip: number;
    ttft: number;
    batchSeqs: number;
    mfu: number;
    fracOfCeiling: number;
    boundBy: Boundedness;
    components: ComponentTimes;
    // the visible split of the phase time after overlap (sums to the time)
    visible: ComponentTimes;
  };
  decode?: {
    tokPerSecPerChip: number;
    tokPerSecPerUser: number;
    tpot: number;
    stepTime: number;
    batchPerStage: number;
    residentSeqs: number;
    mfu: number;
    // model bandwidth utilization: weight + KV bytes streamed per step over
    // what the chip's peak HBM bandwidth could move in a step
    mbu: number;
    fracOfCeiling: number;
    // operating tok/s/chip over this config's own B -> inf rate (KV gate
    // off); low = throughput is KV-room-starved, not sharding-limited
    batchSaturation?: number;
    boundBy: Boundedness;
    components: ComponentTimes;
    // the visible split of the step time after overlap (sums to the time)
    visible: ComponentTimes;
  };
}

export function boundBy(b: ComponentTimes): Boundedness {
  return b.compute >= b.memory && b.compute >= b.comms
    ? 'compute'
    : b.memory >= b.comms
      ? 'memory'
      : 'comms';
}

// A decode-search winner filled out into one config row: prefill is
// evaluated on the same deployment (throughput at a full-machine batch of
// sequences, plus a single-sequence pass for TTFT). The candidate's own
// backend prices both, so the row is the search's numbers, not a mirror.
export function enrichCandidate(
  model: ModelSpec,
  chip: ChipSpec,
  key: string,
  nChips: number,
  workload: { prefillLen: number; generateLen: number },
  c: Candidate<ResourceCostBackend>,
  backend: ResourceCostBackend = c.backend,
): ConfigResult | undefined {
  const { mesh } = c.deployment;
  const input = { model, deployment: c.deployment, workload };
  const dec = c.result;
  if (!('tpot' in dec)) return undefined;

  const runnableModel = runnableOn(model, chip);
  const hw = roofline(model, chip, workload, chip.realizableFlopsFrac)!;
  const dpa = roleSize(mesh, 'DPA');
  const pp = roleSize(mesh, 'PP');
  const ctxAvg = workload.prefillLen + workload.generateLen / 2;
  const prefillBatchSeqs = dpa * Math.max(1, Math.ceil(hw.critTokens / workload.prefillLen));

  const opts = { costBackend: backend };
  const pfThrough = evaluatePrefill(input, prefillBatchSeqs, 'throughput', opts);
  const pfSingle = evaluatePrefill(input, 1, 'ttft', opts);
  // the same config re-priced at an effectively infinite batch (KV gate
  // off) — its own batch-scaling ceiling
  const sat = evaluateDecodeAtBatch(input, 65536 * dpa, pp, { ...opts, ignoreKvCapacity: true });

  // the expert plane is structural noise on dense models (never read by
  // lowering), so hide it from the displayed sharding
  const shown = ROLES.filter(
    (r) => roleSize(mesh, r) > 1 && (hasMoeLayers(model) || (r !== 'EP' && r !== 'ETP')),
  );
  const sizes = Object.fromEntries(shown.map((r) => [r, roleSize(mesh, r)]));
  // DCP is not a role: it owns no dims, it re-spends TP's on the sequence.
  // It still belongs in the identity and the label, since two rows can
  // otherwise differ only by it.
  const dcp = c.deployment.decodeContextParallel ?? 1;
  const placement =
    shown.map((r) => `${r}[${mesh.roles[r].join('')}]`).join(' ') +
    (dcp > 1 ? ` DCP=${dcp} of TP` : '');

  return {
    id: `${key}|${JSON.stringify(sizes)}|dcp${dcp}`,
    chipId: chip.id,
    sizes,
    decodeContextParallel: dcp,
    placement,
    dispatch: c.deployment.moeDispatch,
    mesh,
    nChips,
    workload,
    diagnostics: dec.diags,
    memory: {
      weightBytesPerChip: dec.memory.weightBytesPerChip,
      kvSpaceBytesPerChip: Math.max(0, chip.hbmCapacity - dec.memory.weightBytesPerChip),
      kvBytesPerSeqPerChip: dec.memory.kvBytesPerSeqPerChip,
      pagedKvBytesPerSeqPerChip: dec.memory.pagedKvBytesPerSeqPerChip,
      stateBytesPerSeqPerChip: dec.memory.stateBytesPerSeqPerChip,
      stateSlotsPerSeq: dec.memory.stateSlotsPerSeq,
      maxResidentSeqs: dec.memory.maxResidentSeqsPerChip * dpa,
    },
    decode: {
      tokPerSecPerChip: dec.tokPerSecPerChip,
      tokPerSecPerUser: 1 / dec.tpot,
      tpot: dec.tpot,
      stepTime: dec.stepTime,
      batchPerStage: c.batch!,
      residentSeqs: c.batch! * pp,
      mfu:
        dec.tokPerSecPerChip * matmulSeconds(flopsPerDecodeToken(runnableModel, ctxAvg), chip, 1)!,
      // like MFU, quoted against the datasheet peak, not the realizable fraction
      mbu: (dec.traffic.weightBytes + dec.traffic.kvBytes) / dec.stepTime / chip.hbmBandwidth,
      fracOfCeiling: dec.tokPerSecPerChip / hw.decodeCeilingOverlapped,
      batchSaturation: sat.ok ? dec.tokPerSecPerChip / sat.tokPerSecPerChip : undefined,
      boundBy: boundBy(dec.cost.busy),
      components: dec.cost.busy,
      visible: dec.cost.parts,
    },
    prefill: pfThrough.ok
      ? {
          tokPerSecPerChip: pfThrough.tokPerSecPerChip,
          ttft: pfSingle.ok ? pfSingle.latency : pfThrough.latency,
          batchSeqs: prefillBatchSeqs,
          mfu:
            pfThrough.tokPerSecPerChip *
            matmulSeconds(flopsPerPrefillToken(runnableModel, workload.prefillLen), chip, 1)!,
          fracOfCeiling: pfThrough.tokPerSecPerChip / hw.prefillCeiling,
          boundBy: boundBy(pfThrough.cost.busy),
          components: pfThrough.cost.busy,
          visible: pfThrough.cost.parts,
        }
      : undefined,
  };
}
