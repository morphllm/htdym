import {
  enrichCandidate,
  Boundedness,
  boundBy,
  ComponentTimes,
} from '../core/engine/optimizer/enrich';
import type { Candidate, SearchPhase } from '../core/engine/optimizer/search';
import type { Scheduler, OverlapOptions } from '../core/engine/sim/cost/select';
import type { ResourceCostBackend } from '../core/engine/sim/cost/types';
import { ROLES, ShardingRole } from '../core/engine/sim/ir/sharding/roles';
import type { StatePoolOptions } from '../core/engine/surface/api';
import { Diagnostic, Mesh, MoeDispatch, roleSize } from '../core/engine/surface/deploy';
import { hasMoeLayers } from '../core/model/utils';
import { ModelSpec } from '../core/model/models';
import type { Context } from './context';
import { nullable, round } from './format';
import { VERSION } from './version';

// One estimated configuration as the CLI emits it (a JSON line, a table
// row). Field names follow the sweep output the tab fleet tooling reads.
export interface EstimateRow {
  model: string;
  chip: string;
  machine: string;
  nChips: number;
  phase: 'decode' | 'prefill';
  slo: number | null;
  feasible: boolean;
  // "DPA=16 TP=2 EP=32": roles above 1, expert roles hidden on dense models
  sharding: string | null;
  sizes: Record<ShardingRole, number> | null;
  dcp: number | null;
  moeDispatch: MoeDispatch | null;
  placement: string | null;
  batch: number | null;
  tokPerSecPerChip: number | null;
  tokPerSecMachine: number | null;
  tpotMs: number | null;
  tokPerSecPerUser: number | null;
  stepTimeMs: number | null;
  boundBy: Boundedness | null;
  busyMs: ComponentTimes | null;
  visibleMs: ComponentTimes | null;
  mfu: number | null;
  mbu: number | null;
  weightGbPerChip: number | null;
  kvMbPerSeqPerChip: number | null;
  pagedKvMbPerSeqPerChip: number | null;
  stateMbPerSeqPerChip: number | null;
  stateSlotsPerSeq: number | null;
  maxResidentSeqs: number | null;
  // the same deployment's prefill (for a decode row: a full-machine batch
  // plus a single-sequence TTFT pass; for a prefill row: the searched pass)
  prefill: {
    tokPerSecPerChip: number;
    ttftMs: number | null;
    latencyMs: number | null;
    batchSeqs: number;
    boundBy: Boundedness;
  } | null;
  costPerHour: number | null;
  usdPerMtok: number | null;
  workload: { prefillLen: number; generateLen: number };
  overlap: OverlapOptions;
  statePool: Partial<StatePoolOptions>;
  scheduler: Scheduler;
  diagnostics: Diagnostic[];
  error: string | null;
  id: string | null;
  version: string;
}

// $/Mtok at one chip's rental rate and rate, the fleet tooling's own
// expression (cost_per_million_tokens in tab/clis/fleet/batch_floor.py)
export function usdPerMtok(costPerHour: number, tokPerSecPerChip: number): number {
  return (costPerHour / (tokPerSecPerChip * 3600)) * 1e6;
}

export function sizesOf(mesh: Mesh): Record<ShardingRole, number> {
  return Object.fromEntries(ROLES.map((r) => [r, roleSize(mesh, r)])) as Record<
    ShardingRole,
    number
  >;
}

// the expert plane is structural noise on dense models, so it is hidden
export function shardingLabel(model: ModelSpec, mesh: Mesh): string {
  const shown = ROLES.filter(
    (r) => roleSize(mesh, r) > 1 && (hasMoeLayers(model) || (r !== 'EP' && r !== 'ETP')),
  );
  return shown.map((r) => `${r}=${roleSize(mesh, r)}`).join(' ') || 'single-chip';
}

export function placementLabel(model: ModelSpec, mesh: Mesh, dcp: number): string {
  const shown = ROLES.filter(
    (r) => roleSize(mesh, r) > 1 && (hasMoeLayers(model) || (r !== 'EP' && r !== 'ETP')),
  );
  return (
    shown.map((r) => `${r}[${mesh.roles[r].join('')}]`).join(' ') +
    (dcp > 1 ? ` DCP=${dcp} of TP` : '')
  );
}

export const sloOf = (phase: SearchPhase): number | null =>
  phase.kind === 'decode' ? (phase.policy?.sloTokPerSecPerUser ?? null) : null;

function base(ctx: Context, phase: SearchPhase): EstimateRow {
  return {
    model: ctx.model.name,
    chip: ctx.chip.id,
    machine: ctx.machineName,
    nChips: ctx.nChips,
    phase: phase.kind,
    slo: sloOf(phase),
    feasible: false,
    sharding: null,
    sizes: null,
    dcp: null,
    moeDispatch: null,
    placement: null,
    batch: null,
    tokPerSecPerChip: null,
    tokPerSecMachine: null,
    tpotMs: null,
    tokPerSecPerUser: null,
    stepTimeMs: null,
    boundBy: null,
    busyMs: null,
    visibleMs: null,
    mfu: null,
    mbu: null,
    weightGbPerChip: null,
    kvMbPerSeqPerChip: null,
    pagedKvMbPerSeqPerChip: null,
    stateMbPerSeqPerChip: null,
    stateSlotsPerSeq: null,
    maxResidentSeqs: null,
    prefill: null,
    costPerHour: ctx.chip.costPerHour ?? null,
    usdPerMtok: null,
    workload: ctx.workload,
    overlap: ctx.opts.overlap,
    statePool: ctx.opts.statePool,
    scheduler: ctx.opts.overlap.scheduler ?? 'naive',
    diagnostics: [],
    error: null,
    id: null,
    version: VERSION,
  };
}

export function infeasibleRow(ctx: Context, phase: SearchPhase, error?: string): EstimateRow {
  return { ...base(ctx, phase), error: error ?? null };
}

const ms = (c: ComponentTimes): ComponentTimes => ({
  compute: round(c.compute * 1e3, 4),
  memory: round(c.memory * 1e3, 4),
  comms: round(c.comms * 1e3, 4),
});

export function candidateRow(
  ctx: Context,
  phase: SearchPhase,
  c: Candidate<ResourceCostBackend>,
): EstimateRow {
  const { mesh } = c.deployment;
  const dcp = c.deployment.decodeContextParallel ?? 1;
  const sizes = sizesOf(mesh);
  const price = ctx.chip.costPerHour;
  const rate = c.result.tokPerSecPerChip;
  const row: EstimateRow = {
    ...base(ctx, phase),
    feasible: true,
    sharding: shardingLabel(ctx.model, mesh),
    sizes,
    dcp,
    moeDispatch: sizes.EP > 1 ? c.deployment.moeDispatch : null,
    placement: placementLabel(ctx.model, mesh, dcp),
    batch: c.batch ?? null,
    tokPerSecPerChip: round(rate, 1),
    tokPerSecMachine: round(rate * ctx.nChips, 0),
    stepTimeMs: round(c.result.stepTime * 1e3, 3),
    boundBy: boundBy(c.result.cost.busy),
    busyMs: ms(c.result.cost.busy),
    visibleMs: ms(c.result.cost.parts),
    costPerHour: price ?? null,
    usdPerMtok: price !== undefined ? round(usdPerMtok(price, rate), 4) : null,
    diagnostics: c.result.diags,
    id: `${ctx.chip.id}|${ctx.machineName}|${JSON.stringify(sizes)}|dcp${dcp}`,
  };

  if (!('tpot' in c.result)) {
    // a searched prefill pass: its own latency is the TTFT only when it
    // was a single-sequence pass
    const single = phase.kind === 'prefill' && phase.mode === 'ttft';
    row.prefill = {
      tokPerSecPerChip: round(rate, 1),
      ttftMs: single ? round(c.result.latency * 1e3, 3) : null,
      latencyMs: round(c.result.latency * 1e3, 3),
      batchSeqs: phase.kind === 'prefill' ? phase.seqs : 0,
      boundBy: boundBy(c.result.cost.busy),
    };
    return row;
  }

  const full = enrichCandidate(ctx.model, ctx.chip, ctx.chip.id, ctx.nChips, ctx.workload, c);
  const dec = full?.decode;
  const mem = full?.memory;
  return {
    ...row,
    tpotMs: round(c.result.tpot * 1e3, 3),
    tokPerSecPerUser: round(1 / c.result.tpot, 2),
    mfu: nullable(dec?.mfu, 4),
    mbu: nullable(dec?.mbu, 4),
    weightGbPerChip: round(c.result.memory.weightBytesPerChip / 1e9, 2),
    kvMbPerSeqPerChip: round(c.result.memory.kvBytesPerSeqPerChip / 1e6, 3),
    pagedKvMbPerSeqPerChip: round(c.result.memory.pagedKvBytesPerSeqPerChip / 1e6, 3),
    stateMbPerSeqPerChip: round(c.result.memory.stateBytesPerSeqPerChip / 1e6, 3),
    stateSlotsPerSeq: c.result.memory.stateSlotsPerSeq,
    maxResidentSeqs: mem?.maxResidentSeqs ?? null,
    prefill: full?.prefill
      ? {
          tokPerSecPerChip: round(full.prefill.tokPerSecPerChip, 1),
          ttftMs: round(full.prefill.ttft * 1e3, 3),
          latencyMs: null,
          batchSeqs: full.prefill.batchSeqs,
          boundBy: full.prefill.boundBy,
        }
      : null,
  };
}
