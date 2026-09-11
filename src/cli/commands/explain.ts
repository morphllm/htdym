import {
  Candidate,
  decodeContextParallels,
  evaluateWithSearchOpts,
  SearchPhase,
} from '../../core/engine/optimizer/search';
import type { ResourceCostBackend } from '../../core/engine/sim/cost/types';
import { ShardingRole } from '../../core/engine/sim/ir/sharding/roles';
import { evaluateDecodeAtBatch } from '../../core/engine/sim/run/decode';
import { Deployment, MoeDispatch, validateSizes } from '../../core/engine/surface/deploy';
import { enumeratePlacements } from '../../core/engine/surface/placements';
import { deployedAxes } from '../../core/hardware/topology';
import { hasMoeLayers } from '../../core/model/utils';
import { Context, evalOptions } from '../context';
import { fix, int } from '../format';
import { CliError } from '../resolve';
import { candidateRow, EstimateRow } from '../row';
import { SearchArgs, searchPhase } from './search';

export interface ExplainArgs extends SearchArgs {
  // the roles given; the rest fill in from the machine (PP 1, DPA the
  // remainder of the attention plane, EP the whole stage on MoE, ETP the
  // remainder of the expert plane)
  sizes: Partial<Record<ShardingRole, number>>;
  dcp?: number;
  dispatch?: MoeDispatch;
  // decode at exactly this batch instead of the policy's operating batch
  batch?: number;
}

export interface Explanation {
  // the best placement/dispatch/DCP of the sizes, the way a search scores it
  best: EstimateRow;
  // every placement priced, best first
  all: EstimateRow[];
  sizes: Record<ShardingRole, number>;
  hints: string[];
}

export function fillSizes(ctx: Context, given: Partial<Record<ShardingRole, number>>) {
  const n = ctx.nChips;
  const PP = given.PP ?? 1;
  const stage = n / PP;
  if (!Number.isInteger(stage)) throw new CliError(`PP=${PP} does not divide ${n} chips`, 2);
  const TP = given.TP ?? (given.DPA ? stage / given.DPA : 1);
  const DPA = given.DPA ?? stage / TP;
  const moe = hasMoeLayers(ctx.model);
  const EP = given.EP ?? (given.ETP ? stage / given.ETP : moe ? stage : 1);
  const ETP = given.ETP ?? stage / EP;
  const sizes = { PP, DPA, TP, EP, ETP };
  for (const [r, v] of Object.entries(sizes))
    if (!Number.isInteger(v) || v < 1) throw new CliError(`${r}=${v} is not a whole role size`, 2);
  if (DPA * TP !== stage)
    throw new CliError(`DPA=${DPA} x TP=${TP} is not the ${stage} chips of a stage`, 2);
  if (EP * ETP !== stage)
    throw new CliError(`EP=${EP} x ETP=${ETP} is not the ${stage} chips of a stage`, 2);
  return sizes;
}

export function explain(ctx: Context, a: ExplainArgs): Explanation {
  const sizes = fillSizes(ctx, a.sizes);
  const diags = validateSizes(ctx.model, { ...sizes, DCP: a.dcp });
  const errors = diags.filter((d) => d.severity === 'error');
  if (errors.length) throw new CliError(errors.map((d) => d.message).join('; '), 2);

  const axes = deployedAxes(ctx.chip.interconnect, ctx.machine);
  const phase = searchPhase(ctx, a);
  const opts = { ...evalOptions(ctx), phase };
  const dispatches: MoeDispatch[] = a.dispatch
    ? [a.dispatch]
    : hasMoeLayers(ctx.model) && sizes.EP > 1
      ? ['ring-of-experts', 'coalesced-a2a']
      : ['ring-of-experts'];
  const dcps = a.dcp !== undefined ? [a.dcp] : decodeContextParallels(ctx.model, sizes.TP);

  const cands: Candidate<ResourceCostBackend>[] = [];
  for (const mesh of enumeratePlacements(axes, sizes))
    for (const moeDispatch of dispatches)
      for (const decodeContextParallel of dcps) {
        const deployment: Deployment = { chip: ctx.chip, mesh, moeDispatch, decodeContextParallel };
        const input = { model: ctx.model, deployment, workload: ctx.workload };
        const cand =
          a.batch !== undefined && phase.kind === 'decode'
            ? atBatch(input, a.batch, sizes.PP, opts)
            : evaluateWithSearchOpts(input, opts);
        if (cand) cands.push(cand);
      }
  if (!cands.length)
    throw new CliError(
      `no feasible placement of ${Object.entries(sizes)
        .map(([r, v]) => `${r}=${v}`)
        .join(' ')} on ${ctx.chip.id} ${ctx.machineName}` +
        (a.slo ? ` at >= ${a.slo} tok/s/user` : '') +
        (a.batch ? ` at batch ${a.batch}` : ''),
    );
  cands.sort((x, y) => y.score - x.score);
  const all = cands.map((c) => candidateRow(ctx, phase, c));
  return { best: all[0], all, sizes, hints: engineHints(sizes, all[0].dcp ?? 1) };
}

// a fixed decode batch, scored the way the search scores its operating one
function atBatch(
  input: Parameters<typeof evaluateDecodeAtBatch>[0],
  batch: number,
  pp: number,
  opts: Parameters<typeof evaluateWithSearchOpts<ResourceCostBackend>>[1],
): Candidate<ResourceCostBackend> | null {
  const res = evaluateDecodeAtBatch(input, batch, pp, opts);
  if (!res.ok) return null;
  return {
    deployment: input.deployment,
    backend: opts.costBackend,
    batch,
    result: res,
    score: res.tokPerSecPerChip,
  };
}

// how the roles spell as engine flags (the mapping roles.ts documents)
export function engineHints(s: Record<ShardingRole, number>, dcp: number): string[] {
  const world = s.TP * s.DPA;
  const vllm = [`-tp ${s.TP}`, `-dp ${s.DPA}`];
  if (s.PP > 1) vllm.push(`-pp ${s.PP}`);
  if (s.EP > 1) vllm.push('--enable-expert-parallel');
  const sglang = [`--tp ${world}`, `--dp ${s.DPA}`];
  if (s.DPA > 1) sglang.push('--enable-dp-attention');
  if (s.PP > 1) sglang.push(`--pp-size ${s.PP}`);
  if (s.EP > 1) sglang.push(`--ep ${s.EP}`);
  if (dcp > 1) sglang.push(`--decode-context-parallel-size ${dcp}`);
  const out = [`vLLM:   ${vllm.join(' ')}`, `SGLang: ${sglang.join(' ')}`];
  if (s.ETP > 1) out.push(`TRT-LLM: moe_tp ${s.ETP} (ETP has no vLLM/SGLang flag)`);
  return out;
}

export function renderExplanation(e: Explanation, phase: SearchPhase): string {
  const r = e.best;
  const lines: string[] = [];
  const kv = (k: string, v: string) => lines.push(`${k.padEnd(22)} ${v}`);
  kv('model', r.model);
  kv('chip', `${r.chip} ${r.machine} (${r.nChips} chips)`);
  kv('sharding', `${r.sharding}` + (r.dcp && r.dcp > 1 ? ` dcp=${r.dcp}` : ''));
  kv('placement', `${r.placement}` + (r.moeDispatch ? `, ${r.moeDispatch}` : ''));
  kv('phase', r.phase + (r.slo !== null ? ` at >= ${r.slo} tok/s/user` : ''));
  if (r.weightGbPerChip !== null) {
    kv('weights / chip', `${fix(r.weightGbPerChip, 1)} GB`);
    kv(
      'kv / seq / chip',
      `${fix(r.kvMbPerSeqPerChip, 1)} MB (paged ${fix(r.pagedKvMbPerSeqPerChip, 1)} MB + state ${fix(r.stateMbPerSeqPerChip, 1)} MB over ${r.stateSlotsPerSeq} slot${r.stateSlotsPerSeq === 1 ? '' : 's'})`,
    );
    kv('max resident seqs', `${int(r.maxResidentSeqs)} on the machine`);
  }
  if (r.phase === 'decode') {
    kv('batch', `${int(r.batch)} sequences`);
    kv('step time', `${fix(r.stepTimeMs, 3)} ms`);
    kv('tpot', `${fix(r.tpotMs, 3)} ms  (${fix(r.tokPerSecPerUser, 1)} tok/s/user)`);
    kv('tok/s/chip', `${int(r.tokPerSecPerChip)}  (${int(r.tokPerSecMachine)} on the machine)`);
    kv('mfu / mbu', `${fix((r.mfu ?? NaN) * 100, 1)}% / ${fix((r.mbu ?? NaN) * 100, 1)}%`);
  } else {
    kv('sequences', `${int(r.prefill?.batchSeqs)}`);
    kv('pass latency', `${fix(r.prefill?.latencyMs, 3)} ms`);
    kv('tok/s/chip', `${int(r.tokPerSecPerChip)}  (${int(r.tokPerSecMachine)} on the machine)`);
  }
  if (r.busyMs)
    kv(
      'bound by',
      `${r.boundBy}  (busy ms: compute ${fix(r.busyMs.compute, 3)}, memory ${fix(r.busyMs.memory, 3)}, comms ${fix(r.busyMs.comms, 3)})`,
    );
  if (r.usdPerMtok !== null) kv('$/Mtok', `${fix(r.usdPerMtok, 4)} at $${r.costPerHour}/chip-hr`);
  if (r.phase === 'decode' && r.prefill)
    kv(
      'prefill (same config)',
      `${int(r.prefill.tokPerSecPerChip)} tok/s/chip at ${r.prefill.batchSeqs} seqs, TTFT ${fix(r.prefill.ttftMs, 1)} ms`,
    );
  kv(
    'scheduler',
    r.scheduler +
      (r.scheduler === 'naive'
        ? ` (overlap ${r.overlap.memoryOverlap}/${r.overlap.commsOverlap})`
        : ''),
  );
  if (Object.keys(r.statePool).length) kv('state pool', JSON.stringify(r.statePool));
  for (const d of r.diagnostics) kv(d.severity === 'error' ? 'error' : d.severity, d.message);
  lines.push('', ...e.hints);
  if (e.all.length > 1) {
    lines.push('', `${e.all.length} placements priced (${phase.kind}):`);
    for (const x of e.all)
      lines.push(
        `  ${int(x.tokPerSecPerChip).padStart(6)} tok/s/chip  ${x.placement}` +
          (x.moeDispatch ? `, ${x.moeDispatch}` : ''),
      );
  }
  return lines.join('\n');
}
