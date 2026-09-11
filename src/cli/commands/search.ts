import {
  Candidate,
  SearchPhase,
  searchShardings,
  searchTuples,
} from '../../core/engine/optimizer/search';
import type { ResourceCostBackend } from '../../core/engine/sim/cost/types';
import { Context, evalOptions } from '../context';
import { fix, int, table } from '../format';
import { CliError } from '../resolve';
import { candidateRow, EstimateRow, shardingLabel } from '../row';

export interface SearchArgs {
  phase: 'decode' | 'prefill';
  // decode: minimum tok/s/user; the batch backs off to meet it
  slo?: number;
  batching?: 'max' | 'b1';
  // prefill: a single-sequence TTFT pass instead of a full-machine batch
  ttft?: boolean;
  // prefill: sequences per throughput pass (default one per chip)
  prefillSeqs?: number;
  perDollar?: boolean;
}

export function searchPhase(ctx: Context, a: SearchArgs): SearchPhase {
  if (a.phase === 'decode')
    return { kind: 'decode', policy: { sloTokPerSecPerUser: a.slo, batching: a.batching } };
  return a.ttft
    ? { kind: 'prefill', seqs: 1, mode: 'ttft' }
    : { kind: 'prefill', seqs: a.prefillSeqs ?? ctx.nChips, mode: 'throughput' };
}

export interface SearchStep {
  done: number;
  total: number;
  candidate?: Candidate<ResourceCostBackend>;
  // is this tuple's candidate the best so far
  best: boolean;
  elapsedMs: number;
}

export function searchSize(ctx: Context): number {
  return searchTuples(ctx.model, ctx.nChips).length;
}

// Every feasible tuple's best candidate, best first. onStep sees each
// tuple the moment it prices.
export function rankedCandidates(
  ctx: Context,
  a: SearchArgs,
  onStep?: (s: SearchStep) => void,
): Candidate<ResourceCostBackend>[] {
  if (a.perDollar && ctx.chip.costPerHour === undefined)
    throw new CliError(`${ctx.chip.id} has no price; pass --cost-per-hour to rank per dollar`, 2);
  const phase = searchPhase(ctx, a);
  const t0 = performance.now();
  const all: Candidate<ResourceCostBackend>[] = [];
  let best = -Infinity;
  for (const step of searchShardings(ctx.model, ctx.chip, ctx.machine, ctx.workload, {
    ...evalOptions(ctx),
    phase,
    rank: a.perDollar ? 'perDollar' : 'perChip',
  })) {
    const c = step.candidate;
    const isBest = !!c && c.score > best;
    if (c) {
      all.push(c);
      if (isBest) best = c.score;
    }
    onStep?.({ ...step, best: isBest, elapsedMs: performance.now() - t0 });
  }
  return all.sort((x, y) => y.score - x.score);
}

export function rankedRows(
  ctx: Context,
  a: SearchArgs,
  top: number,
  onStep?: (s: SearchStep) => void,
): EstimateRow[] {
  const phase = searchPhase(ctx, a);
  const cands = rankedCandidates(ctx, a, onStep);
  return (top > 0 ? cands.slice(0, top) : cands).map((c) => candidateRow(ctx, phase, c));
}

// one streamed line per priced tuple, the scripts/search.ts format
export function stepLine(ctx: Context, s: SearchStep): string {
  const c = s.candidate;
  const head =
    `[+${s.elapsedMs.toFixed(0).padStart(6)}ms] ` +
    `${String(s.done).padStart(3)}/${s.total} ${s.best ? '*' : ' '} `;
  if (!c) return head + 'infeasible';
  const label = shardingLabel(ctx.model, c.deployment.mesh);
  const dcp = c.deployment.decodeContextParallel ?? 1;
  return (
    head +
    `${Math.round(c.result.tokPerSecPerChip).toString().padStart(6)} tok/s/chip  ` +
    `${label.padEnd(30)}  ` +
    (c.batch !== undefined ? `batch=${c.batch} ` : '') +
    (dcp > 1 ? `dcp=${dcp} ` : '') +
    (c.deployment.mesh.roles.EP.length ? `${c.deployment.moeDispatch} ` : '')
  );
}

export function renderRows(rows: EstimateRow[], feasibleTotal: number): string {
  if (!rows.length) return 'no feasible configuration';
  const decode = rows[0].phase === 'decode';
  const head = decode
    ? ['#', 'tok/s/chip', 'tok/s/user', 'batch', 'residents', '$/Mtok', 'bound', 'sharding']
    : ['#', 'tok/s/chip', 'latency ms', 'seqs', '$/Mtok', 'bound', 'sharding'];
  const body = rows.map((r, i) => {
    const sharding =
      `${r.sharding}` +
      (r.dcp && r.dcp > 1 ? ` dcp=${r.dcp}` : '') +
      (r.moeDispatch ? ` ${r.moeDispatch}` : '');
    return decode
      ? [
          String(i + 1),
          int(r.tokPerSecPerChip),
          fix(r.tokPerSecPerUser, 1),
          int(r.batch),
          int(r.maxResidentSeqs),
          fix(r.usdPerMtok, 3),
          r.boundBy ?? '-',
          sharding,
        ]
      : [
          String(i + 1),
          int(r.tokPerSecPerChip),
          fix(r.prefill?.latencyMs, 2),
          int(r.prefill?.batchSeqs),
          fix(r.usdPerMtok, 3),
          r.boundBy ?? '-',
          sharding,
        ];
  });
  const align = decode
    ? (['r', 'r', 'r', 'r', 'r', 'r', 'l', 'l'] as const)
    : (['r', 'r', 'r', 'r', 'r', 'l', 'l'] as const);
  const r0 = rows[0];
  const title =
    `${r0.model} on ${r0.chip} ${r0.machine} (${r0.nChips} chips), ${r0.phase}` +
    (r0.slo !== null ? ` at >= ${r0.slo} tok/s/user` : '') +
    `: top ${rows.length} of ${feasibleTotal} feasible`;
  return `${title}\n${table(head, body, [...align])}`;
}
