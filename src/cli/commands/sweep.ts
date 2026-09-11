import { Worker } from 'node:worker_threads';
import { Context, makeContext, SharedOptions } from '../context';
import { fix, int, table } from '../format';
import { EstimateRow, infeasibleRow } from '../row';
import { rankedRows, SearchArgs, searchPhase } from './search';

// one search: a model on a chip's machine, one phase, one SLO
export interface SweepCell {
  model: string;
  chip: string;
  machine: string;
  phase: 'decode' | 'prefill';
  slo: number | null;
}

export interface SweepSpec {
  model: string;
  chips: string[];
  machines: string[];
  phases: ('decode' | 'prefill')[];
  // null = no SLO, batch to capacity
  slos: (number | null)[];
}

export function sweepCells(s: SweepSpec): SweepCell[] {
  const cells: SweepCell[] = [];
  for (const chip of s.chips)
    for (const machine of s.machines)
      for (const phase of s.phases)
        for (const slo of phase === 'decode' ? s.slos : [null])
          cells.push({ model: s.model, chip, machine, phase, slo });
  return cells;
}

export function cellName(c: SweepCell): string {
  return `${c.chip}-${c.machine}-${c.phase}-${c.slo ?? 'none'}`;
}

// One cell's best configuration, or an infeasible row carrying the reason.
// Never throws: a sweep reports every cell.
export function runCell(cell: SweepCell, shared: SharedOptions): EstimateRow {
  const a: SearchArgs = { phase: cell.phase, slo: cell.slo ?? undefined };
  let ctx: Context;
  try {
    ctx = makeContext(cell.model, cell.chip, cell.machine, shared);
  } catch (err) {
    return {
      ...infeasibleRowFor(cell, shared),
      error: err instanceof Error ? err.message : String(err),
    };
  }
  try {
    const [row] = rankedRows(ctx, a, 1);
    return row ?? infeasibleRow(ctx, searchPhase(ctx, a));
  } catch (err) {
    return infeasibleRow(
      ctx,
      searchPhase(ctx, a),
      err instanceof Error ? err.message : String(err),
    );
  }
}

// a row for a cell whose chip or machine did not even resolve
function infeasibleRowFor(cell: SweepCell, shared: SharedOptions): EstimateRow {
  const ctx = {
    model: { name: cell.model },
    chip: { id: cell.chip },
    machineName: cell.machine,
    nChips: 0,
    workload: { prefillLen: shared.prefillLen, generateLen: shared.genLen },
    opts: shared,
  } as unknown as Context;
  return infeasibleRow(
    ctx,
    cell.phase === 'decode'
      ? { kind: 'decode', policy: { sloTokPerSecPerUser: cell.slo ?? undefined } }
      : { kind: 'prefill', seqs: 0, mode: 'throughput' },
  );
}

export interface WorkerJob {
  kind: 'sweep';
  cells: SweepCell[];
  shared: SharedOptions;
}

// Run the cells, one worker thread per chip (a chip's warm reshard-plan
// caches stay with the thread that will be asked about it again), at most
// `jobs` at a time. jobs <= 1 or no worker script runs everything inline.
export async function runSweep(
  cells: SweepCell[],
  shared: SharedOptions,
  jobs: number,
  onRow: (row: EstimateRow) => void,
  workerScript?: URL,
): Promise<EstimateRow[]> {
  const rows: EstimateRow[] = [];
  const emit = (row: EstimateRow) => {
    rows.push(row);
    onRow(row);
  };
  if (jobs <= 1 || !workerScript) {
    for (const cell of cells) emit(runCell(cell, shared));
    return rows;
  }

  const byChip = new Map<string, SweepCell[]>();
  for (const c of cells) byChip.set(c.chip, [...(byChip.get(c.chip) ?? []), c]);
  const queue = [...byChip.values()];
  const runOne = (group: SweepCell[]) =>
    new Promise<void>((resolve, reject) => {
      const w = new Worker(workerScript, {
        workerData: { kind: 'sweep', cells: group, shared } satisfies WorkerJob,
      });
      w.on('message', (row: EstimateRow) => emit(row));
      w.on('error', reject);
      w.on('exit', (code) =>
        code === 0 ? resolve() : reject(new Error(`sweep worker exited with ${code}`)),
      );
    });
  const lanes = Array.from({ length: Math.min(jobs, queue.length) }, async () => {
    for (let g = queue.shift(); g; g = queue.shift()) await runOne(g);
  });
  await Promise.all(lanes);
  return rows;
}

// The grouped tables of a sweep: decode at each SLO, then prefill, best
// first within each, then the cells that had no feasible configuration.
export function renderReport(rows: EstimateRow[]): string {
  const ok = rows.filter((r) => r.feasible);
  const bad = rows.filter((r) => !r.feasible);
  const out: string[] = [];
  const section = (title: string, sub: EstimateRow[]) => {
    if (!sub.length) return;
    sub.sort((a, b) => (b.tokPerSecPerChip ?? 0) - (a.tokPerSecPerChip ?? 0));
    out.push(
      `=== ${title} ===`,
      table(
        ['chip', 'machine', 'n', 'tok/s/chip', 'tok/s/user', 'batch', '$/Mtok', 'sharding'],
        sub.map((r) => [
          r.chip,
          r.machine,
          String(r.nChips),
          int(r.tokPerSecPerChip),
          fix(r.tokPerSecPerUser, 1),
          int(r.batch ?? r.prefill?.batchSeqs),
          fix(r.usdPerMtok, 3),
          `${r.sharding}` +
            (r.dcp && r.dcp > 1 ? ` dcp=${r.dcp}` : '') +
            (r.moeDispatch ? ` ${r.moeDispatch}` : ''),
        ]),
        ['l', 'l', 'r', 'r', 'r', 'r', 'r', 'l'],
      ),
      '',
    );
  };
  section(
    'DECODE max throughput (no SLO)',
    ok.filter((r) => r.phase === 'decode' && r.slo === null),
  );
  const slos = [...new Set(ok.filter((r) => r.slo !== null).map((r) => r.slo!))].sort(
    (a, b) => a - b,
  );
  for (const slo of slos)
    section(
      `DECODE at >= ${slo} tok/s/user`,
      ok.filter((r) => r.phase === 'decode' && r.slo === slo),
    );
  section(
    'PREFILL throughput',
    ok.filter((r) => r.phase === 'prefill'),
  );
  out.push(`${rows.length} cells, ${ok.length} feasible, ${bad.length} infeasible`);
  for (const r of bad)
    out.push(
      `  infeasible: ${r.chip} ${r.machine} ${r.phase} slo=${r.slo ?? 'none'}` +
        (r.error ? `: ${r.error}` : ''),
    );
  return out.join('\n');
}
