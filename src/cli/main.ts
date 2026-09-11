import { isMainThread, parentPort, workerData } from 'node:worker_threads';
import { readdirSync, readFileSync, statSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { ShardingRole } from '../core/engine/sim/ir/sharding/roles';
import type { MoeDispatch } from '../core/engine/surface/deploy';
import {
  flag,
  list,
  num,
  oneOf,
  parse,
  SHARED_HELP,
  SHARED_OPTIONS,
  sharedOptions,
  str,
  Values,
} from './args';
import { makeContext } from './context';
import { CliError, resolveModel } from './resolve';
import { VERSION } from './version';
import { chipRows, renderChips } from './commands/chips';
import { explain, renderExplanation } from './commands/explain';
import { modelRows, renderModels } from './commands/models';
import { rankedRows, renderRows, searchPhase, searchSize, stepLine } from './commands/search';
import { cellName, renderReport, runCell, runSweep, sweepCells, WorkerJob } from './commands/sweep';
import type { EstimateRow } from './row';

const USAGE = `estimate ${VERSION}: static roofline estimates of LLM serving (htdym)

usage
  estimate models [--json]
  estimate chips [--model M] [--json]
  estimate search  M CHIP MACHINE [--phase decode|prefill] [--slo N] [--batching max|b1]
                   [--ttft] [--prefill-seqs N] [--top N] [--per-dollar] [--quiet] [--json]
  estimate top     M CHIP MACHINE [same as search; silent, --top 10]
  estimate explain M CHIP MACHINE --sizes DPA=16,TP=2,EP=32 [--dcp N] [--dispatch D]
                   [--batch N | --slo N | --batching max|b1] [--phase decode|prefill] [--json]
  estimate sweep   --model M --chips a,b --machines 32x1,8x2 [--phases decode,prefill]
                   [--slos none,20,50] [--out DIR] [--jobs N] [--table]
  estimate report  <rows.jsonl | DIR>

M is a preset name, a unique prefix, or an alias (k3, dsv4flash, gptoss120b, ...).
CHIP is a chip id (estimate chips). MACHINE is <chips per domain>x<nodes> on a
switched fabric (32x1, 8x2) or a slice name on a ring fabric (4x4x4).

Every number is an estimate from a static roofline model, not a measurement.
${SHARED_HELP}
`;

const out = (s: string) => process.stdout.write(s + '\n');
const err = (s: string) => process.stderr.write(s + '\n');
const jsonl = (rows: EstimateRow[]) => rows.forEach((r) => out(JSON.stringify(r)));

const SEARCH_OPTIONS = {
  ...SHARED_OPTIONS,
  phase: { type: 'string' },
  slo: { type: 'string' },
  batching: { type: 'string' },
  ttft: { type: 'boolean' },
  'prefill-seqs': { type: 'string' },
  top: { type: 'string' },
  'per-dollar': { type: 'boolean' },
  quiet: { type: 'boolean', short: 'q' },
} as const;

function searchArgs(values: Values) {
  return {
    phase: oneOf(values, 'phase', ['decode', 'prefill'] as const, 'decode'),
    slo: num(values, 'slo'),
    batching: str(values, 'batching')
      ? oneOf(values, 'batching', ['max', 'b1'] as const, 'max')
      : undefined,
    ttft: flag(values, 'ttft'),
    prefillSeqs: num(values, 'prefill-seqs'),
    perDollar: flag(values, 'per-dollar'),
  };
}

function needPositionals(p: string[], n: number, what: string) {
  if (p.length < n) throw new CliError(`expected ${what}`, 2);
}

function cmdModels(argv: string[]) {
  const { values } = parse(argv, SHARED_OPTIONS, false);
  const rows = modelRows();
  if (flag(values, 'json')) rows.forEach((r) => out(JSON.stringify(r)));
  else out(renderModels(rows));
}

function cmdChips(argv: string[]) {
  const { values } = parse(argv, { ...SHARED_OPTIONS, model: { type: 'string' } }, false);
  const model = str(values, 'model');
  const rows = chipRows(model ? resolveModel(model) : undefined);
  if (flag(values, 'json')) rows.forEach((r) => out(JSON.stringify(r)));
  else out(renderChips(rows));
}

function cmdSearch(argv: string[], silent: boolean) {
  const { values, positionals } = parse(argv, SEARCH_OPTIONS);
  needPositionals(positionals, 3, 'MODEL CHIP MACHINE');
  const ctx = makeContext(positionals[0], positionals[1], positionals[2], sharedOptions(values));
  const a = searchArgs(values);
  const top = num(values, 'top') ?? (silent ? 10 : 5);
  const quiet = silent || flag(values, 'quiet');
  if (!quiet)
    err(
      `${ctx.model.name} on ${ctx.chip.id} ${ctx.machineName} (${ctx.nChips} chips): ` +
        `${searchSize(ctx)} role-size tuples, ${a.phase}` +
        (a.slo ? ` at >= ${a.slo} tok/s/user` : ''),
    );
  let feasible = 0;
  const rows = rankedRows(ctx, a, top, (s) => {
    if (s.candidate) feasible++;
    if (!quiet) err(stepLine(ctx, s));
  });
  if (flag(values, 'json')) jsonl(rows);
  else out(renderRows(rows, feasible));
  if (!rows.length) throw new CliError('no feasible configuration', 1);
}

function cmdExplain(argv: string[]) {
  const { values, positionals } = parse(argv, {
    ...SEARCH_OPTIONS,
    sizes: { type: 'string' },
    dcp: { type: 'string' },
    dispatch: { type: 'string' },
    batch: { type: 'string' },
  });
  needPositionals(positionals, 3, 'MODEL CHIP MACHINE');
  const ctx = makeContext(positionals[0], positionals[1], positionals[2], sharedOptions(values));
  const sizes: Partial<Record<ShardingRole, number>> = {};
  for (const part of list(values, 'sizes')) {
    const m = /^(PP|DPA|TP|EP|ETP)=(\d+)$/i.exec(part);
    if (!m) throw new CliError(`--sizes takes ROLE=N pairs (DPA=16,TP=2,EP=32), got "${part}"`, 2);
    sizes[m[1].toUpperCase() as ShardingRole] = Number(m[2]);
  }
  const dispatch = str(values, 'dispatch');
  const dispatches = ['ring-of-experts', 'coalesced-a2a', 'expanded-a2a'] as const;
  if (dispatch && !(dispatches as readonly string[]).includes(dispatch))
    throw new CliError(`--dispatch must be one of ${dispatches.join('|')}`, 2);
  const a = {
    ...searchArgs(values),
    sizes,
    dcp: num(values, 'dcp'),
    dispatch: dispatch as MoeDispatch | undefined,
    batch: num(values, 'batch'),
  };
  const e = explain(ctx, a);
  if (flag(values, 'json'))
    out(JSON.stringify({ ...e.best, hints: e.hints, placements: e.all.length }));
  else out(renderExplanation(e, searchPhase(ctx, a)));
}

async function cmdSweep(argv: string[]) {
  const { values } = parse(
    argv,
    {
      ...SHARED_OPTIONS,
      model: { type: 'string' },
      chips: { type: 'string' },
      machines: { type: 'string' },
      phases: { type: 'string' },
      slos: { type: 'string' },
      out: { type: 'string' },
      jobs: { type: 'string' },
      table: { type: 'boolean' },
    },
    false,
  );
  const model = str(values, 'model');
  if (!model) throw new CliError('--model is required', 2);
  const chips = list(values, 'chips');
  const machines = list(values, 'machines');
  if (!chips.length || !machines.length)
    throw new CliError('--chips and --machines are required', 2);
  const phases = list(values, 'phases', ['decode']).map((p) => {
    if (p !== 'decode' && p !== 'prefill') throw new CliError(`--phases takes decode,prefill`, 2);
    return p;
  });
  const slos = list(values, 'slos', ['none']).map((s) => {
    if (s === 'none') return null;
    const n = Number(s);
    if (!Number.isFinite(n) || n <= 0)
      throw new CliError(`--slos takes none or tok/s/user numbers`, 2);
    return n;
  });
  const shared = sharedOptions(values);
  const cells = sweepCells({ model, chips, machines, phases, slos });
  const outDir = str(values, 'out') ?? join(tmpdir(), `estimate-sweep-${Date.now()}`);
  mkdirSync(outDir, { recursive: true });
  const jobs = num(values, 'jobs') ?? Math.min(chips.length, 4);
  err(`${cells.length} cells over ${chips.length} chips, ${jobs} at a time, rows under ${outDir}`);
  const t0 = performance.now();
  const rows = await runSweep(
    cells,
    shared,
    jobs,
    (row) => {
      writeFileSync(join(outDir, `${cellName(rowCell(row))}.json`), JSON.stringify(row) + '\n');
      if (!flag(values, 'table')) out(JSON.stringify(row));
      err(
        `[+${((performance.now() - t0) / 1000).toFixed(1).padStart(6)}s] ${cellName(rowCell(row))}: ` +
          (row.feasible
            ? `${Math.round(row.tokPerSecPerChip!)} tok/s/chip ${row.sharding}`
            : `infeasible${row.error ? ` (${row.error})` : ''}`),
      );
    },
    new URL(import.meta.url),
  );
  if (flag(values, 'table')) out(renderReport(rows));
}

const rowCell = (r: EstimateRow) => ({
  model: r.model,
  chip: r.chip,
  machine: r.machine,
  phase: r.phase,
  slo: r.slo,
});

function cmdReport(argv: string[]) {
  const { positionals } = parse(argv, SHARED_OPTIONS);
  needPositionals(positionals, 1, 'a rows.jsonl file or a directory of row files');
  const rows: EstimateRow[] = [];
  for (const p of positionals) {
    const files = statSync(p).isDirectory()
      ? readdirSync(p)
          .filter((f) => f.endsWith('.json') || f.endsWith('.jsonl'))
          .map((f) => join(p, f))
      : [p];
    for (const f of files)
      for (const line of readFileSync(f, 'utf8').split('\n'))
        if (line.trim().startsWith('{')) rows.push(JSON.parse(line) as EstimateRow);
  }
  out(renderReport(rows));
}

export async function main(argv: string[]): Promise<number> {
  const [cmd, ...rest] = argv;
  try {
    if (!cmd || cmd === '--help' || cmd === '-h' || cmd === 'help') {
      out(USAGE);
      return cmd ? 0 : 2;
    }
    if (cmd === '--version') {
      out(VERSION);
      return 0;
    }
    if (rest.includes('--help') || rest.includes('-h')) {
      out(USAGE);
      return 0;
    }
    switch (cmd) {
      case 'models':
        cmdModels(rest);
        break;
      case 'chips':
        cmdChips(rest);
        break;
      case 'search':
        cmdSearch(rest, false);
        break;
      case 'top':
        cmdSearch(rest, true);
        break;
      case 'explain':
        cmdExplain(rest);
        break;
      case 'sweep':
        await cmdSweep(rest);
        break;
      case 'report':
        cmdReport(rest);
        break;
      default:
        throw new CliError(`unknown command "${cmd}"\n\n${USAGE}`, 2);
    }
    return 0;
  } catch (e) {
    if (e instanceof CliError) {
      err(`estimate: ${e.message}`);
      return e.exitCode;
    }
    throw e;
  }
}

if (!isMainThread && (workerData as WorkerJob | undefined)?.kind === 'sweep') {
  const job = workerData as WorkerJob;
  for (const cell of job.cells) parentPort!.postMessage(runCell(cell, job.shared));
} else if (isMainThread) {
  void main(process.argv.slice(2)).then((code) => {
    process.exitCode = code;
  });
}
