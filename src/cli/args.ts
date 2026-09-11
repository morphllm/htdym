import { parseArgs, type ParseArgsConfig } from 'node:util';
import type { Scheduler } from '../core/engine/sim/cost/select';
import { DEFAULT_SHARED, SharedOptions } from './context';
import { CliError } from './resolve';

type OptionTable = NonNullable<ParseArgsConfig['options']>;
export type Values = Record<string, string | boolean | undefined>;

// the knobs every evaluating subcommand takes
export const SHARED_OPTIONS = {
  'prefill-len': { type: 'string' },
  'gen-len': { type: 'string' },
  overlap: { type: 'string' },
  scheduler: { type: 'string' },
  'cost-per-hour': { type: 'string' },
  'mem-fraction': { type: 'string' },
  'state-slots': { type: 'string' },
  'state-dtype': { type: 'string' },
  'spec-slots': { type: 'string' },
  json: { type: 'boolean' },
  help: { type: 'boolean', short: 'h' },
} satisfies OptionTable;

export const SHARED_HELP = `
shared options
  --prefill-len N        prompt tokens per sequence (default 4096)
  --gen-len N            generated tokens per sequence (default 1024)
  --overlap M,C          naive scheduler: fraction of memory and comms traffic
                         hidden behind the widest stream (default 0.9,0.65)
  --scheduler naive|dag  dag reads overlap off the op graph and ignores --overlap
  --cost-per-hour USD    override the chip's rental price ($/chip-hour)
  --mem-fraction F       fraction of HBM the stack lets weights+cache use
                         (SGLang mem_fraction_static); applied as a smaller chip
  --state-slots N        recurrent-state slots reserved per sequence on
                         linear-attention layers (default 1; SGLang radix cache 5)
  --state-dtype T        capacity dtype of that state: fp32 (default) or bf16
  --spec-slots N         extra state slots per sequence for speculative decoding
  --json                 machine-readable output (one JSON object per line)
`.trimEnd();

export function parse<T extends OptionTable>(
  argv: string[],
  options: T,
  allowPositionals = true,
): { values: Values; positionals: string[] } {
  try {
    const { values, positionals } = parseArgs({
      args: argv,
      options,
      allowPositionals,
      strict: true,
    });
    return { values: values as Values, positionals };
  } catch (err) {
    throw new CliError(err instanceof Error ? err.message : String(err), 2);
  }
}

export function num(values: Values, name: string): number | undefined {
  const v = values[name];
  if (v === undefined || typeof v === 'boolean') return undefined;
  const n = Number(v);
  if (!Number.isFinite(n)) throw new CliError(`--${name} expects a number, got "${v}"`, 2);
  return n;
}

export function str(values: Values, name: string): string | undefined {
  const v = values[name];
  return typeof v === 'string' ? v : undefined;
}

export function flag(values: Values, name: string): boolean {
  return values[name] === true;
}

export function oneOf<const T extends readonly string[]>(
  values: Values,
  name: string,
  allowed: T,
  dflt: T[number],
): T[number] {
  const v = str(values, name) ?? dflt;
  if (!allowed.includes(v))
    throw new CliError(`--${name} must be one of ${allowed.join('|')}, got "${v}"`, 2);
  return v as T[number];
}

export function list(values: Values, name: string, dflt: string[] = []): string[] {
  const v = str(values, name);
  return v === undefined
    ? dflt
    : v
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean);
}

export function sharedOptions(values: Values): SharedOptions {
  const overlapArg = str(values, 'overlap');
  const [memoryOverlap, commsOverlap] = overlapArg
    ? overlapArg.split(',').map(Number)
    : [DEFAULT_SHARED.overlap.memoryOverlap, DEFAULT_SHARED.overlap.commsOverlap];
  if (
    overlapArg &&
    ![memoryOverlap, commsOverlap].every((x) => Number.isFinite(x) && x >= 0 && x <= 1)
  )
    throw new CliError(`--overlap expects two fractions "M,C", got "${overlapArg}"`, 2);
  const scheduler: Scheduler = oneOf(values, 'scheduler', ['naive', 'dag'] as const, 'naive');

  const statePool: SharedOptions['statePool'] = {};
  const slots = num(values, 'state-slots');
  if (slots !== undefined) statePool.slotsPerSeq = slots;
  const spec = num(values, 'spec-slots');
  if (spec !== undefined) statePool.specSlots = spec;
  const dtype = str(values, 'state-dtype');
  if (dtype !== undefined) {
    const bytes = { fp32: 4, bf16: 2, fp16: 2, fp8: 1 }[dtype];
    if (!bytes) throw new CliError(`--state-dtype must be fp32|bf16|fp16|fp8, got "${dtype}"`, 2);
    statePool.stateDtypeBytes = bytes;
  }

  const memFraction = num(values, 'mem-fraction');
  if (memFraction !== undefined && !(memFraction > 0 && memFraction <= 1))
    throw new CliError(`--mem-fraction must be in (0, 1], got ${memFraction}`, 2);

  return {
    prefillLen: num(values, 'prefill-len') ?? DEFAULT_SHARED.prefillLen,
    genLen: num(values, 'gen-len') ?? DEFAULT_SHARED.genLen,
    overlap: { memoryOverlap, commsOverlap, scheduler },
    costPerHour: num(values, 'cost-per-hour'),
    memFraction,
    statePool,
  };
}
