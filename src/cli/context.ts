import { makeCostBackend, OverlapOptions } from '../core/engine/sim/cost/select';
import type { ResourceCostBackend } from '../core/engine/sim/cost/types';
import type { EvalOptions, StatePoolOptions } from '../core/engine/surface/api';
import { ChipSpec } from '../core/hardware/chips';
import { Machine, machineName, machineSize, parseMachine } from '../core/hardware/machines';
import { ModelSpec } from '../core/model/models';
import { resolveChip, resolveModel } from './resolve';

// Every knob the subcommands share, as plain data (it crosses into sweep
// worker threads).
export interface SharedOptions {
  prefillLen: number;
  genLen: number;
  overlap: OverlapOptions;
  // chip overrides: rental price and the fraction of HBM the stack lets the
  // cache have (SGLang's mem_fraction_static), applied as a smaller chip
  costPerHour?: number;
  memFraction?: number;
  statePool: Partial<StatePoolOptions>;
}

export const DEFAULT_SHARED: SharedOptions = {
  prefillLen: 4096,
  genLen: 1024,
  overlap: { memoryOverlap: 0.9, commsOverlap: 0.65, scheduler: 'naive' },
  statePool: {},
};

// One (model, chip, machine) the commands evaluate on, resolved.
export interface Context {
  model: ModelSpec;
  chip: ChipSpec;
  machine: Machine;
  machineName: string;
  nChips: number;
  workload: { prefillLen: number; generateLen: number };
  opts: SharedOptions;
  backend: ResourceCostBackend;
}

export function applyChipOverrides(chip: ChipSpec, opts: SharedOptions): ChipSpec {
  return {
    ...chip,
    ...(opts.memFraction !== undefined ? { hbmCapacity: opts.memFraction * chip.hbmCapacity } : {}),
    ...(opts.costPerHour !== undefined ? { costPerHour: opts.costPerHour } : {}),
  };
}

export function makeContext(
  modelArg: string,
  chipArg: string,
  machineArg: string,
  opts: SharedOptions = DEFAULT_SHARED,
): Context {
  const model = resolveModel(modelArg);
  const chip = applyChipOverrides(resolveChip(chipArg), opts);
  const machine = parseMachine(chip, machineArg);
  return {
    model,
    chip,
    machine,
    machineName: machineName(machine),
    nChips: machineSize(machine),
    workload: { prefillLen: opts.prefillLen, generateLen: opts.genLen },
    opts,
    backend: makeCostBackend(opts.overlap),
  };
}

export function evalOptions(ctx: Context): EvalOptions<ResourceCostBackend> {
  return { costBackend: ctx.backend, statePool: ctx.opts.statePool };
}
