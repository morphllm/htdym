import type { Roofline } from '../core/engine/roofline';
import type { Diagnostic } from '../core/engine/surface/deploy';
import type { ChipSpec } from '../core/hardware/chips';
import type { ChipOnMachine } from '../core/hardware/machines';
import type { OverlapOptions } from '../core/engine/sim/cost/select';
import type { Boundedness, ComponentTimes, ConfigResult } from '../core/engine/optimizer/enrich';

export type { Diagnostic };

// A chip as the UI holds it: the editable spec plus the machine the user
// picked for it.
export type UiChip = ChipOnMachine;

export type { Boundedness, ComponentTimes };

export interface UiWorkload {
  prefillLen: number;
  generateLen: number;
  // 0 = unconstrained sentinel while the field is edited (as the old UI)
  sloTokPerSecPerUser?: number;
  batching?: 'max' | 'b1';
}

// the two overlap constants plus which scheduler reads them ('naive'
// applies them, 'dag' reads overlap off the op graph and ignores them)
export type UiOverlap = OverlapOptions;

// A streamed configuration row: the engine's result plus the efficiency
// figures filled at render time from the live prices and baseline, never
// by the worker.
export interface UiResult extends ConfigResult {
  // cost-efficiency for the whole workload vs the HMVP baseline
  requestEff?: number;
  prefill?: ConfigResult['prefill'] & {
    // (rate ÷ relative price) over the HMVP's rate
    eff?: number;
    // rate over the HMVP's rate — pure speed, price not included
    relRate?: number;
  };
  decode?: ConfigResult['decode'] & { eff?: number; relRate?: number };
}

// One chip's slot in the leaderboard: its fixed machine, the hardware
// rooflines, and the streamed configs (best-first arrival order).
export interface UiGroup {
  key: string;
  chip: ChipSpec;
  machineLabel: string;
  nChips: number;
  hardware: Roofline | null;
  configs: UiResult[];
  // streaming progress for this chip's search
  done: number;
  total: number;
  // the whole chip failed to evaluate (e.g. unsupported dtype)
  error?: string;
}

// glyph and color class for a diagnostic line, by severity
export const DIAG_GLYPH: Record<Diagnostic['severity'], string> = {
  error: '✕',
  warning: '⚠',
  info: 'ⓘ',
};
export const DIAG_CLASS: Record<Diagnostic['severity'], string> = {
  error: 'diag-err',
  warning: 'diag-warn',
  info: 'diag-info',
};

export function hasError(r: UiResult): boolean {
  return r.diagnostics.some((d) => d.severity === 'error');
}

export function hasWarning(r: UiResult): boolean {
  return r.diagnostics.some((d) => d.severity === 'warning');
}

// Traffic-light ramp for attainment gauges (was FabricView's).
export function gaugeColor(frac: number): string {
  return frac >= 0.9 ? 'var(--status-ok)' : frac >= 0.5 ? 'var(--bar-warn)' : 'var(--red)';
}

// Progress-ring tone: red early, yellow past 30% priced, green when complete.
export function ringTone(frac: number): string {
  return frac >= 1 ? 'var(--green)' : frac >= 0.3 ? 'var(--bar-warn)' : 'var(--red)';
}
