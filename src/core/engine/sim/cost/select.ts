import { makeDagScheduleBackend } from './dagSchedule';
import { makeNaiveOpCostSumBackend } from './naiveOpCostSum';
import type { ResourceCostBackend } from './types';

// 'naive' hides fixed fractions of the memory and comms streams behind the
// widest one (the two overlap constants); 'dag' reads the overlap off the
// op graph and ignores the constants.
export type Scheduler = 'naive' | 'dag';

export interface OverlapOptions {
  memoryOverlap: number;
  commsOverlap: number;
  scheduler?: Scheduler;
}

export function makeCostBackend(o: OverlapOptions): ResourceCostBackend {
  return o.scheduler === 'dag' ? makeDagScheduleBackend() : makeNaiveOpCostSumBackend(o);
}
