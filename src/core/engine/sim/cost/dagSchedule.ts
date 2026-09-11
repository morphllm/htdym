import { collectiveCost } from './helpers/collectives';
import { naiveOpCost } from './helpers/naiveOpCost';
import type { Deployment } from '../../surface/deploy';
import type { ExpandedOp, OpId, Segment } from '../ir/ops';
import type { HardwareResource } from '../../surface/api';
import type { OpCost } from './helpers/naiveOpCost';
import type { CostBackend, ResourceTraceCost } from './types';

const RESOURCES: readonly HardwareResource[] = ['compute', 'memory', 'comms'];

// What set a trace's time: one of the three streams ran out of room, or
// the op graph's dependency chain did before any stream filled.
export type ScheduleBound = HardwareResource | 'deps';

export interface DagScheduleTraceCost extends ResourceTraceCost {
  // per-resource work that ran under something else (busy - parts)
  hidden: Record<HardwareResource, number>;
  bound: ScheduleBound;
}

export interface DagScheduleOptions {
  // weight streaming is issued ahead of the ops that need it, so it never
  // sits on the dependency chain: the memory stream's load still counts,
  // the recurrence does not wait for it. Off puts every load back in
  // line, the way a stack without prefetch runs. Default on.
  prefetchWeights?: boolean;
}

// Overlap read off the op graph instead of two hand-set fractions. Each
// op keeps the naive roofline price on its streams; a segment then runs as
// a software-pipelined loop over its repeats, whose period is the larger
// of the resource bound (the fullest stream) and the recurrence bound (the
// longest dependency chain through one iteration, each op as wide as its
// slowest stream). The first iteration also waits for its weights. The
// result is always within [max stream, sum of streams]: nothing hides more
// than a stream can absorb, and nothing serializes past the chain.
export function makeDagScheduleBackend(options: DagScheduleOptions = {}) {
  const prefetch = options.prefetchWeights ?? true;
  return ((deployment: Deployment) => {
    return {
      // collectives are priced exactly as the naive backend prices them, so
      // the two share reshard plans
      priceCollectiveHash: JSON.stringify(['naive-op-cost-sum', deployment.mesh.dims]),
      priceCollective: (kind, over, input, elemBytes) =>
        collectiveCost(kind, over, input, elemBytes, deployment.mesh.dims),
      priceTrace: (trace: Segment[]): DagScheduleTraceCost => {
        const busy = zero();
        const parts = zero();
        const reason: Record<ScheduleBound, number> = { ...zero(), deps: 0 };
        const busyPerOp = new Map<OpId, Record<HardwareResource, number>>();

        for (const s of trace) {
          const costs = new Map<OpId, OpCost>();
          const load = zero();
          for (const op of s.ops) {
            const c = naiveOpCost(op, deployment);
            costs.set(op.id, c);
            for (const r of RESOURCES) {
              load[r] += c[r];
              busy[r] += c[r] * s.repeat;
            }
            busyPerOp.set(op.id, {
              compute: c.compute * s.repeat,
              memory: c.memory * s.repeat,
              comms: c.comms * s.repeat,
            });
          }

          const full = criticalPath(s.ops, costs, () => true);
          const steady = prefetch
            ? criticalPath(s.ops, costs, (op) => op.kind !== 'weight-load')
            : full;

          // the first iteration: streams or the chain with its weights
          charge(parts, reason, load, full, 1);
          // every later one: streams or the chain, weights already there
          charge(parts, reason, load, steady, s.repeat - 1);
        }

        const time = parts.compute + parts.memory + parts.comms;
        const hidden = zero();
        for (const r of RESOURCES) hidden[r] = Math.max(0, busy[r] - parts[r]);
        const bound = (Object.keys(reason) as ScheduleBound[]).reduce((a, b) =>
          reason[b] > reason[a] ? b : a,
        );
        return { time, busy, busyPerOp, parts, hidden, bound };
      },
    };
  }) satisfies CostBackend;
}

const zero = (): Record<HardwareResource, number> => ({ compute: 0, memory: 0, comms: 0 });

// The longest dependency chain through the segment's ops, each op as long
// as its slowest stream. Deps outside the segment finished before it
// started. The chain's time is returned split by the resource each op on
// it was widest on, so the caller can attribute it.
interface Chain {
  time: number;
  parts: Record<HardwareResource, number>;
}

function criticalPath(
  ops: ExpandedOp[],
  costs: Map<OpId, OpCost>,
  include: (op: ExpandedOp) => boolean,
): Chain {
  const byId = new Map(ops.filter(include).map((op) => [op.id, op]));
  const memo = new Map<OpId, Chain>();
  const empty = (): Chain => ({ time: 0, parts: zero() });

  const finish = (id: OpId): Chain => {
    const op = byId.get(id);
    if (!op) return empty();
    const hit = memo.get(id);
    if (hit) return hit;

    let best = empty();
    for (const dep of op.deps) {
      const chain = finish(dep);
      if (chain.time > best.time) best = chain;
    }
    const c = costs.get(id)!;
    const widest = RESOURCES.reduce((a, b) => (c[b] > c[a] ? b : a));
    const out: Chain = {
      time: best.time + c[widest],
      parts: { ...best.parts, [widest]: best.parts[widest] + c[widest] },
    };
    memo.set(id, out);
    return out;
  };

  let best = empty();
  for (const id of byId.keys()) {
    const chain = finish(id);
    if (chain.time > best.time) best = chain;
  }
  return best;
}

// One iteration's period is the larger of the fullest stream and the
// chain: charge that many iterations to whichever it was.
function charge(
  parts: Record<HardwareResource, number>,
  reason: Record<ScheduleBound, number>,
  load: Record<HardwareResource, number>,
  chain: Chain,
  iterations: number,
): void {
  if (iterations <= 0) return;
  const fullest = RESOURCES.reduce((a, b) => (load[b] > load[a] ? b : a));
  if (load[fullest] >= chain.time) {
    parts[fullest] += load[fullest] * iterations;
    reason[fullest] += load[fullest] * iterations;
  } else {
    for (const r of RESOURCES) parts[r] += chain.parts[r] * iterations;
    reason.deps += chain.time * iterations;
  }
}
