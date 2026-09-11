import { expect, test } from 'vitest';
import { makeDagScheduleBackend } from '../src/core/engine/sim/cost/dagSchedule';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { makeCostBackend } from '../src/core/engine/sim/cost/select';
import { naiveOpCost } from '../src/core/engine/sim/cost/helpers/naiveOpCost';
import type { ExpandedOp, OpId, Segment } from '../src/core/engine/sim/ir/ops';
import { tt } from '../src/core/engine/sim/ir/tensors';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { evaluatePrefill } from '../src/core/engine/sim/run/prefill';
import { searchShardings } from '../src/core/engine/optimizer/search';
import { Deployment, makeMesh } from '../src/core/engine/surface/deploy';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { MODEL_PRESETS } from '../src/core/model/models';

const h100 = CHIPS_BY_ID['h100-sxm'];
const naive = makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 });
const serial = makeNaiveOpCostSumBackend({ memoryOverlap: 0, commsOverlap: 0 });
const dag = makeDagScheduleBackend();

function deploymentOn(chip = h100, tp = 1): Deployment {
  const axes = deployedAxes(chip.interconnect, { domain: tp, nodes: 1 });
  const names = tp > 1 ? axes.map((a) => a.name) : [];
  return {
    chip,
    mesh: makeMesh(axes, { DPA: [], TP: names, EP: names, ETP: [], PP: [] }),
    moeDispatch: 'ring-of-experts',
  };
}

const id = (s: string) => s as OpId;
const RES = ['compute', 'memory', 'comms'] as const;
const sum = (c: Record<(typeof RES)[number], number>) => c.compute + c.memory + c.comms;
const max = (c: Record<(typeof RES)[number], number>) => Math.max(c.compute, c.memory, c.comms);

// a decode-shaped gemm: few rows, so it streams more than it multiplies
const gemm = (name: string, deps: string[], m: number, k: number, n: number): ExpandedOp => ({
  id: id(name),
  label: name,
  deps: deps.map(id),
  kind: 'gemm',
  x: tt([m, k]),
  w: tt([k, n]),
  out: tt([m, n]),
  dtype: 'bf16',
});
const weightLoad = (name: string, k: number, n: number): ExpandedOp => ({
  id: id(name),
  label: name,
  deps: [],
  kind: 'weight-load',
  out: tt([k, n]),
  dtype: 'bf16',
  loadFraction: 1,
});

test('an op and an independent weight stream overlap fully', () => {
  const d = deploymentOn();
  const ops = [gemm('g', [], 4096, 8192, 8192), weightLoad('w', 8192, 8192)];
  const trace: Segment[] = [{ label: 's', ops, repeat: 1 }];
  const cost = dag(d).priceTrace(trace);
  const g = naiveOpCost(ops[0], d);
  const w = naiveOpCost(ops[1], d);
  // the streams overlap: the fullest one sets the time
  expect(cost.time).toBeCloseTo(Math.max(g.compute, g.memory + w.memory), 12);
  expect(cost.busy).toEqual(naive(d).priceTrace(trace).busy);
  expect(sum(cost.parts)).toBeCloseTo(cost.time, 12);
});

test('a dependent chain through a collective serializes exactly', () => {
  const d = deploymentOn(h100, 2);
  const axis = d.mesh.dims[0].name;
  const g1 = gemm('g1', [], 64, 8192, 8192);
  const ar: ExpandedOp = {
    id: id('ar'),
    label: 'ar',
    deps: [id('g1')],
    kind: 'collective',
    variant: 'all-reduce',
    axes: [axis],
    x: tt([64, 8192], [[], []], [axis]),
    out: tt([64, 8192]),
    dtype: 'bf16',
  };
  const g2 = gemm('g2', ['ar'], 64, 8192, 8192);
  const trace: Segment[] = [{ label: 's', ops: [g1, ar, g2], repeat: 1 }];
  const cost = dag(d).priceTrace(trace);
  const c1 = naiveOpCost(g1, d);
  const c2 = naiveOpCost(g2, d);
  const cc = naiveOpCost(ar, d);
  expect(cost.time).toBeCloseTo(max(c1) + cc.comms + max(c2), 12);
  expect(cost.bound).toBe('deps');
  expect(sum(cost.parts)).toBeCloseTo(cost.time, 12);
});

test('repeats pipeline: the first iteration pays its weights, the rest do not', () => {
  const d = deploymentOn();
  const w = weightLoad('w', 8192, 8192);
  const g: ExpandedOp = { ...gemm('g', ['w'], 64, 8192, 8192) };
  const one = dag(d).priceTrace([{ label: 's', ops: [w, g], repeat: 1 }]);
  const n = 10;
  const many = dag(d).priceTrace([{ label: 's', ops: [w, g], repeat: n }]);
  const cw = naiveOpCost(w, d);
  const cg = naiveOpCost(g, d);
  // one iteration: the load then the gemm, in that order
  expect(one.time).toBeCloseTo(cw.memory + max(cg), 12);
  // then each later one is bounded by its fullest stream, the memory stream
  // that still carries the prefetched weights
  const steady = Math.max(cg.compute, cg.memory + cw.memory, max(cg));
  expect(many.time).toBeCloseTo(one.time + (n - 1) * steady, 12);
  expect(many.time).toBeLessThan(n * one.time);

  // without prefetch every iteration waits for its weights
  const noPrefetch = makeDagScheduleBackend({ prefetchWeights: false })(d);
  expect(noPrefetch.priceTrace([{ label: 's', ops: [w, g], repeat: n }]).time).toBeCloseTo(
    n * one.time,
    12,
  );
});

test('busy and busyPerOp are the naive backend’s, op for op', () => {
  const gptoss = MODEL_PRESETS.find((m) => m.name.startsWith('gpt-oss-120b'))!;
  const k3 = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K3'))!;
  const h200 = CHIPS_BY_ID['h200-sxm'];
  const b300 = CHIPS_BY_ID['b300'];
  const cases = [
    evaluateDecodeAtBatch(
      {
        model: gptoss,
        deployment: deploymentOn(h200),
        workload: { prefillLen: 4096, generateLen: 1024 },
      },
      64,
      1,
      { costBackend: dag, ignoreKvCapacity: true },
    ),
    evaluatePrefill(
      {
        model: k3,
        deployment: deploymentOn(b300, 8),
        workload: { prefillLen: 4096, generateLen: 1024 },
      },
      1,
      'throughput',
      { costBackend: dag, ignoreKvCapacity: true },
    ),
  ];
  for (const r of cases) {
    if (!r.ok) throw new Error(r.diags.map((x) => x.message).join('; '));
    for (const trace of r.perStageTrace) {
      const d = cases[0] === r ? deploymentOn(h200) : deploymentOn(b300, 8);
      const a = naive(d).priceTrace(trace);
      const b = dag(d).priceTrace(trace);
      expect(b.busy).toEqual(a.busy);
      expect([...b.busyPerOp!.entries()]).toEqual([...a.busyPerOp!.entries()]);
    }
  }
});

test('every preset schedules between its widest stream and its serial sum', () => {
  const chip = CHIPS_BY_ID['b300'];
  const d = deploymentOn(chip, 8);
  const workload = { prefillLen: 2048, generateLen: 512 };
  for (const model of MODEL_PRESETS) {
    for (const phase of ['decode', 'prefill'] as const) {
      const r =
        phase === 'decode'
          ? evaluateDecodeAtBatch({ model, deployment: d, workload }, 32, 1, {
              costBackend: dag,
              ignoreKvCapacity: true,
            })
          : evaluatePrefill({ model, deployment: d, workload }, 1, 'throughput', {
              costBackend: dag,
              ignoreKvCapacity: true,
            });
      if (!r.ok) continue;
      const c = r.cost;
      expect(c.time, `${model.name} ${phase}`).toBeGreaterThanOrEqual(max(c.busy) * (1 - 1e-9));
      expect(c.time, `${model.name} ${phase}`).toBeLessThanOrEqual(sum(c.busy) * (1 + 1e-9));
      expect(sum(c.parts)).toBeCloseTo(c.time, 9);
      for (const k of RES) expect(c.hidden[k]).toBeCloseTo(c.busy[k] - c.parts[k], 9);
    }
  }
});

test('gpt-oss-120b decode on one H200 at batch 64 lands under the measured TPOT', () => {
  const gptoss = MODEL_PRESETS.find((m) => m.name.startsWith('gpt-oss-120b'))!;
  const d = deploymentOn(CHIPS_BY_ID['h200-sxm']);
  const input = { model: gptoss, deployment: d, workload: { prefillLen: 4096, generateLen: 1024 } };
  const r = evaluateDecodeAtBatch(input, 64, 1, { costBackend: dag, ignoreKvCapacity: true });
  if (!r.ok) throw new Error('anchor did not evaluate');
  // vLLM on this cell measures ~22.5 ms per token; the serial sum overshoots
  // it and a fully hidden memory stream undershoots it, the schedule sits
  // between: past the memory floor, under the measurement
  expect(r.stepTime).toBeLessThan(22.5e-3);
  expect(r.stepTime).toBeGreaterThan(r.cost.busy.memory);
  const s = evaluateDecodeAtBatch(input, 64, 1, { costBackend: serial, ignoreKvCapacity: true });
  if (!s.ok) throw new Error('anchor did not evaluate');
  expect(r.stepTime).toBeLessThan(s.stepTime);
});

test('makeCostBackend picks the scheduler', () => {
  const d = deploymentOn();
  const trace: Segment[] = [
    { label: 's', ops: [gemm('g', [], 4096, 8192, 8192), weightLoad('w', 8192, 8192)], repeat: 1 },
  ];
  const o = { memoryOverlap: 0.9, commsOverlap: 0.65 };
  expect(makeCostBackend(o)(d).priceTrace(trace).time).toBe(naive(d).priceTrace(trace).time);
  expect(makeCostBackend({ ...o, scheduler: 'dag' })(d).priceTrace(trace).time).toBe(
    dag(d).priceTrace(trace).time,
  );
  // the dag backend shares the naive backend's reshard plans
  expect(dag(d).priceCollectiveHash).toBe(naive(d).priceCollectiveHash);
});

test('on K3 across a GB300 NVL72 the schedule runs slower than the constants, not faster', () => {
  const k3 = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K3'))!;
  const chip = CHIPS_BY_ID['gb300-nvl72'];
  const workload = { prefillLen: 4096, generateLen: 1024 };
  const best = (backend: typeof dag | typeof naive) => {
    let top = 0;
    for (const step of searchShardings(k3, chip, { domain: 32, nodes: 1 }, workload, {
      costBackend: backend,
      phase: { kind: 'decode', policy: { sloTokPerSecPerUser: 20 } },
    }))
      if (step.candidate) top = Math.max(top, step.candidate.score);
    return top;
  };
  const ratio = best(dag) / best(naive);
  expect(ratio).toBeGreaterThan(0.7);
  expect(ratio).toBeLessThanOrEqual(1);
}, 180_000);
