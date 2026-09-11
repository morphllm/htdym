import { expect, test } from 'vitest';
import { MODEL_PRESETS } from '../src/core/model/models';
import { CHIPS_BY_ID, ChipSpec } from '../src/core/hardware/chips';
import { deployedAxes } from '../src/core/hardware/topology';
import { Deployment, makeMesh, roleSize } from '../src/core/engine/surface/deploy';
import { memoryFootprint } from '../src/core/engine/sim/run/memory';
import { partitionIntoStages } from '../src/core/engine/sim/lowering/stages';
import { evaluateDecodeAtBatch } from '../src/core/engine/sim/run/decode';
import { operatingBatch } from '../src/core/engine/optimizer/policy';
import { searchShardings } from '../src/core/engine/optimizer/search';
import { makeNaiveOpCostSumBackend } from '../src/core/engine/sim/cost/naiveOpCostSum';
import { blockStateBytes } from '../src/core/model/block';
import { DEFAULT_STATE_POOL, StatePoolOptions } from '../src/core/engine/surface/api';

// The anchor cell: Kimi K3 (24 MLA + 69 KDA layers) on one 8x B300 node at
// TP=8, ISL 8192 / OSL 1024, fp8 latents, fp32 recurrent state. SGLang on
// this exact cell admits 101 concurrent requests with radix cache
// (extra_buffer, 5 state slots per request) and mem_fraction_static 0.9,
// and 68 with DSPARK speculative decoding on top (draft block + 1 = 8
// more slots). The simulator's physics floor is one slot per sequence.
const k3 = MODEL_PRESETS.find((m) => m.name.startsWith('Kimi K3'))!;
const b300 = CHIPS_BY_ID['b300'];
const axes = deployedAxes(b300.interconnect, { domain: 8, nodes: 1 });
const names = axes.map((a) => a.name);
const stages = partitionIntoStages(k3, 1);
const workload = { prefillLen: 8192, generateLen: 1024 };
const fullLen = workload.prefillLen + workload.generateLen;
const backend = makeNaiveOpCostSumBackend({ memoryOverlap: 0.9, commsOverlap: 0.65 });

const tp8 = (chip: ChipSpec = b300, dcp = 1): Deployment => ({
  chip,
  mesh: makeMesh(axes, { DPA: [], TP: names, EP: names, ETP: [], PP: [] }),
  moeDispatch: 'ring-of-experts',
  decodeContextParallel: dcp,
});
const footprint = (pool: Partial<StatePoolOptions>, chip = b300, dcp = 1) =>
  memoryFootprint(k3, tp8(chip, dcp), stages, fullLen, pool);
// SGLang's mem_fraction_static is a chip-level headroom, not a per-sequence
// cost, so it is modelled as the chip having that much HBM
const atFraction = (f: number): ChipSpec => ({ ...b300, hbmCapacity: f * b300.hbmCapacity });

const kdaLayers = 69;
const kdaBlock = k3.blocks[0].pattern[0].block;
const statePerSlotPerChip = (kdaLayers * blockStateBytes(kdaBlock)) / 8;

test('the state pool defaults to one slot and reproduces the old footprint', () => {
  expect(DEFAULT_STATE_POOL).toEqual({ slotsPerSeq: 1, specSlots: 0 });
  const before = memoryFootprint(k3, tp8(), stages, fullLen);
  const explicit = footprint({});
  expect(explicit).toEqual(before);
  expect(before.stateSlotsPerSeq).toBe(1);
  expect(before.kvBytesPerSeqPerChip).toBe(
    before.pagedKvBytesPerSeqPerChip + before.stateBytesPerSeqPerChip,
  );
  expect(before.stateBytesPerSeqPerChip).toBeCloseTo(statePerSlotPerChip, 0);
  expect(before.maxResidentSeqsPerChip).toBe(407);
});

test('reserved slots multiply the state part only, by exact byte arithmetic', () => {
  const one = footprint({});
  const five = footprint({ slotsPerSeq: 5 });
  expect(five.stateSlotsPerSeq).toBe(5);
  expect(five.pagedKvBytesPerSeqPerChip).toBe(one.pagedKvBytesPerSeqPerChip);
  expect(five.stateBytesPerSeqPerChip).toBeCloseTo(5 * one.stateBytesPerSeqPerChip, 0);
  expect(five.kvBytesPerSeqPerChip).toBeCloseTo(
    one.pagedKvBytesPerSeqPerChip + 5 * one.stateBytesPerSeqPerChip,
    0,
  );
  expect(five.weightBytesPerChip).toBe(one.weightBytesPerChip);
  expect(five.maxResidentSeqsPerChip).toBe(
    Math.floor((b300.hbmCapacity - five.weightBytesPerChip) / five.kvBytesPerSeqPerChip),
  );
  expect(five.maxResidentSeqsPerChip).toBe(185);

  // spec slots stack on top of the working slots
  const spec = footprint({ slotsPerSeq: 5, specSlots: 8 });
  expect(spec.stateSlotsPerSeq).toBe(13);
  expect(spec.stateBytesPerSeqPerChip).toBeCloseTo(13 * one.stateBytesPerSeqPerChip, 0);
});

test('a narrower state dtype halves the state part and leaves the paged part alone', () => {
  const fp32 = footprint({ slotsPerSeq: 5 });
  const bf16 = footprint({ slotsPerSeq: 5, stateDtypeBytes: 2 });
  expect(bf16.stateBytesPerSeqPerChip).toBeCloseTo(fp32.stateBytesPerSeqPerChip / 2, 0);
  expect(bf16.pagedKvBytesPerSeqPerChip).toBe(fp32.pagedKvBytesPerSeqPerChip);
  expect(bf16.maxResidentSeqsPerChip).toBeGreaterThan(fp32.maxResidentSeqsPerChip);
});

test('DCP shards the paged cache and cannot touch the reserved state', () => {
  const dcp1 = footprint({ slotsPerSeq: 5 });
  const dcp8 = footprint({ slotsPerSeq: 5 }, b300, 8);
  expect(dcp8.pagedKvBytesPerSeqPerChip).toBeCloseTo(dcp1.pagedKvBytesPerSeqPerChip / 8, 0);
  expect(dcp8.stateBytesPerSeqPerChip).toBe(dcp1.stateBytesPerSeqPerChip);
});

test('five slots plus 0.9 static fraction lands near the measured admission', () => {
  const measuredNoSpec = 101;
  const measuredDspark = 68;

  // one slot per sequence overshoots the measured ceiling by more than 2x:
  // this is the bug the knob exists for
  expect(footprint({}, atFraction(0.9)).maxResidentSeqsPerChip).toBeGreaterThan(2 * measuredNoSpec);

  const noSpec = footprint({ slotsPerSeq: 5 }, atFraction(0.9)).maxResidentSeqsPerChip;
  expect(noSpec).toBe(117);
  expect(Math.abs(noSpec - measuredNoSpec) / measuredNoSpec).toBeLessThan(0.25);

  // DSPARK's extra slots are charged per sequence the same way, which is a
  // stricter accounting than the stack's (it shares a pool between the
  // working and draft states): direction and a 30% band, not a hit
  const spec = footprint({ slotsPerSeq: 5, specSlots: 8 }, atFraction(0.9)).maxResidentSeqsPerChip;
  expect(spec).toBeLessThan(noSpec);
  expect(Math.abs(spec - measuredDspark) / measuredDspark).toBeLessThan(0.3);
});

test('the pool changes capacity only: the priced step is byte-identical', () => {
  const input = { model: k3, deployment: tp8(), workload };
  const base = { costBackend: backend, ignoreKvCapacity: true };
  const a = evaluateDecodeAtBatch(input, 64, 1, base);
  const b = evaluateDecodeAtBatch(input, 64, 1, { ...base, statePool: { slotsPerSeq: 5 } });
  if (!a.ok || !b.ok) throw new Error('anchor cell did not evaluate');
  expect(b.stepTime).toBe(a.stepTime);
  expect(b.traffic).toEqual(a.traffic);
  expect(b.cost.busy).toEqual(a.cost.busy);
  expect(b.memory.maxResidentSeqsPerChip).toBe(185);
  expect(a.memory.maxResidentSeqsPerChip).toBe(407);
});

test('the KV gate and the operating batch see the reserved slots', () => {
  const input = { model: k3, deployment: tp8(), workload };
  const opts = { costBackend: backend, statePool: { slotsPerSeq: 5 } };

  expect(evaluateDecodeAtBatch(input, 185, 1, opts).ok).toBe(true);
  const over = evaluateDecodeAtBatch(input, 400, 1, opts);
  expect(over.ok).toBe(false);
  expect(over.diags.map((d) => d.code)).toContain('kv-no-room');
  expect(evaluateDecodeAtBatch(input, 400, 1, { costBackend: backend }).ok).toBe(true);

  expect(operatingBatch(input, { batching: 'max' }, opts)).toBe(185);
  expect(operatingBatch(input, { batching: 'max' }, { costBackend: backend })).toBe(407);
});

test('a search carries the pool into every candidate it yields', () => {
  const gen = searchShardings(k3, b300, { domain: 8, nodes: 1 }, workload, {
    costBackend: backend,
    statePool: { slotsPerSeq: 5 },
    phase: { kind: 'decode', policy: { batching: 'max' } },
  });
  // the first tuples (TP=1) cannot hold K3's replicated weights on one
  // B300, so take the first tuple that priced
  let cand;
  for (const step of gen) if ((cand = step.candidate)) break;
  if (!cand) throw new Error('no feasible tuple on the anchor node');
  if (!('memory' in cand.result)) throw new Error('decode search yielded a prefill result');
  expect(cand.result.memory.stateSlotsPerSeq).toBe(5);
  const dpa = roleSize(cand.deployment.mesh, 'DPA');
  expect(cand.batch).toBeLessThanOrEqual(cand.result.memory.maxResidentSeqsPerChip * dpa);
});
