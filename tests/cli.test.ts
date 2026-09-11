import { expect, test } from 'vitest';
import { CHIPS_BY_ID } from '../src/core/hardware/chips';
import { parseMachine } from '../src/core/hardware/machines';
import { sharedOptions } from '../src/cli/args';
import { DEFAULT_SHARED, makeContext } from '../src/cli/context';
import { CliError, resolveChip, resolveModel } from '../src/cli/resolve';
import { usdPerMtok } from '../src/cli/row';
import { chipRows } from '../src/cli/commands/chips';
import { modelRows } from '../src/cli/commands/models';
import { explain, fillSizes, engineHints } from '../src/cli/commands/explain';
import { rankedRows, renderRows } from '../src/cli/commands/search';
import { renderReport, runSweep, sweepCells } from '../src/cli/commands/sweep';

test('models resolve by name, alias, or unique prefix', () => {
  expect(resolveModel('Kimi K3 MXFP4/MXFP8').name).toBe('Kimi K3 MXFP4/MXFP8');
  expect(resolveModel('k3').name).toBe('Kimi K3 MXFP4/MXFP8');
  expect(resolveModel('kimi k3').name).toBe('Kimi K3 MXFP4/MXFP8');
  expect(resolveModel('gpt-oss-120b').name).toBe('gpt-oss-120b MXFP4/BF16');
  expect(resolveModel('dsv4flash').name).toBe('DeepSeek V4 Flash MXFP4/FP8');
  // "Kimi" alone is K2.6 or K3
  expect(() => resolveModel('kimi')).toThrow(CliError);
  expect(() => resolveModel('nonesuch')).toThrow(/unknown model/);
  expect(resolveChip('gb300-nvl72').id).toBe('gb300-nvl72');
  expect(resolveChip('gb300').id).toBe('gb300-nvl72');
  expect(() => resolveChip('h')).toThrow(/matches/);
});

test('machines parse as domain x nodes on switched fabrics and as slices on rings', () => {
  const gb300 = CHIPS_BY_ID['gb300-nvl72'];
  expect(parseMachine(gb300, '32x1')).toEqual({ domain: 32, nodes: 1 });
  expect(parseMachine(gb300, '64')).toEqual({ domain: 64, nodes: 1 });
  expect(() => parseMachine(gb300, '72x1')).toThrow(/up to 64/);
  expect(() => parseMachine(gb300, '32x2')).toThrow(/scales out to 1 node/);
  const h100 = CHIPS_BY_ID['h100-sxm'];
  expect(parseMachine(h100, '8x2')).toEqual({ domain: 8, nodes: 2 });
  const v5p = CHIPS_BY_ID['tpu-v5p'];
  expect(parseMachine(v5p, '2x2x2')).toMatchObject({ name: '2x2x2', count: 8 });
  expect(() => parseMachine(v5p, '8x1')).toThrow(/no 8x1 slice/);
  expect(makeContext('k3', 'gb300', '32x1').nChips).toBe(32);
});

test('shared flags map onto the engine knobs', () => {
  const o = sharedOptions({
    'state-slots': '5',
    'spec-slots': '8',
    'state-dtype': 'bf16',
    'mem-fraction': '0.9',
    scheduler: 'dag',
    overlap: '0.5,0.25',
    'cost-per-hour': '7',
  });
  expect(o.statePool).toEqual({ slotsPerSeq: 5, specSlots: 8, stateDtypeBytes: 2 });
  expect(o.memFraction).toBe(0.9);
  expect(o.overlap).toEqual({ memoryOverlap: 0.5, commsOverlap: 0.25, scheduler: 'dag' });
  expect(o.costPerHour).toBe(7);
  expect(sharedOptions({})).toEqual(DEFAULT_SHARED);
  const ctx = makeContext('k3', 'b300', '8x1', o);
  expect(ctx.chip.hbmCapacity).toBeCloseTo(0.9 * CHIPS_BY_ID['b300'].hbmCapacity, 0);
  expect(ctx.chip.costPerHour).toBe(7);
  expect(() => sharedOptions({ scheduler: 'magic' })).toThrow(/scheduler/);
  expect(() => sharedOptions({ overlap: '2,0' })).toThrow(/overlap/);
});

test('the chip table carries the estimated GB300 entry', () => {
  const k3 = resolveModel('k3');
  const row = chipRows(k3).find((r) => r.id === 'gb300-nvl72')!;
  expect(row).toMatchObject({ hbmGb: 270, hbmTbps: 8, costPerHour: 6.75, domain: 64 });
  expect(row.minChipsForWeights).toBe(6);
  expect(modelRows().find((m) => m.name.startsWith('Kimi K3'))).toMatchObject({
    layers: 93,
    experts: 896,
    minKvHeads: 1,
  });
});

test('$/Mtok is the fleet tooling’s cost_per_million_tokens', () => {
  // cost_per_million_tokens(gpu_hourly, gpus, tok_s) = gpu_hourly*gpus / (tok_s*3600) * 1e6
  expect(usdPerMtok(6.75, 4371)).toBeCloseTo((6.75 / (4371 * 3600)) * 1e6, 12);
});

test('explain fills the roles the way the search sizes them', () => {
  const ctx = makeContext('k3', 'gb300', '32x1');
  expect(fillSizes(ctx, { DPA: 16, TP: 2, EP: 32 })).toEqual({
    PP: 1,
    DPA: 16,
    TP: 2,
    EP: 32,
    ETP: 1,
  });
  expect(fillSizes(ctx, { TP: 4 })).toEqual({ PP: 1, DPA: 8, TP: 4, EP: 32, ETP: 1 });
  expect(fillSizes(ctx, { PP: 2, TP: 8, ETP: 2 })).toEqual({ PP: 2, DPA: 2, TP: 8, EP: 8, ETP: 2 });
  expect(() => fillSizes(ctx, { TP: 3 })).toThrow(CliError);
  expect(engineHints({ PP: 1, DPA: 16, TP: 2, EP: 32, ETP: 1 }, 2)).toEqual([
    'vLLM:   -tp 2 -dp 16 --enable-expert-parallel',
    'SGLang: --tp 32 --dp 16 --enable-dp-attention --ep 32 --decode-context-parallel-size 2',
  ]);
});

test('the state pool caps explain’s batch near the measured B300 admission', () => {
  const ctx = makeContext('k3', 'b300', '8x1', {
    ...DEFAULT_SHARED,
    prefillLen: 8192,
    genLen: 1024,
    memFraction: 0.9,
    statePool: { slotsPerSeq: 5 },
  });
  const e = explain(ctx, { phase: 'decode', sizes: { TP: 8, EP: 8 }, dcp: 1 });
  expect(e.best.maxResidentSeqs).toBe(117);
  expect(e.best.batch).toBe(117);
  expect(e.best.stateSlotsPerSeq).toBe(5);
  expect(e.best.kvMbPerSeqPerChip).toBeCloseTo(
    e.best.pagedKvMbPerSeqPerChip! + e.best.stateMbPerSeqPerChip!,
    2,
  );
});

test('the pinned GB300 search: K3 on 32 chips at 20 tok/s/user lands on DPA=16 TP=2 EP=32 DCP=2', () => {
  const ctx = makeContext('k3', 'gb300-nvl72', '32x1');
  const [top] = rankedRows(ctx, { phase: 'decode', slo: 20 }, 1);
  expect(top.sizes).toEqual({ PP: 1, DPA: 16, TP: 2, EP: 32, ETP: 1 });
  expect(top.dcp).toBe(2);
  expect(top.moeDispatch).toBe('coalesced-a2a');
  expect(top.sharding).toBe('DPA=16 TP=2 EP=32');
  expect(top.tokPerSecPerUser).toBeGreaterThanOrEqual(20);
  expect(top.tokPerSecPerUser).toBeLessThan(21);
  expect(top.usdPerMtok).toBeCloseTo(usdPerMtok(6.75, top.tokPerSecPerChip!), 3);
  expect(top.batch).toBeLessThanOrEqual(top.maxResidentSeqs!);
  expect(top.prefill?.ttftMs).toBeGreaterThan(0);
  expect(renderRows([top], 1)).toContain('DPA=16 TP=2 EP=32 dcp=2 coalesced-a2a');

  // the row's key set is the contract sweep consumers read
  expect(Object.keys(top).sort()).toEqual(
    [
      'batch',
      'boundBy',
      'busyMs',
      'chip',
      'costPerHour',
      'dcp',
      'diagnostics',
      'error',
      'feasible',
      'id',
      'kvMbPerSeqPerChip',
      'machine',
      'maxResidentSeqs',
      'mbu',
      'mfu',
      'model',
      'moeDispatch',
      'nChips',
      'overlap',
      'pagedKvMbPerSeqPerChip',
      'phase',
      'placement',
      'prefill',
      'scheduler',
      'sharding',
      'sizes',
      'slo',
      'stateMbPerSeqPerChip',
      'stateSlotsPerSeq',
      'statePool',
      'stepTimeMs',
      'tokPerSecMachine',
      'tokPerSecPerChip',
      'tokPerSecPerUser',
      'tpotMs',
      'usdPerMtok',
      'version',
      'visibleMs',
      'weightGbPerChip',
      'workload',
    ].sort(),
  );

  // explain on the winner's sizes reproduces the winner
  const e = explain(ctx, { phase: 'decode', slo: 20, sizes: { DPA: 16, TP: 2, EP: 32 } });
  expect(e.best.batch).toBe(top.batch);
  expect(e.best.tokPerSecPerChip).toBe(top.tokPerSecPerChip);
  expect(e.best.placement).toBe(top.placement);
}, 180_000);

test('a sweep reports every cell and the report groups them', async () => {
  const cells = sweepCells({
    model: 'k3',
    chips: ['b300', 'nonesuch'],
    machines: ['8x1'],
    phases: ['decode', 'prefill'],
    slos: [null, 50],
  });
  expect(cells.map((c) => `${c.chip}/${c.phase}/${c.slo}`)).toEqual([
    'b300/decode/null',
    'b300/decode/50',
    'b300/prefill/null',
    'nonesuch/decode/null',
    'nonesuch/decode/50',
    'nonesuch/prefill/null',
  ]);
  const seen: string[] = [];
  const rows = await runSweep(cells, DEFAULT_SHARED, 1, (r) => seen.push(r.chip));
  expect(seen.length).toBe(6);
  const ok = rows.filter((r) => r.feasible);
  expect(ok.map((r) => `${r.phase}/${r.slo}`)).toEqual([
    'decode/null',
    'decode/50',
    'prefill/null',
  ]);
  const bad = rows.filter((r) => !r.feasible);
  expect(bad).toHaveLength(3);
  expect(bad[0].error).toMatch(/unknown chip/);
  const at50 = ok.find((r) => r.slo === 50)!;
  expect(at50.tokPerSecPerUser).toBeGreaterThanOrEqual(50);
  const report = renderReport(rows);
  expect(report).toContain('DECODE at >= 50 tok/s/user');
  expect(report).toContain('PREFILL throughput');
  expect(report).toContain('3 infeasible');
}, 120_000);
