import { searchShardings, searchTuples } from '../core/engine/optimizer/search';
import { ServingPolicy } from '../core/engine/optimizer/policy';
import { roofline } from '../core/engine/roofline';
import { makeCostBackend } from '../core/engine/sim/cost/select';
import { enrichCandidate } from '../core/engine/optimizer/enrich';
import { MODEL_PRESETS } from '../core/model/models';
import { machineChips, machineKey, machineLabel, machineOf } from '../core/hardware/machines';
import type { SearchRequest, SearchUpdate } from './searchClient';

// the client pins one worker per chip, so this instance only ever sees one
// chip's requests: each supersedes everything older than it
let latest = 0;

const post = (u: SearchUpdate) => (self as { postMessage(m: unknown): void }).postMessage(u);

self.onmessage = (e: MessageEvent<SearchRequest>) => {
  latest = e.data.id;
  void run(e.data);
};

async function run(req: SearchRequest) {
  const model = MODEL_PRESETS.find((m) => m.name === req.modelName);
  if (!model) return;

  const backend = makeCostBackend(req.overlap);
  const policy: ServingPolicy = {
    batching: req.workload.batching,
    sloTokPerSecPerUser: req.workload.sloTokPerSecPerUser || undefined,
  };
  const workload = { prefillLen: req.workload.prefillLen, generateLen: req.workload.generateLen };

  // every group posts upfront with its search size, so the app knows the
  // full totals before any search runs
  const machines = req.chips.map((chip) => {
    const hardware = roofline(model, chip, workload, chip.realizableFlopsFrac);
    const nChips = machineChips(chip);
    const total = hardware ? searchTuples(model, nChips).length : 0;
    const group = {
      id: req.id,
      key: machineKey(chip),
      kind: 'group' as const,
      chipId: chip.id,
      machineLabel: machineLabel(chip),
      nChips,
      hardware,
      total,
      error: hardware ? undefined : `${chip.name} does not support this model's compute dtype`,
    };
    post(group);
    return { chip, hardware, nChips, group };
  });

  for (const { chip, hardware, nChips, group } of machines) {
    if (req.id < latest) return;
    if (!hardware) continue;

    try {
      const gen = searchShardings(model, chip, machineOf(chip), workload, {
        costBackend: backend,
        phase: { kind: 'decode', policy },
      });
      for (const step of gen) {
        if (req.id < latest) return;
        post({
          id: req.id,
          key: group.key,
          kind: 'row',
          done: step.done,
          row:
            step.candidate &&
            enrichCandidate(model, chip, group.key, nChips, workload, step.candidate),
        });
        await new Promise((r) => setTimeout(r));
      }
    } catch (err) {
      post({ ...group, error: String(err) });
    }
  }
}
