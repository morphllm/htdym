import { MODEL_PRESETS, ModelSpec } from '../../core/model/models';
import {
  activeParams,
  layerCount,
  minKvHeads,
  moeExperts,
  totalParams,
  weightBytesTotal,
} from '../../core/model/utils';
import { round, table } from '../format';

export interface ModelRow {
  name: string;
  paramsB: number;
  activeB: number;
  weightGb: number;
  layers: number;
  experts: number;
  minKvHeads: number;
  weights: string;
  activations: string;
  kv: string;
}

const dtypes = (m: ModelSpec, which: 'weights' | 'activations') => {
  const p = m.precision[which];
  const set = [...new Set([p.attention, p.denseMlp, p.routedExperts])];
  return set.join('/');
};

export function modelRows(): ModelRow[] {
  return MODEL_PRESETS.map((m) => ({
    name: m.name,
    paramsB: round(totalParams(m) / 1e9, 1),
    activeB: round(activeParams(m) / 1e9, 1),
    weightGb: round(weightBytesTotal(m) / 1e9, 1),
    layers: layerCount(m),
    experts: moeExperts(m),
    minKvHeads: minKvHeads(m),
    weights: dtypes(m, 'weights'),
    activations: dtypes(m, 'activations'),
    kv: m.precision.kv,
  }));
}

export function renderModels(rows: ModelRow[]): string {
  return table(
    [
      'model',
      'params B',
      'active B',
      'weights GB',
      'layers',
      'experts',
      'kv heads',
      'weights',
      'acts',
      'kv',
    ],
    rows.map((r) => [
      r.name,
      String(r.paramsB),
      String(r.activeB),
      String(r.weightGb),
      String(r.layers),
      r.experts ? String(r.experts) : '-',
      String(r.minKvHeads),
      r.weights,
      r.activations,
      r.kv,
    ]),
    ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'l', 'l', 'l'],
  );
}
