import { CHIPS, ChipSpec, peakFlops } from '../../core/hardware/chips';
import { ModelSpec } from '../../core/model/models';
import { weightBytesTotal } from '../../core/model/utils';
import { fix, round, table } from '../format';

export interface ChipRow {
  id: string;
  name: string;
  vendor: string;
  hbmGb: number;
  hbmTbps: number;
  bf16Pf: number;
  fp8Pf: number | null;
  fp4Pf: number | null;
  linkGbps: number;
  domain: number;
  // switched fabrics: nodes the scale-out tier reaches; ring fabrics: the
  // slices sold
  maxNodes: number | null;
  slices: string[] | null;
  costPerHour: number | null;
  tdp: number | null;
  // with --model: the least chips whose HBM holds the weights
  minChipsForWeights: number | null;
}

const pf = (chip: ChipSpec, d: 'bf16' | 'fp8' | 'fp4') => {
  const v = peakFlops(chip, d) ?? (d === 'fp8' ? peakFlops(chip, 'mxfp8') : undefined);
  return v === undefined ? null : round(v / 1e15, 2);
};

export function chipRows(model?: ModelSpec): ChipRow[] {
  const w = model ? weightBytesTotal(model) : undefined;
  return CHIPS.map((c) => ({
    id: c.id,
    name: c.name,
    vendor: c.vendor,
    hbmGb: round(c.hbmCapacity / 1e9, 0),
    hbmTbps: round(c.hbmBandwidth / 1e12, 2),
    bf16Pf: pf(c, 'bf16')!,
    fp8Pf: pf(c, 'fp8'),
    fp4Pf: pf(c, 'fp4'),
    linkGbps: round(c.interconnect.bandwidthPerChip / 1e9, 0),
    domain: c.interconnect.domainSize,
    maxNodes: c.interconnect.topologies ? null : (c.interconnect.scaleOut?.maxNodes ?? 1),
    slices: c.interconnect.topologies ? c.interconnect.topologies.map((t) => t.name) : null,
    costPerHour: c.costPerHour ?? null,
    tdp: c.tdp ?? null,
    minChipsForWeights: w === undefined ? null : Math.ceil(w / c.hbmCapacity),
  }));
}

export function renderChips(rows: ChipRow[]): string {
  const withModel = rows.some((r) => r.minChipsForWeights !== null);
  const head = [
    'chip',
    'HBM GB',
    'TB/s',
    'bf16 PF',
    'fp8 PF',
    'fp4 PF',
    'link GB/s',
    'domain',
    'machines',
    '$/hr',
    'W',
    ...(withModel ? ['min chips'] : []),
  ];
  const body = rows.map((r) => [
    r.id,
    String(r.hbmGb),
    fix(r.hbmTbps, 2),
    fix(r.bf16Pf, 2),
    fix(r.fp8Pf, 2),
    fix(r.fp4Pf, 2),
    String(r.linkGbps),
    String(r.domain),
    r.slices
      ? `${r.slices.length} slices (${r.slices.slice(0, 3).join(',')}${r.slices.length > 3 ? ',...' : ''})`
      : r.maxNodes === 1
        ? `${r.domain}x1`
        : `${r.domain}x1 to ${r.domain}x${r.maxNodes}`,
    fix(r.costPerHour, 2),
    r.tdp === null ? '-' : String(r.tdp),
    ...(withModel ? [r.minChipsForWeights === null ? '-' : String(r.minChipsForWeights)] : []),
  ]);
  return table(head, body, ['l', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'l', 'r', 'r', 'r']);
}
