import { ChipSpec } from './chips';
import { SliceTopology } from './topology';

// A chip with the machine it deploys on: a ring-fabric slice name or a
// switched-fabric node count, both defaulting in machineOf when unset.
export interface ChipOnMachine extends ChipSpec {
  slice?: string;
  nodes?: number;
}

// The machine a search runs on, as searchShardings takes it.
export type Machine = SliceTopology | { domain: number; nodes: number };

// A machine spelled on a command line: a catalog slice name on a ring
// fabric ("4x4x4"), or "<chips per domain>x<nodes>" ("8x2", "32x1") on a
// switched fabric, where a bare count means one node.
export function parseMachine(chip: ChipSpec, spec: string): Machine {
  const catalog = chip.interconnect.topologies;
  if (catalog) {
    const slice = catalog.find((s) => s.name === spec);
    if (!slice)
      throw new Error(
        `${chip.id} has no ${spec} slice; it offers ${catalog.map((s) => s.name).join(', ')}`,
      );
    return slice;
  }
  const m = /^(\d+)(?:x(\d+))?$/.exec(spec);
  if (!m) throw new Error(`machine ${spec} is not <domain>x<nodes>`);
  const domain = Number(m[1]);
  const nodes = m[2] === undefined ? 1 : Number(m[2]);
  if (domain < 1 || domain > chip.interconnect.domainSize)
    throw new Error(`${chip.id} domains hold up to ${chip.interconnect.domainSize} chips`);
  const maxNodes = chip.interconnect.scaleOut?.maxNodes ?? 1;
  if (nodes < 1 || nodes > maxNodes)
    throw new Error(`${chip.id} scales out to ${maxNodes} node${maxNodes > 1 ? 's' : ''}`);
  return { domain, nodes };
}

// The name parseMachine accepts for a machine, for labels and file names.
export function machineName(m: Machine): string {
  return 'dims' in m ? m.name : `${m.domain}x${m.nodes}`;
}

export function machineSize(m: Machine): number {
  return 'dims' in m ? m.count : m.domain * m.nodes;
}

// Most scale-out nodes selectable on a switched fabric (1 = no scale-out).
export function maxNodesOf(chip: ChipSpec): number {
  return Math.min(8, chip.interconnect.scaleOut?.maxNodes ?? 1);
}

// Ring-fabric slices offered for a chip (absent on switched fabrics).
export function slicesOf(chip: ChipSpec): SliceTopology[] | undefined {
  return chip.interconnect.topologies?.filter((s) => s.count <= 128);
}

// Every machine a chip can be deployed on, as chips the rest of the UI can
// treat independently: one per slice, or one per scale-out node count.
export function machineVariants(chip: ChipOnMachine): ChipOnMachine[] {
  const slices = slicesOf(chip);
  return slices
    ? slices.map((s) => ({ ...chip, slice: s.name }))
    : Array.from({ length: maxNodesOf(chip) }, (_, i) => ({ ...chip, nodes: i + 1 }));
}

// The offered slice closest to a host count's worth of chips (ties go to
// the larger slice). Sliced-fabric chips only.
function sliceAtHosts(chip: ChipOnMachine, hosts: number): SliceTopology {
  const target = hosts * chip.interconnect.chipsPerHost!;
  return slicesOf(chip)!.reduce((a, b) => {
    const da = Math.abs(a.count - target);
    const db = Math.abs(b.count - target);
    return db < da || (db === da && b.count > a.count) ? b : a;
  });
}

// The chip deployed on the machine nearest a global node count: that many
// scale-out nodes on a switched fabric, or the slice closest to the hosts'
// worth of chips on a ring fabric. Fixed-size machines (NVL72, Trainium2)
// resolve to their one machine.
export function machineAtNodes(chip: ChipOnMachine, nodes: number): ChipOnMachine {
  if (!slicesOf(chip)) return { ...chip, nodes: Math.min(nodes, maxNodesOf(chip)) };
  return { ...chip, slice: sliceAtHosts(chip, nodes).name };
}

// Identifies a chip *on a machine*: the key groups are held under, so a
// machine sweep's variants of one chip stay separate rows.
export function machineKey(chip: ChipOnMachine): string {
  const m = machineOf(chip);
  return `${chip.id}|${'dims' in m ? m.name : m.nodes}`;
}

export function machineOf(chip: ChipOnMachine): Machine {
  const catalog = chip.interconnect.topologies;
  // unset machine picks default to one node/host of chips
  if (!catalog)
    return {
      domain: chip.interconnect.domainSize,
      nodes: Math.min(chip.nodes ?? 1, maxNodesOf(chip)),
    };
  const slice = chip.slice ? catalog.find((s) => s.name === chip.slice) : sliceAtHosts(chip, 1);
  if (!slice) throw new Error(`no machine chosen for ${chip.id}`);
  return slice;
}

// empty when the chip offers no scale-out at all: the pill would just say
// "1 node" on every row
export function machineLabel(chip: ChipOnMachine): string {
  const m = machineOf(chip);
  if ('dims' in m) return m.name;
  return maxNodesOf(chip) > 1 ? `${m.nodes} node${m.nodes > 1 ? 's' : ''}` : '';
}

export function machineChips(chip: ChipOnMachine): number {
  return machineSize(machineOf(chip));
}
