// Fixed-width text tables and number cells for the terminal.

export type Align = 'l' | 'r';

export function table(head: string[], rows: string[][], align: Align[] = []): string {
  const widths = head.map((h, i) => Math.max(h.length, ...rows.map((r) => (r[i] ?? '').length)));
  const line = (cells: string[]) =>
    cells
      .map((c, i) => (align[i] === 'r' ? c.padStart(widths[i]) : c.padEnd(widths[i])))
      .join('  ')
      .trimEnd();
  return [line(head), ...rows.map(line)].join('\n');
}

// a number cell, '-' for what a row does not have
export function fix(v: number | null | undefined, digits = 1): string {
  return v === null || v === undefined || !Number.isFinite(v) ? '-' : v.toFixed(digits);
}

export function int(v: number | null | undefined): string {
  return v === null || v === undefined || !Number.isFinite(v) ? '-' : String(Math.round(v));
}

// JSON rows carry rounded numbers so a diff reads
export function round(v: number, digits: number): number {
  return Number(v.toFixed(digits));
}

export function nullable(v: number | undefined, digits: number): number | null {
  return v === undefined || !Number.isFinite(v) ? null : round(v, digits);
}
