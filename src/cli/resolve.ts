import { CHIPS, CHIPS_BY_ID, ChipSpec } from '../core/hardware/chips';
import { MODEL_PRESETS, ModelSpec } from '../core/model/models';

// A failure the user can act on: printed as one line, no stack.
export class CliError extends Error {
  constructor(
    message: string,
    public readonly exitCode: number = 1,
  ) {
    super(message);
  }
}

// short names people type, each to a unique preset-name prefix
const MODEL_ALIASES: Record<string, string> = {
  k3: 'Kimi K3',
  kimik3: 'Kimi K3',
  k2: 'Kimi K2.6',
  'k2.6': 'Kimi K2.6',
  kimik2: 'Kimi K2.6',
  dsv4flash: 'DeepSeek V4 Flash',
  dsv4pro: 'DeepSeek V4 Pro',
  glm53: 'GLM 5.3',
  gptoss120b: 'gpt-oss-120b',
  gptoss20b: 'gpt-oss-20b',
  gemma31b: 'Gemma 4 31B',
  gemma12b: 'Gemma 4 12B',
  gemma26b: 'Gemma 4 26B',
  llama8b: 'LLaMA 3 8B',
  llama70b: 'LLaMA 3 70B',
  llama405b: 'LLaMA 3.1 405B',
  qwen35b: 'Qwen3.6 35B',
  qwen27b: 'Qwen3.8 27B',
  qwen4b: 'Qwen3 4B BF16',
};

const norm = (s: string) => s.toLowerCase().replace(/[\s_-]/g, '');

// exact preset name, an alias, or a unique case-insensitive prefix
export function resolveModel(arg: string): ModelSpec {
  const exact = MODEL_PRESETS.find((m) => m.name === arg);
  if (exact) return exact;
  const target = MODEL_ALIASES[norm(arg)] ?? arg;
  const hits = MODEL_PRESETS.filter((m) => norm(m.name).startsWith(norm(target)));
  if (hits.length === 1) return hits[0];
  if (hits.length > 1)
    throw new CliError(`model "${arg}" matches ${hits.map((m) => `"${m.name}"`).join(', ')}`, 2);
  throw new CliError(
    `unknown model "${arg}"; models: ${MODEL_PRESETS.map((m) => `"${m.name}"`).join(', ')}`,
    2,
  );
}

// exact chip id or a unique prefix of one
export function resolveChip(arg: string): ChipSpec {
  const exact = CHIPS_BY_ID[arg];
  if (exact) return exact;
  const hits = CHIPS.filter((c) => c.id.startsWith(arg.toLowerCase()));
  if (hits.length === 1) return hits[0];
  if (hits.length > 1)
    throw new CliError(`chip "${arg}" matches ${hits.map((c) => c.id).join(', ')}`, 2);
  throw new CliError(`unknown chip "${arg}"; chips: ${CHIPS.map((c) => c.id).join(', ')}`, 2);
}
