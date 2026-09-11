// Bundle the CLI into one node script: the sources use extensionless
// imports under moduleResolution "bundler", which node cannot run as-is.
import { build } from 'esbuild';
import { execSync } from 'node:child_process';
import { mkdirSync } from 'node:fs';

const version = (() => {
  try {
    return execSync('git describe --tags --always --dirty', { stdio: ['ignore', 'pipe', 'ignore'] })
      .toString()
      .trim();
  } catch {
    return 'unknown';
  }
})();

mkdirSync('dist/cli', { recursive: true });
await build({
  entryPoints: ['src/cli/main.ts'],
  outfile: 'dist/cli/estimate.mjs',
  bundle: true,
  platform: 'node',
  format: 'esm',
  target: 'node20',
  banner: { js: '#!/usr/bin/env node' },
  define: { __CLI_VERSION__: JSON.stringify(version) },
  logLevel: 'warning',
});
console.error(`built dist/cli/estimate.mjs (${version})`);
