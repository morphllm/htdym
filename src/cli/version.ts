// Stamped by scripts/build-cli.mjs from git describe; 'dev' when the
// sources run unbundled (vitest imports the commands in-process).
declare const __CLI_VERSION__: string | undefined;

export const VERSION: string = typeof __CLI_VERSION__ === 'string' ? __CLI_VERSION__ : 'dev';
