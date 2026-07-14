import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import type { Plugin } from 'vite';

/**
 * WGSL module plugin (ARCHITECTURE.md §5): WGSL has no `#include`, so shader
 * sources are assembled at build time. Supports:
 *
 *   #include "relative/path.wgsl"
 *
 * Include-once semantics (a file inlined multiple times would redeclare
 * symbols), cycle detection, and dev-server HMR via watch-file registration.
 */
const INCLUDE_RE = /^[ \t]*#include\s+"([^"]+)"[ \t]*$/gm;

function assemble(file: string, stack: string[], seen: Set<string>, watched: Set<string>): string {
  if (stack.includes(file)) {
    throw new Error(`wgsl: circular #include: ${[...stack, file].join(' -> ')}`);
  }
  if (seen.has(file)) return '';
  seen.add(file);
  watched.add(file);

  const source = readFileSync(file, 'utf8');
  return source.replace(INCLUDE_RE, (_match: string, rel: string) => {
    const target = resolve(dirname(file), rel);
    const body = assemble(target, [...stack, file], seen, watched);
    return `// ---- begin include: ${rel} ----\n${body}\n// ---- end include: ${rel} ----`;
  });
}

export function wgsl(): Plugin {
  return {
    name: 'lensforge:wgsl',
    enforce: 'pre',
    transform(_code, id) {
      if (!id.endsWith('.wgsl')) return null;
      const watched = new Set<string>();
      const assembled = assemble(id, [], new Set(), watched);
      for (const f of watched) if (f !== id) this.addWatchFile(f);
      return { code: `export default ${JSON.stringify(assembled)};`, map: null };
    },
  };
}
