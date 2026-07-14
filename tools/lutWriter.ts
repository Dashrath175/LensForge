import { createHash } from 'node:crypto';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

/**
 * LUT output helper: Float32 binary + JSON metadata (with a content hash so
 * regeneration reproducibility is checkable at a glance — DEVELOPMENT.md §4).
 */
export function writeLut(
  relPath: string,
  data: Float32Array,
  meta: Record<string, unknown>,
): void {
  const binPath = resolve(process.cwd(), relPath);
  mkdirSync(dirname(binPath), { recursive: true });
  const bytes = Buffer.from(data.buffer, data.byteOffset, data.byteLength);
  const sha256 = createHash('sha256').update(bytes).digest('hex');
  writeFileSync(binPath, bytes);
  writeFileSync(
    binPath.replace(/\.bin$/, '.json'),
    `${JSON.stringify({ ...meta, elements: data.length, sha256 }, null, 2)}\n`,
  );
  console.log(`${relPath}: ${data.length} floats, sha256 ${sha256.slice(0, 16)}…`);
}
