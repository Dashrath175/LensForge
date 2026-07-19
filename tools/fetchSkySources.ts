/**
 * Fetch the sky-pipeline source datasets (IMPLEMENTATION.md M7) into
 * assets/sky-src/ (gitignored — sources are dev-time inputs, not shipped).
 * URLs and SHA-256 hashes are pinned so the generated assets are reproducible
 * (the M7 risk note: pin tool/data versions).
 *
 *  - NASA/GSFC Scientific Visualization Studio, "Deep Star Maps 2020"
 *    (SVS 4851), celestial-coordinate equirectangular EXR, 4k. Public domain.
 *  - Yale Bright Star Catalogue, 5th revised ed. (Hoffleit & Warren 1991),
 *    via the Harvard-Smithsonian Telescope Data Center. ~9100 stars to V≈6.5.
 */
import { createHash } from 'node:crypto';
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';

export interface SkySource {
  file: string;
  url: string;
  sha256: string;
}

export const SKY_SOURCES: readonly SkySource[] = [
  {
    file: 'assets/sky-src/starmap_2020_4k.exr',
    url: 'https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_4k.exr',
    sha256: '69b841cd048c2ef35543eb1c5819b322530e78aa4b0cab97ec1de3ef56a4ddcd',
  },
  {
    file: 'assets/sky-src/bsc5.dat.gz',
    url: 'http://tdc-www.harvard.edu/catalogs/bsc5.dat.gz',
    sha256: '0471ced07fc241e625613baa689b2fa5f4b59fd733f731003e57d2d436036cdb',
  },
];

function sha256(buf: Buffer): string {
  return createHash('sha256').update(buf).digest('hex');
}

/** Ensure every pinned source is present and hash-verified; download if absent. */
export async function fetchSkySources(): Promise<void> {
  for (const src of SKY_SOURCES) {
    const path = resolve(process.cwd(), src.file);
    if (existsSync(path)) {
      const hash = sha256(readFileSync(path));
      if (hash === src.sha256) {
        console.log(`${src.file}: present, sha256 verified`);
        continue;
      }
      throw new Error(`${src.file}: sha256 mismatch (${hash}) — delete the file to re-download`);
    }
    console.log(`${src.file}: downloading ${src.url} …`);
    const res = await fetch(src.url);
    if (!res.ok) throw new Error(`${src.url}: HTTP ${res.status}`);
    const buf = Buffer.from(await res.arrayBuffer());
    const hash = sha256(buf);
    if (hash !== src.sha256) {
      throw new Error(`${src.file}: downloaded sha256 ${hash} does not match pinned ${src.sha256}`);
    }
    mkdirSync(dirname(path), { recursive: true });
    writeFileSync(path, buf);
    console.log(`${src.file}: ${buf.length} bytes, sha256 verified`);
  }
}

// Run directly: `npm run tools:fetch-sky`
if (process.argv[1]?.endsWith('fetchSkySources.ts')) {
  fetchSkySources().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
