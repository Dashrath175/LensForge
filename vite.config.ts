import { defineConfig } from 'vitest/config';
import { wgsl } from './tools/vitePluginWgsl';

export default defineConfig({
  plugins: [wgsl()],
  build: { target: 'es2022' },
  test: {
    include: ['tests/**/*.test.ts'],
  },
});
