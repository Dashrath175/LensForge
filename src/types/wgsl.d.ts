/** WGSL sources are assembled (with #include resolution) into plain strings by tools/vitePluginWgsl.ts. */
declare module '*.wgsl' {
  const source: string;
  export default source;
}
