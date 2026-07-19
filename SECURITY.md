# Security Policy

LensForge is a fully static, client-side application: no backend, no
serverless functions, no telemetry, no network requests at runtime beyond
fetching its own bundled assets. Its attack surface is correspondingly small,
but not zero.

## Reporting a vulnerability

Please report suspected vulnerabilities privately via
**GitHub → Security → Report a vulnerability** (private vulnerability
reporting) rather than opening a public issue. You can expect an
acknowledgement within a week.

Reports we consider in scope:

- Supply-chain issues in the (deliberately minimal) dependency tree.
- Anything that could make the published site serve content other than the
  built `dist/` output.
- Preset-file parsing (`Load JSON…`): the parser validates structurally, but
  a crafted preset that breaks out of parameter space would be a bug worth a
  private report.
- Issues in the build/release workflows (`.github/workflows/`).

Out of scope: GPU driver crashes or device resets caused by extreme
parameters (the renderer bounds every dispatch specifically to stay under the
OS watchdog — reproducible TDRs are ordinary bugs, please file them
publicly), and browser WebGPU implementation bugs (report those to the
browser vendor; we are happy to help triage).

## Supported versions

Only the latest release is supported. There is no LTS branch.
