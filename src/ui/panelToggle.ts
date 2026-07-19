/**
 * Control-panel visibility toggle (presentation only): the corner button and
 * the H shortcut hide ONLY the right-side panel — the HUD, canvas, and every
 * behavior underneath are untouched. State lives on <body> as a class so the
 * transition is pure CSS (index.html) and survives panel rebuilds after
 * device loss.
 */
export function installPanelToggle(button: HTMLButtonElement): void {
  let visible = true;

  const apply = (): void => {
    document.body.classList.toggle('controls-hidden', !visible);
    const label = visible ? 'Hide control panel (H)' : 'Show control panel (H)';
    button.setAttribute('aria-label', label);
    button.title = label;
  };

  button.addEventListener('click', () => {
    visible = !visible;
    apply();
    button.blur(); // so a following Space/Enter drives the app, not the button
  });

  window.addEventListener('keydown', (ev) => {
    if (ev.key !== 'h' && ev.key !== 'H') return;
    if (ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const target = ev.target;
    if (target instanceof Element && target.closest('input, textarea, select, [contenteditable="true"]')) return;
    visible = !visible;
    apply();
  });
}
