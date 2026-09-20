// Mirrors gerp93/VisualAssault packages/angular/theme-names.ts @ v0.2.0 (same tag
// src/renderer/themes.css is vendored from) - keep in sync when re-vendoring.
export const THEME_NAMES = [
  'blue-oval',
  'bubblegum',
  'commander-keen',
  'electric-lime',
  'flambeau',
  'flambeau-inverse',
  'green-acres',
  'hacker',
  'hawkeye',
  'lava',
  'merica',
  'neon',
  'red-barn',
  'retrowave',
] as const;

export type ThemeName = (typeof THEME_NAMES)[number];

export function themeDisplayName(themeId: string): string {
  return themeId
    .split('-')
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}
