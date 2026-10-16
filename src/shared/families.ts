/**
 * Model family keys. `z-image` was `z-image-turbo` until the key was renamed to name the family (Z-Image)
 * rather than one variant of it (Turbo). Old rows are rewritten by the startup migration (familyMigration.ts);
 * the old key is still accepted wherever a family comes in from outside - an MCP client's saved config, a
 * queued job from before the upgrade - and means the same thing, with no end date.
 */
export const Z_IMAGE_FAMILY = 'z-image';

/** The other text-to-image families. They have a text-to-image workflow only (no image to image, inpainting or outpainting). */
export const TEXT_TO_IMAGE_FAMILIES: readonly string[] = ['qwen-image', 'flux-dev', 'hidream', 'ovis-image'];

/** Old key -> current key. */
export const LEGACY_FAMILY_KEYS: Readonly<Record<string, string>> = {
  'z-image-turbo': Z_IMAGE_FAMILY,
};

/** The current key for a family, mapping a retired one forward and leaving any other value as it is. */
export function canonicalFamily(family: string): string {
  return LEGACY_FAMILY_KEYS[family] ?? family;
}
