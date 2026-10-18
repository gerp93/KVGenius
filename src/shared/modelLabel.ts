import { canonicalFamily } from './families';
import { GIF_FAMILY } from './gif';
import { profileFamily, profileFamilyKey } from './modelFamilies';
import { isUpscaleFamily } from './upscale';

/** The name of the model an item was made with, for its Library tag: the saved model variant it used, else the
 * built-in model of its family (Z Image Turbo, Wan 2.2 text to video, ...). Null when no generation model was
 * involved (an upscale, a GIF) or the family is not known - better no tag than a wrong one. */
export function modelLabel(record: { modelFamily: string; modelName: string | null }): string | null {
  const family = canonicalFamily(record.modelFamily);
  if (isUpscaleFamily(family) || family === GIF_FAMILY) return null;
  const name = record.modelName?.trim();
  if (name) return name;
  return profileFamily(profileFamilyKey(family))?.builtInName ?? null;
}
