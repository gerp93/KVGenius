/**
 * Face reference: a new picture in which the person from a supplied reference picture appears - the same face
 * carried into a different scene. It is an edit-style model that is shown the reference picture and told in the
 * prompt what to do with it ("the woman in the reference photo, sitting in a cafe").
 *
 * Several methods can do this and which keeps a face best is a matter of looking, so each is its own family (its
 * own workflow and model files) and the Generate page lets the person run the same reference and prompt through
 * each and compare. They are made from a supplied picture (see shared/sourceFamilies.ts). They are experimental:
 * no model profiles, and the sampler is the distilled one the template ships with.
 */
export const FACE_KLEIN_FAMILY = 'face-klein';
export const FACE_QWEN_FAMILY = 'face-qwen';

export interface FaceMethod {
  family: string;
  /** Short name for the method picker. */
  label: string;
  /** One line on what to expect, shown under the picker. */
  note: string;
}

export const FACE_METHODS: readonly FaceMethod[] = [
  {
    family: FACE_KLEIN_FAMILY,
    label: 'Flux.2 Klein 4B',
    note: 'Fast (4 steps) and light on memory. Reads the reference with the image model itself.',
  },
  {
    family: FACE_QWEN_FAMILY,
    label: 'Qwen-Image-Edit 2511',
    note: 'Slower and heavier (a 20 GB model); reads the reference with a vision-language model. Often stronger on identity.',
  },
];

export const FACE_FAMILIES: readonly string[] = FACE_METHODS.map((m) => m.family);

export function isFaceReferenceFamily(family: string): boolean {
  return FACE_FAMILIES.includes(family);
}

export function faceMethod(family: string): FaceMethod | undefined {
  return FACE_METHODS.find((m) => m.family === family);
}

/** The method used when none has been chosen. */
export const DEFAULT_FACE_FAMILY = FACE_KLEIN_FAMILY;
