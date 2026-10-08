/**
 * Which model files can work together, worked out from the files themselves instead of from the template's defaults.
 *
 * A .safetensors header lists every tensor's name and shape. From a few of those shapes the app can read what a file *needs*
 * or *produces* - the width of the text embeddings a diffusion model takes in and an encoder gives out, the number of latent
 * channels a diffusion model works in and a VAE encodes to - and a pick in one slot then narrows what the other slots offer.
 *
 * The tensor names below are written from knowledge of the model code (ComfyUI's Z-Image / Wan implementations, Qwen3, UMT5, the
 * Flux and Wan VAEs), NOT checked against the real files: they are the part to verify against real headers
 * (`modelTraits.test.ts` builds headers from the same assumptions, so it cannot catch a wrong name). A file the rules do not
 * recognise is never filtered out - it comes back as "unknown" and stays on offer.
 */

/** What a file is, as far as its tensors say. */
export type ModelArch = 'z-image-dit' | 'wan-dit' | 'qwen3-encoder' | 'umt5-encoder' | 'latent-vae' | 'unknown';

export interface ModelTraits {
  arch: ModelArch;
  /** Diffusion models: width of the text embeddings it takes in. */
  textWidth?: number;
  /** Encoders: width of the embeddings it gives out. */
  outputWidth?: number;
  /** Diffusion models and VAEs: how many channels the latent has. */
  latentChannels?: number;
  /** Wan diffusion models: which of the two it is (they take different inputs, so they cannot swap). */
  variant?: 't2v' | 'i2v';
  /** VAEs: 2 for a picture VAE, 3 for a video VAE (Wan's) - both can have 16 channels and still not swap. */
  vaeDims?: 2 | 3;
}

/** Tensor name -> shape: all the rules look at. */
export type TensorShapes = Record<string, number[]>;

/** The shape of the tensor whose name is `name`, or ends in `.name` (some files carry a prefix such as `model.diffusion_model.`). */
function find(shapes: TensorShapes, name: string): number[] | undefined {
  if (shapes[name]) return shapes[name];
  const tail = `.${name}`;
  for (const key of Object.keys(shapes)) if (key.endsWith(tail)) return shapes[key];
  return undefined;
}

/** The shape of the first tensor whose name matches. */
function findMatching(shapes: TensorShapes, pattern: RegExp): number[] | undefined {
  for (const key of Object.keys(shapes)) if (pattern.test(key)) return shapes[key];
  return undefined;
}

const wide = (shape: number[] | undefined, axis: number): number | undefined => (shape && shape.length > axis && shape[axis] > 0 ? shape[axis] : undefined);

/** Reads what a file is from its tensors. Anything not recognised is `{ arch: 'unknown' }`. */
export function readTraits(shapes: TensorShapes): ModelTraits {
  // Z-Image (a Lumina-style DiT): `cap_embedder.1` is the linear layer taking the text embeddings in; the patch embedder takes
  // patch x patch x latent channels (2 x 2 x 16 = 64 for Z-Image).
  const capEmbed = find(shapes, 'cap_embedder.1.weight');
  const xEmbed = findMatching(shapes, /(^|\.)(x_embedder|all_x_embedder\.[^.]+)\.weight$/);
  if (capEmbed && xEmbed) {
    const patchIn = wide(xEmbed, 1);
    return { arch: 'z-image-dit', textWidth: wide(capEmbed, 1), latentChannels: patchIn && patchIn % 4 === 0 ? patchIn / 4 : undefined };
  }

  // Wan: a 3D patch embedding (its input channels tell text to video - the latent alone - from image to video - latent, a mask and the
  // start picture's latent) and a text embedding.
  const patchEmbed = find(shapes, 'patch_embedding.weight');
  const textEmbed = find(shapes, 'text_embedding.0.weight');
  if (patchEmbed && patchEmbed.length === 5 && textEmbed) {
    const inChannels = wide(patchEmbed, 1);
    const variant = inChannels === 16 ? 't2v' : inChannels === 36 ? 'i2v' : undefined;
    return {
      arch: 'wan-dit',
      textWidth: wide(textEmbed, 1),
      variant,
      // Both 14B models work in Wan 2.1's 16-channel latent (image to video adds the mask and start picture on top of it).
      latentChannels: variant ? 16 : undefined,
    };
  }

  // Qwen3 (Z-Image's text encoder): token embeddings plus the per-head q/k norms Qwen2 does not have.
  const qwenEmbed = find(shapes, 'model.embed_tokens.weight');
  if (qwenEmbed && find(shapes, 'layers.0.self_attn.q_norm.weight')) {
    return { arch: 'qwen3-encoder', outputWidth: wide(find(shapes, 'model.norm.weight'), 0) ?? wide(qwenEmbed, 1) };
  }

  // UMT5 (Wan's text encoder): a shared embedding and T5-style encoder blocks.
  const sharedEmbed = find(shapes, 'shared.weight');
  if (sharedEmbed && find(shapes, 'encoder.block.0.layer.0.SelfAttention.q.weight')) {
    return { arch: 'umt5-encoder', outputWidth: wide(find(shapes, 'encoder.final_layer_norm.weight'), 0) ?? wide(sharedEmbed, 1) };
  }

  // VAEs: what the decoder takes in is the latent. A picture VAE's first decoder layer is `conv_in` (2D); Wan's video VAE has `conv1` (3D).
  const decoderIn = find(shapes, 'decoder.conv_in.weight');
  if (decoderIn && decoderIn.length === 4) return { arch: 'latent-vae', latentChannels: wide(decoderIn, 1), vaeDims: 2 };
  const videoDecoderIn = find(shapes, 'decoder.conv1.weight');
  if (videoDecoderIn && videoDecoderIn.length === 5) return { arch: 'latent-vae', latentChannels: wide(videoDecoderIn, 1), vaeDims: 3 };

  return { arch: 'unknown' };
}

/** What each family's slots must be. */
interface FamilyRule {
  dit: ModelArch;
  /** For a diffusion model that comes in kinds (Wan). */
  variant?: 't2v' | 'i2v';
  encoder: ModelArch;
  vaeDims: 2 | 3;
}

const FAMILY_RULES: Record<string, FamilyRule> = {
  'z-image': { dit: 'z-image-dit', encoder: 'qwen3-encoder', vaeDims: 2 },
  'wan22-i2v': { dit: 'wan-dit', variant: 'i2v', encoder: 'umt5-encoder', vaeDims: 3 },
  'wan22-t2v': { dit: 'wan-dit', variant: 't2v', encoder: 'umt5-encoder', vaeDims: 3 },
};

export type SlotRole = 'diffusion' | 'encoder' | 'vae';

/** The role a slot key plays, or null for a slot nothing is checked on (the LoRAs: their names say little about what they fit). */
export function slotRole(slotKey: string): SlotRole | null {
  if (slotKey === 'diffusionModel' || slotKey === 'highNoiseModel' || slotKey === 'lowNoiseModel') return 'diffusion';
  if (slotKey === 'textEncoder') return 'encoder';
  if (slotKey === 'vae') return 'vae';
  return null;
}

const ARCH_NAMES: Record<ModelArch, string> = {
  'z-image-dit': 'a Z-Image model',
  'wan-dit': 'a Wan video model',
  'qwen3-encoder': 'a Qwen3 text encoder',
  'umt5-encoder': 'a UMT5 text encoder',
  'latent-vae': 'a VAE',
  unknown: 'a file the app does not recognise',
};

const ROLE_NAMES: Record<SlotRole, string> = { diffusion: 'the image or video model', encoder: 'the text encoder', vae: 'the VAE' };

export type Fit = { status: 'fits' } | { status: 'unknown' } | { status: 'conflict'; reason: string };

/** What the other slots of the editor have picked, by role (a Wan model has two diffusion slots). */
export type ChosenTraits = Partial<Record<SlotRole, ModelTraits[]>>;

/**
 * Whether `candidate` (the traits of a file offered in slot `slotKey` of `family`) fits what the family needs there and what is
 * already picked in the other slots. A file whose traits are not known (not read, not recognised) is 'unknown' and stays on offer.
 */
export function checkFit(family: string, slotKey: string, candidate: ModelTraits | null, chosen: ChosenTraits): Fit {
  const rule = FAMILY_RULES[family];
  const role = slotRole(slotKey);
  if (!rule || !role || !candidate || candidate.arch === 'unknown') return { status: 'unknown' };

  const wantArch = role === 'diffusion' ? rule.dit : role === 'encoder' ? rule.encoder : 'latent-vae';
  if (candidate.arch !== wantArch) return { status: 'conflict', reason: `This looks like ${ARCH_NAMES[candidate.arch]}, not ${ARCH_NAMES[wantArch]}.` };

  if (role === 'diffusion' && rule.variant && candidate.variant && candidate.variant !== rule.variant) {
    return { status: 'conflict', reason: candidate.variant === 't2v' ? 'This is a text-to-video model; this needs an image-to-video one.' : 'This is an image-to-video model; this needs a text-to-video one.' };
  }
  if (role === 'vae' && candidate.vaeDims && candidate.vaeDims !== rule.vaeDims) {
    return { status: 'conflict', reason: rule.vaeDims === 3 ? 'This is a picture VAE; a video model needs a video VAE.' : 'This is a video VAE; a picture model needs a picture VAE.' };
  }

  // Against what is already picked.
  for (const other of otherPicks(role, chosen)) {
    const reason = disagreement(role, candidate, other.role, other.traits);
    if (reason) return { status: 'conflict', reason };
  }
  return { status: 'fits' };
}

function otherPicks(role: SlotRole, chosen: ChosenTraits): { role: SlotRole; traits: ModelTraits }[] {
  const picks: { role: SlotRole; traits: ModelTraits }[] = [];
  for (const other of ['diffusion', 'encoder', 'vae'] as const) {
    if (other === role) continue;
    for (const traits of chosen[other] ?? []) if (traits.arch !== 'unknown') picks.push({ role: other, traits });
  }
  return picks;
}

/** Why two files of different roles cannot work together, or null if they can (or nothing can be told). */
function disagreement(role: SlotRole, a: ModelTraits, otherRole: SlotRole, b: ModelTraits): string | null {
  const model = role === 'diffusion' ? a : otherRole === 'diffusion' ? b : null;
  const encoder = role === 'encoder' ? a : otherRole === 'encoder' ? b : null;
  const vae = role === 'vae' ? a : otherRole === 'vae' ? b : null;
  if (model && encoder && model.textWidth && encoder.outputWidth && model.textWidth !== encoder.outputWidth) {
    return role === 'diffusion'
      ? `This model takes ${model.textWidth}-wide text, but ${ROLE_NAMES.encoder} you picked gives ${encoder.outputWidth}.`
      : `This encoder gives ${encoder.outputWidth}-wide text, but ${ROLE_NAMES.diffusion} you picked takes ${model.textWidth}.`;
  }
  if (model && vae && model.latentChannels && vae.latentChannels && model.latentChannels !== vae.latentChannels) {
    return role === 'diffusion'
      ? `This model works in ${model.latentChannels} latent channels, but ${ROLE_NAMES.vae} you picked uses ${vae.latentChannels}.`
      : `This VAE uses ${vae.latentChannels} latent channels, but ${ROLE_NAMES.diffusion} you picked works in ${model.latentChannels}.`;
  }
  return null;
}
