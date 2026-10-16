/**
 * Which model files can work together, worked out from the files themselves instead of from the template's defaults.
 *
 * A .safetensors header lists every tensor's name and shape. From a few of those shapes the app can read what a file *needs*
 * or *produces* - the width of the text embeddings a diffusion model takes in and an encoder gives out, the number of latent
 * channels a diffusion model works in and a VAE encodes to - and a pick in one slot then narrows what the other slots offer.
 *
 * The tensor names and shapes below were checked against the headers of real files (Z-Image Turbo, Qwen-Image and its 2512 release,
 * Flux.1 dev, HiDream I1 Full fp8, Ovis-Image; the Qwen3, Qwen2.5-VL, Ovis 2.5, Llama 3.1, T5 / UMT5 and CLIP encoders; the Flux, Qwen and
 * Wan VAEs) on 2026-10-16. `modelTraits.test.ts` builds its headers from those shapes. A file the rules do not recognise is never
 * filtered out on its own - it comes back as "unrecognised" or "unknown" and stays on offer, with a warning.
 */

/** What a file is, as far as its tensors say. */
export type ModelArch =
  | 'z-image-dit'
  | 'wan-dit'
  | 'qwen-image-dit'
  | 'flux-dit'
  | 'hidream-dit'
  | 'ovis-dit'
  | 'qwen3-encoder'
  | 'qwen25vl-encoder'
  | 'ovis-encoder'
  | 'llama-encoder'
  | 'umt5-encoder'
  | 't5-encoder'
  | 'clip-encoder'
  | 'latent-vae'
  | 'unknown';

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

  // HiDream: its own "stream" block names and a caption projection that takes the text in (the text encoders' 4096-wide output);
  // the final layer writes 2 x 2 x latent channels.
  const hiCaption = find(shapes, 'caption_projection.0.linear.weight');
  if (hiCaption && findMatching(shapes, /(^|\.)double_stream_blocks\./)) {
    const out = wide(find(shapes, 'final_layer.linear.weight'), 0);
    return { arch: 'hidream-dit', textWidth: wide(hiCaption, 1), latentChannels: out && out % 4 === 0 ? out / 4 : undefined };
  }

  // The "img_in / txt_in" family: Qwen-Image (diffusers-style transformer_blocks), Flux (double_blocks, a guidance embedder) and
  // Ovis-Image (double_blocks like Flux, but with a text norm). `img_in` takes 2 x 2 x latent channels.
  const imgIn = find(shapes, 'img_in.weight');
  const txtIn = find(shapes, 'txt_in.weight');
  if (imgIn && txtIn) {
    const patchIn = wide(imgIn, 1);
    const common = { textWidth: wide(txtIn, 1), latentChannels: patchIn && patchIn % 4 === 0 ? patchIn / 4 : undefined };
    if (find(shapes, 'txt_norm.weight') && findMatching(shapes, /(^|\.)transformer_blocks\./)) return { arch: 'qwen-image-dit' as const, ...common };
    if (findMatching(shapes, /(^|\.)double_blocks\./)) return { arch: find(shapes, 'txt_norm.scale') || find(shapes, 'txt_norm.weight') ? ('ovis-dit' as const) : ('flux-dit' as const), ...common };
  }

  // Text encoders that are language models: token embeddings and layers. Qwen3 (Z-Image) has per-head q/k norms; Ovis 2.5 is a Qwen3
  // with a vision tower on top; Qwen2.5-VL (Qwen-Image) has biases on its q/k/v projections instead; Llama has neither.
  const lmEmbed = find(shapes, 'model.embed_tokens.weight');
  // (Gemma has q/k norms and a vision tower too, but also a feed-forward pre-norm none of these have: it is not one of them.)
  if (lmEmbed && findMatching(shapes, /(^|\.)layers\.0\.self_attn\.q_proj\./) && !findMatching(shapes, /(^|\.)layers\.0\.pre_feedforward_layernorm\./)) {
    const outputWidth = wide(find(shapes, 'model.norm.weight'), 0) ?? wide(lmEmbed, 1);
    if (find(shapes, 'layers.0.self_attn.q_norm.weight')) {
      return { arch: findMatching(shapes, /(^|\.)vision_model\./) ? ('ovis-encoder' as const) : ('qwen3-encoder' as const), outputWidth };
    }
    if (find(shapes, 'layers.0.self_attn.q_proj.bias')) return { arch: 'qwen25vl-encoder' as const, outputWidth };
    return { arch: 'llama-encoder' as const, outputWidth };
  }

  // T5 and UMT5 (Wan's): the same blocks, told apart by the size of the vocabulary (32k against 256k).
  const sharedEmbed = find(shapes, 'shared.weight');
  if (sharedEmbed && find(shapes, 'encoder.block.0.layer.0.SelfAttention.q.weight')) {
    return {
      arch: (wide(sharedEmbed, 0) ?? 0) > 100_000 ? ('umt5-encoder' as const) : ('t5-encoder' as const),
      outputWidth: wide(find(shapes, 'encoder.final_layer_norm.weight'), 0) ?? wide(sharedEmbed, 1),
    };
  }

  // CLIP (clip_l is 768 wide, clip_g 1280): a text model with token embeddings.
  const clipEmbed = find(shapes, 'text_model.embeddings.token_embedding.weight');
  if (clipEmbed) return { arch: 'clip-encoder', outputWidth: wide(clipEmbed, 1) };

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
  /** The width of text embeddings the family's workflow takes, which only one text encoder size gives (Z-Image: Qwen3 4B; Wan: UMT5 XXL). */
  textWidth: number;
  /** The latent channels the family works in, which its VAE must decode. */
  latentChannels: number;
  /** For a diffusion model that comes in kinds (Wan). */
  variant?: 't2v' | 'i2v';
  /** What each text encoder slot must hold, by slot key (a family can have several: Flux has two, HiDream four). */
  encoders: Record<string, { arch: ModelArch; width: number }>;
  vaeDims: 2 | 3;
}

const FAMILY_RULES: Record<string, FamilyRule> = {
  'z-image': { dit: 'z-image-dit', textWidth: 2560, latentChannels: 16, encoders: { textEncoder: { arch: 'qwen3-encoder', width: 2560 } }, vaeDims: 2 },
  'wan22-i2v': { dit: 'wan-dit', textWidth: 4096, latentChannels: 16, variant: 'i2v', encoders: { textEncoder: { arch: 'umt5-encoder', width: 4096 } }, vaeDims: 3 },
  'wan22-t2v': { dit: 'wan-dit', textWidth: 4096, latentChannels: 16, variant: 't2v', encoders: { textEncoder: { arch: 'umt5-encoder', width: 4096 } }, vaeDims: 3 },
  // Qwen-Image's VAE is a 3D one (the same kind as Wan's), so a Wan VAE is not ruled out by shape.
  'qwen-image': { dit: 'qwen-image-dit', textWidth: 3584, latentChannels: 16, encoders: { textEncoder: { arch: 'qwen25vl-encoder', width: 3584 } }, vaeDims: 3 },
  'flux-dev': {
    dit: 'flux-dit',
    textWidth: 4096,
    latentChannels: 16,
    encoders: { textEncoder: { arch: 't5-encoder', width: 4096 }, clipL: { arch: 'clip-encoder', width: 768 } },
    vaeDims: 2,
  },
  hidream: {
    dit: 'hidream-dit',
    textWidth: 4096,
    latentChannels: 16,
    encoders: {
      textEncoder: { arch: 't5-encoder', width: 4096 },
      llama: { arch: 'llama-encoder', width: 4096 },
      clipL: { arch: 'clip-encoder', width: 768 },
      clipG: { arch: 'clip-encoder', width: 1280 },
    },
    vaeDims: 2,
  },
  'ovis-image': { dit: 'ovis-dit', textWidth: 2048, latentChannels: 16, encoders: { textEncoder: { arch: 'ovis-encoder', width: 2048 } }, vaeDims: 2 },
};

/** 'encoder': a text encoder whose output the diffusion model takes in; 'auxEncoder': a pooled-embedding encoder (CLIP) that is only checked on its own. */
export type SlotRole = 'diffusion' | 'encoder' | 'auxEncoder' | 'vae';

/** The role a slot key plays, or null for a slot nothing is checked on (the LoRAs: their names say little about what they fit). */
export function slotRole(slotKey: string): SlotRole | null {
  if (slotKey === 'diffusionModel' || slotKey === 'highNoiseModel' || slotKey === 'lowNoiseModel') return 'diffusion';
  if (slotKey === 'textEncoder' || slotKey === 'llama') return 'encoder';
  if (slotKey === 'clipL' || slotKey === 'clipG') return 'auxEncoder';
  if (slotKey === 'vae') return 'vae';
  return null;
}

const ARCH_NAMES: Record<ModelArch, string> = {
  'z-image-dit': 'a Z-Image model',
  'wan-dit': 'a Wan video model',
  'qwen-image-dit': 'a Qwen-Image model',
  'flux-dit': 'a Flux model',
  'hidream-dit': 'a HiDream model',
  'ovis-dit': 'an Ovis-Image model',
  'qwen3-encoder': 'a Qwen3 text encoder',
  'qwen25vl-encoder': 'a Qwen2.5-VL text encoder',
  'ovis-encoder': 'an Ovis 2.5 text encoder',
  'llama-encoder': 'a Llama text encoder',
  'umt5-encoder': 'a UMT5 text encoder',
  't5-encoder': 'a T5 text encoder',
  'clip-encoder': 'a CLIP text encoder',
  'latent-vae': 'a VAE',
  unknown: 'a file the app does not recognise',
};

const ROLE_NAMES: Record<SlotRole, string> = { diffusion: 'the image or video model', encoder: 'the text encoder', auxEncoder: 'the CLIP text encoder', vae: 'the VAE' };

/**
 * 'conflict': the file is positively something else, or disagrees with the family or with what is picked - it cannot work.
 * 'unrecognised': its header was read and the app could not tell what it is - hidden from the lists, but only a warning, since the
 * rules are written from knowledge of the model code and a real file may name its tensors differently.
 * 'unknown': nothing could be read (not a .safetensors file, or the folder is not reachable), so nothing is known either way.
 */
export type Fit = { status: 'fits' } | { status: 'unknown' } | { status: 'conflict'; reason: string } | { status: 'unrecognised'; reason: string };

/** What the other slots of the editor have picked, by role (a Wan model has two diffusion slots). */
export type ChosenTraits = Partial<Record<SlotRole, ModelTraits[]>>;

/** ComfyUI lists these in its VAE loader, but they are placeholders and live previews, not VAE files. */
const BUILT_IN_VAE_NAMES = new Set(['pixel_space', 'taesd', 'taesdxl', 'taesd3', 'taef1']);

/**
 * Whether `candidate` (the traits of a file offered in slot `slotKey` of `family`) fits what the family needs there and what is
 * already picked in the other slots. `fileName` (as ComfyUI lists it) catches built-in names that are not files at all.
 */
export function checkFit(family: string, slotKey: string, candidate: ModelTraits | null, chosen: ChosenTraits, fileName?: string): Fit {
  const rule = FAMILY_RULES[family];
  const role = slotRole(slotKey);
  if (!rule || !role) return { status: 'unknown' };
  if (role === 'vae' && fileName && BUILT_IN_VAE_NAMES.has(fileName.toLowerCase())) {
    return { status: 'conflict', reason: `"${fileName}" is a built-in entry in ComfyUI's list, not a VAE file.` };
  }
  if (!candidate) return { status: 'unknown' };

  const encoderRule = rule.encoders[slotKey];
  if ((role === 'encoder' || role === 'auxEncoder') && !encoderRule) return { status: 'unknown' };
  const wantArch = role === 'diffusion' ? rule.dit : role === 'vae' ? 'latent-vae' : encoderRule.arch;
  if (candidate.arch === 'unknown') return { status: 'unrecognised', reason: `The app cannot tell that this is ${ARCH_NAMES[wantArch]}.` };
  if (candidate.arch !== wantArch) return { status: 'conflict', reason: `This looks like ${ARCH_NAMES[candidate.arch]}, not ${ARCH_NAMES[wantArch]}.` };
  if ((role === 'encoder' || role === 'auxEncoder') && candidate.outputWidth && candidate.outputWidth !== encoderRule.width) {
    return { status: 'conflict', reason: `This encoder gives ${candidate.outputWidth}-wide text, but this kind of model needs ${encoderRule.width} here. It is a different size of the same encoder.` };
  }
  if (role === 'diffusion' && candidate.textWidth && candidate.textWidth !== rule.textWidth) {
    return { status: 'conflict', reason: `This model takes ${candidate.textWidth}-wide text, but this family's workflow gives ${rule.textWidth}.` };
  }
  if ((role === 'vae' || role === 'diffusion') && candidate.latentChannels && candidate.latentChannels !== rule.latentChannels) {
    return { status: 'conflict', reason: `This works in ${candidate.latentChannels} latent channels, but this kind of model uses ${rule.latentChannels}.` };
  }
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
