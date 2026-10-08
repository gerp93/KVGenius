/**
 * The tools KVGenius exposes to outside clients (see docs/mcp-plan.md). Defined once here and used
 * by the local HTTP API (which runs them) and the MCP stdio shim (which advertises them), so the
 * two can never disagree. Plain JSON Schema, nothing client-specific.
 */
export interface ToolDefinition {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, unknown>;
    required?: string[];
  };
}

const batchProp = {
  type: 'string',
  description: 'Optional label grouping related work (e.g. "music-video-1"). Filter jobs and library items by it later.',
};

const itemIdProp = (what: string) => ({
  type: 'string',
  description: `${what} - a library item id such as "gen-12" (generated) or "imp-5" (imported), as returned by other tools.`,
});

export const TOOLS: ToolDefinition[] = [
  {
    name: 'list_capabilities',
    description:
      'Lists the model families KVGenius can run (image or video), the fields each accepts with their limits and defaults, and whether ComfyUI and ffmpeg are available right now. Call this first.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'import_folder',
    description:
      'Imports local image, video and audio files into the library so other tools can refer to them by id. Accepts a folder (files are returned sorted by name, natural order: img2 before img10) or a single file. Paths must be absolute on the machine running KVGenius. Files are referenced in place, not copied.',
    inputSchema: {
      type: 'object',
      properties: {
        path: { type: 'string', description: 'Absolute path of a folder or a file. A leading ~ is expanded.' },
        recursive: { type: 'boolean', description: 'Also import files in sub-folders. Default false.' },
        batch: batchProp,
      },
      required: ['path'],
    },
  },
  {
    name: 'generate_image',
    description:
      'Queues a text-to-image generation - or, with `source`, an image-to-image one - and returns immediately with a job id. Generation takes seconds to minutes; poll with get_job.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'What to generate.' },
        style: {
          type: 'string',
          description:
            'Optional. The name of one of the user\'s saved styles (see list_styles; case-insensitive). Its wording is appended to the prompt, so describe only the subject in `prompt`. Omit to send the prompt exactly as written.',
        },
        source: itemIdProp('Optional. A library picture to start from (image to image). The new picture is drawn from it and the prompt, instead of from nothing; width and height then default to its shape'),
        strength: {
          type: 'number',
          description:
            'With `source`: how much of the picture is re-drawn, 0.05-1. Small keeps most of it, 1 ignores it. Default 0.6.',
        },
        extend: {
          type: 'object',
          description:
            'With `source`: outpainting - extend the picture beyond its frame. Pixels to add on each side of the source (any may be 0, up to 2048; at least one must be above 0). The original is kept as it is and only the new area is drawn, continuing the picture and following the prompt. The size is the whole extended picture (long side 1024-1536), so width, height and strength are ignored.',
          properties: {
            left: { type: 'integer' },
            top: { type: 'integer' },
            right: { type: 'integer' },
            bottom: { type: 'integer' },
          },
        },
        model: {
          type: 'string',
          description:
            'Optional. The name of one of the user\'s saved models (see list_models; case-insensitive), a variant of the family such as a fine-tune. Its files and sampler are used, and `steps` and `cfg` default to its own. Omit for the built-in model.',
        },
        family: { type: 'string', description: 'Model family (see list_capabilities). Default "z-image" (the old name "z-image-turbo" still works).' },
        width: { type: 'integer', description: 'Pixels, snapped to a multiple of 64 (256-2048). Default 1024.' },
        height: { type: 'integer', description: 'Pixels, snapped to a multiple of 64 (256-2048). Default 1024.' },
        seed: { type: 'integer', description: 'Random if omitted. Reuse a seed to reproduce a result.' },
        steps: { type: 'integer', description: 'Sampling steps, 1-20 (1-100 with a `model`). Default 8, or the model\'s own.' },
        cfg: { type: 'number', description: 'Guidance, 0.5-3 (0-30 with a `model`). Default 1, or the model\'s own.' },
        batch: batchProp,
      },
      required: ['prompt'],
    },
  },
  {
    name: 'list_models',
    description:
      'Lists the models available to generate_image and generate_video: the built-in ones and any variants the user saved in KVGenius (a fine-tune or a different checkpoint of the same family), with the steps and CFG an image model is set up for. Pass a model\'s name as `model` to the tool listed for it.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'list_styles',
    description:
      'Lists the prompt styles the user saved in KVGenius (a name and the wording it adds after a prompt, e.g. "1930s movie poster"). Pass a style\'s name as `style` to generate_image to apply it.',
    inputSchema: { type: 'object', properties: {} },
  },
  {
    name: 'generate_video',
    description:
      'Queues a video generation and returns immediately with a job id: image-to-video when `source` (a library image) is given, text-to-video from the prompt alone when it is left out. A clip takes several minutes; jobs run one at a time in the order submitted, so queue a whole batch and poll with list_jobs.',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: { type: 'string', description: 'With a source: the motion and action to add to the image. Without one: the whole scene and its motion.' },
        source: itemIdProp('Optional. The image to animate (image-to-video). Leave out for text-to-video'),
        family: { type: 'string', description: 'Video family (see list_capabilities). Default "wan22-i2v" with a source, "wan22-t2v" without.' },
        model: {
          type: 'string',
          description:
            'Optional. The name of one of the user\'s saved video models (see list_models; case-insensitive) - a variant that swaps the Wan model files. Omit for the built-in one.',
        },
        seconds: { type: 'number', description: 'Clip length, 1-12 seconds (snapped to quarter seconds). Default 5.' },
        width: { type: 'integer', description: 'Snapped to a multiple of 16. Default: keeps the source aspect ratio at ~640px on the long side (640 without a source).' },
        height: { type: 'integer', description: 'Snapped to a multiple of 16. Default: keeps the source aspect ratio at ~640px on the long side (640 without a source).' },
        seed: { type: 'integer', description: 'Random if omitted.' },
        batch: batchProp,
      },
      required: ['prompt'],
    },
  },
  {
    name: 'list_jobs',
    description:
      'Lists the generation jobs submitted through this interface, newest first. Use it to see what is queued, running, finished or failed, including jobs submitted earlier in another conversation. Work done in the KVGenius app itself is never shown.',
    inputSchema: {
      type: 'object',
      properties: {
        batch: batchProp,
        status: { type: 'string', enum: ['queued', 'running', 'done', 'failed', 'cancelled', 'interrupted'] },
        limit: { type: 'integer', description: 'Default 100, max 500.' },
      },
    },
  },
  {
    name: 'get_job',
    description:
      'Returns one job: its status, live progress while running, the error if it failed, and (once done) the resulting library item. wait_seconds holds the call open until the job finishes or the time runs out.',
    inputSchema: {
      type: 'object',
      properties: {
        job_id: { type: 'integer' },
        wait_seconds: { type: 'number', description: 'Wait up to this long (max 60) for the job to finish. Default 0.' },
      },
      required: ['job_id'],
    },
  },
  {
    name: 'cancel_job',
    description:
      'Cancels one waiting or running generation job (job_id), every waiting job of a batch (batch), or a running assembly (assembly_id). The running job is interrupted on the GPU.',
    inputSchema: {
      type: 'object',
      properties: {
        job_id: { type: 'integer' },
        batch: batchProp,
        assembly_id: { type: 'integer' },
      },
    },
  },
  {
    name: 'list_library',
    description:
      'Lists the library items created through this interface (generated by its jobs, imported, or assembled), newest first. Each has an id usable in other tools. Images and videos made in the KVGenius app itself are not included.',
    inputSchema: {
      type: 'object',
      properties: {
        kind: { type: 'string', enum: ['image', 'video', 'audio'] },
        origin: { type: 'string', enum: ['generated', 'imported', 'assembled'] },
        batch: batchProp,
        limit: { type: 'integer', description: 'Default 50, max 200.' },
      },
    },
  },
  {
    name: 'get_item',
    description:
      'Returns one library item with its file path and details. For images and videos it also returns a small preview image (a frame, for video) so you can check the result by eye.',
    inputSchema: {
      type: 'object',
      properties: {
        item_id: itemIdProp('The item'),
        preview: { type: 'boolean', description: 'Include the preview image. Default true.' },
      },
      required: ['item_id'],
    },
  },
  {
    name: 'probe_media',
    description: 'Returns duration, resolution, frame rate and codecs of a library video, audio or image item. Use it to plan how clips fit a music track.',
    inputSchema: {
      type: 'object',
      properties: { item_id: itemIdProp('The item to inspect') },
      required: ['item_id'],
    },
  },
  {
    name: 'assemble_video',
    description:
      'Stitches video clips together in the order given and lays one audio file (the full track) under them, producing a new mp4 in the library. Runs in the background: returns an assembly id at once (or the finished result if it completes within wait_seconds); poll with get_assembly. Clips are normalised to the first clip\'s size and frame rate.',
    inputSchema: {
      type: 'object',
      properties: {
        clips: {
          type: 'array',
          minItems: 1,
          description: 'Video items in playback order. Each entry is an item id, or {item_id, seconds} to use only the first `seconds` of that clip.',
          items: {
            anyOf: [
              { type: 'string' },
              {
                type: 'object',
                properties: { item_id: { type: 'string' }, seconds: { type: 'number' } },
                required: ['item_id'],
              },
            ],
          },
        },
        audio: itemIdProp('The backing track (optional)'),
        transition: { type: 'string', enum: ['cut', 'crossfade'], description: 'Default "cut".' },
        crossfade_seconds: { type: 'number', description: 'Length of each crossfade, default 0.5.' },
        end: {
          type: 'string',
          enum: ['trim_to_video', 'trim_to_audio', 'fade_out'],
          description:
            'How the ending is decided when clips and audio differ in length. trim_to_video (default): the video sets the length and the audio is cut. trim_to_audio: the audio sets the length; video is cut, or its last frame held, to match. fade_out: like trim_to_video with a fade to black/silence at the end.',
        },
        fade_seconds: { type: 'number', description: 'Fade length for end="fade_out", default 2.' },
        name: { type: 'string', description: 'Output file name (without extension). Default assembled-<timestamp>.' },
        batch: batchProp,
        wait_seconds: { type: 'number', description: 'Wait up to this long (max 60) for it to finish. Default 0.' },
      },
      required: ['clips'],
    },
  },
  {
    name: 'get_assembly',
    description: 'Returns the status of an assemble_video run: progress while running, the error if it failed, and the resulting library item when done.',
    inputSchema: {
      type: 'object',
      properties: {
        assembly_id: { type: 'integer' },
        wait_seconds: { type: 'number', description: 'Wait up to this long (max 60). Default 0.' },
      },
      required: ['assembly_id'],
    },
  },
];

/** What a tool call returns over the local API. `images` are previews a client may show inline. */
export interface ToolResult {
  data: unknown;
  images?: Array<{ mimeType: string; data: string }>;
}

export type ApiResponse =
  | ({ ok: true } & ToolResult)
  | { ok: false; error: { code: string; message: string } };
