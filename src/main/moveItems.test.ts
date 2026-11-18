import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { getGenerationById, initDatabase, insertGeneration, listGenerations, setGenerationFavorite } from './db';
import { MoveContext, ensureWorkspaceDirs, mergeWorkspaces, moveGenerations, removeWorkspaceFolder, sweepStaleWorkspaceFolders } from './moveItems';
import { restoreFromTrash, moveToTrash } from './trash';
import { listStyles, saveStyle } from './styles';
import { upsertImport } from './library';
import { insertJob } from './jobStore';
import { DEFAULT_WORKSPACE_ID, createWorkspace, getWorkspace, getWorkspacePromptSlots, saveWorkspacePromptSlots, workspaceDirsOf } from './workspaces';
import { newPromptSlot } from '../shared/promptSlots';
import { workspaceDirs } from './workspaceFolders';

const VIDEO_FAMILIES = ['wan22-i2v'];

interface Fixture {
  db: DatabaseSync;
  root: string;
  ctx: MoveContext;
  /** A stand-in for the operating system's Recycle Bin: files "sent" there end up in this folder. */
  bin: string;
  /** Makes a generation in a workspace with a real file in that workspace's own folder. */
  add: (name: string, over?: { workspaceId?: number; family?: string; bytes?: string }) => { id: number; file: string };
}

function fixture(): Fixture {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-move-'));
  const bin = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-bin-'));
  const db = initDatabase(':memory:');
  let binned = 0;
  const dirsOf = (id: number) => workspaceDirsOf(db, root, id);
  let seed = 0;
  return {
    db,
    root,
    bin,
    ctx: {
      db,
      outputRoot: root,
      dirsOf,
      videoFamilies: VIDEO_FAMILIES,
      recycle: async (file) => {
        fs.renameSync(file, path.join(bin, `${++binned}-${path.basename(file)}`));
      },
    },
    add(name, over = {}) {
      const workspaceId = over.workspaceId ?? DEFAULT_WORKSPACE_ID;
      const family = over.family ?? 'z-image';
      const dirs = dirsOf(workspaceId);
      const dir = VIDEO_FAMILIES.includes(family) ? dirs.videos : dirs.images;
      fs.mkdirSync(dir, { recursive: true });
      const file = path.join(dir, name);
      fs.writeFileSync(file, over.bytes ?? name);
      const record = insertGeneration(db, { prompt: name, width: 8, height: 8, seed: ++seed, steps: 1, cfg: 1, ...(VIDEO_FAMILIES.includes(family) ? { length: 81 } : {}) }, family, file, null, workspaceId);
      return { id: record.id, file };
    },
  };
}

test('moving an item puts its file in the other workspace\'s folder and updates the record', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const a = f.add('a.png');
  const video = f.add('v.mp4', { family: 'wan22-i2v' });

  assert.deepEqual(moveGenerations(f.ctx, [a.id, video.id], other), { moved: 2, skipped: 0, failed: 0 });
  const dirs = workspaceDirsOf(f.db, f.root, other);
  assert.equal(getGenerationById(f.db, a.id)!.workspaceId, other);
  assert.equal(getGenerationById(f.db, a.id)!.imagePath, path.join(dirs.images, 'a.png'));
  assert.equal(getGenerationById(f.db, video.id)!.imagePath, path.join(dirs.videos, 'v.mp4'));
  assert.equal(fs.existsSync(a.file), false, 'the file moved, it was not copied');
  assert.equal(fs.readFileSync(path.join(dirs.images, 'a.png'), 'utf8'), 'a.png');
  assert.deepEqual(listGenerations(f.db, VIDEO_FAMILIES, 'image', 50, null, false, DEFAULT_WORKSPACE_ID), []);
  assert.deepEqual(listGenerations(f.db, VIDEO_FAMILIES, 'image', 50, null, false, other).map((r) => r.id), [a.id]);
});

test('a favorite lands in the favorites folder, a pin and the flags come along, and a name clash gets a new name', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const a = f.add('a.png');
  setGenerationFavorite(f.db, a.id, true);
  f.db.prepare('UPDATE generations SET pinned_at = ? WHERE id = ?').run('2026-01-01', a.id);
  // Something already in the other workspace has the same file name.
  const dirs = workspaceDirsOf(f.db, f.root, other);
  fs.mkdirSync(path.join(dirs.images, 'favorites'), { recursive: true });
  fs.writeFileSync(path.join(dirs.images, 'favorites', 'a.png'), 'already here');

  moveGenerations(f.ctx, [a.id], other);
  const moved = getGenerationById(f.db, a.id)!;
  assert.equal(moved.imagePath, path.join(dirs.images, 'favorites', 'a-2.png'));
  assert.deepEqual([moved.favorite, moved.pinned], [true, true]);
  assert.equal(fs.readFileSync(path.join(dirs.images, 'favorites', 'a.png'), 'utf8'), 'already here', 'nothing is overwritten');
});

test('an item in the Trash moves to the other workspace\'s Trash and can still be restored there', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const a = f.add('a.png');
  moveToTrash(f.db, [a.id], (id) => workspaceDirsOf(f.db, f.root, id).trash);
  assert.ok(getGenerationById(f.db, a.id)!.trashedAt);

  moveGenerations(f.ctx, [a.id], other);
  const dirs = workspaceDirsOf(f.db, f.root, other);
  const trashed = getGenerationById(f.db, a.id)!;
  assert.equal(trashed.workspaceId, other);
  assert.equal(path.dirname(trashed.imagePath), dirs.trash);
  assert.deepEqual(restoreFromTrash(f.db, [a.id]), { restored: 1, failed: 0 });
  assert.equal(getGenerationById(f.db, a.id)!.imagePath, path.join(dirs.images, 'a.png'), 'back in the new workspace\'s images folder');
});

test('kept source images are copied across, and the old copy stays while another item still uses it', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const from = workspaceDirsOf(f.db, f.root, DEFAULT_WORKSPACE_ID);
  fs.mkdirSync(from.sources, { recursive: true });
  const shared = path.join(from.sources, 'abc123.png');
  fs.writeFileSync(shared, 'source bytes');
  const a = f.add('a.png');
  const b = f.add('b.png');
  f.db.prepare('UPDATE generations SET source_image_path = ? WHERE id IN (?, ?)').run(shared, a.id, b.id);

  moveGenerations(f.ctx, [a.id], other);
  const to = workspaceDirsOf(f.db, f.root, other);
  assert.equal(getGenerationById(f.db, a.id)!.sourceImagePath, path.join(to.sources, 'abc123.png'));
  assert.equal(fs.readFileSync(path.join(to.sources, 'abc123.png'), 'utf8'), 'source bytes');
  assert.equal(fs.existsSync(shared), true, 'b still uses it');

  moveGenerations(f.ctx, [b.id], other);
  assert.equal(fs.existsSync(shared), false, 'nothing uses the old copy any more');
  assert.equal(getGenerationById(f.db, b.id)!.sourceImagePath, path.join(to.sources, 'abc123.png'));
});

test('a video-to-video result keeps pointing at its source video when that video moves', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const source = f.add('source.mp4', { family: 'wan22-i2v' });
  const result = f.add('result.mp4', { family: 'wan22-i2v' });
  f.db.prepare('UPDATE generations SET source_video_path = ? WHERE id = ?').run(source.file, result.id);

  moveGenerations(f.ctx, [source.id], other);
  assert.equal(getGenerationById(f.db, result.id)!.sourceVideoPath, getGenerationById(f.db, source.id)!.imagePath);
});

test('items already in place, items that are gone, files outside the output folder and a missing file are handled quietly', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const inPlace = f.add('a.png');
  assert.deepEqual(moveGenerations(f.ctx, [inPlace.id, 99999], DEFAULT_WORKSPACE_ID), { moved: 0, skipped: 2, failed: 0 });

  const outsideDir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-outside-'));
  const outside = path.join(outsideDir, 'o.png');
  fs.writeFileSync(outside, 'x');
  const o = insertGeneration(f.db, { prompt: 'o', width: 8, height: 8, seed: 1, steps: 1, cfg: 1 }, 'z-image', outside);
  const gone = insertGeneration(f.db, { prompt: 'g', width: 8, height: 8, seed: 2, steps: 1, cfg: 1 }, 'z-image', path.join(f.root, 'nowhere.png'));
  assert.deepEqual(moveGenerations(f.ctx, [o.id, gone.id], other), { moved: 2, skipped: 0, failed: 0 });
  assert.equal(getGenerationById(f.db, o.id)!.workspaceId, other);
  assert.equal(getGenerationById(f.db, o.id)!.imagePath, outside, 'a file outside the app\'s folders stays where it is');
  assert.equal(fs.existsSync(outside), true);
  assert.equal(getGenerationById(f.db, gone.id)!.workspaceId, other);
});

test('when the record cannot be updated the file goes back, and the item is counted as failed', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const a = f.add('a.png');
  const b = f.add('b.png');
  f.db.exec(`CREATE TRIGGER refuse_a BEFORE UPDATE ON generations WHEN OLD.id = ${a.id} BEGIN SELECT RAISE(ABORT, 'refused'); END`);

  assert.deepEqual(moveGenerations(f.ctx, [a.id, b.id], other), { moved: 1, skipped: 0, failed: 1 });
  assert.equal(fs.existsSync(a.file), true, 'back where it was');
  assert.equal(getGenerationById(f.db, a.id)!.workspaceId, DEFAULT_WORKSPACE_ID);
  assert.equal(getGenerationById(f.db, a.id)!.imagePath, a.file);
  assert.equal(getGenerationById(f.db, b.id)!.workspaceId, other, 'one failure does not stop the rest');
});

// ---- Merging ------------------------------------------------------------------------------------------

test('merging moves everything across - the Trash, imports, queue history and styles - and deletes the emptied workspace and its folders', async () => {
  const f = fixture();
  const from = createWorkspace(f.db, 'From').id;
  const a = f.add('a.png', { workspaceId: from });
  const trashed = f.add('t.png', { workspaceId: from });
  moveToTrash(f.db, [trashed.id], (id) => workspaceDirsOf(f.db, f.root, id).trash);
  const stay = f.add('stay.png');
  upsertImport(f.db, { path: '/x/clip.mp4', kind: 'video', origin: 'imported', workspaceId: from });
  insertJob(f.db, { family: 'z-image', params: { prompt: 'x', width: 8, height: 8, seed: 1, steps: 1, cfg: 1 }, source: 'ui', workspaceId: from });
  saveStyle(f.db, { name: 'Local piece', text: 'x', kind: 'element' }, null, from);
  saveStyle(f.db, { name: 'Same name', text: 'x', kind: 'element' }, null, from);
  saveStyle(f.db, { name: 'same name', text: 'y', kind: 'element' }, null, DEFAULT_WORKSPACE_ID);
  const fromRoot = workspaceDirs(f.root, 'From').root;
  assert.equal(fs.existsSync(fromRoot), true);

  const result = await mergeWorkspaces(f.ctx, from, DEFAULT_WORKSPACE_ID);
  assert.deepEqual(result, { moved: 2, skipped: 0, failed: 0, deleted: true, renamedStyles: 1 });
  assert.equal(getWorkspace(f.db, from), null);
  assert.equal(fs.existsSync(fromRoot), false, 'the emptied folders are removed');
  const dirs = workspaceDirsOf(f.db, f.root, DEFAULT_WORKSPACE_ID);
  assert.equal(path.dirname(getGenerationById(f.db, a.id)!.imagePath), dirs.images);
  assert.equal(path.dirname(getGenerationById(f.db, trashed.id)!.imagePath), dirs.trash);
  assert.ok(getGenerationById(f.db, trashed.id)!.trashedAt, 'still in the Trash');
  assert.equal(getGenerationById(f.db, stay.id)!.workspaceId, DEFAULT_WORKSPACE_ID);
  assert.equal((f.db.prepare('SELECT COUNT(*) AS n FROM imports WHERE workspace_id = ?').get(DEFAULT_WORKSPACE_ID) as { n: number }).n, 1);
  assert.equal((f.db.prepare('SELECT COUNT(*) AS n FROM jobs WHERE workspace_id = ?').get(DEFAULT_WORKSPACE_ID) as { n: number }).n, 1);
  const names = listStyles(f.db, DEFAULT_WORKSPACE_ID).map((s) => s.name).sort();
  assert.deepEqual(names, ['Local piece', 'Same name (from From)', 'same name']);
});

test('a merge that cannot move everything keeps the workspace, with what is left in it', async () => {
  const f = fixture();
  const from = createWorkspace(f.db, 'From').id;
  const a = f.add('a.png', { workspaceId: from });
  const b = f.add('b.png', { workspaceId: from });
  f.db.exec(`CREATE TRIGGER refuse_a BEFORE UPDATE ON generations WHEN OLD.id = ${a.id} BEGIN SELECT RAISE(ABORT, 'refused'); END`);

  const result = await mergeWorkspaces(f.ctx, from, DEFAULT_WORKSPACE_ID);
  assert.deepEqual([result.moved, result.failed, result.deleted], [1, 1, false]);
  assert.notEqual(getWorkspace(f.db, from), null);
  assert.equal(getGenerationById(f.db, a.id)!.workspaceId, from);
  assert.equal(getGenerationById(f.db, b.id)!.workspaceId, DEFAULT_WORKSPACE_ID);
  assert.equal(fs.existsSync(a.file), true);
});

test('a workspace cannot be merged into itself or into one that is gone', async () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  await assert.rejects(mergeWorkspaces(f.ctx, other, other), /different workspace/);
  await assert.rejects(mergeWorkspaces(f.ctx, other, 999), /no longer exists/);
});

// ---- Folders: made with the workspace, removed with it -----------------------------------------------------

test('a workspace\'s folders can be made ahead of its first picture', () => {
  const f = fixture();
  const made = createWorkspace(f.db, 'Fresh');
  const dirs = workspaceDirsOf(f.db, f.root, made.id);
  assert.equal(fs.existsSync(dirs.root), false);
  ensureWorkspaceDirs(dirs);
  for (const dir of [dirs.images, dirs.videos, dirs.gifs, dirs.sources, dirs.trash]) assert.equal(fs.statSync(dir).isDirectory(), true);
  ensureWorkspaceDirs(dirs); // again: nothing changes, nothing fails
});

test('merging removes the emptied workspace\'s whole folder, including files nothing needs (they go to the Recycle Bin)', async () => {
  const f = fixture();
  const from = createWorkspace(f.db, 'From').id;
  const dirs = workspaceDirsOf(f.db, f.root, from);
  ensureWorkspaceDirs(dirs);
  f.add('a.png', { workspaceId: from });
  fs.writeFileSync(path.join(dirs.sources, 'orphan.png'), 'nothing uses me');
  fs.mkdirSync(path.join(dirs.images, 'favorites'), { recursive: true });
  fs.writeFileSync(path.join(dirs.images, 'favorites', 'stray.png'), 'stray');

  const result = await mergeWorkspaces(f.ctx, from, DEFAULT_WORKSPACE_ID);
  assert.equal(result.deleted, true);
  assert.equal(fs.existsSync(dirs.root), false, 'no folder left behind');
  assert.deepEqual(fs.readdirSync(f.bin).map((n) => n.replace(/^\d+-/, '')).sort(), ['orphan.png', 'stray.png']);
  assert.equal(fs.existsSync(getGenerationById(f.db, 1)!.imagePath), true, 'what a record needs is not touched');
});

test('a kept source image that another workspace\'s item still uses is copied to that workspace before the folder goes', async () => {
  const f = fixture();
  const from = createWorkspace(f.db, 'From').id;
  const into = createWorkspace(f.db, 'Into').id;
  const fromDirs = workspaceDirsOf(f.db, f.root, from);
  ensureWorkspaceDirs(fromDirs);
  const shared = path.join(fromDirs.sources, 'abc123.png');
  fs.writeFileSync(shared, 'source bytes');
  const keeper = f.add('keeper.png', { workspaceId: into });
  f.db.prepare('UPDATE generations SET source_image_path = ? WHERE id = ?').run(shared, keeper.id);

  const removal = await removeWorkspaceFolder(f.ctx, fromDirs.root);
  assert.equal(removal.removed, true);
  assert.equal(fs.existsSync(fromDirs.root), false);
  const moved = getGenerationById(f.db, keeper.id)!.sourceImagePath!;
  assert.equal(moved, path.join(workspaceDirsOf(f.db, f.root, into).sources, 'abc123.png'));
  assert.equal(fs.readFileSync(moved, 'utf8'), 'source bytes');
  assert.equal(fs.readdirSync(f.bin).length, 1, 'the old copy, now used by nothing, went to the Recycle Bin - after the new one was made');
});

test('a folder is kept, and says why, when a file in it cannot be dealt with', async () => {
  const f = fixture();
  const from = createWorkspace(f.db, 'From').id;
  const dirs = workspaceDirsOf(f.db, f.root, from);
  ensureWorkspaceDirs(dirs);
  fs.writeFileSync(path.join(dirs.images, 'stuck.png'), 'x');
  const failing = { ...f.ctx, recycle: async () => Promise.reject(new Error('in use')) };
  const removal = await removeWorkspaceFolder(failing, dirs.root);
  assert.deepEqual(removal, { removed: false, recycled: 0, kept: 1 });
  assert.equal(fs.existsSync(path.join(dirs.images, 'stuck.png')), true);
});

test('folders left by a workspace that no longer exists are swept at start-up, and nothing else is touched', async () => {
  const f = fixture();
  const live = createWorkspace(f.db, 'Live');
  ensureWorkspaceDirs(workspaceDirsOf(f.db, f.root, live.id));
  // Left behind by merged or deleted workspaces: one empty shape, one with an orphan file.
  const emptyShape = path.join(f.root, 'Gone one');
  fs.mkdirSync(path.join(emptyShape, 'images'), { recursive: true });
  fs.mkdirSync(path.join(emptyShape, 'trash'), { recursive: true });
  const withFile = path.join(f.root, 'Gone two');
  fs.mkdirSync(path.join(withFile, 'sources'), { recursive: true });
  fs.writeFileSync(path.join(withFile, 'sources', 'orphan.png'), 'x');
  // Not ours: a folder with a stray file of the user's own, and the old flat folders.
  const notOurs = path.join(f.root, 'My notes');
  fs.mkdirSync(path.join(notOurs, 'images'), { recursive: true });
  fs.writeFileSync(path.join(notOurs, 'readme.txt'), 'mine');
  const legacy = path.join(f.root, 'images');
  fs.mkdirSync(legacy);
  fs.writeFileSync(path.join(legacy, 'old.png'), 'old');

  assert.equal(await sweepStaleWorkspaceFolders(f.ctx), 2);
  assert.equal(fs.existsSync(emptyShape), false);
  assert.equal(fs.existsSync(withFile), false);
  assert.equal(fs.readdirSync(f.bin).length, 1);
  assert.equal(fs.existsSync(workspaceDirsOf(f.db, f.root, live.id).root), true, 'a workspace\'s own folder stays, empty or not');
  assert.equal(fs.existsSync(path.join(notOurs, 'readme.txt')), true);
  assert.equal(fs.existsSync(path.join(legacy, 'old.png')), true);
  assert.equal(await sweepStaleWorkspaceFolders(f.ctx), 0, 'nothing more to sweep');
});

// ---- Generate tabs belong to a workspace ----------------------------------------------------------------------

test('each workspace has its own Generate tabs, and a new one starts with a single empty tab', () => {
  const f = fixture();
  const other = createWorkspace(f.db, 'Other').id;
  const mine = newPromptSlot();
  mine.data.prompt = 'a red fox';
  saveWorkspacePromptSlots(f.db, DEFAULT_WORKSPACE_ID, [mine], mine.id);

  assert.deepEqual(getWorkspacePromptSlots(f.db, DEFAULT_WORKSPACE_ID).slots.map((s) => s.data.prompt), ['a red fox']);
  assert.equal(getWorkspacePromptSlots(f.db, DEFAULT_WORKSPACE_ID).activeId, mine.id);
  const fresh = getWorkspacePromptSlots(f.db, other);
  assert.equal(fresh.slots.length, 1);
  assert.equal(fresh.slots[0].data.prompt, '');
  assert.equal(fresh.activeId, null);
});

test('merging adds the other workspace\'s typed tabs after the target\'s and leaves out empty ones', async () => {
  const f = fixture();
  const from = createWorkspace(f.db, 'From').id;
  const keep = newPromptSlot();
  keep.data.prompt = 'target tab';
  saveWorkspacePromptSlots(f.db, DEFAULT_WORKSPACE_ID, [keep], keep.id);
  const typed = newPromptSlot();
  typed.data.prompt = 'typed in From';
  const named = newPromptSlot();
  named.name = 'Named';
  saveWorkspacePromptSlots(f.db, from, [typed, named, newPromptSlot()], typed.id);

  await mergeWorkspaces(f.ctx, from, DEFAULT_WORKSPACE_ID);
  const after = getWorkspacePromptSlots(f.db, DEFAULT_WORKSPACE_ID);
  assert.deepEqual(after.slots.map((s) => s.name ?? s.data.prompt), ['target tab', 'typed in From', 'Named']);
  assert.equal(new Set(after.slots.map((s) => s.id)).size, 3, 'every tab has its own id');
  assert.equal(after.activeId, keep.id);
});
