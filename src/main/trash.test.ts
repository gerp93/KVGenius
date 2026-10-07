import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { DEFAULT_CLEANUP_SETTINGS, CleanupSettings } from '../shared/cleanup';
import {
  countGenerations,
  getGenerationById,
  initDatabase,
  insertGeneration,
  listFavoriteIds,
  listGenerationRefs,
  listGenerations,
  listImageExtensions,
  listPinnedGenerations,
  setGenerationFavorite,
  setGenerationPinned,
} from './db';
import { getItem, listItems } from './library';
import { CleanupDeps, runAutoCleanupIfDue } from './cleanupScheduler';
import { keepSourceImage } from './sourceImages';
import {
  Recycle,
  cleanupCandidates,
  deleteFromTrash,
  emptyTrash,
  listTrashed,
  moveToTrash,
  previewCleanup,
  restoreFromTrash,
  runCleanup,
  trashStats,
} from './trash';

const VIDEO_FAMILIES = ['wan22-i2v'];
const NOW = new Date('2026-10-05T12:00:00.000Z');
const daysAgo = (n: number) => new Date(NOW.getTime() - n * 24 * 60 * 60 * 1000).toISOString();

interface Fixture {
  db: DatabaseSync;
  images: string;
  trash: string;
  sources: string;
  /** A stand-in for the operating system's Recycle Bin: files "sent" there end up in this folder. */
  bin: string;
  recycle: Recycle;
  /** Adds a generation with a real file of `size` bytes, `age` days old. */
  add: (name: string, age: number, size?: number) => { id: number; file: string };
}

function fixture(): Fixture {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-trash-'));
  const images = path.join(root, 'images');
  const bin = path.join(root, 'recycle-bin');
  fs.mkdirSync(images);
  fs.mkdirSync(bin);
  const db = initDatabase(':memory:');
  let seed = 0;
  return {
    db,
    images,
    trash: path.join(root, 'trash'),
    sources: path.join(root, 'sources'),
    bin,
    recycle: async (file) => {
      fs.renameSync(file, path.join(bin, `${Date.now()}-${path.basename(file)}`));
    },
    add(name, age, size = 10) {
      const file = path.join(images, name);
      fs.writeFileSync(file, 'x'.repeat(size));
      const record = insertGeneration(db, { prompt: name, width: 8, height: 8, seed: ++seed, steps: 1, cfg: 1 }, 'z-image', file);
      db.prepare('UPDATE generations SET created_at = ? WHERE id = ?').run(daysAgo(age), record.id);
      return { id: record.id, file };
    },
  };
}

test('a cleanup only finds old items nobody kept', () => {
  const f = fixture();
  const old = f.add('old.png', 40);
  const fav = f.add('fav.png', 40);
  const pinned = f.add('pinned.png', 40);
  const recent = f.add('recent.png', 5);
  const already = f.add('already.png', 40);
  setGenerationFavorite(f.db, fav.id, true);
  setGenerationPinned(f.db, pinned.id, true);
  moveToTrash(f.db, [already.id], f.trash, { now: NOW });

  assert.deepEqual(cleanupCandidates(f.db, 30, NOW).map((c) => c.id), [old.id]);
  assert.deepEqual(cleanupCandidates(f.db, 3, NOW).map((c) => c.id).sort(), [old.id, recent.id].sort());
});

test('the preview counts what would move and how much space it takes, and moves nothing', () => {
  const f = fixture();
  f.add('a.png', 40, 100);
  f.add('b.png', 40, 250);
  f.add('new.png', 1, 999);
  assert.deepEqual(previewCleanup(f.db, 30, NOW), { count: 2, bytes: 350 });
  assert.equal(trashStats(f.db).count, 0);
  assert.equal(fs.existsSync(f.trash), false);
});

test('moving to the Trash takes the file and the record out of the Library but keeps both', () => {
  const f = fixture();
  const item = f.add('shot.png', 40);
  const keep = f.add('keep.png', 40);

  const result = moveToTrash(f.db, [item.id], f.trash, { now: NOW });
  assert.deepEqual(result, { moved: 1, skipped: 0, failed: 0 });

  const record = getGenerationById(f.db, item.id)!;
  assert.equal(record.trashedAt, NOW.toISOString());
  assert.equal(path.dirname(record.imagePath), f.trash);
  assert.equal(fs.existsSync(item.file), false);
  assert.equal(fs.existsSync(record.imagePath), true);

  // Gone from every Library view; the other item is untouched.
  assert.deepEqual(listGenerations(f.db, VIDEO_FAMILIES, 'image', 50, null, false, true).map((r) => r.id), [keep.id]);
  assert.deepEqual(listGenerationRefs(f.db, VIDEO_FAMILIES, 'image', false, true).map((r) => r.id), [keep.id]);
  assert.equal(countGenerations(f.db, VIDEO_FAMILIES, false, true).image, 1);
  assert.deepEqual(listImageExtensions(f.db, VIDEO_FAMILIES), ['png']);
  assert.deepEqual(listTrashed(f.db, 50, null).map((r) => r.id), [item.id]);
  assert.equal(trashStats(f.db).count, 1);
});

test('the automatic cleanup never trashes a favorite or a pinned item', () => {
  const f = fixture();
  const fav = f.add('fav.png', 400);
  const pinned = f.add('pinned.png', 400);
  const plain = f.add('plain.png', 400);
  setGenerationFavorite(f.db, fav.id, true);
  setGenerationPinned(f.db, pinned.id, true);

  assert.deepEqual(runCleanup(f.db, 30, f.trash, NOW), { moved: 1, skipped: 0, failed: 0 });
  assert.equal(fs.existsSync(fav.file), true);
  assert.equal(fs.existsSync(pinned.file), true);
  assert.equal(getGenerationById(f.db, fav.id)!.trashedAt, null);
  assert.equal(getGenerationById(f.db, pinned.id)!.trashedAt, null);
  assert.notEqual(getGenerationById(f.db, plain.id)!.trashedAt, null);
  assert.deepEqual(listPinnedGenerations(f.db, true).map((r) => r.id), [pinned.id]);

  // Asking for them by id without saying the user chose them is also refused, and nothing is trashed twice.
  assert.deepEqual(moveToTrash(f.db, [fav.id, pinned.id, plain.id], f.trash, { now: NOW }), { moved: 0, skipped: 3, failed: 0 });
});

test('deleting a favorite or pinned item yourself moves it to the Trash, and restoring brings it back as it was', () => {
  const f = fixture();
  const favDir = path.join(f.images, 'favorites');
  fs.mkdirSync(favDir);
  const fav = f.add('fav.png', 1);
  const favFile = path.join(favDir, 'fav.png');
  fs.renameSync(fav.file, favFile);
  f.db.prepare('UPDATE generations SET image_path = ? WHERE id = ?').run(favFile, fav.id);
  setGenerationFavorite(f.db, fav.id, true);
  const pinned = f.add('pinned.png', 1);
  setGenerationPinned(f.db, pinned.id, true);

  assert.deepEqual(moveToTrash(f.db, [fav.id, pinned.id], f.trash, { now: NOW, includeKept: true }), { moved: 2, skipped: 0, failed: 0 });
  assert.equal(fs.existsSync(favFile), false);
  assert.deepEqual(listPinnedGenerations(f.db, true), [], 'a trashed pin is out of Library > Prompts');
  assert.deepEqual(listFavoriteIds(f.db), [], 'the startup favorites sync must leave a trashed favorite in the Trash');

  assert.deepEqual(restoreFromTrash(f.db, [fav.id, pinned.id]), { restored: 2, failed: 0 });
  const back = getGenerationById(f.db, fav.id)!;
  assert.equal(back.favorite, true);
  assert.equal(back.imagePath, favFile, 'back in the favorites folder');
  assert.equal(getGenerationById(f.db, pinned.id)!.pinned, true);
  assert.deepEqual(listFavoriteIds(f.db), [fav.id]);
});

test('an item whose file is already gone is still trashed, and two files with one name both survive', () => {
  const f = fixture();
  const ghost = f.add('ghost.png', 40);
  fs.rmSync(ghost.file);
  assert.deepEqual(moveToTrash(f.db, [ghost.id], f.trash, { now: NOW }), { moved: 1, skipped: 0, failed: 0 });
  assert.notEqual(getGenerationById(f.db, ghost.id)!.trashedAt, null);

  // Same file name from two different folders.
  const other = path.join(path.dirname(f.images), 'elsewhere');
  fs.mkdirSync(other);
  const a = f.add('same.png', 40);
  const b = f.add('same2.png', 40);
  const moved = path.join(other, 'same.png');
  fs.renameSync(path.join(f.images, 'same2.png'), moved);
  f.db.prepare('UPDATE generations SET image_path = ? WHERE id = ?').run(moved, b.id);
  moveToTrash(f.db, [a.id, b.id], f.trash, { now: NOW });
  assert.equal(fs.readdirSync(f.trash).filter((n) => n.startsWith('same')).length, 2);
});

test('the cleanup moves what the preview promised', () => {
  const f = fixture();
  f.add('a.png', 40);
  f.add('b.png', 40);
  f.add('new.png', 1);
  assert.deepEqual(runCleanup(f.db, 30, f.trash, NOW), { moved: 2, skipped: 0, failed: 0 });
  assert.equal(trashStats(f.db).count, 2);
  assert.equal(countGenerations(f.db, VIDEO_FAMILIES, false, true).image, 1);
});

test('restoring puts the file and the record back where they were', () => {
  const f = fixture();
  const item = f.add('shot.png', 40);
  moveToTrash(f.db, [item.id], f.trash, { now: NOW });

  assert.deepEqual(restoreFromTrash(f.db, [item.id]), { restored: 1, failed: 0 });
  const record = getGenerationById(f.db, item.id)!;
  assert.equal(record.trashedAt, null);
  assert.equal(record.imagePath, item.file);
  assert.equal(fs.existsSync(item.file), true);
  assert.deepEqual(listGenerations(f.db, VIDEO_FAMILIES, 'image', 50, null, false, true).map((r) => r.id), [item.id]);
});

test('an item whose file vanished from the Trash cannot be restored, and a taken name is not overwritten', () => {
  const f = fixture();
  const lost = f.add('lost.png', 40);
  moveToTrash(f.db, [lost.id], f.trash, { now: NOW });
  fs.rmSync(getGenerationById(f.db, lost.id)!.imagePath);
  assert.deepEqual(restoreFromTrash(f.db, [lost.id]), { restored: 0, failed: 1 });
  assert.notEqual(getGenerationById(f.db, lost.id)!.trashedAt, null);

  const clash = f.add('clash.png', 40);
  moveToTrash(f.db, [clash.id], f.trash, { now: NOW });
  fs.writeFileSync(clash.file, 'someone else got here');
  assert.deepEqual(restoreFromTrash(f.db, [clash.id]), { restored: 1, failed: 0 });
  assert.equal(fs.readFileSync(clash.file, 'utf8'), 'someone else got here');
  assert.notEqual(getGenerationById(f.db, clash.id)!.imagePath, clash.file);
});

test('removing from the Trash sends the file to the Recycle Bin, and only reaches items that are in the Trash', async () => {
  const f = fixture();
  const trashed = f.add('trashed.png', 40);
  const live = f.add('live.png', 40);
  moveToTrash(f.db, [trashed.id], f.trash, { now: NOW });
  const trashedPath = getGenerationById(f.db, trashed.id)!.imagePath;

  assert.deepEqual(await deleteFromTrash(f.db, [trashed.id, live.id], f.sources, f.recycle), { deleted: 1, failed: 0 });
  assert.equal(getGenerationById(f.db, trashed.id), null);
  assert.equal(fs.existsSync(trashedPath), false);
  assert.equal(fs.readdirSync(f.bin).length, 1, 'recoverable from the Recycle Bin, not hard-deleted');
  assert.notEqual(getGenerationById(f.db, live.id), null);
  assert.equal(fs.existsSync(live.file), true);
});

test('an item the Recycle Bin will not take stays in the Trash, whole', async () => {
  const f = fixture();
  const stuck = f.add('stuck.png', 40);
  const fine = f.add('fine.png', 40);
  moveToTrash(f.db, [stuck.id, fine.id], f.trash, { now: NOW });
  const stuckPath = getGenerationById(f.db, stuck.id)!.imagePath;
  const picky: Recycle = async (file) => {
    if (file === stuckPath) throw new Error('locked');
    await f.recycle(file);
  };

  assert.deepEqual(await emptyTrash(f.db, f.sources, picky), { deleted: 1, failed: 1 });
  assert.notEqual(getGenerationById(f.db, stuck.id), null, 'record kept');
  assert.equal(fs.existsSync(stuckPath), true, 'file kept');
  assert.notEqual(getGenerationById(f.db, stuck.id)!.trashedAt, null, 'still restorable');
  assert.equal(getGenerationById(f.db, fine.id), null);
});

test('an item whose file is already gone is removed without needing the Recycle Bin', async () => {
  const f = fixture();
  const ghost = f.add('ghost.png', 40);
  moveToTrash(f.db, [ghost.id], f.trash, { now: NOW });
  fs.rmSync(getGenerationById(f.db, ghost.id)!.imagePath);
  const never: Recycle = async () => {
    throw new Error('should not be asked');
  };
  assert.deepEqual(await emptyTrash(f.db, f.sources, never), { deleted: 1, failed: 0 });
  assert.equal(getGenerationById(f.db, ghost.id), null);
});

test('removing for good also drops a video\'s kept source image once nothing else uses it', async () => {
  const f = fixture();
  const original = path.join(f.images, 'source-original.png');
  fs.writeFileSync(original, 'source');
  const kept = keepSourceImage(original, f.sources)!;
  const file = path.join(f.images, 'clip.mp4');
  fs.writeFileSync(file, 'v');
  const video = insertGeneration(f.db, { prompt: 'v', width: 8, height: 8, seed: 1, steps: 1, cfg: 1, length: 81 }, 'wan22-i2v', file, null, false, kept);
  f.db.prepare('UPDATE generations SET created_at = ? WHERE id = ?').run(daysAgo(40), video.id);

  moveToTrash(f.db, [video.id], f.trash, { now: NOW });
  assert.equal(fs.existsSync(kept), true, 'still needed while the video can be restored');
  await deleteFromTrash(f.db, [video.id], f.sources, f.recycle);
  assert.equal(fs.existsSync(kept), false);
});

test('emptying the Trash can be limited to what has sat there long enough', async () => {
  const f = fixture();
  const longAgo = f.add('long-ago.png', 90);
  const justNow = f.add('just-now.png', 90);
  moveToTrash(f.db, [longAgo.id], f.trash, { now: new Date(NOW.getTime() - 40 * 24 * 60 * 60 * 1000) });
  moveToTrash(f.db, [justNow.id], f.trash, { now: NOW });

  assert.deepEqual(await emptyTrash(f.db, f.sources, f.recycle, 30, NOW), { deleted: 1, failed: 0 });
  assert.equal(getGenerationById(f.db, longAgo.id), null);
  assert.notEqual(getGenerationById(f.db, justNow.id), null);
  assert.deepEqual(await emptyTrash(f.db, f.sources, f.recycle), { deleted: 1, failed: 0 }, 'with no limit, everything goes');
  assert.equal(trashStats(f.db).count, 0);
});

test('outside clients cannot see anything in the Trash', () => {
  const f = fixture();
  const item = f.add('shot.png', 40);
  assert.ok(getItem(f.db, `gen-${item.id}`));
  moveToTrash(f.db, [item.id], f.trash, { now: NOW });
  assert.equal(getItem(f.db, `gen-${item.id}`), null);
  assert.equal(listItems(f.db).filter((i) => i.id === `gen-${item.id}`).length, 0);
});

// ---- The automatic schedules ------------------------------------------------------------

function schedule(f: Fixture, settings: Partial<CleanupSettings>) {
  let current: CleanupSettings = { ...DEFAULT_CLEANUP_SETTINGS, ...settings };
  const deps: CleanupDeps = {
    getDb: () => f.db,
    getSettings: () => current,
    saveSettings: (s) => {
      current = s;
    },
    trashDir: () => f.trash,
    sourcesDir: () => f.sources,
    recycle: f.recycle,
  };
  return { deps, settings: () => current };
}

test('the schedules do nothing at all unless turned on', async () => {
  const f = fixture();
  const old = f.add('old.png', 400);
  moveToTrash(f.db, [f.add('trashed.png', 400).id], f.trash, { now: new Date(NOW.getTime() - 400 * 24 * 60 * 60 * 1000) });
  const s = schedule(f, {});
  assert.deepEqual(await runAutoCleanupIfDue(s.deps, NOW), { trashed: null, emptied: null });
  assert.equal(getGenerationById(f.db, old.id)!.trashedAt, null);
  assert.equal(trashStats(f.db).count, 1);
  assert.equal(fs.readdirSync(f.bin).length, 0);
  assert.equal(s.settings().lastAutoTrashRun, null);
  assert.equal(s.settings().lastAutoEmptyRun, null);
});

test('only moving to the Trash is on: old items go to the Trash and nothing reaches the Recycle Bin', async () => {
  const f = fixture();
  const old = f.add('old.png', 100);
  const stale = f.add('stale.png', 100);
  moveToTrash(f.db, [stale.id], f.trash, { now: new Date(NOW.getTime() - 99 * 24 * 60 * 60 * 1000) });
  const s = schedule(f, { autoTrashEnabled: true, olderThanDays: 30 });

  const result = await runAutoCleanupIfDue(s.deps, NOW);
  assert.equal(result.trashed, 'Moved 1 item to the Trash.');
  assert.equal(result.emptied, null);
  assert.notEqual(getGenerationById(f.db, old.id)!.trashedAt, null);
  assert.notEqual(getGenerationById(f.db, stale.id), null, 'long in the Trash, but emptying is off');
  assert.equal(fs.readdirSync(f.bin).length, 0);
  assert.equal(s.settings().lastAutoTrashRun, NOW.toISOString());
  assert.equal(s.settings().lastAutoEmptyRun, null);
});

test('only emptying is on: long-trashed items go to the Recycle Bin and nothing new is trashed', async () => {
  const f = fixture();
  const old = f.add('old.png', 100);
  const stale = f.add('stale.png', 100);
  const recent = f.add('recent.png', 100);
  moveToTrash(f.db, [stale.id], f.trash, { now: new Date(NOW.getTime() - 45 * 24 * 60 * 60 * 1000) });
  moveToTrash(f.db, [recent.id], f.trash, { now: new Date(NOW.getTime() - 2 * 24 * 60 * 60 * 1000) });
  const s = schedule(f, { autoEmptyEnabled: true, trashRetentionDays: 30 });

  const result = await runAutoCleanupIfDue(s.deps, NOW);
  assert.equal(result.emptied, 'Sent 1 item to the Recycle Bin.');
  assert.equal(result.trashed, null);
  assert.equal(getGenerationById(f.db, stale.id), null);
  assert.equal(fs.readdirSync(f.bin).length, 1);
  assert.notEqual(getGenerationById(f.db, recent.id), null, 'not there long enough');
  assert.equal(getGenerationById(f.db, old.id)!.trashedAt, null, 'moving to the Trash is off');
  assert.equal(s.settings().lastAutoEmptyRun, NOW.toISOString());
});

test('each schedule runs at most once a day, on its own clock', async () => {
  const f = fixture();
  f.add('old.png', 100);
  const s = schedule(f, { autoTrashEnabled: true, olderThanDays: 30, autoEmptyEnabled: true, trashRetentionDays: 30 });

  const first = await runAutoCleanupIfDue(s.deps, NOW);
  assert.equal(first.trashed, 'Moved 1 item to the Trash.');
  assert.equal(first.emptied, 'Sent 0 items to the Recycle Bin.');

  f.add('another-old.png', 100);
  const later = new Date(NOW.getTime() + 60 * 60 * 1000);
  assert.deepEqual(await runAutoCleanupIfDue(s.deps, later), { trashed: null, emptied: null });
  const nextDay = new Date(NOW.getTime() + 25 * 60 * 60 * 1000);
  assert.equal((await runAutoCleanupIfDue(s.deps, nextDay)).trashed, 'Moved 1 item to the Trash.');
});
