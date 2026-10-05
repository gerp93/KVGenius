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
  listGenerationRefs,
  listGenerations,
  listImageExtensions,
  listPinnedGenerations,
  setGenerationPinned,
} from './db';
import { setGenerationFavorite } from './db';
import { getItem, listItems } from './library';
import { CleanupDeps, runAutoCleanupIfDue } from './cleanupScheduler';
import { keepSourceImage } from './sourceImages';
import { cleanupCandidates, deleteFromTrash, emptyTrash, listTrashed, moveToTrash, previewCleanup, restoreFromTrash, runCleanup, trashStats } from './trash';

const VIDEO_FAMILIES = ['wan22-i2v'];
const NOW = new Date('2026-10-05T12:00:00.000Z');
const daysAgo = (n: number) => new Date(NOW.getTime() - n * 24 * 60 * 60 * 1000).toISOString();

interface Fixture {
  db: DatabaseSync;
  images: string;
  trash: string;
  sources: string;
  /** Adds a generation with a real file of `size` bytes, `age` days old. */
  add: (name: string, age: number, size?: number) => { id: number; file: string };
}

function fixture(): Fixture {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-trash-'));
  const images = path.join(root, 'images');
  fs.mkdirSync(images);
  const db = initDatabase(':memory:');
  let seed = 0;
  return {
    db,
    images,
    trash: path.join(root, 'trash'),
    sources: path.join(root, 'sources'),
    add(name, age, size = 10) {
      const file = path.join(images, name);
      fs.writeFileSync(file, 'x'.repeat(size));
      const record = insertGeneration(db, { prompt: name, width: 8, height: 8, seed: ++seed, steps: 1, cfg: 1 }, 'z-image-turbo', file);
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
  moveToTrash(f.db, [already.id], f.trash, NOW);

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

  const result = moveToTrash(f.db, [item.id], f.trash, NOW);
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

test('favorites and pinned items are never moved to the Trash, whoever asks', () => {
  const f = fixture();
  const fav = f.add('fav.png', 400);
  const pinned = f.add('pinned.png', 400);
  const plain = f.add('plain.png', 400);
  setGenerationFavorite(f.db, fav.id, true);
  setGenerationPinned(f.db, pinned.id, true);

  assert.deepEqual(moveToTrash(f.db, [fav.id, pinned.id, plain.id], f.trash, NOW), { moved: 1, skipped: 2, failed: 0 });
  assert.equal(fs.existsSync(fav.file), true);
  assert.equal(fs.existsSync(pinned.file), true);
  assert.equal(getGenerationById(f.db, fav.id)!.trashedAt, null);
  assert.equal(getGenerationById(f.db, pinned.id)!.trashedAt, null);
  assert.deepEqual(listPinnedGenerations(f.db, true).map((r) => r.id), [pinned.id]);

  // An item that is already in the Trash is skipped, not moved twice.
  assert.deepEqual(moveToTrash(f.db, [plain.id], f.trash, NOW), { moved: 0, skipped: 1, failed: 0 });
});

test('an item whose file is already gone is still trashed, and two files with one name both survive', () => {
  const f = fixture();
  const ghost = f.add('ghost.png', 40);
  fs.rmSync(ghost.file);
  assert.deepEqual(moveToTrash(f.db, [ghost.id], f.trash, NOW), { moved: 1, skipped: 0, failed: 0 });
  assert.notEqual(getGenerationById(f.db, ghost.id)!.trashedAt, null);

  // Same file name from two different folders.
  const other = path.join(path.dirname(f.images), 'elsewhere');
  fs.mkdirSync(other);
  const a = f.add('same.png', 40);
  const b = f.add('same2.png', 40);
  const moved = path.join(other, 'same.png');
  fs.renameSync(path.join(f.images, 'same2.png'), moved);
  f.db.prepare('UPDATE generations SET image_path = ? WHERE id = ?').run(moved, b.id);
  moveToTrash(f.db, [a.id, b.id], f.trash, NOW);
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
  moveToTrash(f.db, [item.id], f.trash, NOW);

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
  moveToTrash(f.db, [lost.id], f.trash, NOW);
  fs.rmSync(getGenerationById(f.db, lost.id)!.imagePath);
  assert.deepEqual(restoreFromTrash(f.db, [lost.id]), { restored: 0, failed: 1 });
  assert.notEqual(getGenerationById(f.db, lost.id)!.trashedAt, null);

  const clash = f.add('clash.png', 40);
  moveToTrash(f.db, [clash.id], f.trash, NOW);
  fs.writeFileSync(clash.file, 'someone else got here');
  assert.deepEqual(restoreFromTrash(f.db, [clash.id]), { restored: 1, failed: 0 });
  assert.equal(fs.readFileSync(clash.file, 'utf8'), 'someone else got here');
  assert.notEqual(getGenerationById(f.db, clash.id)!.imagePath, clash.file);
});

test('deleting from the Trash is for good, and only reaches items that are in the Trash', () => {
  const f = fixture();
  const trashed = f.add('trashed.png', 40);
  const live = f.add('live.png', 40);
  moveToTrash(f.db, [trashed.id], f.trash, NOW);
  const trashedPath = getGenerationById(f.db, trashed.id)!.imagePath;

  assert.equal(deleteFromTrash(f.db, [trashed.id, live.id], f.sources), 1);
  assert.equal(getGenerationById(f.db, trashed.id), null);
  assert.equal(fs.existsSync(trashedPath), false);
  assert.notEqual(getGenerationById(f.db, live.id), null);
  assert.equal(fs.existsSync(live.file), true);
});

test('deleting for good also removes a video\'s kept source image once nothing else uses it', () => {
  const f = fixture();
  const original = path.join(f.images, 'source-original.png');
  fs.writeFileSync(original, 'source');
  const kept = keepSourceImage(original, f.sources)!;
  const file = path.join(f.images, 'clip.mp4');
  fs.writeFileSync(file, 'v');
  const video = insertGeneration(f.db, { prompt: 'v', width: 8, height: 8, seed: 1, steps: 1, cfg: 1, length: 81 }, 'wan22-i2v', file, null, false, kept);
  f.db.prepare('UPDATE generations SET created_at = ? WHERE id = ?').run(daysAgo(40), video.id);

  moveToTrash(f.db, [video.id], f.trash, NOW);
  assert.equal(fs.existsSync(kept), true, 'still needed while the video can be restored');
  deleteFromTrash(f.db, [video.id], f.sources);
  assert.equal(fs.existsSync(kept), false);
});

test('emptying the Trash can be limited to what has sat there long enough', () => {
  const f = fixture();
  const longAgo = f.add('long-ago.png', 90);
  const justNow = f.add('just-now.png', 90);
  moveToTrash(f.db, [longAgo.id], f.trash, new Date(NOW.getTime() - 40 * 24 * 60 * 60 * 1000));
  moveToTrash(f.db, [justNow.id], f.trash, NOW);

  assert.equal(emptyTrash(f.db, f.sources, 30, NOW), 1);
  assert.equal(getGenerationById(f.db, longAgo.id), null);
  assert.notEqual(getGenerationById(f.db, justNow.id), null);
  assert.equal(emptyTrash(f.db, f.sources), 1, 'with no limit, everything goes');
  assert.equal(trashStats(f.db).count, 0);
});

test('outside clients cannot see anything in the Trash', () => {
  const f = fixture();
  const item = f.add('shot.png', 40);
  assert.ok(getItem(f.db, `gen-${item.id}`));
  moveToTrash(f.db, [item.id], f.trash, NOW);
  assert.equal(getItem(f.db, `gen-${item.id}`), null);
  assert.equal(listItems(f.db).filter((i) => i.id === `gen-${item.id}`).length, 0);
});

// ---- The automatic schedule -------------------------------------------------------------

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
  };
  return { deps, settings: () => current };
}

test('the schedule does nothing at all unless it was turned on', () => {
  const f = fixture();
  f.add('old.png', 400);
  const s = schedule(f, {});
  assert.equal(runAutoCleanupIfDue(s.deps, NOW), null);
  assert.equal(trashStats(f.db).count, 0);
  assert.equal(countGenerations(f.db, VIDEO_FAMILIES, false, true).image, 1);
  assert.equal(s.settings().lastAutoRun, null);
});

test('when on and due it moves old items to the Trash, deletes old Trash items, and records it', () => {
  const f = fixture();
  const old = f.add('old.png', 100);
  const recent = f.add('recent.png', 2);
  const stale = f.add('stale.png', 100);
  moveToTrash(f.db, [stale.id], f.trash, new Date(NOW.getTime() - 45 * 24 * 60 * 60 * 1000));
  const s = schedule(f, { autoEnabled: true, olderThanDays: 30, trashRetentionDays: 30 });

  const summary = runAutoCleanupIfDue(s.deps, NOW);
  assert.match(summary ?? '', /Moved 1 item to the Trash and deleted 1 item from it for good/);
  assert.notEqual(getGenerationById(f.db, old.id)!.trashedAt, null);
  assert.equal(getGenerationById(f.db, recent.id)!.trashedAt, null);
  assert.equal(getGenerationById(f.db, stale.id), null);
  assert.equal(s.settings().lastAutoRun, NOW.toISOString());
  assert.equal(s.settings().lastAutoSummary, summary);

  // And not again within the day.
  f.add('another-old.png', 100);
  assert.equal(runAutoCleanupIfDue(s.deps, new Date(NOW.getTime() + 60 * 60 * 1000)), null);
  assert.equal(runAutoCleanupIfDue(s.deps, new Date(NOW.getTime() + 25 * 60 * 60 * 1000))?.startsWith('Moved 1 item'), true);
});
