import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import {
  deleteGeneration,
  getGenerationById,
  initDatabase,
  insertGeneration,
  listGenerationRefs,
  listPinnedGenerations,
  migrateSavedPrompts,
  setGenerationHidden,
  setGenerationPinned,
} from './db';

const params = (prompt: string, seed = 1) => ({ prompt, width: 64, height: 64, seed, steps: 4, cfg: 1 });

function legacyTable(db: DatabaseSync, rows: { name: string | null; prompt: string; tags?: string; at: string }[]): void {
  db.exec(`CREATE TABLE saved_prompts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, prompt TEXT NOT NULL, negative_prompt TEXT,
    tags TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
  )`);
  const insert = db.prepare('INSERT INTO saved_prompts (name, prompt, tags, created_at) VALUES (?, ?, ?, ?)');
  for (const r of rows) insert.run(r.name, r.prompt, r.tags ?? '[]', r.at);
}

const hasLegacyTable = (db: DatabaseSync) =>
  db.prepare("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'saved_prompts'").get() !== undefined;

test('pinning flags a generation, shows it in the pinned list, and unpinning removes it', () => {
  const db = initDatabase(':memory:');
  const a = insertGeneration(db, params('a'), 'z-image-turbo', '/out/a.png');
  const b = insertGeneration(db, params('b'), 'z-image-turbo', '/out/b.png');
  assert.equal(a.pinned, false);

  setGenerationPinned(db, a.id, true);
  assert.equal(getGenerationById(db, a.id)?.pinned, true);
  assert.equal(getGenerationById(db, b.id)?.pinned, false);
  assert.deepEqual(listPinnedGenerations(db, false).map((r) => r.id), [a.id]);

  setGenerationPinned(db, a.id, false);
  assert.equal(getGenerationById(db, a.id)?.pinned, false);
  assert.deepEqual(listPinnedGenerations(db, false), []);
});

test('the most recently pinned item comes first, and re-pinning keeps its place', async () => {
  const db = initDatabase(':memory:');
  const a = insertGeneration(db, params('a'), 'z-image-turbo', '/out/a.png');
  const b = insertGeneration(db, params('b'), 'z-image-turbo', '/out/b.png');
  setGenerationPinned(db, a.id, true);
  await new Promise((resolve) => setTimeout(resolve, 5));
  setGenerationPinned(db, b.id, true);
  assert.deepEqual(listPinnedGenerations(db, false).map((r) => r.id), [b.id, a.id]);

  await new Promise((resolve) => setTimeout(resolve, 5));
  setGenerationPinned(db, a.id, true); // already pinned: not moved to the front
  assert.deepEqual(listPinnedGenerations(db, false).map((r) => r.id), [b.id, a.id]);
});

test('hidden pinned items are left out unless asked for', () => {
  const db = initDatabase(':memory:');
  const a = insertGeneration(db, params('a'), 'z-image-turbo', '/out/a.png');
  const b = insertGeneration(db, params('b'), 'z-image-turbo', '/out/b.png');
  setGenerationPinned(db, a.id, true);
  setGenerationPinned(db, b.id, true);
  setGenerationHidden(db, a.id, true);
  assert.deepEqual(listPinnedGenerations(db, false).map((r) => r.id), [b.id]);
  assert.deepEqual(listPinnedGenerations(db, true).map((r) => r.id).sort(), [a.id, b.id].sort());
});

test('deleting a pinned generation takes its pin with it, and selection refs report the pin', () => {
  const db = initDatabase(':memory:');
  const a = insertGeneration(db, params('a'), 'z-image-turbo', '/out/a.png');
  const b = insertGeneration(db, params('b'), 'z-image-turbo', '/out/b.png');
  setGenerationPinned(db, a.id, true);
  const refs = listGenerationRefs(db, ['wan22-i2v'], 'image', false, false);
  assert.deepEqual(refs.map((r) => [r.id, r.pinned]).sort(), [[a.id, true], [b.id, false]].sort());

  deleteGeneration(db, a.id);
  assert.deepEqual(listPinnedGenerations(db, true), []);
});

test('a database from before pinning gets the column added and keeps its rows', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-pin-')), 'old.db');
  const old = new DatabaseSync(file);
  old.exec(`CREATE TABLE generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL, negative_prompt TEXT, width INTEGER NOT NULL,
    height INTEGER NOT NULL, seed INTEGER NOT NULL, steps INTEGER NOT NULL, cfg REAL NOT NULL, length INTEGER,
    model_family TEXT NOT NULL, image_path TEXT NOT NULL, favorite INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0, timing_id INTEGER, created_at TEXT NOT NULL)`);
  old.prepare(
    "INSERT INTO generations (prompt, width, height, seed, steps, cfg, model_family, image_path, created_at) VALUES ('kept', 64, 64, 1, 4, 1, 'z-image-turbo', '/out/k.png', 'x')"
  ).run();
  old.close();

  const db = initDatabase(file);
  const record = getGenerationById(db, 1);
  assert.equal(record?.prompt, 'kept');
  assert.equal(record?.pinned, false);
  assert.equal(record?.sourceImagePath, null);
  db.close();
});

test('saved prompts pin the newest matching generation and the old table is dropped', () => {
  const db = initDatabase(':memory:');
  const old = insertGeneration(db, params('castle at dusk', 1), 'z-image-turbo', '/out/old.png');
  const newest = insertGeneration(db, params('castle at dusk', 2), 'z-image-turbo', '/out/new.png');
  const other = insertGeneration(db, params('a fox', 3), 'z-image-turbo', '/out/fox.png');
  legacyTable(db, [
    { name: 'Castle', prompt: 'castle at dusk', at: '2026-01-01T00:00:00.000Z' },
    { name: null, prompt: 'a fox', at: '2026-02-01T00:00:00.000Z' },
  ]);

  assert.deepEqual(migrateSavedPrompts(db, null), { pinned: 2, exported: 0 });
  assert.equal(hasLegacyTable(db), false);
  assert.equal(getGenerationById(db, old.id)?.pinned, false);
  assert.equal(getGenerationById(db, newest.id)?.pinned, true);
  // Newest saved prompt first, as the saved prompts were listed.
  assert.deepEqual(listPinnedGenerations(db, false).map((r) => r.id), [other.id, newest.id]);
});

test('a hidden match is only used when nothing visible matches', () => {
  const db = initDatabase(':memory:');
  const shown = insertGeneration(db, params('same', 1), 'z-image-turbo', '/out/shown.png');
  const hidden = insertGeneration(db, params('same', 2), 'z-image-turbo', '/out/hidden.png');
  setGenerationHidden(db, hidden.id, true);
  legacyTable(db, [{ name: 'x', prompt: 'same', at: '2026-01-01T00:00:00.000Z' }]);
  migrateSavedPrompts(db, null);
  assert.equal(getGenerationById(db, shown.id)?.pinned, true);
  assert.equal(getGenerationById(db, hidden.id)?.pinned, false);
});

test('saved prompts without a picture are exported before the table is dropped', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-pin-'));
  const exportPath = path.join(dir, 'unpinned.txt');
  const db = initDatabase(':memory:');
  insertGeneration(db, params('has a picture'), 'z-image-turbo', '/out/a.png');
  legacyTable(db, [
    { name: 'With picture', prompt: 'has a picture', at: '2026-01-01T00:00:00.000Z' },
    { name: 'Lonely', prompt: 'never generated', tags: '["moody","rain"]', at: '2026-01-02T00:00:00.000Z' },
    { name: null, prompt: 'also never generated', at: '2026-01-03T00:00:00.000Z' },
  ]);

  assert.deepEqual(migrateSavedPrompts(db, exportPath), { pinned: 1, exported: 2 });
  assert.equal(hasLegacyTable(db), false);
  const text = fs.readFileSync(exportPath, 'utf8');
  assert.match(text, /# Lonely {2}\[moody, rain\]\nnever generated/);
  assert.match(text, /# \(untitled\)\nalso never generated/);
  assert.doesNotMatch(text, /has a picture/);
});

test('nothing is dropped when unmatched prompts cannot be exported', () => {
  const db = initDatabase(':memory:');
  legacyTable(db, [{ name: 'Lonely', prompt: 'never generated', at: '2026-01-01T00:00:00.000Z' }]);

  // No export location at all.
  assert.equal(migrateSavedPrompts(db, null), null);
  assert.equal(hasLegacyTable(db), true);

  // A location that cannot be written (its folder does not exist).
  const missing = path.join(os.tmpdir(), 'kvg-no-such-dir', 'x', 'unpinned.txt');
  assert.equal(migrateSavedPrompts(db, missing), null);
  assert.equal(hasLegacyTable(db), true);

  // Once it can be written, it goes through.
  const ok = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-pin-')), 'unpinned.txt');
  assert.deepEqual(migrateSavedPrompts(db, ok), { pinned: 0, exported: 1 });
  assert.equal(hasLegacyTable(db), false);
});

test('with no old table the migration does nothing', () => {
  const db = initDatabase(':memory:');
  assert.equal(migrateSavedPrompts(db, null), null);
});
