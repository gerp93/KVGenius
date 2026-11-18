import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { getGenerationById, initDatabase, insertGeneration } from './db';
import { followLegacyPath, migrateOutputLayout } from './outputLayout';
import { DEFAULT_WORKSPACE_ID, createWorkspace, workspaceDirsOf } from './workspaces';

function setup() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-layout-'));
  const db = initDatabase(':memory:');
  const ctx = { db, outputRoot: root, dirsOf: (id: number) => workspaceDirsOf(db, root, id), videoFamilies: ['wan22-i2v'] };
  /** A file in the old flat layout, with its record. */
  const flat = (sub: string, name: string, workspaceId = DEFAULT_WORKSPACE_ID) => {
    const dir = path.join(root, sub);
    fs.mkdirSync(dir, { recursive: true });
    const file = path.join(dir, name);
    fs.writeFileSync(file, name);
    return { file, id: insertGeneration(db, { prompt: name, width: 8, height: 8, seed: 1, steps: 1, cfg: 1 }, 'z-image', file, null, workspaceId).id };
  };
  return { root, db, ctx, flat };
}

test('a path in the old flat layout is rewritten into the workspace folder, and nothing else is', () => {
  const root = path.join(os.tmpdir(), 'out');
  assert.equal(followLegacyPath(path.join(root, 'images', 'favorites', 'a.png'), root, 'Default'), path.join(root, 'Default', 'images', 'favorites', 'a.png'));
  assert.equal(followLegacyPath(path.join(root, 'Default', 'images', 'a.png'), root, 'Default'), null, 'already there');
  assert.equal(followLegacyPath(path.join(os.tmpdir(), 'elsewhere', 'images', 'a.png'), root, 'Default'), null, 'outside the output folder');
  assert.equal(followLegacyPath(null, root, 'Default'), null);
});

test('the flat folders move whole under the first workspace and every stored path follows', () => {
  const s = setup();
  const a = s.flat('images', 'a.png');
  const t = s.flat('trash', 't.png');
  const v = s.flat('videos', 'v.mp4');
  s.db.prepare('UPDATE generations SET trashed_at = ?, trash_from = ? WHERE id = ?').run('2026-01-01', path.join(s.root, 'images', 't.png'), t.id);
  fs.mkdirSync(path.join(s.root, 'sources'), { recursive: true });
  const src = path.join(s.root, 'sources', 'abc.png');
  fs.writeFileSync(src, 'src');
  s.db.prepare('UPDATE generations SET source_image_path = ? WHERE id = ?').run(src, a.id);

  const result = migrateOutputLayout(s.ctx);
  const dirs = workspaceDirsOf(s.db, s.root, DEFAULT_WORKSPACE_ID);
  assert.deepEqual([result.foldersMoved, result.problems], [4, 0]);
  assert.equal(getGenerationById(s.db, a.id)!.imagePath, path.join(dirs.images, 'a.png'));
  assert.equal(getGenerationById(s.db, a.id)!.sourceImagePath, path.join(dirs.sources, 'abc.png'));
  assert.equal(getGenerationById(s.db, v.id)!.imagePath, path.join(dirs.videos, 'v.mp4'));
  assert.equal(path.dirname(getGenerationById(s.db, t.id)!.imagePath), dirs.trash);
  assert.equal((s.db.prepare('SELECT trash_from FROM generations WHERE id = ?').get(t.id) as { trash_from: string }).trash_from, path.join(dirs.images, 't.png'));
  for (const rec of [a, t, v]) assert.equal(fs.existsSync(getGenerationById(s.db, rec.id)!.imagePath), true, 'every file is where its record says');
  for (const old of ['images', 'videos', 'sources', 'trash']) assert.equal(fs.existsSync(path.join(s.root, old)), false);
});

test('running it again changes nothing', () => {
  const s = setup();
  s.flat('images', 'a.png');
  migrateOutputLayout(s.ctx);
  assert.deepEqual(migrateOutputLayout(s.ctx), { foldersMoved: 0, rowsRewritten: 0, itemsRelocated: 0, problems: 0, backup: null });
});

test('a start that was interrupted after the folders moved but before the records changed is finished on the next one', () => {
  const s = setup();
  const a = s.flat('images', 'a.png');
  const dirs = workspaceDirsOf(s.db, s.root, DEFAULT_WORKSPACE_ID);
  // The folder was renamed but the record still has the old path.
  fs.mkdirSync(dirs.root, { recursive: true });
  fs.renameSync(path.join(s.root, 'images'), dirs.images);

  migrateOutputLayout(s.ctx);
  assert.equal(getGenerationById(s.db, a.id)!.imagePath, path.join(dirs.images, 'a.png'));
});

test('when the new folder already exists the files are moved one by one and nothing is overwritten', () => {
  const s = setup();
  const a = s.flat('images', 'a.png');
  const dirs = workspaceDirsOf(s.db, s.root, DEFAULT_WORKSPACE_ID);
  fs.mkdirSync(dirs.images, { recursive: true });
  fs.writeFileSync(path.join(dirs.images, 'new.png'), 'new');
  fs.writeFileSync(path.join(dirs.images, 'clash.png'), 'kept');
  const clash = s.flat('images', 'clash.png');

  const result = migrateOutputLayout(s.ctx);
  assert.equal(fs.readFileSync(path.join(dirs.images, 'clash.png'), 'utf8'), 'kept');
  assert.equal(fs.existsSync(clash.file), true, 'the one that could not move is left, and reported');
  assert.equal(result.problems, 1);
  assert.equal(getGenerationById(s.db, a.id)!.imagePath, path.join(dirs.images, 'a.png'));
});

test('items of another workspace end up in that workspace\'s own folder', () => {
  const s = setup();
  const hidden = createWorkspace(s.db, 'Hidden');
  const mine = s.flat('images', 'mine.png');
  const theirs = s.flat('images', 'theirs.png', hidden.id);

  const result = migrateOutputLayout(s.ctx);
  assert.equal(result.itemsRelocated, 1);
  assert.equal(getGenerationById(s.db, mine.id)!.imagePath, path.join(workspaceDirsOf(s.db, s.root, DEFAULT_WORKSPACE_ID).images, 'mine.png'));
  const hiddenDirs = workspaceDirsOf(s.db, s.root, hidden.id);
  assert.equal(getGenerationById(s.db, theirs.id)!.imagePath, path.join(hiddenDirs.images, 'theirs.png'));
  assert.equal(fs.readFileSync(path.join(hiddenDirs.images, 'theirs.png'), 'utf8'), 'theirs.png');
});

test('a new install with no output folder yet has nothing to do', () => {
  const s = setup();
  assert.deepEqual(migrateOutputLayout(s.ctx), { foldersMoved: 0, rowsRewritten: 0, itemsRelocated: 0, problems: 0, backup: null });
});

test('the database is copied before the first change, and not again once nothing is left to move', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-layout-bk-'));
  const dbPath = path.join(root, 'kvgenius.db');
  const db = initDatabase(dbPath);
  const output = path.join(root, 'output');
  fs.mkdirSync(path.join(output, 'images'), { recursive: true });
  const file = path.join(output, 'images', 'a.png');
  fs.writeFileSync(file, 'a');
  insertGeneration(db, { prompt: 'a', width: 8, height: 8, seed: 1, steps: 1, cfg: 1 }, 'z-image', file);
  const ctx = { db, outputRoot: output, dirsOf: (id: number) => workspaceDirsOf(db, output, id), videoFamilies: [] };

  const first = migrateOutputLayout(ctx, dbPath);
  assert.equal(first.backup, `${dbPath}.pre-workspace-folders`);
  const copy = new DatabaseSync(first.backup!);
  assert.equal((copy.prepare('SELECT image_path FROM generations').get() as { image_path: string }).image_path, file, 'the copy has the paths as they were');
  copy.close();
  assert.equal(migrateOutputLayout(ctx, dbPath).backup, null);
  db.close();
});
