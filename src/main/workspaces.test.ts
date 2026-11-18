import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { DEFAULT_CLEANUP_SETTINGS } from '../shared/cleanup';
import { DEFAULT_WORKSPACE_NAME, HIDDEN_WORKSPACE_NAME } from '../shared/workspaces';
import { countGenerations, findDuplicateGeneration, getGenerationById, initDatabase, insertGeneration, listGenerations, listImageExtensions, listStyleTags } from './db';
import { insertJob, listJobs } from './jobStore';
import { listStyles, saveStyle } from './styles';
import {
  DEFAULT_WORKSPACE_ID,
  createWorkspace,
  currentWorkspaceId,
  defaultWorkspaceId,
  deleteWorkspace,
  getWorkspace,
  listWorkspaces,
  saveWorkspaceCleanup,
  setActiveWorkspaceId,
  styleExamplesWorkspaceId,
  summarizeWorkspaces,
  updateWorkspace,
} from './workspaces';

const VIDEO_FAMILIES = ['wan22-i2v'];
const params = (prompt: string, seed = 1) => ({ prompt, width: 64, height: 64, seed, steps: 4, cfg: 1 });

test('a new database starts with one workspace, Default, with the cleanup off', () => {
  const db = initDatabase(':memory:');
  const all = listWorkspaces(db, true);
  assert.deepEqual(all.map((w) => [w.id, w.name, w.hidden, w.theme]), [[DEFAULT_WORKSPACE_ID, DEFAULT_WORKSPACE_NAME, false, null]]);
  assert.deepEqual(all[0].cleanup, DEFAULT_CLEANUP_SETTINGS);
  assert.equal(defaultWorkspaceId(db), DEFAULT_WORKSPACE_ID);
});

test('workspaces are made, renamed, themed and hidden; names are unique ignoring case', () => {
  const db = initDatabase(':memory:');
  const logos = createWorkspace(db, '  Logo   work ');
  assert.equal(logos.name, 'Logo work');
  assert.throws(() => createWorkspace(db, 'logo WORK'), /already exists/);
  assert.throws(() => createWorkspace(db, '   '), /name/);
  assert.throws(() => updateWorkspace(db, logos.id, { name: 'default' }), /already exists/);

  const edited = updateWorkspace(db, logos.id, { name: 'Logos', theme: 'lava', hidden: true });
  assert.deepEqual([edited.name, edited.theme, edited.hidden], ['Logos', 'lava', true]);
  assert.equal(updateWorkspace(db, logos.id, { theme: null }).theme, null, 'null goes back to the app theme');
  assert.equal(updateWorkspace(db, logos.id, {}).name, 'Logos', 'an empty change keeps everything');

  assert.deepEqual(listWorkspaces(db, false).map((w) => w.name), ['Default']);
  assert.deepEqual(listWorkspaces(db, true).map((w) => w.name), ['Default', 'Logos']);
  assert.throws(() => updateWorkspace(db, 999, { name: 'x' }), /no longer exists/);
});

test("the library, counts, extensions, duplicates and style filter each see only their workspace", () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  const a = insertGeneration(db, { ...params('a fox'), styleName: 'Noir' }, 'z-image', '/out/a.png');
  const b = insertGeneration(db, { ...params('a fox'), styleName: 'Pastel' }, 'z-image', '/out/b.gif', null, other);

  assert.deepEqual(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, DEFAULT_WORKSPACE_ID).map((r) => r.id), [a.id]);
  assert.deepEqual(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, other).map((r) => r.id), [b.id]);
  assert.deepEqual(countGenerations(db, VIDEO_FAMILIES, false, other), { image: 1, video: 0 });
  assert.deepEqual(listImageExtensions(db, VIDEO_FAMILIES, DEFAULT_WORKSPACE_ID), ['png']);
  assert.deepEqual(listImageExtensions(db, VIDEO_FAMILIES, other), ['gif']);
  assert.deepEqual(listStyleTags(db, DEFAULT_WORKSPACE_ID), ['Noir']);
  assert.deepEqual(listStyleTags(db, other), ['Pastel']);
  assert.equal(getGenerationById(db, b.id)?.workspaceId, other);

  // The same settings are a repeat only inside the workspace they were made in.
  const query = { ...params('a fox'), workspaceId: other };
  assert.equal(findDuplicateGeneration(db, 'z-image', query)?.id, b.id);
  assert.equal(findDuplicateGeneration(db, 'z-image', { ...query, workspaceId: DEFAULT_WORKSPACE_ID })?.id, a.id);
  assert.equal(findDuplicateGeneration(db, 'z-image', { ...query, workspaceId: 999 }), null);
});

test('a job keeps the workspace it was queued in', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  const job = insertJob(db, { family: 'z-image', params: params('x'), source: 'ui', workspaceId: other });
  assert.equal(job.workspaceId, other);
  assert.equal(insertJob(db, { family: 'z-image', params: params('y'), source: 'ui' }).workspaceId, DEFAULT_WORKSPACE_ID);
  assert.deepEqual(listJobs(db, { workspaceId: other }).map((j) => j.id), [job.id]);
});

test('the current workspace is the one the window is in, else the oldest', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  setActiveWorkspaceId(null);
  assert.equal(currentWorkspaceId(db), DEFAULT_WORKSPACE_ID);
  setActiveWorkspaceId(other);
  assert.equal(currentWorkspaceId(db), other);
  setActiveWorkspaceId(12345);
  assert.equal(currentWorkspaceId(db), DEFAULT_WORKSPACE_ID, 'a workspace that is gone falls back');
  setActiveWorkspaceId(null);
});

test('each workspace has its own cleanup rules', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  saveWorkspaceCleanup(db, other, { ...DEFAULT_CLEANUP_SETTINGS, autoTrashEnabled: true, olderThanDays: 7 });
  assert.equal(getWorkspace(db, other)!.cleanup.olderThanDays, 7);
  assert.equal(getWorkspace(db, DEFAULT_WORKSPACE_ID)!.cleanup.autoTrashEnabled, false);
});

test('only an empty workspace can be deleted, and the last one stays', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  insertGeneration(db, params('x'), 'z-image', '/out/x.png', null, other);
  assert.throws(() => deleteWorkspace(db, other), /empty/);
  db.prepare('DELETE FROM generations').run();
  deleteWorkspace(db, other);
  assert.equal(getWorkspace(db, other), null);
  assert.throws(() => deleteWorkspace(db, DEFAULT_WORKSPACE_ID), /only workspace/);
});

test('deleting a workspace takes its own styles with it but never the shared ones', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  saveStyle(db, { name: 'Shared look', text: 'x', kind: 'style' }, null, other);
  saveStyle(db, { name: 'Local piece', text: 'y', kind: 'element' }, null, other);
  deleteWorkspace(db, other);
  assert.deepEqual(listStyles(db, DEFAULT_WORKSPACE_ID).map((s) => s.name), ['Shared look']);
});

test("the grid's summary counts a workspace's items and picks its newest picture as the cover", () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  insertGeneration(db, params('a', 1), 'z-image', '/out/old.png');
  insertGeneration(db, params('b', 2), 'z-image', '/out/new.png');
  insertGeneration(db, { ...params('v', 3), length: 81 }, 'wan22-i2v', '/out/newest.mp4');
  const summary = summarizeWorkspaces(db, false, VIDEO_FAMILIES);
  const first = summary.find((w) => w.id === DEFAULT_WORKSPACE_ID)!;
  assert.equal(first.itemCount, 3);
  assert.equal(first.coverPath, '/out/new.png', 'a video is never the cover');
  const second = summary.find((w) => w.id === other)!;
  assert.deepEqual([second.itemCount, second.coverPath], [0, null]);
});

test('the Styles page examples live in a workspace of the app\'s own that is never listed', () => {
  const db = initDatabase(':memory:');
  const examples = styleExamplesWorkspaceId(db);
  assert.notEqual(examples, DEFAULT_WORKSPACE_ID);
  assert.equal(styleExamplesWorkspaceId(db), examples, 'made once');
  assert.equal(listWorkspaces(db, true).some((w) => w.id === examples), false);
  assert.equal(getWorkspace(db, examples), null, 'it cannot be opened either');
  assert.equal(summarizeWorkspaces(db, true, VIDEO_FAMILIES).some((w) => w.id === examples), false);
});

// ---- A database from before workspaces ---------------------------------------------------------------

/** A database as it was before workspaces: the old tables (with the per-item hidden flag and one global styles list). */
function legacyDatabase(): string {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-ws-')), 'old.db');
  const old = new DatabaseSync(file);
  old.exec(`CREATE TABLE generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, prompt TEXT NOT NULL, negative_prompt TEXT, width INTEGER NOT NULL,
    height INTEGER NOT NULL, seed INTEGER NOT NULL, steps INTEGER NOT NULL, cfg REAL NOT NULL, length INTEGER,
    model_family TEXT NOT NULL, image_path TEXT NOT NULL, favorite INTEGER NOT NULL DEFAULT 0,
    hidden INTEGER NOT NULL DEFAULT 0, timing_id INTEGER, created_at TEXT NOT NULL)`);
  const add = old.prepare(
    "INSERT INTO generations (prompt, width, height, seed, steps, cfg, model_family, image_path, hidden, created_at) VALUES (?, 64, 64, ?, 4, 1, 'z-image', ?, ?, 'x')"
  );
  add.run('shown', 1, '/out/shown.png', 0);
  add.run('private', 2, '/out/private.png', 1);
  add.run('a style example', 3, '/out/example.png', 1);
  old.exec(`CREATE TABLE styles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL COLLATE NOCASE, text TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'style', created_at TEXT NOT NULL, text_changed_at TEXT NOT NULL DEFAULT '')`);
  old.exec("CREATE UNIQUE INDEX idx_styles_name ON styles (name COLLATE NOCASE)");
  old.exec("INSERT INTO styles (name, text, kind, created_at) VALUES ('Anime', 'cel shaded', 'style', 'x'), ('Red coat', 'a red coat', 'element', 'x')");
  old.exec('CREATE TABLE style_samples (style_id INTEGER PRIMARY KEY, generation_id INTEGER NOT NULL, text_snapshot TEXT NOT NULL, prompt_snapshot TEXT NOT NULL, seed_snapshot INTEGER NOT NULL, rendered_at TEXT NOT NULL)');
  old.exec("INSERT INTO style_samples VALUES (1, 3, 'cel shaded', 'p', 1, 'x')");
  old.close();
  return file;
}

test('an older database moves into the Default workspace: styles stay shared, elements go to it, hidden items go to a hidden workspace', () => {
  const file = legacyDatabase();
  const rules = { ...DEFAULT_CLEANUP_SETTINGS, autoTrashEnabled: true, olderThanDays: 45 };
  const db = initDatabase(file, rules);

  const all = listWorkspaces(db, true);
  assert.deepEqual(all.map((w) => [w.name, w.hidden]).sort(), [[DEFAULT_WORKSPACE_NAME, false], [HIDDEN_WORKSPACE_NAME, true]]);
  const first = all.find((w) => w.name === DEFAULT_WORKSPACE_NAME)!;
  const hidden = all.find((w) => w.name === HIDDEN_WORKSPACE_NAME)!;
  assert.equal(first.id, DEFAULT_WORKSPACE_ID);
  assert.deepEqual(first.cleanup, rules, 'the cleanup rules that used to be global are the first workspace\'s');

  const at = (id: number) => getGenerationById(db, id)!.workspaceId;
  assert.equal(at(1), first.id, 'an ordinary item');
  assert.equal(at(2), hidden.id, 'a hand-hidden item goes where it stays out of sight');
  assert.equal(at(3), styleExamplesWorkspaceId(db), 'a style example goes to the app\'s own workspace, not the Hidden one');
  assert.deepEqual(listGenerations(db, VIDEO_FAMILIES, 'image', 50, null, false, first.id).map((r) => r.id), [1]);

  assert.deepEqual(listStyles(db, first.id).map((s) => [s.name, s.workspaceId]), [['Anime', null], ['Red coat', first.id]]);
  const second = createWorkspace(db, 'Second').id;
  assert.deepEqual(listStyles(db, second).map((s) => s.name), ['Anime'], 'another workspace sees the shared style but not the element');
  db.close();
});

test('moving an older database into workspaces is done once: a second start changes nothing', () => {
  const file = legacyDatabase();
  const once = initDatabase(file);
  const second = createWorkspace(once, 'Second').id;
  once.prepare('UPDATE generations SET workspace_id = ? WHERE id = 1').run(second);
  once.close();

  const again = initDatabase(file);
  assert.equal(getGenerationById(again, 1)!.workspaceId, second, 'an item moved since stays where it was put');
  assert.equal(listWorkspaces(again, true).filter((w) => w.name === HIDDEN_WORKSPACE_NAME).length, 1);
  again.close();
});
