import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as path from 'path';
import { folderNameFor, workspaceDirs } from './workspaceFolders';
import { createWorkspace, updateWorkspace, workspaceDirsOf, workspaceFolder, DEFAULT_WORKSPACE_ID } from './workspaces';
import * as fs from 'fs';
import * as os from 'os';
import { DatabaseSync } from 'node:sqlite';
import { initDatabase } from './db';

test('a folder name keeps what a folder can hold and is different from every taken one, ignoring case', () => {
  assert.equal(folderNameFor('Album covers', []), 'Album covers');
  assert.equal(folderNameFor('  a/b\\c:d*e?  ', []), 'a b c d e');
  assert.equal(folderNameFor('Logos', ['logos']), 'Logos-2');
  assert.equal(folderNameFor('Logos', ['logos', 'Logos-2']), 'Logos-3');
  assert.equal(folderNameFor('...', []), 'workspace', 'nothing usable left');
  assert.equal(folderNameFor('x'.repeat(100), []).length, 40);
});

test('the names of the old flat folders and Windows device names are never used', () => {
  assert.equal(folderNameFor('Images', []), 'Images-2');
  assert.equal(folderNameFor('trash', []), 'trash-2');
  assert.equal(folderNameFor('CON', []), 'CON-2');
});

test('a workspace has images, videos, gifs, sources and trash folders under its own folder', () => {
  const dirs = workspaceDirs(path.join('/data', 'output'), 'Default');
  assert.equal(dirs.root, path.join('/data', 'output', 'Default'));
  assert.equal(dirs.images, path.join('/data', 'output', 'Default', 'images'));
  assert.equal(dirs.trash, path.join('/data', 'output', 'Default', 'trash'));
});

test('workspaces get their own folder when made, and renaming one never changes it', () => {
  const db = initDatabase(':memory:');
  assert.equal(workspaceFolder(db, DEFAULT_WORKSPACE_ID), 'Default');
  const logos = createWorkspace(db, 'Logo work');
  assert.equal(workspaceFolder(db, logos.id), 'Logo work');
  updateWorkspace(db, logos.id, { name: 'Logos 2027' });
  assert.equal(workspaceFolder(db, logos.id), 'Logo work', 'the label changed, the folder did not');
  assert.equal(workspaceDirsOf(db, '/o', logos.id).images, path.join('/o', 'Logo work', 'images'));
  assert.throws(() => workspaceDirsOf(db, '/o', 999), /no longer exists/);
});

test('a database from when workspaces had no folders gets one for each, named from the workspace', () => {
  const file = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-folders-')), 'old.db');
  const old = new DatabaseSync(file);
  old.exec(`CREATE TABLE workspaces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, theme TEXT, hidden INTEGER NOT NULL DEFAULT 0,
    system INTEGER NOT NULL DEFAULT 0, cleanup TEXT, created_at TEXT NOT NULL)`);
  old.exec("INSERT INTO workspaces (id, name, hidden, system, created_at) VALUES (1, 'Default', 0, 0, 'x'), (2, 'Album covers', 0, 0, 'x'), (3, 'Hidden', 1, 0, 'x'), (4, 'Style examples', 1, 1, 'x')");
  old.close();

  const db = initDatabase(file);
  assert.deepEqual([1, 2, 3, 4].map((id) => workspaceFolder(db, id)), ['Default', 'Album covers', 'Hidden', 'Style examples']);
  db.close();
});
