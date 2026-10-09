import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initDatabase, insertGeneration, getGenerationById } from './db';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { deleteStyle, findStyleByName, getStyle, listStyles, saveStyle } from './styles';

const params = (prompt: string, styleName?: string) => ({ prompt, width: 64, height: 64, seed: 1, steps: 4, cfg: 1, ...(styleName ? { styleName } : {}) });

test('styles are created, listed alphabetically, edited and deleted', () => {
  const db = initDatabase(':memory:');
  assert.deepEqual(listStyles(db), []);
  const poster = saveStyle(db, { name: ' 1930s movie poster ', text: ' bold lithograph ' });
  const anime = saveStyle(db, { name: 'anime', text: 'cel shaded' });
  assert.equal(poster.name, '1930s movie poster');
  assert.equal(poster.text, 'bold lithograph');
  assert.deepEqual(listStyles(db).map((s) => s.name), ['1930s movie poster', 'anime']);

  const edited = saveStyle(db, { name: 'Anime', text: 'cel shaded, flat colors' }, anime.id);
  assert.equal(edited.id, anime.id);
  assert.equal(getStyle(db, anime.id)?.text, 'cel shaded, flat colors');

  deleteStyle(db, poster.id);
  assert.equal(getStyle(db, poster.id), null);
  assert.deepEqual(listStyles(db).map((s) => s.name), ['Anime']);
});

test('names are unique ignoring case, and a style can keep its own name when edited', () => {
  const db = initDatabase(':memory:');
  const a = saveStyle(db, { name: 'Noir', text: 'black and white' });
  assert.throws(() => saveStyle(db, { name: 'noir', text: 'other' }), /already exists/);
  const b = saveStyle(db, { name: 'Pastel', text: 'soft' });
  assert.throws(() => saveStyle(db, { name: 'NOIR', text: 'x' }, b.id), /already exists/);
  assert.equal(saveStyle(db, { name: 'Noir', text: 'high contrast' }, a.id).text, 'high contrast');
});

test('saving rejects blank input and editing a style that is gone', () => {
  const db = initDatabase(':memory:');
  assert.throws(() => saveStyle(db, { name: '', text: 'x' }), /name/i);
  assert.throws(() => saveStyle(db, { name: 'x', text: ' ' }), /style text/i);
  assert.throws(() => saveStyle(db, { name: 'x', text: 'y' }, 999), /no longer exists/);
});

test('a style can be found by name ignoring case', () => {
  const db = initDatabase(':memory:');
  const s = saveStyle(db, { name: '1930s Poster', text: 'x' });
  assert.equal(findStyleByName(db, '  1930s poster ')?.id, s.id);
  assert.equal(findStyleByName(db, 'nope'), null);
});

test('a generation records the style name, and none when no style was used', () => {
  const db = initDatabase(':memory:');
  const styled = insertGeneration(db, params('a fox, bold lithograph', '1930s movie poster'), 'z-image', '/out/a.png');
  const plain = insertGeneration(db, params('a fox'), 'z-image', '/out/b.png');
  assert.equal(getGenerationById(db, styled.id)?.styleName, '1930s movie poster');
  assert.equal(getGenerationById(db, styled.id)?.prompt, 'a fox, bold lithograph', 'the stored prompt is the full text sent');
  assert.equal(plain.styleName, null);
  assert.equal(getGenerationById(db, plain.id)?.styleName, null);
});

test('deleting a style leaves past generations alone', () => {
  const db = initDatabase(':memory:');
  const s = saveStyle(db, { name: 'Noir', text: 'black and white' });
  const g = insertGeneration(db, params('a fox, black and white', 'Noir'), 'z-image', '/out/a.png');
  deleteStyle(db, s.id);
  assert.equal(getGenerationById(db, g.id)?.prompt, 'a fox, black and white');
  assert.equal(getGenerationById(db, g.id)?.styleName, 'Noir');
});

test('a style or element keeps its kind, and an old table without one is upgraded', () => {
  const db = initDatabase(':memory:');
  const coat = saveStyle(db, { name: 'Coat', text: 'red trench coat', kind: 'element' });
  const anime = saveStyle(db, { name: 'Anime', text: 'cel shaded' });
  assert.equal(coat.kind, 'element');
  assert.equal(anime.kind, 'style', 'no kind means a style');
  assert.equal(getStyle(db, coat.id)?.kind, 'element');
  assert.equal(saveStyle(db, { name: 'Coat', text: 'red trench coat', kind: 'style' }, coat.id).kind, 'style');
  assert.throws(() => saveStyle(db, { name: 'anime', text: 'x', kind: 'element' }), /style or element named/);
});

test('a database made before kinds existed gains the column with every style as a style', () => {
  const dir = mkdtempSync(join(tmpdir(), 'kv-styles-'));
  const path = join(dir, 'old.db');
  const old = new DatabaseSync(path);
  old.exec("CREATE TABLE styles (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL COLLATE NOCASE UNIQUE, text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')))");
  old.exec("INSERT INTO styles (name, text) VALUES ('Noir', 'black and white')");
  old.close();
  const db = initDatabase(path);
  assert.deepEqual(listStyles(db).map((s) => [s.name, s.kind]), [['Noir', 'style']]);
});
