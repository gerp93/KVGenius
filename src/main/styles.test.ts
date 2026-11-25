import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initDatabase, insertGeneration, getGenerationById, findDuplicateGeneration } from './db';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { deleteStyle, findStyleByName, getStyle, listStyles, saveStyle } from './styles';
import { createWorkspace } from './workspaces';

const params = (prompt: string, styleName?: string) => ({ prompt, width: 64, height: 64, seed: 1, steps: 4, cfg: 1, ...(styleName ? { styleName } : {}) });

test('styles are created, listed alphabetically, edited and deleted', () => {
  const db = initDatabase(':memory:');
  assert.deepEqual(listStyles(db, 1), []);
  const poster = saveStyle(db, { name: ' 1930s movie poster ', text: ' bold lithograph ' }, null, 1);
  const anime = saveStyle(db, { name: 'anime', text: 'cel shaded' }, null, 1);
  assert.equal(poster.name, '1930s movie poster');
  assert.equal(poster.text, 'bold lithograph');
  assert.deepEqual(listStyles(db, 1).map((s) => s.name), ['1930s movie poster', 'anime']);

  const edited = saveStyle(db, { name: 'Anime', text: 'cel shaded, flat colors' }, anime.id, 1);
  assert.equal(edited.id, anime.id);
  assert.equal(getStyle(db, anime.id)?.text, 'cel shaded, flat colors');

  deleteStyle(db, poster.id);
  assert.equal(getStyle(db, poster.id), null);
  assert.deepEqual(listStyles(db, 1).map((s) => s.name), ['Anime']);
});

test('names are unique ignoring case, and a style can keep its own name when edited', () => {
  const db = initDatabase(':memory:');
  const a = saveStyle(db, { name: 'Noir', text: 'black and white' }, null, 1);
  assert.throws(() => saveStyle(db, { name: 'noir', text: 'other' }, null, 1), /already exists/);
  const b = saveStyle(db, { name: 'Pastel', text: 'soft' }, null, 1);
  assert.throws(() => saveStyle(db, { name: 'NOIR', text: 'x' }, b.id, 1), /already exists/);
  assert.equal(saveStyle(db, { name: 'Noir', text: 'high contrast' }, a.id, 1).text, 'high contrast');
});

test('saving rejects blank input and editing a style that is gone', () => {
  const db = initDatabase(':memory:');
  assert.throws(() => saveStyle(db, { name: '', text: 'x' }, null, 1), /name/i);
  assert.throws(() => saveStyle(db, { name: 'x', text: ' ' }, null, 1), /style text/i);
  assert.throws(() => saveStyle(db, { name: 'x', text: 'y' }, 999, 1), /no longer exists/);
});

test('a style can be found by name ignoring case', () => {
  const db = initDatabase(':memory:');
  const s = saveStyle(db, { name: '1930s Poster', text: 'x' }, null, 1);
  assert.equal(findStyleByName(db, '  1930s poster ', 1)?.id, s.id);
  assert.equal(findStyleByName(db, 'nope', 1), null);
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
  const s = saveStyle(db, { name: 'Noir', text: 'black and white' }, null, 1);
  const g = insertGeneration(db, params('a fox, black and white', 'Noir'), 'z-image', '/out/a.png');
  deleteStyle(db, s.id);
  assert.equal(getGenerationById(db, g.id)?.prompt, 'a fox, black and white');
  assert.equal(getGenerationById(db, g.id)?.styleName, 'Noir');
});

test('a style or element keeps its kind, and an old table without one is upgraded', () => {
  const db = initDatabase(':memory:');
  const coat = saveStyle(db, { name: 'Coat', text: 'red trench coat', kind: 'element' }, null, 1);
  const anime = saveStyle(db, { name: 'Anime', text: 'cel shaded' }, null, 1);
  assert.equal(coat.kind, 'element');
  assert.equal(anime.kind, 'style', 'no kind means a style');
  assert.equal(getStyle(db, coat.id)?.kind, 'element');
  assert.equal(saveStyle(db, { name: 'Coat', text: 'red trench coat', kind: 'style' }, coat.id, 1).kind, 'style');
  assert.throws(() => saveStyle(db, { name: 'anime', text: 'x', kind: 'element' }, null, 1), /style or element named/);
});

test('a database made before kinds existed gains the column with every style as a style', () => {
  const dir = mkdtempSync(join(tmpdir(), 'kv-styles-'));
  const path = join(dir, 'old.db');
  const old = new DatabaseSync(path);
  old.exec("CREATE TABLE styles (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL COLLATE NOCASE UNIQUE, text TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT (datetime('now')), updated_at TEXT NOT NULL DEFAULT (datetime('now')))");
  old.exec("INSERT INTO styles (name, text) VALUES ('Noir', 'black and white')");
  old.close();
  const db = initDatabase(path);
  assert.deepEqual(listStyles(db, 1).map((s) => [s.name, s.kind]), [['Noir', 'style']]);
});

test('an element is for the workspace it was made in, a style is shared - unless the scope says otherwise', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  const look = saveStyle(db, { name: 'Noir', text: 'x' }, null, 1);
  const piece = saveStyle(db, { name: 'Red coat', text: 'y', kind: 'element' }, null, 1);
  const localLook = saveStyle(db, { name: 'Local look', text: 'z', scope: 'workspace' }, null, 1);
  const sharedPiece = saveStyle(db, { name: 'Shared piece', text: 'w', kind: 'element', scope: 'all' }, null, 1);
  assert.deepEqual([look.workspaceId, piece.workspaceId, localLook.workspaceId, sharedPiece.workspaceId], [null, 1, 1, null]);

  assert.deepEqual(listStyles(db, 1).map((s) => s.name), ['Local look', 'Noir', 'Red coat', 'Shared piece']);
  assert.deepEqual(listStyles(db, other).map((s) => s.name), ['Noir', 'Shared piece']);
  assert.equal(findStyleByName(db, 'red coat', 1)?.id, piece.id);
  assert.equal(findStyleByName(db, 'red coat', other), null, 'not visible from the other workspace');
});

test('editing a style does not change who it is for, and the same name can be used in two workspaces', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  const piece = saveStyle(db, { name: 'Red coat', text: 'y', kind: 'element' }, null, 1);
  assert.equal(saveStyle(db, { name: 'Red coat', text: 'a longer red coat', kind: 'element', scope: 'all' }, piece.id, other).workspaceId, 1);
  const twin = saveStyle(db, { name: 'red coat', text: 'a different one', kind: 'element' }, null, other);
  assert.equal(twin.workspaceId, other);
  assert.equal(findStyleByName(db, 'Red coat', other)?.id, twin.id);
});

test('a shared name cannot be taken twice, and a shared style cannot take a workspace style\'s name', () => {
  const db = initDatabase(':memory:');
  const other = createWorkspace(db, 'Other').id;
  saveStyle(db, { name: 'Noir', text: 'x' }, null, 1);
  assert.throws(() => saveStyle(db, { name: 'noir', text: 'y', scope: 'workspace' }, null, other), /already exists/, 'a workspace one cannot repeat a shared name');
  saveStyle(db, { name: 'Piece', text: 'x', kind: 'element' }, null, other);
  assert.throws(() => saveStyle(db, { name: 'PIECE', text: 'y', scope: 'all' }, null, 1), /already exists/, 'a shared one cannot repeat any workspace\'s name');
});

test('an element can carry a LoRA, which is kept, edited, cleared and never held by a style', () => {
  const db = initDatabase(':memory:');
  const lora = { file: 'ffl-twilek.safetensors', strength: 0.8, families: ['flux-dev'] };
  const twilek = saveStyle(db, { name: "Twi'lek", text: "ffl-twi'lek-mk.1", kind: 'element', lora }, null, 1);
  assert.deepEqual(twilek.lora, lora);
  assert.deepEqual(getStyle(db, twilek.id)?.lora, lora);
  assert.deepEqual(listStyles(db, 1).find((s) => s.id === twilek.id)?.lora, lora);
  assert.deepEqual(findStyleByName(db, "twi'lek", 1)?.lora, lora);

  // An edit that does not mention the LoRA leaves it alone; a new strength replaces it; null removes it.
  assert.deepEqual(saveStyle(db, { name: "Twi'lek", text: 'new words', kind: 'element' }, twilek.id, 1).lora, lora);
  assert.equal(saveStyle(db, { name: "Twi'lek", text: 'new words', kind: 'element', lora: { ...lora, strength: -0.5 } }, twilek.id, 1).lora?.strength, -0.5);
  assert.equal(saveStyle(db, { name: "Twi'lek", text: 'new words', kind: 'element', lora: null }, twilek.id, 1).lora, null);

  // A style has none even if one is sent, and turning an element into a style drops it.
  assert.equal(saveStyle(db, { name: 'Noir', text: 'x', kind: 'style', lora }, null, 1).lora, null);
  const piece = saveStyle(db, { name: 'Piece', text: 'y', kind: 'element', lora }, null, 1);
  assert.equal(saveStyle(db, { name: 'Piece', text: 'y', kind: 'style' }, piece.id, 1).lora, null);
  assert.throws(() => saveStyle(db, { name: 'Bad', text: 'z', kind: 'element', lora: { ...lora, strength: 50 } }, null, 1), /strength/i);
});

test('a database from before element LoRAs gains the columns and keeps its styles and pictures', () => {
  const dir = mkdtempSync(join(tmpdir(), 'kvg-lora-migrate-'));
  try {
    const file = join(dir, 'old.db');
    const first = initDatabase(file);
    saveStyle(first, { name: 'Red coat', text: 'wearing red', kind: 'element' }, null, 1);
    insertGeneration(first, params('a fox'), 'flux-dev', '/out/a.png');
    // Make it look like an older database: no LoRA columns at all.
    first.exec('ALTER TABLE styles DROP COLUMN lora; ALTER TABLE generations DROP COLUMN loras;');
    first.close();

    const db = initDatabase(file);
    const styleColumns = (db.prepare('PRAGMA table_info(styles)').all() as unknown as { name: string }[]).map((c) => c.name);
    const generationColumns = (db.prepare('PRAGMA table_info(generations)').all() as unknown as { name: string }[]).map((c) => c.name);
    assert.ok(styleColumns.includes('lora'));
    assert.ok(generationColumns.includes('loras'));
    assert.equal(listStyles(db, 1)[0].lora, null, 'an old element has no LoRA');
    assert.equal(getGenerationById(db, 1)?.loras, null);
    const edited = saveStyle(db, { name: 'Red coat', text: 'wearing red', kind: 'element', lora: { file: 'a.safetensors', strength: 1, families: ['flux-dev'] } }, 1, 1);
    assert.equal(edited.lora?.file, 'a.safetensors');
    // Running it again is harmless.
    db.close();
    initDatabase(file).close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('a generation records the LoRAs it was made with, and a different LoRA is a different picture to the duplicate guard', () => {
  const db = initDatabase(':memory:');
  const loras = [{ file: 'a.safetensors', strength: 0.8 }, { file: 'b.safetensors', strength: 1 }];
  const made = insertGeneration(db, { ...params('a twi\'lek'), loras }, 'flux-dev', '/out/a.png');
  assert.deepEqual(made.loras, loras);
  assert.deepEqual(getGenerationById(db, made.id)?.loras, loras);
  assert.equal(insertGeneration(db, params('plain'), 'flux-dev', '/out/b.png').loras, null);

  const query = { workspaceId: 1, prompt: "a twi'lek", width: 64, height: 64, seed: 1, steps: 4, cfg: 1 };
  assert.equal(findDuplicateGeneration(db, 'flux-dev', { ...query, loras })?.id, made.id);
  assert.equal(findDuplicateGeneration(db, 'flux-dev', query), null, 'the same settings without the LoRAs are not a repeat');
  assert.equal(findDuplicateGeneration(db, 'flux-dev', { ...query, loras: [{ ...loras[0], strength: 0.5 }, loras[1]] }), null, 'nor with another strength');
});
