import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import { DatabaseSync } from 'node:sqlite';
import { initDatabase } from './db';
import { migrateFamilyKeys } from './familyMigration';

function seedOldRows(db: DatabaseSync): void {
  db.exec(
    `INSERT INTO generations (prompt, width, height, seed, steps, cfg, model_family, image_path, created_at) VALUES ('old', 64, 64, 1, 8, 1, 'z-image-turbo', '/out/a.png', 'x')`,
  );
  db.exec(
    `INSERT INTO generations (prompt, width, height, seed, steps, cfg, model_family, image_path, created_at) VALUES ('wan', 64, 64, 1, 4, 1, 'wan22-i2v', '/out/b.mp4', 'x')`,
  );
  db.exec(`INSERT INTO jobs (source, family, params, status, created_at) VALUES ('ui', 'z-image-turbo', '{}', 'queued', 'x')`);
  db.exec(
    `INSERT INTO timing_stats (created_at, family, kind, width, height, steps, cfg, actual_ms) VALUES ('x', 'z-image-turbo', 'image', 64, 64, 8, 1, 1000)`,
  );
}

function families(db: DatabaseSync): { generations: string[]; jobs: string[]; timing: string[] } {
  const col = (sql: string) => (db.prepare(sql).all() as unknown as { f: string }[]).map((r) => r.f);
  return {
    generations: col('SELECT model_family AS f FROM generations ORDER BY id'),
    jobs: col('SELECT family AS f FROM jobs ORDER BY id'),
    timing: col('SELECT family AS f FROM timing_stats ORDER BY id'),
  };
}

test('old keys in all three tables are renamed, other families are left alone, and a second run does nothing', () => {
  const db = initDatabase(':memory:');
  seedOldRows(db);
  const first = migrateFamilyKeys(db, ':memory:');
  assert.equal(first.renamed, 3);
  assert.equal(first.backup, null);
  assert.deepEqual(families(db), { generations: ['z-image', 'wan22-i2v'], jobs: ['z-image'], timing: ['z-image'] });
  assert.equal(migrateFamilyKeys(db, ':memory:').renamed, 0);
});

test('a database with nothing to rename is left alone and gets no backup', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-fam-'));
  try {
    const dbPath = path.join(dir, 'kvgenius.db');
    const db = initDatabase(dbPath);
    assert.deepEqual(migrateFamilyKeys(db, dbPath), { renamed: 0, backup: null });
    assert.deepEqual(fs.readdirSync(dir).filter((f) => f.includes('pre-family-rename')), []);
    db.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('a file database is copied first, the copy keeps the old keys, and it is never removed or overwritten', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-fam-'));
  try {
    const dbPath = path.join(dir, 'kvgenius.db');
    const db = initDatabase(dbPath);
    seedOldRows(db);
    const first = migrateFamilyKeys(db, dbPath);
    assert.equal(first.backup, `${dbPath}.pre-family-rename`);
    assert.ok(fs.existsSync(first.backup!));

    const copy = new DatabaseSync(first.backup!);
    assert.deepEqual(families(copy).generations, ['z-image-turbo', 'wan22-i2v']);
    copy.close();
    assert.deepEqual(families(db).generations, ['z-image', 'wan22-i2v']);

    // Old rows appearing again (say, an older app version wrote them) get a second, separate backup.
    seedOldRows(db);
    const second = migrateFamilyKeys(db, dbPath);
    assert.equal(second.backup, `${dbPath}.pre-family-rename-2`);
    assert.ok(fs.existsSync(first.backup!));
    db.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});

test('initDatabase runs the migration by itself', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'kvg-fam-'));
  try {
    const dbPath = path.join(dir, 'kvgenius.db');
    const old = initDatabase(dbPath);
    seedOldRows(old);
    old.close();
    const reopened = initDatabase(dbPath);
    assert.deepEqual(families(reopened).generations, ['z-image', 'wan22-i2v']);
    assert.ok(fs.existsSync(`${dbPath}.pre-family-rename`));
    reopened.close();
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
});
