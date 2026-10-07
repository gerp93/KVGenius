import { test } from 'node:test';
import assert from 'node:assert/strict';
import { migrationBackups } from './dbBackups';

test('only migration backups of this database are listed', () => {
  const files = ['kvgenius.db', 'kvgenius.db-wal', 'kvgenius.db.pre-family-rename', 'kvgenius.db.pre-family-rename-2', 'other.db.pre-family-rename', 'notes.txt'];
  assert.deepEqual(migrationBackups('kvgenius.db', files), ['kvgenius.db.pre-family-rename', 'kvgenius.db.pre-family-rename-2']);
  assert.deepEqual(migrationBackups('kvgenius.db', ['kvgenius.db']), []);
});
