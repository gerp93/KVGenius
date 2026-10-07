import * as fs from 'fs';
import { DatabaseSync } from 'node:sqlite';
import { LEGACY_FAMILY_KEYS } from '../shared/families';

/** Every place a family key is stored as plain text. */
const FAMILY_COLUMNS: { table: string; column: string }[] = [
  { table: 'generations', column: 'model_family' },
  { table: 'jobs', column: 'family' },
  { table: 'timing_stats', column: 'family' },
];

export interface FamilyMigrationResult {
  /** Rows rewritten, across the three tables. 0 when there was nothing to do. */
  renamed: number;
  /** Where the pre-migration copy of the database was left, if one was made. */
  backup: string | null;
}

function countLegacy(db: DatabaseSync): number {
  const legacy = Object.keys(LEGACY_FAMILY_KEYS);
  const marks = legacy.map(() => '?').join(', ');
  let total = 0;
  for (const { table, column } of FAMILY_COLUMNS) {
    const row = db.prepare(`SELECT COUNT(*) AS n FROM ${table} WHERE ${column} IN (${marks})`).get(...legacy) as unknown as { n: number };
    total += row.n;
  }
  return total;
}

/** A backup name that does not exist yet (VACUUM INTO refuses to overwrite). */
function freeBackupPath(dbPath: string): string {
  const base = `${dbPath}.pre-family-rename`;
  if (!fs.existsSync(base)) return base;
  for (let n = 2; ; n++) if (!fs.existsSync(`${base}-${n}`)) return `${base}-${n}`;
}

/**
 * Rewrites a retired family key (z-image-turbo) to its current one in every table that stores it.
 * Runs at every startup and is a no-op once nothing uses an old key. Before the first real change it
 * copies the database next to itself (VACUUM INTO, so content still in the write-ahead log is
 * included); the app never deletes that copy - the user removes it when satisfied. If the copy cannot
 * be made nothing is changed, and the old key keeps working through canonicalFamily() meanwhile.
 * All the rewrites happen in one transaction: all or nothing.
 */
export function migrateFamilyKeys(db: DatabaseSync, dbPath: string): FamilyMigrationResult {
  if (countLegacy(db) === 0) return { renamed: 0, backup: null };

  let backup: string | null = null;
  if (dbPath !== ':memory:') {
    backup = freeBackupPath(dbPath);
    db.exec(`VACUUM INTO '${backup.replace(/'/g, "''")}'`);
  }

  let renamed = 0;
  db.exec('BEGIN');
  try {
    for (const [legacy, current] of Object.entries(LEGACY_FAMILY_KEYS)) {
      for (const { table, column } of FAMILY_COLUMNS) {
        const result = db.prepare(`UPDATE ${table} SET ${column} = ? WHERE ${column} = ?`).run(current, legacy);
        renamed += Number(result.changes);
      }
    }
    db.exec('COMMIT');
  } catch (err) {
    db.exec('ROLLBACK');
    throw err;
  }
  return { renamed, backup };
}
