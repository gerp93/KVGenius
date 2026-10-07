/** Copies of the database that KVGenius made before a one-time migration (see familyMigration.ts). The app never
 * deletes them: they are listed in Settings > Library & Data so the user can remove them when satisfied. */
export const MIGRATION_BACKUP = /\.pre-[a-z-]+(-\d+)?$/;

/** Which of a folder's file names are such backups of the database called `dbFileName`. */
export function migrationBackups(dbFileName: string, folderFiles: string[]): string[] {
  return folderFiles.filter((name) => name.startsWith(`${dbFileName}.`) && MIGRATION_BACKUP.test(name)).sort();
}
