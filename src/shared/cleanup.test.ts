import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_CLEANUP_SETTINGS,
  MAX_CLEANUP_DAYS,
  autoEmptyDue,
  autoTrashDue,
  cutoffIso,
  normalizeCleanupSettings,
  normalizeDays,
  updateCleanupSettings,
} from './cleanup';

const NOW = new Date('2026-10-05T12:00:00.000Z');

test('cleanup starts out manual: neither schedule is on by default', () => {
  assert.equal(DEFAULT_CLEANUP_SETTINGS.autoTrashEnabled, false);
  assert.equal(DEFAULT_CLEANUP_SETTINGS.autoEmptyEnabled, false);
  assert.deepEqual(normalizeCleanupSettings(undefined), DEFAULT_CLEANUP_SETTINGS);
  assert.deepEqual(normalizeCleanupSettings('nonsense'), DEFAULT_CLEANUP_SETTINGS);
});

test('days are whole numbers within range, with a fallback for junk', () => {
  assert.equal(normalizeDays(14, 30), 14);
  assert.equal(normalizeDays('45', 30), 45);
  assert.equal(normalizeDays(7.6, 30), 8);
  assert.equal(normalizeDays(0, 30), 1);
  assert.equal(normalizeDays(-5, 30), 1);
  assert.equal(normalizeDays(99999, 30), MAX_CLEANUP_DAYS);
  assert.equal(normalizeDays('abc', 30), 30);
  assert.equal(normalizeDays(NaN, 30), 30);
  assert.equal(normalizeDays(undefined, 30), 30);
});

test('only an explicit true turns a schedule on, and stored dates must be real', () => {
  assert.equal(normalizeCleanupSettings({ autoTrashEnabled: 'yes' }).autoTrashEnabled, false);
  assert.equal(normalizeCleanupSettings({ autoEmptyEnabled: 1 }).autoEmptyEnabled, false);
  assert.equal(normalizeCleanupSettings({ autoTrashEnabled: true }).autoTrashEnabled, true);
  assert.equal(normalizeCleanupSettings({ autoEmptyEnabled: true }).autoEmptyEnabled, true);
  assert.equal(normalizeCleanupSettings({ lastAutoTrashRun: 'not a date' }).lastAutoTrashRun, null);
  assert.equal(normalizeCleanupSettings({ lastAutoEmptyRun: NOW.toISOString() }).lastAutoEmptyRun, NOW.toISOString());
});

test('the cutoff is that many days back', () => {
  assert.equal(cutoffIso(30, NOW), '2026-09-05T12:00:00.000Z');
  assert.equal(cutoffIso(1, NOW), '2026-10-04T12:00:00.000Z');
});

test('the two schedules are independent', () => {
  const trashOnly = updateCleanupSettings(DEFAULT_CLEANUP_SETTINGS, { autoTrashEnabled: true }, NOW);
  assert.equal(trashOnly.autoTrashEnabled, true);
  assert.equal(trashOnly.autoEmptyEnabled, false);
  assert.equal(trashOnly.lastAutoEmptyRun, null);

  const emptyOnly = updateCleanupSettings(DEFAULT_CLEANUP_SETTINGS, { autoEmptyEnabled: true }, NOW);
  assert.equal(emptyOnly.autoTrashEnabled, false);
  assert.equal(emptyOnly.lastAutoTrashRun, null);
  assert.equal(emptyOnly.lastAutoEmptyRun, NOW.toISOString());
});

test('turning a schedule on starts its clock, so nothing runs by surprise', () => {
  const turnedOn = updateCleanupSettings(DEFAULT_CLEANUP_SETTINGS, { autoTrashEnabled: true }, NOW);
  assert.equal(turnedOn.lastAutoTrashRun, NOW.toISOString());
  assert.equal(autoTrashDue(turnedOn, NOW), false, 'not due until a day has passed');

  // Changing a threshold while it is already on does not restart the clock.
  const later = new Date(NOW.getTime() + 3 * 60 * 60 * 1000);
  const changed = updateCleanupSettings(turnedOn, { olderThanDays: 14 }, later);
  assert.equal(changed.olderThanDays, 14);
  assert.equal(changed.lastAutoTrashRun, NOW.toISOString());

  // Turning it off keeps the thresholds; junk is cleaned up.
  const off = updateCleanupSettings(changed, { autoTrashEnabled: false, trashRetentionDays: 0 }, later);
  assert.equal(off.autoTrashEnabled, false);
  assert.equal(off.trashRetentionDays, 1);
  assert.equal(off.olderThanDays, 14);
});

test('each schedule only runs when on, and not more than once a day', () => {
  const on = { ...DEFAULT_CLEANUP_SETTINGS, autoTrashEnabled: true, autoEmptyEnabled: true };
  assert.equal(autoTrashDue(DEFAULT_CLEANUP_SETTINGS, NOW), false, 'off by default');
  assert.equal(autoEmptyDue(DEFAULT_CLEANUP_SETTINGS, NOW), false, 'off by default');
  assert.equal(autoTrashDue(on, NOW), true, 'never ran');
  assert.equal(autoEmptyDue(on, NOW), true, 'never ran');
  assert.equal(autoTrashDue({ ...on, lastAutoTrashRun: cutoffIso(0.5, NOW) }, NOW), false, 'ran 12 hours ago');
  assert.equal(autoTrashDue({ ...on, lastAutoTrashRun: cutoffIso(1, NOW) }, NOW), true, 'ran exactly a day ago');
  // One having just run says nothing about the other.
  assert.equal(autoEmptyDue({ ...on, lastAutoTrashRun: NOW.toISOString() }, NOW), true);
});
