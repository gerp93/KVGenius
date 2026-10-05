import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_CLEANUP_SETTINGS,
  MAX_CLEANUP_DAYS,
  autoCleanupDue,
  cutoffIso,
  normalizeCleanupSettings,
  normalizeDays,
  updateCleanupSettings,
} from './cleanup';

const NOW = new Date('2026-10-05T12:00:00.000Z');

test('cleanup starts out manual: the schedule is off by default', () => {
  assert.equal(DEFAULT_CLEANUP_SETTINGS.autoEnabled, false);
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

test('only an explicit true turns the schedule on, and stored dates must be real', () => {
  assert.equal(normalizeCleanupSettings({ autoEnabled: 'yes' }).autoEnabled, false);
  assert.equal(normalizeCleanupSettings({ autoEnabled: 1 }).autoEnabled, false);
  assert.equal(normalizeCleanupSettings({ autoEnabled: true }).autoEnabled, true);
  assert.equal(normalizeCleanupSettings({ lastAutoRun: 'not a date' }).lastAutoRun, null);
  assert.equal(normalizeCleanupSettings({ lastAutoRun: NOW.toISOString() }).lastAutoRun, NOW.toISOString());
});

test('the cutoff is that many days back', () => {
  assert.equal(cutoffIso(30, NOW), '2026-09-05T12:00:00.000Z');
  assert.equal(cutoffIso(1, NOW), '2026-10-04T12:00:00.000Z');
});

test('turning the schedule on starts its clock, so nothing runs by surprise', () => {
  const turnedOn = updateCleanupSettings(DEFAULT_CLEANUP_SETTINGS, { autoEnabled: true }, NOW);
  assert.equal(turnedOn.autoEnabled, true);
  assert.equal(turnedOn.lastAutoRun, NOW.toISOString());
  assert.equal(autoCleanupDue(turnedOn, NOW), false, 'not due until a day has passed');

  // Changing a threshold while it is already on does not restart the clock.
  const later = new Date(NOW.getTime() + 3 * 60 * 60 * 1000);
  const changed = updateCleanupSettings(turnedOn, { olderThanDays: 14 }, later);
  assert.equal(changed.olderThanDays, 14);
  assert.equal(changed.lastAutoRun, NOW.toISOString());

  // Turning it off keeps the thresholds; junk is cleaned up.
  const off = updateCleanupSettings(changed, { autoEnabled: false, trashRetentionDays: 0 }, later);
  assert.equal(off.autoEnabled, false);
  assert.equal(off.trashRetentionDays, 1);
  assert.equal(off.olderThanDays, 14);
});

test('the schedule only runs when on, and not more than once a day', () => {
  const on = { ...DEFAULT_CLEANUP_SETTINGS, autoEnabled: true };
  assert.equal(autoCleanupDue(DEFAULT_CLEANUP_SETTINGS, NOW), false, 'off by default');
  assert.equal(autoCleanupDue({ ...DEFAULT_CLEANUP_SETTINGS, lastAutoRun: null }, NOW), false, 'off stays off');
  assert.equal(autoCleanupDue(on, NOW), true, 'never ran');
  assert.equal(autoCleanupDue({ ...on, lastAutoRun: cutoffIso(0.5, NOW) }, NOW), false, 'ran 12 hours ago');
  assert.equal(autoCleanupDue({ ...on, lastAutoRun: cutoffIso(1, NOW) }, NOW), true, 'ran exactly a day ago');
  assert.equal(autoCleanupDue({ ...on, lastAutoRun: cutoffIso(3, NOW) }, NOW), true);
});
