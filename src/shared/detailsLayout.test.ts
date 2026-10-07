import assert from 'node:assert/strict';
import { test } from 'node:test';
import { shouldSplitDetails } from './detailsLayout';

test('a panel too narrow for a picture column stays stacked', () => {
  assert.equal(shouldSplitDetails({ panelWidth: 400, panelHeight: 900, aspect: 0.7 }), false);
  assert.equal(shouldSplitDetails({ panelWidth: 600, panelHeight: 900, aspect: 0.7 }), false);
});

test('nothing measured yet stays stacked', () => {
  assert.equal(shouldSplitDetails({ panelWidth: 1200, panelHeight: 0, aspect: 0.7 }), false);
  assert.equal(shouldSplitDetails({ panelWidth: 1200, panelHeight: 900, aspect: 0 }), false);
});

test('a big panel splits for a tall picture, which gains the full height', () => {
  // 3440 wide: a 1278 x 1100 panel; the stacked picture would only get ~620px of height.
  assert.equal(shouldSplitDetails({ panelWidth: 1278, panelHeight: 1100, aspect: 0.7 }), true);
  assert.equal(shouldSplitDetails({ panelWidth: 1278, panelHeight: 1100, aspect: 1 }), true);
});

test('a wide picture stays stacked where stacking shows it larger', () => {
  assert.equal(shouldSplitDetails({ panelWidth: 1000, panelHeight: 900, aspect: 1.8 }), false);
});

test('a mid-size panel stays stacked when the split would not show the picture larger', () => {
  // 2560 wide: a 892 x 1300 panel - the split column (508px) is narrower than the stacked picture.
  assert.equal(shouldSplitDetails({ panelWidth: 892, panelHeight: 1300, aspect: 0.7 }), false);
});
