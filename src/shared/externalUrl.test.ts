import { test } from 'node:test';
import assert from 'node:assert/strict';
import { isExternalWebUrl } from './externalUrl';

test('web links may be opened in the browser', () => {
  assert.equal(isExternalWebUrl('https://docs.comfy.org/installation/desktop'), true);
  assert.equal(isExternalWebUrl('http://localhost:8000'), true);
});

test('anything that is not a web link is refused', () => {
  assert.equal(isExternalWebUrl('file:///etc/passwd'), false);
  assert.equal(isExternalWebUrl('javascript:alert(1)'), false);
  assert.equal(isExternalWebUrl('kvimage://x'), false);
  assert.equal(isExternalWebUrl('not a url'), false);
  assert.equal(isExternalWebUrl(undefined), false);
  assert.equal(isExternalWebUrl(42), false);
});
