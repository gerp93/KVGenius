import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ElementLora,
  LORA_FAMILIES,
  MAX_LORAS,
  parseElementLora,
  parseLoras,
  resolveLoras,
  serializeElementLora,
  serializeLoras,
  validateElementLora,
} from './loras';
import { validateStyleInput } from './styles';

const twilek: ElementLora = { file: 'ffl-twilek.safetensors', strength: 0.8, families: ['flux-dev'] };
const el = (name: string, lora: ElementLora | null) => ({ name, lora });

test('an element LoRA is checked: file required, strength defaulted and bounded, families narrowed to the supported ones', () => {
  assert.deepEqual(validateElementLora(null), { ok: true, value: null });
  assert.deepEqual(validateElementLora({ file: '   ' }), { ok: true, value: null }, 'no file means no LoRA');
  assert.deepEqual(validateElementLora({ file: ' a.safetensors ' }), { ok: true, value: { file: 'a.safetensors', strength: 1, families: [...LORA_FAMILIES] } });
  assert.deepEqual(validateElementLora({ file: 'a', strength: -1.5, families: ['flux-dev', 'nope', 'flux-dev'] }), { ok: true, value: { file: 'a', strength: -1.5, families: ['flux-dev'] } });
  assert.equal(validateElementLora({ file: 'a', strength: 3 }).ok, false);
  assert.equal(validateElementLora({ file: 'a', strength: 'strong' }).ok, false);
  assert.equal(validateElementLora({ file: 'a\nb' }).ok, false);
  assert.equal(validateElementLora('x').ok, false);
});

test('only an element keeps a LoRA when a style input is validated', () => {
  const element = validateStyleInput({ name: 'Twi\'lek', text: 'ffl-twi\'lek-mk.1', kind: 'element', lora: { file: 'a.safetensors', strength: 0.7, families: ['flux-dev'] } });
  assert.ok(element.ok && element.value.lora?.file === 'a.safetensors');
  const style = validateStyleInput({ name: 'Noir', text: 'black and white', kind: 'style', lora: { file: 'a.safetensors', strength: 1, families: ['flux-dev'] } });
  assert.ok(style.ok && style.value.lora === null);
  const untouched = validateStyleInput({ name: 'Twi\'lek', text: 'x', kind: 'element' });
  assert.ok(untouched.ok && !('lora' in untouched.value), 'no lora in the input leaves a saved one alone');
  const bad = validateStyleInput({ name: 'x', text: 'y', kind: 'element', lora: { file: 'a', strength: 99, families: [] } });
  assert.equal(bad.ok, false);
});

test('LoRAs round-trip through their stored form, and nothing unreadable becomes one', () => {
  assert.deepEqual(parseElementLora(serializeElementLora(twilek)), twilek);
  assert.equal(serializeElementLora(null), null);
  assert.equal(parseElementLora('{not json'), null);
  assert.equal(parseElementLora(null), null);

  const list = [{ file: 'a.safetensors', strength: 1 }, { file: 'b.safetensors', strength: 0.5 }];
  assert.deepEqual(parseLoras(serializeLoras(list)), list);
  assert.equal(serializeLoras([]), null);
  assert.equal(serializeLoras(undefined), null);
  assert.equal(parseLoras('[{"file":3}]'), null);
  assert.equal(parseLoras('nope'), null);
  // Equal LoRAs serialize equally (the duplicate guard compares the text), different strengths do not.
  assert.equal(serializeLoras(list), serializeLoras(list.map((l) => ({ ...l }))));
  assert.notEqual(serializeLoras(list), serializeLoras([{ file: 'a.safetensors', strength: 0.9 }, list[1]]));
});

test('the chosen elements bring their LoRAs in the order picked, for a family that can load them', () => {
  const second: ElementLora = { file: 'style.safetensors', strength: 0.6, families: ['flux-dev'] };
  const { applied, notes } = resolveLoras([el('Twi\'lek', twilek), el('Plain', null), el('Look', second)], 'flux-dev');
  assert.deepEqual(applied, [{ file: 'ffl-twilek.safetensors', strength: 0.8 }, { file: 'style.safetensors', strength: 0.6 }]);
  assert.deepEqual(notes, []);
});

test('with a family that cannot load LoRAs they are left out with a note, never an error', () => {
  for (const family of ['z-image', 'qwen-image', 'hidream', 'ovis-image', 'z-image-i2i']) {
    const { applied, notes } = resolveLoras([el('Twi\'lek', twilek)], family);
    assert.deepEqual(applied, [], family);
    assert.equal(notes.length, 1, family);
    assert.match(notes[0], /Twi'lek/);
    assert.match(notes[0], /Flux dev/);
    assert.match(notes[0], /not used/);
  }
  assert.deepEqual(resolveLoras([el('Plain', null)], 'z-image'), { applied: [], notes: [] }, 'an element without a LoRA needs no note');
});

test('an element that names other families is left out for this one', () => {
  const other: ElementLora = { file: 'x.safetensors', strength: 1, families: ['qwen-image'] };
  const { applied, notes } = resolveLoras([el('Other', other)], 'flux-dev');
  assert.deepEqual(applied, []);
  assert.equal(notes.length, 1);
});

test('the same file twice loads once, and no more than the maximum are used', () => {
  const dup = resolveLoras([el('A', twilek), el('B', { ...twilek, strength: 0.3 })], 'flux-dev');
  assert.deepEqual(dup.applied, [{ file: twilek.file, strength: 0.8 }]);
  assert.equal(dup.notes.length, 1);

  const many = Array.from({ length: MAX_LORAS + 1 }, (_, i) => el(`E${i}`, { ...twilek, file: `f${i}.safetensors` }));
  const capped = resolveLoras(many, 'flux-dev');
  assert.equal(capped.applied.length, MAX_LORAS);
  assert.equal(capped.notes.length, 1);
});
