import { test } from 'node:test';
import assert from 'node:assert/strict';
import { initDatabase } from './db';
import { deletePromptTemplate, getPromptTemplate, listPromptTemplates, savePromptTemplate } from './promptTemplates';

const sections = [{ label: 'Subject' }, { label: 'Outfit', hint: 'what they wear' }];

test('templates are created, listed alphabetically, edited and deleted', () => {
  const db = initDatabase(':memory:');
  assert.deepEqual(listPromptTemplates(db), []);
  const zed = savePromptTemplate(db, { name: ' Zed ', sections });
  const alpha = savePromptTemplate(db, { name: 'alpha', sections: [{ label: 'One' }], separator: 'newline' });
  assert.equal(zed.name, 'Zed');
  assert.equal(zed.separator, 'comma');
  assert.deepEqual(zed.sections.map((s) => [s.label, s.hint]), [['Subject', ''], ['Outfit', 'what they wear']]);
  assert.deepEqual(listPromptTemplates(db).map((t) => t.name), ['alpha', 'Zed']);

  // Editing keeps section ids, so what was typed into a section stays with it.
  const edited = savePromptTemplate(db, { name: 'Zed', sections: [{ ...zed.sections[1], label: 'Clothes' }, zed.sections[0]], separator: 'space' }, zed.id);
  assert.deepEqual(edited.sections.map((s) => s.id), [zed.sections[1].id, zed.sections[0].id]);
  assert.equal(getPromptTemplate(db, zed.id)?.separator, 'space');

  deletePromptTemplate(db, alpha.id);
  assert.equal(getPromptTemplate(db, alpha.id), null);
  assert.equal(listPromptTemplates(db).length, 1);
});

test('template names are unique ignoring case, and invalid input is explained', () => {
  const db = initDatabase(':memory:');
  savePromptTemplate(db, { name: 'Portrait', sections });
  assert.throws(() => savePromptTemplate(db, { name: 'portrait', sections }), /already exists/);
  assert.throws(() => savePromptTemplate(db, { name: 'X', sections: [] }), /at least one section/);
  assert.throws(() => savePromptTemplate(db, { name: 'X', sections }, 999), /no longer exists/);
});
