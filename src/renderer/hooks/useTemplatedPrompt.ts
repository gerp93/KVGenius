import { useEffect, useRef, useState } from 'react';
import { PromptTemplate, TemplateValues, joinSections, valuesFromPrompt } from '../../shared/promptTemplates';

interface Args {
  /** Bumped when a template is added, edited or deleted, so the list reloads. */
  templatesVersion: number;
  activeSlotId: string;
  prompt: string;
  setPrompt: (prompt: string) => void;
  templateId: number | null;
  setTemplateId: (id: number | null) => void;
  values: TemplateValues;
  setValues: (values: TemplateValues) => void;
}

/**
 * Lets the active tab write its prompt in a template's sections instead of the one big box. The tab's `prompt` stays the real thing - the
 * sections joined into one ordinary prompt - so everything else (the queue, the Library, Re-rack) never knows a template was used.
 *
 * Whenever the prompt changes some other way (a recall, a Library pick), the sections no longer describe it, so the tab simply goes back to the single
 * box with that prompt in it. A template that is edited or deleted meanwhile is followed (the prompt is re-joined) or let go of.
 */
export function useTemplatedPrompt({ templatesVersion, activeSlotId, prompt, setPrompt, templateId, setTemplateId, values, setValues }: Args) {
  const [templates, setTemplates] = useState<PromptTemplate[] | null>(null);
  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .listPromptTemplates()
      .then((list) => {
        if (!cancelled) setTemplates(list);
      })
      .catch(() => {
        if (!cancelled) setTemplates([]);
      });
    return () => {
      cancelled = true;
    };
  }, [templatesVersion]);

  const template = templateId === null ? null : (templates?.find((t) => t.id === templateId) ?? null);

  // The last prompt this hook can account for (one it wrote, or the one a tab came with). A different prompt was put there by something else.
  const accountedFor = useRef(prompt);
  useEffect(() => {
    accountedFor.current = prompt;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeSlotId]);

  useEffect(() => {
    if (templates === null || templateId === null) return;
    const release = () => {
      setTemplateId(null);
      setValues({});
    };
    if (!template) return release(); // deleted
    const joined = joinSections(template, values);
    if (prompt === joined) {
      accountedFor.current = joined;
    } else if (prompt === accountedFor.current) {
      // The template was edited (or this tab was saved under an older version of it): write the prompt the sections make now.
      accountedFor.current = joined;
      setPrompt(joined);
    } else {
      release();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [templates, templateId, template, values, prompt]);

  function write(next: TemplateValues, using: PromptTemplate) {
    const joined = joinSections(using, next);
    accountedFor.current = joined;
    setValues(next);
    setPrompt(joined);
  }

  /** Picks a template (anything already in the prompt box moves into its first section), or null for the single box (the prompt stays as it is). */
  function pickTemplate(id: number | null) {
    const picked = id === null ? null : (templates?.find((t) => t.id === id) ?? null);
    if (!picked) {
      setTemplateId(null);
      setValues({});
      return;
    }
    setTemplateId(picked.id);
    write(valuesFromPrompt(picked, prompt), picked);
  }

  function setSection(sectionId: string, text: string) {
    if (template) write({ ...values, [sectionId]: text }, template);
  }

  return { templates: templates ?? [], template, pickTemplate, setSection };
}
