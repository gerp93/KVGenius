import { useEffect, useState } from 'react';
import { GenerationKind } from '../../shared/types';
import { MAX_PROMPT_SLOTS, PromptSlot, PromptSlotData, newPromptSlot, slotLabel } from '../../shared/promptSlots';

/** How long to wait after the last edit before writing the tabs to disk - typing shouldn't hit
 * the filesystem on every keystroke. */
const PERSIST_DEBOUNCE_MS = 600;

/**
 * The Generate page's prompt "tabs": each holds a full copy of the left-hand form (prompt, mode,
 * size, seed, steps, CFG, video length, source image, batch size), switchable and persisted
 * across restarts. Mirrors the individual useState calls Generate.tsx used to own directly, so it
 * can be dropped in with the same field names.
 */
export function usePromptSlots() {
  const [loaded, setLoaded] = useState(false);
  const [slots, setSlots] = useState<PromptSlot[]>([]);
  const [activeSlotId, setActiveSlotId] = useState<string>('');

  const [mode, setMode] = useState<GenerationKind>('image');
  const [prompt, setPrompt] = useState('');
  const [width, setWidth] = useState(1024);
  const [height, setHeight] = useState(1024);
  const [seed, setSeed] = useState(0);
  const [seedLocked, setSeedLocked] = useState(false);
  const [steps, setSteps] = useState(8);
  const [cfg, setCfg] = useState(1);
  const [lengthSeconds, setLengthSeconds] = useState(5);
  const [sourceImagePath, setSourceImagePath] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [customSize, setCustomSize] = useState(false);
  const [batchSize, setBatchSize] = useState(1);
  const [lastRunSignature, setLastRunSignature] = useState<string | null>(null);

  function applySnapshot(data: PromptSlotData) {
    setMode(data.mode);
    setPrompt(data.prompt);
    setWidth(data.width);
    setHeight(data.height);
    setSeed(data.seed);
    setSeedLocked(data.seedLocked);
    setSteps(data.steps);
    setCfg(data.cfg);
    setLengthSeconds(data.lengthSeconds);
    setSourceImagePath(data.sourceImagePath);
    setAdvancedOpen(data.advancedOpen);
    setCustomSize(data.customSize);
    setBatchSize(data.batchSize);
    setLastRunSignature(data.lastRunSignature ?? null);
  }

  // Load once on mount (Generate stays mounted for the whole session, so this never re-runs).
  useEffect(() => {
    let cancelled = false;
    window.kvgenius
      .getPromptSlots()
      .then(({ slots: loadedSlots, activeId }) => {
        if (cancelled) return;
        const list = loadedSlots.length > 0 ? loadedSlots : [newPromptSlot()];
        const active = list.find((s) => s.id === activeId) ?? list[0];
        setSlots(list);
        setActiveSlotId(active.id);
        applySnapshot(active.data);
        setLoaded(true);
      })
      .catch(() => {
        if (cancelled) return;
        const fresh = newPromptSlot();
        setSlots([fresh]);
        setActiveSlotId(fresh.id);
        applySnapshot(fresh.data);
        setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function snapshotFromState(): PromptSlotData {
    return {
      mode,
      prompt,
      width,
      height,
      seed,
      seedLocked,
      steps,
      cfg,
      lengthSeconds,
      sourceImagePath,
      advancedOpen,
      customSize,
      batchSize,
      lastRunSignature,
    };
  }

  // Autosave: whenever a field of the active tab changes, write the whole set back to disk after
  // a short pause. Uses the functional form of setSlots so a burst of edits (or React 18 Strict
  // Mode's double-invoke in dev) can't clobber a concurrent add/close/switch with a stale array.
  useEffect(() => {
    if (!loaded) return;
    const timer = setTimeout(() => {
      setSlots((prev) => {
        const merged = prev.map((s) => (s.id === activeSlotId ? { ...s, data: snapshotFromState() } : s));
        void window.kvgenius.savePromptSlots(merged, activeSlotId);
        return merged;
      });
    }, PERSIST_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    loaded,
    activeSlotId,
    mode,
    prompt,
    width,
    height,
    seed,
    seedLocked,
    steps,
    cfg,
    lengthSeconds,
    sourceImagePath,
    advancedOpen,
    customSize,
    batchSize,
    lastRunSignature,
  ]);

  function switchTo(id: string) {
    if (id === activeSlotId) return;
    const target = slots.find((s) => s.id === id);
    if (!target) return;
    const next = slots.map((s) => (s.id === activeSlotId ? { ...s, data: snapshotFromState() } : s));
    setSlots(next);
    setActiveSlotId(id);
    applySnapshot(target.data);
    void window.kvgenius.savePromptSlots(next, id);
  }

  function addSlot() {
    if (slots.length >= MAX_PROMPT_SLOTS) return;
    const fresh = newPromptSlot('image');
    const next = slots.map((s) => (s.id === activeSlotId ? { ...s, data: snapshotFromState() } : s)).concat(fresh);
    setSlots(next);
    setActiveSlotId(fresh.id);
    applySnapshot(fresh.data);
    void window.kvgenius.savePromptSlots(next, fresh.id);
  }

  function closeSlot(id: string) {
    if (slots.length <= 1) return;
    const merged = slots.map((s) => (s.id === activeSlotId ? { ...s, data: snapshotFromState() } : s));
    const closingIndex = merged.findIndex((s) => s.id === id);
    const remaining = merged.filter((s) => s.id !== id);
    let nextActiveId = activeSlotId;
    if (id === activeSlotId) {
      const neighbor = remaining[Math.max(0, closingIndex - 1)] ?? remaining[0];
      nextActiveId = neighbor.id;
      applySnapshot(neighbor.data);
    }
    setSlots(remaining);
    setActiveSlotId(nextActiveId);
    void window.kvgenius.savePromptSlots(remaining, nextActiveId);
  }

  function renameSlot(id: string, rawName: string) {
    const trimmed = rawName.trim().slice(0, 40);
    const name = trimmed === '' ? null : trimmed;
    const withSnapshot = slots.map((s) => (s.id === activeSlotId ? { ...s, data: snapshotFromState() } : s));
    const next = withSnapshot.map((s) => (s.id === id ? { ...s, name } : s));
    setSlots(next);
    void window.kvgenius.savePromptSlots(next, activeSlotId);
  }

  function labelFor(slot: PromptSlot): string {
    return slotLabel(slot, slot.id === activeSlotId ? prompt : slot.data.prompt);
  }

  return {
    loaded,
    slots,
    activeSlotId,
    switchTo,
    addSlot,
    closeSlot,
    renameSlot,
    labelFor,

    mode,
    setMode,
    prompt,
    setPrompt,
    width,
    setWidth,
    height,
    setHeight,
    seed,
    setSeed,
    seedLocked,
    setSeedLocked,
    steps,
    setSteps,
    cfg,
    setCfg,
    lengthSeconds,
    setLengthSeconds,
    sourceImagePath,
    setSourceImagePath,
    advancedOpen,
    setAdvancedOpen,
    customSize,
    setCustomSize,
    batchSize,
    setBatchSize,
    lastRunSignature,
    setLastRunSignature,
  };
}
