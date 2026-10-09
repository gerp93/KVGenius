import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import { GenerationRecord } from '../../shared/types';
import GeneratedVideo from './GeneratedVideo';
import './LibraryPicker.css';

const PAGE_SIZE = 48;

interface Props {
  /** What the picture is for, e.g. "Choose the source image" - shown as the dialog title. */
  title: string;
  /** What to choose from: pictures (the default) or videos (to re-draw one). */
  kind?: 'image' | 'video';
  onPick: (record: GenerationRecord) => void;
  onClose: () => void;
}

/**
 * A dialog to choose one picture from the Library (images only; videos and GIFs cannot be a source - or, with `kind="video"`, one video), newest
 * first, so an image-to-image or a video's source image can be taken from what was already made instead of
 * hunting for the file on disk. Esc or a click outside leaves.
 */
export default function LibraryPicker({ title, kind = 'image', onPick, onClose }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [favoritesOnly, setFavoritesOnly] = useState(false);
  const [loading, setLoading] = useState(true);
  const [more, setMore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function load(beforeId: number | null, replace: boolean) {
    setLoading(true);
    setError(null);
    try {
      const page = await window.kvgenius.listGenerations(kind, PAGE_SIZE, beforeId, favoritesOnly, false);
      setMore(page.length === PAGE_SIZE);
      const usable = kind === 'video' ? page : page.filter((r) => !r.imagePath.toLowerCase().endsWith('.gif'));
      setRecords((prev) => (replace ? usable : [...prev, ...usable]));
      return page;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      return [];
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void load(null, true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [favoritesOnly]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  async function handleMore() {
    // Page by the last item the library returned, not the last one shown (GIFs are left out of the grid).
    const last = records[records.length - 1];
    if (last) await load(last.id, false);
  }

  return createPortal(
    <div className="library-picker" onClick={onClose}>
      <div className="library-picker__dialog" role="dialog" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className="library-picker__head">
          <h3>{title}</h3>
          <label className="library-picker__filter">
            <input type="checkbox" checked={favoritesOnly} onChange={(e) => setFavoritesOnly(e.target.checked)} /> Favorites only
          </label>
          <button type="button" onClick={onClose}>
            ✕
          </button>
        </div>
        {error && <p className="library-picker__note">{error}</p>}
        {!loading && !error && records.length === 0 && (
          <p className="library-picker__note">{kind === 'video' ? (favoritesOnly ? 'No favorite videos yet.' : 'No videos in the Library yet.') : favoritesOnly ? 'No favorite pictures yet.' : 'No pictures in the Library yet.'}</p>
        )}
        <div className="library-picker__grid">
          {records.map((record) => (
            <button
              key={record.id}
              type="button"
              className="library-picker__tile"
              onClick={() => onPick(record)}
              title={record.prompt}
            >
              {kind === 'video' ? (
                <GeneratedVideo src={window.kvgenius.imageUrlFor(record.imagePath)} filePath={record.imagePath} thumbnail />
              ) : (
                <img src={window.kvgenius.thumbUrlFor(record.imagePath)} alt={record.prompt} loading="lazy" decoding="async" />
              )}
            </button>
          ))}
        </div>
        {loading && <p className="library-picker__note">Loading...</p>}
        {!loading && more && (
          <button type="button" className="library-picker__more" onClick={() => void handleMore()}>
            Load more
          </button>
        )}
      </div>
    </div>,
    document.body
  );
}
