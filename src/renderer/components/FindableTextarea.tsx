import {
  ChangeEvent,
  CSSProperties,
  Fragment,
  KeyboardEvent,
  TextareaHTMLAttributes,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import { findMatches, matchNearCaret, stepMatch } from '../../shared/textFind';

type Props = Omit<TextareaHTMLAttributes<HTMLTextAreaElement>, 'value' | 'onChange'> & {
  value: string;
  onChange: (value: string) => void;
  /** Style for the box around the textarea (the textarea's own `style` is untouched). */
  boxStyle?: CSSProperties;
};

/** Text styles the highlight layer copies from the textarea so both wrap the words identically. */
const MIRRORED_STYLES = [
  'fontFamily',
  'fontSize',
  'fontWeight',
  'fontStyle',
  'lineHeight',
  'letterSpacing',
  'wordSpacing',
  'textIndent',
  'textTransform',
  'tabSize',
  'paddingTop',
  'paddingLeft',
] as const;

/** A textarea you can search: Ctrl+F (or the magnifier) opens a find bar over it that highlights
 * every match and steps through them with Enter / Shift+Enter. Nothing here changes the text. */
export default function FindableTextarea({ value, onChange, boxStyle, onKeyDown, ...rest }: Props) {
  const areaRef = useRef<HTMLTextAreaElement>(null);
  const layerRef = useRef<HTMLDivElement>(null);
  const currentMarkRef = useRef<HTMLElement>(null);
  const queryRef = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState('');
  const [chosen, setChosen] = useState(0);
  // Bumped each time the bar is asked for, so its input takes focus once it exists.
  const [focusRequest, setFocusRequest] = useState(0);

  const matches = useMemo(() => (open ? findMatches(value, query) : []), [open, value, query]);
  const current = matches.length === 0 ? -1 : Math.min(chosen, matches.length - 1);

  function openBar() {
    const area = areaRef.current;
    if (area && !open) {
      const picked = area.value.slice(area.selectionStart, area.selectionEnd);
      // Search for the words the person has selected, when that is a single line.
      if (picked !== '' && !picked.includes('\n')) {
        setQuery(picked);
        setChosen(matchNearCaret(findMatches(area.value, picked), area.selectionStart));
      } else {
        setChosen(matchNearCaret(findMatches(area.value, query), area.selectionStart));
      }
    }
    setOpen(true);
    setFocusRequest((n) => n + 1);
  }

  /** Closes the bar and leaves the text selected at the current match, ready to edit. */
  function closeBar() {
    const area = areaRef.current;
    const match = current >= 0 ? matches[current] : null;
    setOpen(false);
    if (!area) return;
    area.focus({ preventScroll: true });
    if (match) area.setSelectionRange(match.start, match.end);
  }

  function step(direction: 1 | -1) {
    setChosen(stepMatch(current, matches.length, direction));
  }

  function changeQuery(next: string) {
    setQuery(next);
    const caret = areaRef.current?.selectionStart ?? 0;
    setChosen(matchNearCaret(findMatches(value, next), caret));
  }

  function handleAreaKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    onKeyDown?.(event);
    if (event.defaultPrevented) return;
    if ((event.ctrlKey || event.metaKey) && !event.altKey && event.key.toLowerCase() === 'f') {
      event.preventDefault();
      openBar();
    } else if (event.key === 'F3' && open && query !== '') {
      event.preventDefault();
      step(event.shiftKey ? -1 : 1);
    }
  }

  function handleQueryKeyDown(event: KeyboardEvent<HTMLInputElement>) {
    if (event.key === 'Enter' || event.key === 'F3') {
      event.preventDefault();
      step(event.shiftKey ? -1 : 1);
    } else if (event.key === 'Escape') {
      event.preventDefault();
      event.stopPropagation();
      closeBar();
    } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'f') {
      event.preventDefault();
      queryRef.current?.select();
    }
  }

  useEffect(() => {
    if (focusRequest === 0) return;
    queryRef.current?.focus();
    queryRef.current?.select();
  }, [focusRequest]);

  // Lay the highlight layer exactly over the text: same type, same padding, and room for the
  // textarea's scrollbar so long lines wrap in the same places.
  useLayoutEffect(() => {
    const area = areaRef.current;
    const layer = layerRef.current;
    if (!open || !area || !layer) return;
    function fit() {
      if (!area || !layer) return;
      const computed = window.getComputedStyle(area);
      for (const name of MIRRORED_STYLES) layer.style[name] = computed[name];
      const scrollbar = area.offsetWidth - area.clientWidth - parseFloat(computed.borderLeftWidth) - parseFloat(computed.borderRightWidth);
      layer.style.paddingRight = `${parseFloat(computed.paddingRight) + Math.max(0, scrollbar)}px`;
      layer.scrollTop = area.scrollTop;
    }
    fit();
    const observer = new ResizeObserver(fit);
    observer.observe(area);
    return () => observer.disconnect();
  }, [open, value, matches.length]);

  // Bring the chosen match into view, scrolling the textarea only when it is out of sight.
  useEffect(() => {
    const area = areaRef.current;
    const mark = currentMarkRef.current;
    if (!open || !area || !mark) return;
    const top = mark.offsetTop;
    const bottom = top + mark.offsetHeight;
    if (top < area.scrollTop || bottom > area.scrollTop + area.clientHeight) {
      area.scrollTop = Math.max(0, top - area.clientHeight / 2);
      // The scroll event that would move the highlights along arrives a frame later; do it now.
      if (layerRef.current) layerRef.current.scrollTop = area.scrollTop;
    }
  }, [open, current, matches]);

  let layerContent: JSX.Element | null = null;
  if (open) {
    const pieces: JSX.Element[] = [];
    let cursor = 0;
    matches.forEach((match, index) => {
      pieces.push(
        <Fragment key={index}>
          {value.slice(cursor, match.start)}
          <mark
            ref={index === current ? currentMarkRef : undefined}
            className={index === current ? 'findable__mark findable__mark--current' : 'findable__mark'}
          >
            {value.slice(match.start, match.end)}
          </mark>
        </Fragment>
      );
      cursor = match.end;
    });
    // The extra space keeps a trailing newline from collapsing, so the layer ends where the text does.
    layerContent = (
      <>
        {pieces}
        {value.slice(cursor)}
        {'\n '}
      </>
    );
  }

  return (
    <div className={open ? 'findable findable--open' : 'findable'} style={boxStyle}>
      {open && (
        <div ref={layerRef} className="findable__layer" aria-hidden>
          {layerContent}
        </div>
      )}
      <textarea
        {...rest}
        ref={areaRef}
        value={value}
        onChange={(e: ChangeEvent<HTMLTextAreaElement>) => onChange(e.target.value)}
        onKeyDown={handleAreaKeyDown}
        onScroll={(e) => {
          if (layerRef.current) layerRef.current.scrollTop = e.currentTarget.scrollTop;
          rest.onScroll?.(e);
        }}
      />
      {!open && (
        <button
          type="button"
          className="findable__open"
          title="Find in this text (Ctrl+F)"
          aria-label="Find in this text"
          onClick={openBar}
        >
          🔍
        </button>
      )}
      {open && (
        <div className="findable__bar" role="search">
          <input
            ref={queryRef}
            type="text"
            value={query}
            placeholder="Find in text"
            aria-label="Find in text"
            onChange={(e) => changeQuery(e.target.value)}
            onKeyDown={handleQueryKeyDown}
          />
          <span className="findable__count" aria-live="polite">
            {query === '' ? '' : matches.length === 0 ? 'No matches' : `${current + 1} of ${matches.length}`}
          </span>
          <button type="button" title="Previous match (Shift+Enter)" aria-label="Previous match" disabled={matches.length === 0} onClick={() => step(-1)}>
            ▲
          </button>
          <button type="button" title="Next match (Enter)" aria-label="Next match" disabled={matches.length === 0} onClick={() => step(1)}>
            ▼
          </button>
          <button type="button" title="Close (Esc)" aria-label="Close find" onClick={closeBar}>
            ✕
          </button>
        </div>
      )}
    </div>
  );
}
