import { ReactNode, useState } from 'react';
import { Link } from 'react-router-dom';
import './Stepper.css';

export interface StepLink {
  label: string;
  /** An in-app route... */
  to?: string;
  /** ...or a web address, opened in the default browser. */
  href?: string;
}

export interface Step {
  title: string;
  /** Small label above the title ("Step 1", "Optional"). */
  kicker?: string;
  body: ReactNode;
  links?: StepLink[];
}

interface Props {
  title: string;
  subtitle?: string;
  steps: Step[];
  finishLabel: string;
  onFinish: () => void;
}

/** A link that opens in the system browser (the app window itself never navigates away). */
export function ExternalLink({ href, children, className }: { href: string; children: ReactNode; className?: string }) {
  return (
    <a
      href={href}
      className={className}
      onClick={(event) => {
        event.preventDefault();
        void window.kvgenius.openExternal(href).catch((err) => console.error('Could not open the link', err));
      }}
    >
      {children}
    </a>
  );
}

/** A guide shown one step at a time: progress bar, clickable dots, Back / Next, and a primary
 * action on the last step. Content is plain data (`Step[]`) so a guide stays easy to edit. */
export default function Stepper({ title, subtitle, steps, finishLabel, onFinish }: Props) {
  const [index, setIndex] = useState(0);
  const step = steps[index];
  const isFirst = index === 0;
  const isLast = index === steps.length - 1;

  return (
    <div className="stepper">
      <header>
        <h2 className="stepper__title">{title}</h2>
        {subtitle && <p className="stepper__subtitle">{subtitle}</p>}
      </header>

      <div className="stepper__progress" aria-live="polite">
        <span className="stepper__count">
          Step {index + 1} of {steps.length}
        </span>
        <div className="stepper__track" aria-hidden>
          <div className="stepper__fill" style={{ width: `${((index + 1) / steps.length) * 100}%` }} />
        </div>
      </div>

      <nav className="stepper__dots" aria-label="Setup steps">
        {steps.map((s, i) => (
          <button
            key={s.title}
            type="button"
            className={`stepper__dot${i === index ? ' active' : ''}${i < index ? ' done' : ''}`}
            aria-label={`Step ${i + 1}: ${s.title}`}
            aria-current={i === index ? 'step' : undefined}
            title={s.title}
            onClick={() => setIndex(i)}
          />
        ))}
      </nav>

      <article className="stepper__card">
        {step.kicker && <p className="stepper__kicker">{step.kicker}</p>}
        <h3 className="stepper__step-title">{step.title}</h3>
        <div className="stepper__body">{step.body}</div>
        {step.links && step.links.length > 0 && (
          <footer className="stepper__links">
            {step.links.map((link) =>
              link.to ? (
                <Link key={link.label} to={link.to} className="stepper__pill stepper__pill--primary">
                  {link.label}
                </Link>
              ) : link.href ? (
                <ExternalLink key={link.label} href={link.href} className="stepper__pill">
                  {link.label} ↗
                </ExternalLink>
              ) : null,
            )}
          </footer>
        )}
      </article>

      <div className={`stepper__nav${isFirst ? ' stepper__nav--forward-only' : ''}`}>
        {!isFirst && (
          <button type="button" onClick={() => setIndex(index - 1)}>
            Back
          </button>
        )}
        {isLast ? (
          <button type="button" className="primary" onClick={onFinish}>
            {finishLabel}
          </button>
        ) : (
          <button type="button" className="primary" onClick={() => setIndex(index + 1)}>
            Next
          </button>
        )}
      </div>
    </div>
  );
}
