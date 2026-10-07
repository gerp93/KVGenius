import { ReactNode } from 'react';

interface Props {
  title: string;
  /** One or two sentences on what the section controls. Longer detail belongs in `children`. */
  description?: ReactNode;
  children: ReactNode;
}

/** A titled card within a Settings tab. */
export default function SettingsSection({ title, description, children }: Props) {
  return (
    <section className="settings-section">
      <h3 className="settings-section__title">{title}</h3>
      {description && <p className="settings-hint">{description}</p>}
      {children}
    </section>
  );
}
