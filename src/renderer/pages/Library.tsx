import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GenerationRecord } from '../../shared/types';

interface Props {
  onRecall: (record: GenerationRecord) => void;
}

export default function Library({ onRecall }: Props) {
  const [records, setRecords] = useState<GenerationRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    window.kvgenius
      .listGenerations()
      .then(setRecords)
      .catch((err) => setError(err instanceof Error ? err.message : String(err)));
  }, []);

  function handleClick(record: GenerationRecord) {
    onRecall(record);
    navigate('/');
  }

  return (
    <div className="page">
      {error && <p style={{ color: 'var(--color-accent-red)' }}>{error}</p>}
      {records.length === 0 && !error && (
        <p style={{ color: 'var(--color-text-muted)' }}>No generations yet - go make something.</p>
      )}
      <div className="library-grid">
        {records.map((record) => (
          <div key={record.id} className="library-card" onClick={() => handleClick(record)}>
            <img src={window.kvgenius.imageUrlFor(record.imagePath)} alt={record.prompt} />
            <div className="library-card__info" title={record.prompt}>
              {record.prompt}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
