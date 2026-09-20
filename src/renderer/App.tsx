import { useEffect, useState } from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import Generate from './pages/Generate';
import Library from './pages/Library';
import { GenerationRecord } from '../shared/types';

const DEFAULT_THEME = 'neon';

export default function App() {
  const [recallRecord, setRecallRecord] = useState<GenerationRecord | null>(null);

  useEffect(() => {
    document.body.className = `${DEFAULT_THEME}-theme`;
  }, []);

  return (
    <div className="app-shell">
      <div className="top-bar">
        <span className="top-bar__title">KVGenius</span>
        <NavLink to="/" end className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Generate
        </NavLink>
        <NavLink to="/library" className={({ isActive }) => `top-bar__link${isActive ? ' active' : ''}`}>
          Library
        </NavLink>
      </div>
      <Routes>
        <Route path="/" element={<Generate recallRecord={recallRecord} onRecalled={() => setRecallRecord(null)} />} />
        <Route path="/library" element={<Library onRecall={setRecallRecord} />} />
      </Routes>
    </div>
  );
}
