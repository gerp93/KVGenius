import React from 'react';
import ReactDOM from 'react-dom/client';
import { HashRouter } from 'react-router-dom';
import App from './App';
import './themes.css';
import './index.css';

// A file dropped anywhere outside a drop zone would make Electron navigate the window to it. Zones handle
// their own drops (and stop them here); this swallows the rest.
for (const type of ['dragover', 'drop'] as const) {
  window.addEventListener(type, (event) => {
    if (event.dataTransfer && Array.from(event.dataTransfer.types).includes('Files')) event.preventDefault();
  });
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <HashRouter>
      <App />
    </HashRouter>
  </React.StrictMode>
);
