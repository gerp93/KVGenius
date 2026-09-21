import { defineConfig, Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

/**
 * Content-Security-Policy for the built renderer, injected at build time only (the dev server
 * needs what a useful CSP forbids: react-refresh evaluates code, Vite's HMR opens a websocket).
 * `img-src`/`media-src` allow the `kvimage:` scheme generated images are served over (see the
 * protocol.handle registration in main.ts) - the renderer never talks to ComfyUI directly,
 * that all happens in the main process, so `connect-src` stays 'self' only.
 *
 * `frame-src` allows Hardpoint's loopback UI only (`:3921` packaged; `:5174` when Hardpoint
 * is in Vite dev and redirects). `frame-src 'none'` blanked the Hardpoint tab iframe in
 * packaged builds even when Hardpoint itself was reachable.
 */
const CSP = [
  "default-src 'self'",
  "script-src 'self'",
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' kvimage: data:",
  "media-src 'self' kvimage: http://127.0.0.1:*",
  "font-src 'self' data:",
  "connect-src 'self'",
  "object-src 'none'",
  "frame-src http://127.0.0.1:3921 http://localhost:3921 http://127.0.0.1:5174 http://localhost:5174",
  "base-uri 'none'",
  "form-action 'none'",
].join('; ');

function cspPlugin(): Plugin {
  return {
    name: 'kvgenius-csp',
    apply: 'build',
    transformIndexHtml(html) {
      return html.replace(
        '<head>',
        `<head>\n    <meta http-equiv="Content-Security-Policy" content="${CSP}" />`
      );
    },
  };
}

export default defineConfig({
  plugins: [react(), cspPlugin()],
  base: './',
  build: {
    outDir: 'dist/renderer',
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src/renderer'),
      '@shared': path.resolve(__dirname, './src/shared'),
    },
  },
  server: {
    port: 5173,
  },
});
