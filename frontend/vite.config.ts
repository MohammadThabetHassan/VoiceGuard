import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In production the app is served behind Nginx at the domain root and calls the
// backend at the same-origin "/api/" prefix (Nginx proxies /api/ -> :8000 and
// strips the prefix; /api/ws/ gets WebSocket upgrade headers). In dev, Vite
// reproduces that: it proxies /api -> the local uvicorn on :8000, stripping /api
// so the backend's bare routes (/detect, /token, /ws/stream, ...) are hit —
// ws: true lets the live-mic WebSocket upgrade through the same entry.
export default defineConfig(({ mode }) => ({
  base: '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: mode === 'development',
    rollupOptions: {
      output: {
        // Split heavy vendors into their own chunks for better caching and to
        // keep the main bundle under the size-warning threshold.
        manualChunks: {
          react: ['react', 'react-dom'],
          recharts: ['recharts'],
        },
      },
    },
  },
}))
