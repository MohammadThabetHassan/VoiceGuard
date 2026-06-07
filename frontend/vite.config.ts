import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In production the app is served behind Nginx at the domain root and calls the
// backend at the same-origin "/api/" prefix (Nginx proxies /api/ -> :8000 and
// strips the prefix). In dev, Vite reproduces that: it proxies /api -> the local
// uvicorn on :8000, stripping /api so the backend's bare routes (/detect, /token,
// ...) are hit, and proxies /ws for the live-mic WebSocket.
export default defineConfig(({ mode }) => ({
  base: '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ''),
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
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
