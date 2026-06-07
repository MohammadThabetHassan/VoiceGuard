import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // GitHub Pages serves project sites under /<repo>/ (case-sensitive). The repo
  // is "VoiceGuard"; the deploy workflow sets VITE_BASE_PATH explicitly. For
  // local dev the default '/' is used.
  base: process.env.VITE_BASE_PATH || '/',
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/detect': 'http://localhost:8000',
      '/synthesize': 'http://localhost:8000',
      '/forensic': 'http://localhost:8000',
      '/token': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
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
})
