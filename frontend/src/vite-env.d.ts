/// <reference types="vite/client" />

interface ImportMetaEnv {
  // Same-origin API base path (default "/api"); set by the build (setup.sh).
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
