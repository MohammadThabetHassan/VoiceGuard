/**
 * API configuration.
 *
 * The frontend is served behind Nginx at the same origin as the backend, which
 * proxies the "/api/" prefix to FastAPI. The base is therefore fixed (no runtime
 * URL picker): "/api" in production, overridable at build time via VITE_API_BASE.
 * In dev, Vite's proxy maps /api -> http://localhost:8000.
 */

/** API base path (no trailing slash). e.g. "/api". */
export const API_BASE = ((import.meta.env.VITE_API_BASE as string) || '/api').replace(/\/+$/, '')

const TOKEN_KEY = 'vg_token'

/**
 * WebSocket URL for a backend path, derived from the current page origin so it
 * auto-switches between ws:// and wss://. Pass e.g. "/ws/stream".
 */
export const getWsUrl = (path: string): string => {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws'
  return `${proto}://${window.location.host}${path}`
}

/** JWT bearer token, if the user has authenticated. */
export const getToken = (): string => localStorage.getItem(TOKEN_KEY) || ''

/** Persist a JWT (from POST /api/token). */
export const setToken = (token: string): void => localStorage.setItem(TOKEN_KEY, token)

/** Remove the stored JWT (log out). */
export const clearToken = (): void => localStorage.removeItem(TOKEN_KEY)

/** Whether a JWT is currently stored. */
export const hasToken = (): boolean => getToken().length > 0

/** Error carrying the HTTP status so callers can special-case 401/501/etc. */
export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/**
 * fetch wrapper that prepends the API base and attaches the bearer token.
 * `path` must start with '/' (e.g. '/detect'). Non-2xx responses raise ApiError
 * with the backend's `detail` message; network failures raise the underlying
 * TypeError.
 */
export const apiFetch = (path: string, init: RequestInit = {}): Promise<Response> => {
  const token = getToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${API_BASE}${path}`, { ...init, headers })
}
