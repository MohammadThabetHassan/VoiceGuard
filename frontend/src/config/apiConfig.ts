/**
 * Runtime API configuration.
 *
 * The frontend is hosted statically (GitHub Pages) and cannot know the user's
 * local backend URL at build time. The URL is therefore resolved at runtime:
 *   localStorage  ->  VITE_API_URL (build-time env)  ->  http://localhost:8000
 *
 * The user sets/overrides it through the ApiSettings panel (gear button).
 */

const DEFAULT_API = (import.meta.env.VITE_API_URL as string) || 'http://localhost:8000'

const URL_KEY = 'voiceguard_api_url'
const TOKEN_KEY = 'vg_token'

/** Current API base URL (no trailing slash). */
export const getApiUrl = (): string => {
  const stored = localStorage.getItem(URL_KEY)
  return (stored && stored.length > 0 ? stored : DEFAULT_API).replace(/\/+$/, '')
}

/** Persist a new API base URL (trailing slashes stripped). */
export const setApiUrl = (url: string): void => {
  localStorage.setItem(URL_KEY, url.trim().replace(/\/+$/, ''))
}

/** WebSocket base URL derived from the API URL (http->ws, https->wss). */
export const getWsUrl = (): string => getApiUrl().replace(/^http/, 'ws')

/** JWT bearer token, if the user has authenticated. */
export const getToken = (): string => localStorage.getItem(TOKEN_KEY) || ''

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
 * fetch wrapper that prepends the runtime API URL and attaches the bearer
 * token. `path` must start with '/'. Non-2xx responses raise ApiError with the
 * backend's `detail` message; network failures raise the underlying TypeError.
 */
export const apiFetch = (path: string, init: RequestInit = {}): Promise<Response> => {
  const token = getToken()
  const headers = new Headers(init.headers)
  if (token) headers.set('Authorization', `Bearer ${token}`)
  return fetch(`${getApiUrl()}${path}`, { ...init, headers })
}
