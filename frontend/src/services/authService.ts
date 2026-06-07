/** Authentication API calls (POST /token). */
import { getApiUrl, setToken, ApiError } from '../config/apiConfig'

/**
 * Exchange username/password for a JWT and store it (as `vg_token`).
 * The backend's /token endpoint expects an OAuth2 password form
 * (application/x-www-form-urlencoded).
 * @throws ApiError on bad credentials (401) or other failures.
 */
export const login = async (username: string, password: string): Promise<void> => {
  const body = new URLSearchParams({ username, password })
  const res = await fetch(`${getApiUrl()}/token`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    body,
  })
  if (!res.ok) {
    const data = await res.json().catch(() => ({}))
    throw new ApiError(res.status, data.detail || `Login failed (${res.status})`)
  }
  const data = await res.json()
  if (!data.access_token) throw new ApiError(500, 'No access_token in response')
  setToken(data.access_token)
}
