/** Voice synthesis API calls (POST /synthesize). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type SynthesisResult = {
  audio_url: string
  watermark_id?: string
}

/**
 * Synthesise speech from text (XTTS v2 + C2PA watermark on the backend).
 * @throws ApiError — notably status 501 when the GPU synthesis stack is
 *         unavailable on the current backend.
 */
export const synthesize = async (text: string): Promise<SynthesisResult> => {
  const res = await apiFetch('/synthesize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<SynthesisResult>
}
