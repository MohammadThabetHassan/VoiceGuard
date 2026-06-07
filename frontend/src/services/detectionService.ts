/** Deepfake detection API calls (POST /detect). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type DetectionResult = {
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string
}

/**
 * Upload an audio file for deepfake detection.
 * @throws ApiError with the HTTP status (e.g. 401 unauthenticated, 503 model
 *         unavailable) and the backend's detail message.
 */
export const detectAudio = async (file: File): Promise<DetectionResult> => {
  const form = new FormData()
  form.append('file', file)

  const res = await apiFetch('/detect', { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<DetectionResult>
}
