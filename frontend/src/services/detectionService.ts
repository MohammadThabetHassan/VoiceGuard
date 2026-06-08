/** Deepfake detection API calls (POST /detect). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type AttributionSegment = {
  start_s: number
  end_s: number
  importance: number
}

export type Explanation = {
  method: string
  baseline: string
  target_class: number
  frame_duration_ms: number
  attribution_frames: number[]
  top_segments: AttributionSegment[]
}

export type DetectionResult = {
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string
  explanation?: Explanation | null
}

/**
 * Upload an audio file for deepfake detection.
 * @param explain request Integrated-Gradients attribution (slower).
 * @throws ApiError with the HTTP status (401 unauthenticated, 422 unusable
 *         audio, 503 model unavailable) and the backend's detail message.
 */
export const detectAudio = async (file: File, explain = false): Promise<DetectionResult> => {
  const form = new FormData()
  form.append('file', file)

  const res = await apiFetch(`/detect?explain=${explain ? 'true' : 'false'}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<DetectionResult>
}
