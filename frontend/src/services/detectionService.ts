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
  narrative?: string | null
}

export type DetectionResult = {
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string
  explanation?: Explanation | null
}

export type ModelInfo = {
  key: string
  available: boolean
}

/** Human-readable labels for the detector model keys. */
export const MODEL_LABELS: Record<string, string> = {
  xls_r_aasist: 'XLS-R-300M + AASIST "v9c" (flagship)',
  wav2vec2_spoof: 'wav2vec2-large (anti-spoofing)',
  classical: 'Classical (SM2026 XGBoost)',
  xls_r: 'XLS-R-300M',
  wav2vec2_large: 'wav2vec2-large',
  wav2vec2: 'wav2vec2-base',
  wavlm_large: 'WavLM-large',
  wavlm_base_plus: 'WavLM-base+',
  aasist: 'AASIST',
  dsfnet_v2: 'DSFNet v2',
  dsfnet: 'DSFNet',
}

/** Fetch the detector registry and its per-model checkpoint availability. */
export const getModels = async (): Promise<ModelInfo[]> => {
  const res = await apiFetch('/models')
  if (!res.ok) throw new ApiError(res.status, `Error ${res.status}`)
  const raw = (await res.json()) as Record<string, { available: boolean }>
  return Object.entries(raw).map(([key, v]) => ({ key, available: !!v.available }))
}

/**
 * Upload an audio file for deepfake detection.
 * @param explain request attribution (slower).
 * @param model detector model key (default 'classical').
 * @throws ApiError with the HTTP status (401 unauthenticated, 422 unusable
 *         audio, 503 model unavailable) and the backend's detail message.
 */
export const detectAudio = async (
  file: File,
  explain = false,
  model = 'classical',
): Promise<DetectionResult> => {
  const form = new FormData()
  form.append('file', file)

  const res = await apiFetch(
    `/detect?explain=${explain ? 'true' : 'false'}&model=${encodeURIComponent(model)}`,
    {
      method: 'POST',
      body: form,
    },
  )
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<DetectionResult>
}
