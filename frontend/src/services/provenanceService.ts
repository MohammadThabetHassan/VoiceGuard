/** Provenance verification API calls (POST /watermark/verify). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type ProvenanceVerdict =
  | 'voiceguard-generated'
  | 'ai-generated'
  | 'no-provenance-found'
  | 'unknown'

export type WatermarkVerifyResult = {
  spectral_checked: boolean
  spectral_detected: boolean
  spectral_correlation: number | null
  c2pa_has_manifest: boolean
  c2pa_validation_state: string | null
  c2pa_ai_generated: boolean | null
  c2pa_software_agent: string | null
  verdict: ProvenanceVerdict
}

/**
 * Verify whether an audio file carries VoiceGuard provenance marks: the keyed
 * spectral watermark (needs the watermark_id shown on the Generate tab) and/or
 * an embedded C2PA manifest (no key needed).
 * @throws ApiError (401 unauthenticated, 415 not audio).
 */
export const verifyProvenance = async (
  file: File,
  watermarkId = '',
): Promise<WatermarkVerifyResult> => {
  const form = new FormData()
  form.append('file', file)
  if (watermarkId.trim()) form.append('watermark_id', watermarkId.trim())

  const res = await apiFetch('/watermark/verify', { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<WatermarkVerifyResult>
}
