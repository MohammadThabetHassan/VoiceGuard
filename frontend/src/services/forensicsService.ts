/** Forensic report API calls (POST /forensic/report). */
import { apiFetch, ApiError } from '../config/apiConfig'
import type { DetectionResult } from './detectionService'

export type ForensicReport = {
  report_url: string
  chain_of_custody_hash: string
}

/**
 * Request a NIST SP 800-86 forensic PDF for a detection result. Returns the
 * served PDF URL + a chain-of-custody hash.
 * @throws ApiError with the HTTP status and the backend's detail message.
 */
export const generateReport = async (result: DetectionResult): Promise<ForensicReport> => {
  const res = await apiFetch('/forensic/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio_hash: result.audio_hash,
      analyst_name: 'VoiceGuard Analyst',
      detection_result: {
        label: result.label,
        confidence: result.confidence,
        model: result.model,
        latency_ms: result.latency_ms,
      },
    }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<ForensicReport>
}
