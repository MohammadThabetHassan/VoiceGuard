/** Forensic report API calls (POST /forensic/report). */
import { apiFetch, ApiError } from '../config/apiConfig'
import type { DetectionResult } from './detectionService'

export type ForensicReport = {
  report_url: string
  chain_of_custody_hash: string
}

/**
 * Request a NIST SP 800-86 forensic PDF for a prior detection. The verdict comes
 * from VoiceGuard's own server-side record for this audio hash (set by /detect),
 * so only `audio_hash` is sent — a client value cannot forge the verdict.
 * @throws ApiError with the HTTP status and the backend's detail message
 *   (e.g. 404 "run /detect first" when there is no server-side record).
 */
export const generateReport = async (result: DetectionResult): Promise<ForensicReport> => {
  const res = await apiFetch('/forensic/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      audio_hash: result.audio_hash,
      analyst_name: 'VoiceGuard Analyst',
    }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<ForensicReport>
}
