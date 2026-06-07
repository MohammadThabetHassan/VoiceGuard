/** Forensic report API calls (POST /forensic/report). */
import { apiFetch, ApiError } from '../config/apiConfig'

/**
 * Request a NIST SP 800-86 forensic PDF for a detection result.
 * Returns the PDF as a Blob (caller can objectURL + download).
 * @throws ApiError — notably 501 until forensic generation is enabled on the
 *         backend.
 */
export const generateReport = async (
  audioHash: string,
  results: Record<string, unknown>,
): Promise<Blob> => {
  const res = await apiFetch('/forensic/report', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hash: audioHash, results }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.blob()
}
