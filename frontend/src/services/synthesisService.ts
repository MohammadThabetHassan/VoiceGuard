/** Voice synthesis API calls (GET /synthesis/engines, POST /synthesize). */
import { apiFetch, ApiError } from '../config/apiConfig'

export type EngineInfo = {
  name: string
  label: string
  requires_reference: boolean
  available: boolean
  preset_voices: string[]
  languages: string[]
  description: string
}

export type SynthesisResult = {
  audio_url: string
  watermark_id?: string
  synthesis_latency_ms?: number
  engine?: string
}

export type SynthesizeParams = {
  text: string
  engine: string
  voice?: string
  language?: string
  reference?: File | null
  consent?: boolean
}

/** List synthesis engines and their availability. */
export const getEngines = async (): Promise<EngineInfo[]> => {
  const res = await apiFetch('/synthesis/engines')
  if (!res.ok) throw new ApiError(res.status, `Error ${res.status}`)
  return res.json() as Promise<EngineInfo[]>
}

/**
 * Synthesise speech (preset TTS or zero-shot voice cloning). Cloning engines
 * require a `reference` audio clip. The result is watermarked on the backend.
 * @throws ApiError — 501 engine unavailable, 422 missing/short reference.
 */
export const synthesize = async (p: SynthesizeParams): Promise<SynthesisResult> => {
  const form = new FormData()
  form.append('text', p.text)
  form.append('engine', p.engine)
  if (p.voice) form.append('voice', p.voice)
  if (p.language) form.append('language', p.language)
  form.append('consent', p.consent ? 'true' : 'false')
  if (p.reference) form.append('reference', p.reference)

  const res = await apiFetch('/synthesize', { method: 'POST', body: form })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new ApiError(res.status, body.detail || `Error ${res.status}`)
  }
  return res.json() as Promise<SynthesisResult>
}
