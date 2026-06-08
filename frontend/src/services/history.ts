/** Client-side detection history (localStorage). Powers the Results tab table. */
import type { DetectionResult } from './detectionService'

const KEY = 'vg_history'
const MAX = 50

export type HistoryEntry = {
  ts: number
  filename: string
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string
}

export const getHistory = (): HistoryEntry[] => {
  try {
    return JSON.parse(localStorage.getItem(KEY) || '[]') as HistoryEntry[]
  } catch {
    return []
  }
}

export const addHistory = (filename: string, r: DetectionResult): void => {
  const entry: HistoryEntry = {
    ts: Date.now(),
    filename,
    label: r.label,
    confidence: r.confidence,
    model: r.model,
    latency_ms: r.latency_ms,
    audio_hash: r.audio_hash,
  }
  const next = [entry, ...getHistory()].slice(0, MAX)
  localStorage.setItem(KEY, JSON.stringify(next))
}

export const clearHistory = (): void => localStorage.removeItem(KEY)
