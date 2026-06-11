import { describe, it, expect, beforeEach } from 'vitest'
import { getHistory, addHistory } from './history'
import type { DetectionResult } from './detectionService'

const result = (overrides: Partial<DetectionResult> = {}): DetectionResult => ({
  label: 'fake',
  confidence: 0.97,
  model: 'xls_r_aasist',
  latency_ms: 120,
  audio_hash: 'abc123',
  ...overrides,
})

describe('history', () => {
  beforeEach(() => localStorage.clear())

  it('starts empty and stores entries newest-first', () => {
    expect(getHistory()).toEqual([])
    addHistory('a.wav', result())
    addHistory('b.wav', result({ label: 'real', confidence: 0.9 }))
    const h = getHistory()
    expect(h).toHaveLength(2)
    expect(h[0].filename).toBe('b.wav')
    expect(h[0].label).toBe('real')
  })

  it('caps the stored history at 50 entries', () => {
    for (let i = 0; i < 60; i++) addHistory(`clip-${i}.wav`, result())
    const h = getHistory()
    expect(h).toHaveLength(50)
    expect(h[0].filename).toBe('clip-59.wav')
  })

  it('survives corrupted storage', () => {
    localStorage.setItem('vg_history', '{not json')
    expect(getHistory()).toEqual([])
  })
})
