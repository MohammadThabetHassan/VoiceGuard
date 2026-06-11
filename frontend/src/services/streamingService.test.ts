import { describe, it, expect } from 'vitest'
import { resampleTo16k, floatToInt16 } from './streamingService'

describe('resampleTo16k', () => {
  it('passes 16kHz input through untouched', () => {
    const input = new Float32Array([0.1, 0.2, 0.3])
    expect(resampleTo16k(input, 16000)).toBe(input)
  })

  it('downsamples 48kHz to a third of the samples', () => {
    const input = new Float32Array(4800).fill(0.5)
    const out = resampleTo16k(input, 48000)
    expect(out.length).toBe(1600)
    expect(out[0]).toBeCloseTo(0.5)
    expect(out[out.length - 1]).toBeCloseTo(0.5)
  })

  it('interpolates between samples (44.1kHz)', () => {
    const input = new Float32Array(441).map((_, i) => i / 441)
    const out = resampleTo16k(input, 44100)
    expect(out.length).toBe(160)
    // Monotonic ramp must stay monotonic after linear interpolation.
    for (let i = 1; i < out.length; i++) {
      expect(out[i]).toBeGreaterThanOrEqual(out[i - 1])
    }
  })
})

describe('floatToInt16', () => {
  it('maps the float range onto int16', () => {
    const out = floatToInt16(new Float32Array([0, 1, -1]))
    expect(out[0]).toBe(0)
    expect(out[1]).toBe(0x7fff)
    expect(out[2]).toBe(-0x8000)
  })

  it('clips out-of-range samples instead of wrapping', () => {
    const out = floatToInt16(new Float32Array([2.5, -2.5]))
    expect(out[0]).toBe(0x7fff)
    expect(out[1]).toBe(-0x8000)
  })
})
