import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'
import VerifyTab from './VerifyTab'
import type { WatermarkVerifyResult } from '../services/provenanceService'

const verifyMock = vi.hoisted(() => vi.fn())
vi.mock('../services/provenanceService', async (importOriginal) => ({
  ...(await importOriginal<object>()),
  verifyProvenance: verifyMock,
}))

const verified: WatermarkVerifyResult = {
  spectral_checked: true,
  spectral_detected: true,
  spectral_correlation: 0.1234,
  c2pa_has_manifest: true,
  c2pa_validation_state: 'Trusted',
  c2pa_ai_generated: true,
  c2pa_software_agent: 'VoiceGuard/kokoro',
  verdict: 'voiceguard-generated',
}

const pickFile = (container: HTMLElement) => {
  const input = container.querySelector('input[type="file"]') as HTMLInputElement
  const file = new File(['RIFFxxxxWAVE'], 'clip.wav', { type: 'audio/wav' })
  fireEvent.change(input, { target: { files: [file] } })
}

describe('VerifyTab', () => {
  beforeEach(() => {
    cleanup()
    verifyMock.mockReset()
  })

  it('disables the verify button until a file is chosen', () => {
    const { container } = render(<VerifyTab />)
    const button = screen.getByRole('button', { name: /verify provenance/i })
    expect(button).toHaveProperty('disabled', true)
    pickFile(container)
    expect(button).toHaveProperty('disabled', false)
  })

  it('shows the verdict panel after a successful verification', async () => {
    verifyMock.mockResolvedValue(verified)
    const { container } = render(<VerifyTab />)
    pickFile(container)
    fireEvent.click(screen.getByRole('button', { name: /verify provenance/i }))
    await waitFor(() => {
      expect(screen.getByText(/VoiceGuard-generated/i)).toBeTruthy()
    })
    expect(screen.getByText(/detected \(corr 0\.1234\)/i)).toBeTruthy()
    expect(verifyMock).toHaveBeenCalledOnce()
  })

  it('surfaces API errors inline', async () => {
    const { ApiError } = await import('../config/apiConfig')
    verifyMock.mockRejectedValue(new ApiError(415, 'File content is not audio.'))
    const { container } = render(<VerifyTab />)
    pickFile(container)
    fireEvent.click(screen.getByRole('button', { name: /verify provenance/i }))
    await waitFor(() => {
      expect(screen.getByText(/not audio/i)).toBeTruthy()
    })
  })
})
