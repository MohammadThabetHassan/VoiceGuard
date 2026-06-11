import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, cleanup } from '@testing-library/react'
import LiveTab from './LiveTab'
import { setToken, clearToken } from '../config/apiConfig'

describe('LiveTab', () => {
  beforeEach(() => {
    cleanup()
    clearToken()
  })

  it('blocks streaming and explains why when not logged in', () => {
    render(<LiveTab />)
    expect(screen.getByText(/log in first/i)).toBeTruthy()
    const button = screen.getByRole('button', { name: /start live analysis/i })
    expect(button).toHaveProperty('disabled', true)
  })

  it('enables the start button once authenticated', () => {
    setToken('jwt-abc')
    render(<LiveTab />)
    expect(screen.queryByText(/log in first/i)).toBeNull()
    const button = screen.getByRole('button', { name: /start live analysis/i })
    expect(button).toHaveProperty('disabled', false)
  })
})
