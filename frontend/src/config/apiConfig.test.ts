import { describe, it, expect, beforeEach } from 'vitest'
import { getWsUrl, getToken, setToken, clearToken, hasToken, ApiError } from './apiConfig'

describe('getWsUrl', () => {
  it('derives ws:// from the page origin (http in jsdom)', () => {
    expect(getWsUrl('/ws/stream')).toBe(`ws://${window.location.host}/ws/stream`)
  })
})

describe('token storage', () => {
  beforeEach(() => clearToken())

  it('round-trips a token through localStorage', () => {
    expect(hasToken()).toBe(false)
    setToken('jwt-abc')
    expect(getToken()).toBe('jwt-abc')
    expect(hasToken()).toBe(true)
    clearToken()
    expect(hasToken()).toBe(false)
  })
})

describe('ApiError', () => {
  it('carries the HTTP status', () => {
    const err = new ApiError(429, 'slow down')
    expect(err.status).toBe(429)
    expect(err.message).toBe('slow down')
    expect(err.name).toBe('ApiError')
  })
})
