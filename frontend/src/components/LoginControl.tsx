import { useState } from 'react'
import { hasToken, clearToken, ApiError } from '../config/apiConfig'
import { login } from '../services/authService'

/**
 * Header login control. Detection endpoints require a JWT, so the user logs in
 * here (demo creds: admin / voiceguard2026). Shows a "Log in" button when signed
 * out and a "Log out" button when signed in. The API URL is fixed (same-origin
 * /api behind Nginx), so there is no URL configuration here.
 */
export default function LoginControl() {
  const [open, setOpen] = useState(false)
  const [loggedIn, setLoggedIn] = useState(hasToken())
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [msg, setMsg] = useState('')

  const handleLogin = async () => {
    setMsg('')
    try {
      await login(username, password)
      setLoggedIn(true)
      setPassword('')
      setOpen(false)
    } catch (e) {
      setMsg(e instanceof ApiError ? e.message : 'Login failed (network error)')
    }
  }

  const handleLogout = () => {
    clearToken()
    setLoggedIn(false)
  }

  if (loggedIn) {
    return (
      <button
        type="button"
        onClick={handleLogout}
        className="rounded-lg border border-gray-700 bg-gray-800 px-3 py-1.5 text-xs font-medium text-gray-300 transition hover:bg-gray-700"
      >
        Log out
      </button>
    )
  }

  return (
    <>
      <button
        type="button"
        onClick={() => {
          setMsg('')
          setOpen(true)
        }}
        className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-medium text-white transition hover:bg-indigo-700"
      >
        Log in
      </button>

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="w-full max-w-sm rounded-2xl border border-gray-800 bg-gray-950 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Sign in</h3>
              <button
                type="button"
                aria-label="Close"
                onClick={() => setOpen(false)}
                className="text-gray-500 transition hover:text-white"
              >
                ✕
              </button>
            </div>

            <label className="mb-1 block text-sm text-gray-400">Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              className="mb-3 w-full rounded-lg border border-gray-700 bg-gray-800 p-2.5 text-sm text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
            />

            <label className="mb-1 block text-sm text-gray-400">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
              autoComplete="current-password"
              className="mb-4 w-full rounded-lg border border-gray-700 bg-gray-800 p-2.5 text-sm text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
            />

            <button
              type="button"
              onClick={handleLogin}
              disabled={!username || !password}
              className="w-full rounded-lg bg-indigo-600 py-2.5 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:opacity-50"
            >
              Log in
            </button>

            {msg && <p className="mt-3 text-xs text-red-400">{msg}</p>}
          </div>
        </div>
      )}
    </>
  )
}
