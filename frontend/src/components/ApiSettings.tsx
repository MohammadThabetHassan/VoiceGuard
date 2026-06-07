import { useEffect, useState } from 'react'
import { getApiUrl, setApiUrl, hasToken, clearToken, ApiError } from '../config/apiConfig'
import { login } from '../services/authService'

type ConnState = 'untested' | 'testing' | 'connected' | 'failed'

const DOT_COLOR: Record<ConnState, string> = {
  untested: 'bg-gray-500',
  testing: 'bg-yellow-400 animate-pulse',
  connected: 'bg-green-500',
  failed: 'bg-red-500',
}

export default function ApiSettings() {
  const [open, setOpen] = useState(false)
  const [url, setUrl] = useState('')
  const [conn, setConn] = useState<ConnState>('untested')
  const [message, setMessage] = useState('')
  const [showAdvanced, setShowAdvanced] = useState(false)

  // Auth state
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loggedIn, setLoggedIn] = useState(false)
  const [authMsg, setAuthMsg] = useState('')

  // Populate the input with the currently saved URL whenever the modal opens.
  useEffect(() => {
    if (open) {
      setUrl(getApiUrl())
      setConn('untested')
      setMessage('')
      setLoggedIn(hasToken())
      setAuthMsg('')
    }
  }, [open])

  const handleLogin = async () => {
    setAuthMsg('')
    try {
      await login(username, password)
      setLoggedIn(true)
      setPassword('')
      setAuthMsg('Logged in — detection enabled')
    } catch (e) {
      setAuthMsg(e instanceof ApiError ? e.message : 'Login failed (network error)')
    }
  }

  const handleLogout = () => {
    clearToken()
    setLoggedIn(false)
    setAuthMsg('Logged out')
  }

  const testConnection = async () => {
    const target = url.trim().replace(/\/+$/, '')
    if (!target) return
    setConn('testing')
    setMessage('')
    const controller = new AbortController()
    const timer = setTimeout(() => controller.abort(), 3000)
    try {
      const res = await fetch(`${target}/health`, { signal: controller.signal })
      if (res.ok) {
        const data = await res.json().catch(() => ({}))
        setConn('connected')
        setMessage(`Connected · API v${data.version ?? '?'}`)
      } else {
        setConn('failed')
        setMessage(`Reached server but got HTTP ${res.status}`)
      }
    } catch {
      setConn('failed')
      setMessage('Could not reach the API (timeout or CORS/network error)')
    } finally {
      clearTimeout(timer)
    }
  }

  const save = () => {
    setApiUrl(url)
    setMessage(`Saved: ${getApiUrl()}`)
  }

  return (
    <>
      {/* Floating gear button (subtle until hovered) */}
      <button
        type="button"
        aria-label="API settings"
        onClick={() => setOpen(true)}
        className="fixed bottom-5 right-5 z-40 flex h-11 w-11 items-center justify-center rounded-full border border-gray-700 bg-gray-900/70 text-gray-400 opacity-60 shadow-lg transition hover:opacity-100 hover:text-white"
      >
        <span className="text-lg">⚙</span>
        <span
          className={`absolute -top-0.5 -right-0.5 h-3 w-3 rounded-full border-2 border-gray-900 ${DOT_COLOR[conn]}`}
        />
      </button>

      {/* Modal */}
      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 px-4"
          onClick={() => setOpen(false)}
        >
          <div
            className="w-full max-w-md rounded-2xl border border-gray-800 bg-gray-950 p-6 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Backend API Settings</h3>
              <button
                type="button"
                aria-label="Close"
                onClick={() => setOpen(false)}
                className="text-gray-500 transition hover:text-white"
              >
                ✕
              </button>
            </div>

            <label className="mb-1 block text-sm text-gray-400">Local PC API URL</label>
            <input
              type="text"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="http://localhost:8000"
              spellCheck={false}
              className="w-full rounded-lg border border-gray-700 bg-gray-800 p-2.5 font-mono text-sm text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
            />

            <div className="mt-3 flex gap-2">
              <button
                type="button"
                onClick={testConnection}
                disabled={conn === 'testing'}
                className="flex-1 rounded-lg border border-gray-700 bg-gray-800 py-2 text-sm font-medium text-gray-200 transition hover:bg-gray-700 disabled:opacity-50"
              >
                {conn === 'testing' ? 'Testing…' : 'Test Connection'}
              </button>
              <button
                type="button"
                onClick={save}
                className="flex-1 rounded-lg bg-indigo-600 py-2 text-sm font-medium text-white transition hover:bg-indigo-700"
              >
                Save
              </button>
            </div>

            {/* Connection status */}
            <div className="mt-3 flex items-center gap-2 text-sm">
              <span className={`h-2.5 w-2.5 rounded-full ${DOT_COLOR[conn]}`} />
              <span
                className={
                  conn === 'connected'
                    ? 'text-green-400'
                    : conn === 'failed'
                      ? 'text-red-400'
                      : 'text-gray-400'
                }
              >
                {message ||
                  (conn === 'connected'
                    ? 'Connected'
                    : conn === 'failed'
                      ? 'Failed'
                      : 'Not tested')}
              </span>
            </div>

            {/* Authentication — /detect requires a JWT */}
            <div className="mt-5 rounded-lg border border-gray-800 bg-gray-900 p-3">
              <div className="mb-2 flex items-center justify-between">
                <p className="text-xs font-medium text-gray-400">Authentication (for Detect)</p>
                <span
                  className={`text-[11px] ${loggedIn ? 'text-green-400' : 'text-gray-500'}`}
                >
                  {loggedIn ? '● logged in' : '○ not logged in'}
                </span>
              </div>
              {loggedIn ? (
                <button
                  type="button"
                  onClick={handleLogout}
                  className="w-full rounded-lg border border-gray-700 bg-gray-800 py-1.5 text-xs text-gray-300 transition hover:bg-gray-700"
                >
                  Log out
                </button>
              ) : (
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    placeholder="username"
                    autoComplete="username"
                    className="min-w-0 flex-1 rounded-lg border border-gray-700 bg-gray-800 p-2 text-sm text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
                  />
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
                    placeholder="password"
                    autoComplete="current-password"
                    className="min-w-0 flex-1 rounded-lg border border-gray-700 bg-gray-800 p-2 text-sm text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
                  />
                  <button
                    type="button"
                    onClick={handleLogin}
                    disabled={!username || !password}
                    className="rounded-lg bg-indigo-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-indigo-700 disabled:opacity-50"
                  >
                    Log in
                  </button>
                </div>
              )}
              {authMsg && <p className="mt-2 text-[11px] text-gray-400">{authMsg}</p>}
            </div>

            {/* Backend start hint */}
            <div className="mt-3 rounded-lg border border-gray-800 bg-gray-900 p-3">
              <p className="mb-1 text-xs font-medium text-gray-400">Start the backend on your PC:</p>
              <pre className="overflow-x-auto whitespace-pre-wrap break-words text-xs text-indigo-300">
{`cd voiceguard
uvicorn voiceguard.api.main:app --host 0.0.0.0 --port 8000`}
              </pre>
              <p className="mt-2 text-[11px] text-gray-600">
                No audio leaves your machine — inference runs locally.
              </p>
            </div>

            {/* Advanced / ngrok */}
            <button
              type="button"
              onClick={() => setShowAdvanced((v) => !v)}
              className="mt-3 text-xs text-gray-500 transition hover:text-gray-300"
            >
              {showAdvanced ? '▾' : '▸'} Advanced — share with others (ngrok)
            </button>
            {showAdvanced && (
              <div className="mt-2 rounded-lg border border-gray-800 bg-gray-900 p-3">
                <p className="mb-1 text-[11px] text-gray-500">
                  Expose your local API publicly, then paste the URL above:
                </p>
                <pre className="overflow-x-auto text-xs text-indigo-300">{`ngrok http 8000`}</pre>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  )
}
