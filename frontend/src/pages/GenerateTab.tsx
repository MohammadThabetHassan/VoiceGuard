import { useState } from 'react'

export default function GenerateTab() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [watermarkId, setWatermarkId] = useState<string | null>(null)

  const handleSynthesize = async () => {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setAudioUrl(null)

    const token = localStorage.getItem('vg_token') || ''
    try {
      const resp = await fetch('/synthesize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ text }),
      })
      if (resp.status === 501) {
        setError(
          'Speech synthesis requires GPU (Coqui XTTS v2). ' +
          'Not available on the current CPU instance. Run the GPU training plan first.'
        )
        return
      }
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        setError(body.detail || `Error ${resp.status}`)
        return
      }
      const data = await resp.json()
      setAudioUrl(data.audio_url)
      setWatermarkId(data.watermark_id)
    } catch {
      setError('Network error')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Synthesise Voice</h2>
        <p className="text-sm text-gray-400">
          Uses Coqui XTTS v2 with C2PA v1.4 spectral watermarking.
          All synthesised audio is cryptographically marked as AI-generated.
        </p>
      </div>

      <div>
        <label className="block text-sm text-gray-400 mb-2">Text to synthesise</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={5}
          maxLength={2000}
          placeholder="Enter text here (max 2000 characters)…"
          className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-gray-100 placeholder-gray-600 resize-none focus:outline-none focus:border-indigo-500"
        />
        <p className="text-xs text-gray-600 mt-1 text-right">{text.length}/2000</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Engine</label>
          <select className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-300 text-sm">
            <option value="xtts_v2">Coqui XTTS v2</option>
          </select>
        </div>
        <div>
          <label className="block text-xs text-gray-500 mb-1">Language</label>
          <select className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-300 text-sm">
            <option value="en">English</option>
            <option value="ar">Arabic</option>
          </select>
        </div>
      </div>

      <button
        onClick={handleSynthesize}
        disabled={!text.trim() || loading}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
      >
        {loading ? 'Generating…' : 'Synthesise'}
      </button>

      {error && (
        <div className="bg-yellow-900/40 border border-yellow-700 rounded-lg p-4 text-yellow-300 text-sm">
          {error}
        </div>
      )}

      {audioUrl && (
        <div className="bg-gray-800 rounded-xl p-6 space-y-3">
          <p className="text-sm text-gray-300 font-medium">Synthesised audio</p>
          <audio controls src={audioUrl} className="w-full" />
          {watermarkId && (
            <div className="flex items-center gap-2 text-xs text-green-400">
              <span>✓</span>
              <span>C2PA watermark embedded · ID: {watermarkId}</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
