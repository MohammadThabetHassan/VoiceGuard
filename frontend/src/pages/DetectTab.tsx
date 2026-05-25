import { useState, useRef } from 'react'

type DetectionResult = {
  label: 'real' | 'fake'
  confidence: number
  model: string
  latency_ms: number
  audio_hash: string
}

const API = '/detect'

function ConfidenceGauge({ confidence, label }: { confidence: number; label: string }) {
  const pct = Math.round(confidence * 100)
  const color = label === 'fake' ? 'text-red-400' : 'text-green-400'
  const ring = label === 'fake' ? 'border-red-500' : 'border-green-500'
  return (
    <div className={`flex flex-col items-center gap-2 p-6 rounded-xl border-2 ${ring} bg-gray-800`}>
      <span className={`text-5xl font-bold font-mono ${color}`}>{pct}%</span>
      <span className={`text-lg font-semibold uppercase tracking-widest ${color}`}>
        {label}
      </span>
      <span className="text-xs text-gray-500">confidence</span>
    </div>
  )
}

export default function DetectTab() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    const token = localStorage.getItem('vg_token') || ''

    try {
      const resp = await fetch(API, {
        method: 'POST',
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        body: formData,
      })
      if (resp.status === 401) {
        setError('Not authenticated. Please log in first.')
        return
      }
      if (!resp.ok) {
        const body = await resp.json().catch(() => ({}))
        setError(body.detail || `Error ${resp.status}`)
        return
      }
      const data: DetectionResult = await resp.json()
      setResult(data)
    } catch {
      setError('Network error — is the API running?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Upload Audio for Analysis</h2>
        <p className="text-sm text-gray-400">
          Supported formats: WAV, MP3, FLAC · Max 100MB · Audio auto-deleted after 60s (PDPL)
        </p>
      </div>

      {/* Upload area */}
      <div
        className="border-2 border-dashed border-gray-700 rounded-xl p-10 text-center cursor-pointer hover:border-indigo-500 transition-colors"
        onClick={() => fileRef.current?.click()}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".wav,.mp3,.flac,audio/*"
          className="hidden"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
        {file ? (
          <div className="space-y-1">
            <p className="text-white font-medium">{file.name}</p>
            <p className="text-gray-400 text-sm">{(file.size / 1024).toFixed(1)} KB</p>
          </div>
        ) : (
          <div className="space-y-2">
            <div className="text-4xl">🎙️</div>
            <p className="text-gray-300">Click to select audio file</p>
            <p className="text-gray-500 text-sm">or drag and drop</p>
          </div>
        )}
      </div>

      {/* Model selector + submit */}
      <div className="flex gap-3">
        <button
          onClick={handleUpload}
          disabled={!file || loading}
          className="flex-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
        >
          {loading ? 'Analysing…' : 'Detect Deepfake'}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {/* Result */}
      {result && (
        <div className="space-y-4">
          <ConfidenceGauge confidence={result.confidence} label={result.label} />
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              { k: 'Model', v: result.model },
              { k: 'Latency', v: `${result.latency_ms.toFixed(1)}ms` },
              { k: 'Audio Hash', v: result.audio_hash.slice(0, 8) + '…' },
            ].map(({ k, v }) => (
              <div key={k} className="bg-gray-800 rounded-lg p-3">
                <p className="text-xs text-gray-500">{k}</p>
                <p className="text-sm text-white font-mono truncate">{v}</p>
              </div>
            ))}
          </div>

          {/* Grad-CAM placeholder */}
          <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
            <p className="text-sm text-gray-400 mb-2">Spectrogram + Grad-CAM overlay</p>
            <div className="h-32 bg-gray-900 rounded flex items-center justify-center text-gray-600 text-sm">
              Available after DSFNet training (Phase 2 GPU run)
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
