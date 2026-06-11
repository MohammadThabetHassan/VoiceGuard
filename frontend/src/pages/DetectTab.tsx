import { useState, useRef } from 'react'
import { detectAudio, type DetectionResult } from '../services/detectionService'
import { generateReport } from '../services/forensicsService'
import { addHistory } from '../services/history'
import { ApiError } from '../config/apiConfig'

function ConfidenceGauge({ confidence, label }: { confidence: number; label: string }) {
  const pct = Math.round(confidence * 100)
  const color = label === 'fake' ? 'text-red-400' : 'text-green-400'
  const ring = label === 'fake' ? 'border-red-500' : 'border-green-500'
  return (
    <div className={`flex flex-col items-center gap-2 p-6 rounded-xl border-2 ${ring} bg-gray-800`}>
      <span className={`text-5xl font-bold font-mono ${color}`}>{pct}%</span>
      <span className={`text-lg font-semibold uppercase tracking-widest ${color}`}>{label}</span>
      <span className="text-xs text-gray-500">confidence</span>
    </div>
  )
}

function AttributionView({
  frames,
  segments,
}: {
  frames: number[]
  segments: { start_s: number; end_s: number; importance: number }[]
}) {
  // Downsample to ~120 bars for a clean sparkline of per-frame importance.
  const step = Math.max(1, Math.ceil(frames.length / 120))
  const bars: number[] = []
  for (let i = 0; i < frames.length; i += step) {
    bars.push(Math.max(...frames.slice(i, i + step)))
  }
  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-3">
      <p className="text-sm font-medium text-gray-300">Why — most influential moments</p>
      <div className="flex items-end gap-px h-16">
        {bars.map((v, i) => (
          <div
            key={i}
            className="flex-1 rounded-sm bg-indigo-500"
            style={{ height: `${Math.max(4, v * 100)}%`, opacity: 0.4 + v * 0.6 }}
            title={`importance ${v.toFixed(2)}`}
          />
        ))}
      </div>
      {segments.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-1">
          {segments.slice(0, 4).map((s, i) => (
            <span
              key={i}
              className="text-xs font-mono px-2 py-1 rounded bg-gray-900 border border-gray-700 text-indigo-300"
            >
              {s.start_s.toFixed(1)}s–{s.end_s.toFixed(1)}s · {Math.round(s.importance * 100)}%
            </span>
          ))}
        </div>
      )}
      <p className="text-[11px] text-gray-600">
        Integrated-Gradients attribution — taller/brighter = more influence on the verdict.
      </p>
    </div>
  )
}

export default function DetectTab() {
  const [file, setFile] = useState<File | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [explain, setExplain] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [reportBusy, setReportBusy] = useState(false)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await detectAudio(file, explain)
      setResult(data)
      addHistory(file.name, data)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 401
            ? 'Not authenticated — use the Log in button (top-right) first.'
            : e.message,
        )
      } else {
        setError('Could not reach the detection API. Please try again in a moment.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleReport = async () => {
    if (!result) return
    setReportBusy(true)
    try {
      const report = await generateReport(result)
      window.open(report.report_url, '_blank')
    } catch (e) {
      setError(e instanceof ApiError ? e.message : 'Could not generate the report.')
    } finally {
      setReportBusy(false)
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
          accept=".wav,.mp3,.flac,.ogg,audio/*"
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
            <p className="text-gray-500 text-sm">or drag and drop · use a few seconds of speech</p>
          </div>
        )}
      </div>

      {/* Options + submit */}
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-sm text-gray-400 select-none cursor-pointer">
          <input
            type="checkbox"
            checked={explain}
            onChange={(e) => setExplain(e.target.checked)}
            className="accent-indigo-500"
          />
          Explain decision
        </label>
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

          {/* Verdict explainer */}
          <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
            <p className="text-sm font-medium text-gray-300 mb-1">
              {result.label === 'fake'
                ? 'This audio shows synthetic-speech artifacts.'
                : 'This audio is consistent with genuine human speech.'}
            </p>
            <p className="text-xs text-gray-500">
              Scored by the {result.model} detector (XLS-R self-supervised front-end + AASIST
              graph-attention back-end). Uploaded audio is deleted within 60 seconds (PDPL).
            </p>
          </div>

          {/* Explainability */}
          {result.explanation && (
            <AttributionView
              frames={result.explanation.attribution_frames}
              segments={result.explanation.top_segments}
            />
          )}

          {/* Forensic report */}
          <button
            onClick={handleReport}
            disabled={reportBusy}
            className="w-full border border-gray-700 bg-gray-800 hover:bg-gray-700 disabled:opacity-50 text-gray-200 font-medium py-2.5 px-6 rounded-lg transition-colors text-sm"
          >
            {reportBusy ? 'Generating…' : '⬇ Download forensic PDF report'}
          </button>
        </div>
      )}
    </div>
  )
}
