import { useState, useRef, useEffect } from 'react'
import {
  detectAudio,
  getModels,
  MODEL_LABELS,
  type DetectionResult,
  type Explanation,
  type ModelInfo,
} from '../services/detectionService'
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
  explanation,
  label,
}: {
  explanation: Explanation
  label: string
}) {
  const { attribution_frames: frames, top_segments: segments, method, frame_duration_ms } = explanation
  const isFake = label === 'fake'
  const barColor = isFake ? 'bg-rose-500' : 'bg-emerald-500'
  const accentText = isFake ? 'text-rose-300' : 'text-emerald-300'

  // Total clip duration (frames × frame length) → for the time axis.
  const totalS = (frames.length * (frame_duration_ms || 10)) / 1000
  const methodLabel =
    method === 'integrated_gradients' ? 'Integrated Gradients' : method === 'occlusion' ? 'Occlusion' : method

  // Downsample to ~140 bars for a clean timeline of per-frame importance.
  const step = Math.max(1, Math.ceil(frames.length / 140))
  const bars: number[] = []
  for (let i = 0; i < frames.length; i += step) bars.push(Math.max(...frames.slice(i, i + step)))

  const top = [...segments].sort((a, b) => b.importance - a.importance).slice(0, 5)
  const strongest = top[0]

  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <p className="text-sm font-medium text-gray-200">Why this verdict — moment-by-moment</p>
        <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-gray-900 border border-gray-700 text-gray-400">
          {methodLabel} · {frame_duration_ms || 10}ms frames
        </span>
      </div>

      {/* LLM forensic narrative */}
      {explanation.narrative && (
        <div className="rounded-lg border border-indigo-500/30 bg-indigo-500/5 p-3">
          <p className="text-[11px] uppercase tracking-wider text-indigo-300/80 mb-1 flex items-center gap-1.5">
            <span aria-hidden>✨</span> AI analysis
          </p>
          <p className="text-sm text-gray-200 leading-relaxed">{explanation.narrative}</p>
        </div>
      )}

      {/* plain-language interpretation */}
      {strongest && (
        <p className="text-sm text-gray-400">
          The detector weighed the whole clip and leaned{' '}
          <span className={`font-semibold ${accentText}`}>{label.toUpperCase()}</span> most strongly on the audio
          around{' '}
          <span className="font-mono text-gray-200">
            {strongest.start_s.toFixed(1)}s–{strongest.end_s.toFixed(1)}s
          </span>
          {top.length > 1 && (
            <>
              , with further support at{' '}
              <span className="font-mono text-gray-200">
                {top[1].start_s.toFixed(1)}s–{top[1].end_s.toFixed(1)}s
              </span>
            </>
          )}
          .
        </p>
      )}

      {/* timeline */}
      <div>
        <div className="flex items-end gap-px h-20">
          {bars.map((v, i) => (
            <div
              key={i}
              className={`flex-1 rounded-sm ${barColor}`}
              style={{ height: `${Math.max(3, v * 100)}%`, opacity: 0.35 + v * 0.65 }}
              title={`importance ${v.toFixed(2)}`}
            />
          ))}
        </div>
        <div className="flex justify-between text-[10px] font-mono text-gray-600 mt-1">
          <span>0.0s</span>
          <span>{(totalS / 2).toFixed(1)}s</span>
          <span>{totalS.toFixed(1)}s</span>
        </div>
      </div>

      {/* ranked suspicious moments with importance bars */}
      {top.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-[11px] uppercase tracking-wider text-gray-500">
            Most influential moments
          </p>
          {top.map((s, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs font-mono text-gray-500 w-4 text-right">{i + 1}</span>
              <span className="text-xs font-mono text-gray-300 w-24 shrink-0">
                {s.start_s.toFixed(1)}–{s.end_s.toFixed(1)}s
              </span>
              <div className="flex-1 h-2 rounded-full bg-gray-900 overflow-hidden">
                <div
                  className={`h-full ${barColor}`}
                  style={{ width: `${Math.max(4, s.importance * 100)}%` }}
                />
              </div>
              <span className={`text-xs font-mono ${accentText} w-10 text-right`}>
                {Math.round(s.importance * 100)}%
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="text-[11px] text-gray-600">
        {methodLabel === 'Integrated Gradients'
          ? 'Integrated Gradients traces the verdict back through the model to each 10ms frame — taller/brighter and higher-ranked = more influence.'
          : 'Occlusion silences each time window in turn and measures how much the verdict shifts — bigger shift = more influence.'}
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
  const [models, setModels] = useState<ModelInfo[]>([])
  const [model, setModel] = useState('classical')
  const fileRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    getModels()
      .then((ms) => {
        // Show the flagship models first; available ones bubble up.
        const order = Object.keys(MODEL_LABELS)
        ms.sort(
          (a, b) =>
            Number(b.available) - Number(a.available) ||
            order.indexOf(a.key) - order.indexOf(b.key),
        )
        setModels(ms)
        const first = ms.find((m) => m.available)
        if (first) setModel(first.key)
      })
      .catch(() => setModels([]))
  }, [])

  const handleUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const data = await detectAudio(file, explain, model)
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

      {/* Model picker */}
      <div className="space-y-1">
        <label className="text-sm text-gray-400">Detector model</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          className="w-full bg-gray-800 border border-gray-700 text-gray-200 rounded-lg px-3 py-2 text-sm"
        >
          {models.map((m) => (
            <option key={m.key} value={m.key} disabled={!m.available}>
              {(MODEL_LABELS[m.key] ?? m.key) + (m.available ? '' : ' — checkpoint unavailable')}
            </option>
          ))}
        </select>
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
              explanation={result.explanation}
              label={result.label}
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
