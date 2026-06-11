import { useRef, useState } from 'react'
import {
  verifyProvenance,
  type WatermarkVerifyResult,
  type ProvenanceVerdict,
} from '../services/provenanceService'
import { ApiError } from '../config/apiConfig'

const VERDICT_VIEW: Record<ProvenanceVerdict, { label: string; cls: string; blurb: string }> = {
  'voiceguard-generated': {
    label: 'VoiceGuard-generated',
    cls: 'border-indigo-500 text-indigo-300',
    blurb: 'The keyed spectral watermark matches — this clip was synthesised by VoiceGuard.',
  },
  'ai-generated': {
    label: 'AI-generated (C2PA)',
    cls: 'border-amber-500 text-amber-300',
    blurb: 'A signed C2PA manifest marks this audio as trained-algorithmic media.',
  },
  unknown: {
    label: 'Manifest present — origin unclear',
    cls: 'border-gray-500 text-gray-300',
    blurb: 'The file carries a C2PA manifest but no AI-generation assertion.',
  },
  'no-provenance-found': {
    label: 'No provenance found',
    cls: 'border-gray-600 text-gray-400',
    blurb:
      'No watermark or manifest detected. This does NOT prove the audio is real — most ' +
      'recordings and most deepfakes carry no provenance marks. Use the Detect tab for a verdict.',
  },
}

function ResultPanel({ result }: { result: WatermarkVerifyResult }) {
  const view = VERDICT_VIEW[result.verdict]
  const rows: { k: string; v: string }[] = [
    {
      k: 'Spectral watermark',
      v: result.spectral_checked
        ? result.spectral_detected
          ? `detected (corr ${result.spectral_correlation?.toFixed(4)})`
          : `not detected (corr ${result.spectral_correlation?.toFixed(4)})`
        : 'not checked — paste the watermark ID from the Generate tab',
    },
    {
      k: 'C2PA manifest',
      v: result.c2pa_has_manifest
        ? `present · ${result.c2pa_validation_state ?? 'unvalidated'}`
        : 'absent',
    },
  ]
  if (result.c2pa_has_manifest) {
    rows.push({
      k: 'C2PA assertion',
      v: result.c2pa_ai_generated
        ? `AI-generated · ${result.c2pa_software_agent ?? 'unknown agent'}`
        : 'no AI-generation assertion',
    })
  }
  return (
    <div className="space-y-4">
      <div className={`rounded-xl border-2 bg-gray-800 p-6 ${view.cls}`}>
        <p className="text-lg font-semibold uppercase tracking-wide">{view.label}</p>
        <p className="text-sm text-gray-400 mt-1">{view.blurb}</p>
      </div>
      <div className="bg-gray-800 rounded-xl border border-gray-700 divide-y divide-gray-700">
        {rows.map(({ k, v }) => (
          <div key={k} className="flex items-center justify-between gap-4 px-5 py-3">
            <span className="text-sm text-gray-400">{k}</span>
            <span className="text-sm text-white font-mono text-right">{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default function VerifyTab() {
  const [file, setFile] = useState<File | null>(null)
  const [watermarkId, setWatermarkId] = useState('')
  const [result, setResult] = useState<WatermarkVerifyResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const fileRef = useRef<HTMLInputElement>(null)

  const handleVerify = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      setResult(await verifyProvenance(file, watermarkId))
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 401
            ? 'Not authenticated — use the Log in button (top-right) first.'
            : e.message,
        )
      } else {
        setError('Could not reach the verification API. Please try again in a moment.')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Verify Provenance</h2>
        <p className="text-sm text-gray-400">
          The read side of Generate: check an audio file for VoiceGuard's spectral watermark and
          embedded C2PA manifest. Files are auto-deleted after 60s (PDPL).
        </p>
      </div>

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
            <div className="text-4xl">🔏</div>
            <p className="text-gray-300">Click to select audio file</p>
            <p className="text-gray-500 text-sm">e.g. a clip downloaded from the Generate tab</p>
          </div>
        )}
      </div>

      <div>
        <label className="block text-sm text-gray-400 mb-1" htmlFor="vg-watermark-id">
          Watermark ID <span className="text-gray-600">(optional — shown after generating)</span>
        </label>
        <input
          id="vg-watermark-id"
          type="text"
          value={watermarkId}
          onChange={(e) => setWatermarkId(e.target.value)}
          placeholder="e.g. 1f3a9c…"
          className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white font-mono text-sm focus:outline-none focus:border-indigo-500"
        />
      </div>

      <button
        onClick={handleVerify}
        disabled={!file || loading}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
      >
        {loading ? 'Verifying…' : 'Verify provenance'}
      </button>

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {result && <ResultPanel result={result} />}
    </div>
  )
}
