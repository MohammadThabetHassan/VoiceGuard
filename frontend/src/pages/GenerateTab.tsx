import { useEffect, useMemo, useState } from 'react'
import { getEngines, synthesize, type EngineInfo } from '../services/synthesisService'
import { detectAudio, type DetectionResult } from '../services/detectionService'
import { apiFetch, ApiError } from '../config/apiConfig'

export default function GenerateTab() {
  const [text, setText] = useState('')
  const [engines, setEngines] = useState<EngineInfo[]>([])
  const [engineName, setEngineName] = useState('kokoro')
  const [voice, setVoice] = useState('af_heart')
  const [language, setLanguage] = useState('en')
  const [reference, setReference] = useState<File | null>(null)
  const [consent, setConsent] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [audioUrl, setAudioUrl] = useState<string | null>(null)
  const [watermarkId, setWatermarkId] = useState<string | null>(null)
  const [verdict, setVerdict] = useState<DetectionResult | null>(null)
  const [testing, setTesting] = useState(false)

  useEffect(() => {
    getEngines()
      .then((es) => {
        setEngines(es)
        const first = es.find((e) => e.available) ?? es[0]
        if (first) {
          setEngineName(first.name)
          if (first.preset_voices[0]) setVoice(first.preset_voices[0])
        }
      })
      .catch(() => setEngines([]))
  }, [])

  const engine = useMemo(
    () => engines.find((e) => e.name === engineName),
    [engines, engineName],
  )
  const needsRef = !!engine?.requires_reference
  const canSubmit =
    !!text.trim() && !loading && (!needsRef || (!!reference && consent))

  const handleSynthesize = async () => {
    if (!canSubmit) return
    setLoading(true)
    setError(null)
    setAudioUrl(null)
    setVerdict(null)
    try {
      const data = await synthesize({ text, engine: engineName, voice, language, reference, consent })
      setAudioUrl(data.audio_url)
      setWatermarkId(data.watermark_id ?? null)
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          e.status === 501
            ? `The "${engine?.label ?? engineName}" engine is not installed on this instance.`
            : e.message,
        )
      } else {
        setError('Could not reach the synthesis API. Please try again in a moment.')
      }
    } finally {
      setLoading(false)
    }
  }

  const handleTestDetector = async () => {
    if (!audioUrl) return
    setTesting(true)
    setVerdict(null)
    try {
      const res = await apiFetch(audioUrl.replace(/^\/api/, ''))
      const blob = await res.blob()
      const file = new File([blob], 'synth.wav', { type: 'audio/wav' })
      setVerdict(await detectAudio(file))
    } catch {
      setError('Could not run the detector on the generated clip.')
    } finally {
      setTesting(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Synthesise Voice</h2>
        <p className="text-sm text-gray-400">
          Preset TTS or zero-shot voice cloning. Every clip is spectrally watermarked and
          flagged as AI-generated, then can be tested against the detector.
        </p>
      </div>

      <div>
        <label className="block text-sm text-gray-400 mb-2">Text to synthesise</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={4}
          maxLength={2000}
          placeholder="Enter text here (max 2000 characters)…"
          className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-gray-100 placeholder-gray-600 resize-none focus:outline-none focus:border-indigo-500"
        />
        <p className="text-xs text-gray-600 mt-1 text-right">{text.length}/2000</p>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="block text-xs text-gray-500 mb-1">Engine</label>
          <select
            value={engineName}
            onChange={(e) => {
              setEngineName(e.target.value)
              const en = engines.find((x) => x.name === e.target.value)
              if (en?.preset_voices[0]) setVoice(en.preset_voices[0])
            }}
            className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-300 text-sm"
          >
            {engines.map((e) => (
              <option key={e.name} value={e.name} disabled={!e.available}>
                {e.label}
                {e.available ? '' : ' — not installed'}
              </option>
            ))}
          </select>
        </div>
        {engine && engine.preset_voices.length > 0 ? (
          <div>
            <label className="block text-xs text-gray-500 mb-1">Voice</label>
            <select
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-300 text-sm"
            >
              {engine.preset_voices.map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
        ) : (
          <div>
            <label className="block text-xs text-gray-500 mb-1">Language</label>
            <select
              value={language}
              onChange={(e) => setLanguage(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-2 text-gray-300 text-sm"
            >
              {(engine?.languages ?? ['en']).map((l) => (
                <option key={l} value={l}>
                  {l}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {needsRef && (
        <div className="bg-gray-800/60 border border-gray-700 rounded-lg p-4 space-y-3">
          <label className="block text-sm text-gray-300">
            Reference voice (≥ 3s) — the voice to clone
          </label>
          <input
            type="file"
            accept="audio/*"
            onChange={(e) => setReference(e.target.files?.[0] ?? null)}
            className="block w-full text-sm text-gray-400 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-indigo-600 file:text-white hover:file:bg-indigo-700"
          />
          <label className="flex items-start gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={consent}
              onChange={(e) => setConsent(e.target.checked)}
              className="mt-0.5"
            />
            <span>
              I am authorised to clone this voice and will use the watermarked output only for
              testing and research, not impersonation.
            </span>
          </label>
        </div>
      )}

      <button
        onClick={handleSynthesize}
        disabled={!canSubmit}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white font-medium py-3 px-6 rounded-lg transition-colors"
      >
        {loading ? 'Generating…' : needsRef ? 'Clone & Synthesise' : 'Synthesise'}
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
              <span>Spectral watermark embedded · ID: {watermarkId}</span>
            </div>
          )}
          <button
            onClick={handleTestDetector}
            disabled={testing}
            className="text-sm bg-gray-700 hover:bg-gray-600 disabled:opacity-60 text-gray-100 font-medium py-2 px-4 rounded-lg transition-colors"
          >
            {testing ? 'Testing…' : '🔎 Test against detector'}
          </button>
          {verdict && (
            <div
              className={`text-sm rounded-lg p-3 ${
                verdict.label === 'fake'
                  ? 'bg-red-900/40 border border-red-700 text-red-300'
                  : 'bg-green-900/40 border border-green-700 text-green-300'
              }`}
            >
              Detector verdict: <b>{verdict.label.toUpperCase()}</b> ·{' '}
              {Math.round(verdict.confidence * 100)}% confidence · model {verdict.model}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
