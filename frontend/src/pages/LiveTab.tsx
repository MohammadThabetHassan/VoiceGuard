import { useEffect, useRef, useState } from 'react'
import {
  LiveMicStream,
  type StreamEvent,
  type StreamStatus,
} from '../services/streamingService'
import { hasToken } from '../config/apiConfig'

const MAX_TIMELINE = 40

const STATUS_LABEL: Record<StreamStatus, string> = {
  idle: 'Idle',
  connecting: 'Connecting…',
  authenticating: 'Authenticating…',
  live: 'LIVE — listening',
  busy: 'Server busy',
  closed: 'Stream closed',
  error: 'Error',
}

function VerdictCard({ event }: { event: StreamEvent }) {
  const pct = Math.round(event.confidence * 100)
  const fake = event.label === 'fake'
  return (
    <div
      className={`flex flex-col items-center gap-2 p-6 rounded-xl border-2 bg-gray-800 ${
        fake ? 'border-red-500' : 'border-green-500'
      }`}
    >
      <span className={`text-5xl font-bold font-mono ${fake ? 'text-red-400' : 'text-green-400'}`}>
        {pct}%
      </span>
      <span
        className={`text-lg font-semibold uppercase tracking-widest ${
          fake ? 'text-red-400' : 'text-green-400'
        }`}
      >
        {event.label}
      </span>
      <span className="text-xs text-gray-500">
        window #{event.window_id} · {event.model ?? 'model'}
      </span>
    </div>
  )
}

function Timeline({ events }: { events: StreamEvent[] }) {
  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 space-y-2">
      <p className="text-sm font-medium text-gray-300">Last {MAX_TIMELINE} windows (1s hop)</p>
      <div className="flex items-end gap-px h-12">
        {events.map((e) => (
          <div
            key={e.window_id}
            className={`flex-1 rounded-sm ${e.label === 'fake' ? 'bg-red-500' : 'bg-green-500'}`}
            style={{ height: `${Math.max(15, e.confidence * 100)}%`, opacity: 0.45 + e.confidence * 0.55 }}
            title={`#${e.window_id} ${e.label} ${(e.confidence * 100).toFixed(0)}%`}
          />
        ))}
      </div>
      <p className="text-[11px] text-gray-600">
        Each bar is a 3-second window scored every second — green real, red fake.
      </p>
    </div>
  )
}

export default function LiveTab() {
  const [status, setStatus] = useState<StreamStatus>('idle')
  const [events, setEvents] = useState<StreamEvent[]>([])
  const [error, setError] = useState<string | null>(null)
  const streamRef = useRef<LiveMicStream | null>(null)

  // Never leave the mic open when the tab unmounts.
  useEffect(() => () => streamRef.current?.stop(), [])

  const startStop = async () => {
    if (status === 'live' || status === 'connecting' || status === 'authenticating') {
      streamRef.current?.stop()
      return
    }
    setError(null)
    setEvents([])
    const stream = new LiveMicStream({
      onStatus: setStatus,
      onError: setError,
      onEvent: (e) => setEvents((prev) => [...prev.slice(-(MAX_TIMELINE - 1)), e]),
    })
    streamRef.current = stream
    await stream.start()
  }

  const running = status === 'live' || status === 'connecting' || status === 'authenticating'
  const latest = events.length > 0 ? events[events.length - 1] : null

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Live Microphone Analysis</h2>
        <p className="text-sm text-gray-400">
          Streams your microphone to the detector in real time — a verdict per second after a
          3-second warm-up. Audio is scored in memory and never written to disk.
        </p>
      </div>

      {!hasToken() && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4 text-yellow-300 text-sm">
          Log in first (top-right) — the live stream requires authentication.
        </div>
      )}

      <div className="flex items-center gap-4">
        <button
          onClick={startStop}
          disabled={!hasToken()}
          className={`flex-1 font-medium py-3 px-6 rounded-lg transition-colors text-white disabled:bg-gray-700 disabled:cursor-not-allowed ${
            running ? 'bg-red-600 hover:bg-red-700' : 'bg-indigo-600 hover:bg-indigo-700'
          }`}
        >
          {running ? '■ Stop listening' : '🎙 Start live analysis'}
        </button>
        <span
          className={`text-sm font-mono px-3 py-2 rounded-lg border ${
            status === 'live'
              ? 'text-green-400 border-green-700 bg-green-900/20'
              : 'text-gray-400 border-gray-700 bg-gray-800'
          }`}
        >
          {STATUS_LABEL[status]}
        </span>
      </div>

      {error && (
        <div className="bg-red-900/40 border border-red-700 rounded-lg p-4 text-red-300 text-sm">
          {error}
        </div>
      )}

      {status === 'live' && !latest && (
        <div className="bg-gray-800 rounded-xl p-5 border border-gray-700 text-sm text-gray-400">
          Listening… first verdict arrives after ~3 seconds of speech.
        </div>
      )}

      {latest && (
        <div className="space-y-4">
          <VerdictCard event={latest} />
          <Timeline events={events} />
        </div>
      )}

      <p className="text-[11px] text-gray-600">
        Sessions are capped server-side (concurrent slots + 15 min max) to keep the shared
        detector responsive.
      </p>
    </div>
  )
}
