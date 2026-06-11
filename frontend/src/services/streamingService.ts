/**
 * Live microphone streaming to WS /ws/stream.
 *
 * Protocol: open the socket, send `{"token": "<jwt>"}` as the first message
 * (keeps the JWT out of proxy access logs), wait for `{"type": "auth_ok"}`,
 * then stream raw PCM (int16, 16 kHz, mono). The server re-scores the session
 * from its start as audio arrives (first verdict at 3s, then every ~2s) and
 * replies with StreamDetectionEvent JSON; the event with `final: true` covers
 * the full scoring cap and is the last one — later audio is not analyzed.
 */
import { API_BASE, getWsUrl, getToken } from '../config/apiConfig'

export type StreamEvent = {
  timestamp_ms: number
  window_id: number
  label: 'real' | 'fake'
  confidence: number
  model: string | null
  final?: boolean
}

export type StreamStatus =
  | 'idle'
  | 'connecting'
  | 'authenticating'
  | 'live'
  | 'busy' // server's concurrent-stream slots are full (close code 1013)
  | 'closed'
  | 'error'

export type StreamHandlers = {
  onEvent: (e: StreamEvent) => void
  onStatus: (s: StreamStatus) => void
  onError: (message: string) => void
}

const TARGET_RATE = 16000
/** Send ~256ms of audio per WS frame — small enough to feel live, large
 *  enough to keep frame overhead negligible. */
const CHUNK_SAMPLES = 4096

/** AudioWorklet that forwards raw input frames to the main thread. Inlined as
 *  a Blob so the build needs no separate worklet asset. */
const WORKLET_SOURCE = `
class VgPcmForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0]
    if (ch) this.port.postMessage(ch.slice(0))
    return true
  }
}
registerProcessor('vg-pcm-forwarder', VgPcmForwarder)
`

/** Linear-interpolation resample to 16 kHz (AudioContext may refuse to run at
 *  16 kHz on some platforms and hand us 44.1/48 kHz instead). */
export const resampleTo16k = (input: Float32Array, fromRate: number): Float32Array => {
  if (fromRate === TARGET_RATE) return input
  const outLen = Math.floor((input.length * TARGET_RATE) / fromRate)
  const out = new Float32Array(outLen)
  const ratio = fromRate / TARGET_RATE
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio
    const i0 = Math.floor(pos)
    const i1 = Math.min(i0 + 1, input.length - 1)
    out[i] = input[i0] + (input[i1] - input[i0]) * (pos - i0)
  }
  return out
}

/** Float32 [-1, 1] → int16 little-endian bytes. The concrete ArrayBuffer-backed
 *  type keeps the result assignable to WebSocket.send's BufferSource. */
export const floatToInt16 = (input: Float32Array): Int16Array<ArrayBuffer> => {
  const out = new Int16Array(input.length)
  for (let i = 0; i < input.length; i++) {
    const v = Math.max(-1, Math.min(1, input[i]))
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff
  }
  return out
}

export class LiveMicStream {
  private ws: WebSocket | null = null
  private ctx: AudioContext | null = null
  private mediaStream: MediaStream | null = null
  private pending: Float32Array[] = []
  private pendingSamples = 0
  private stopped = false

  constructor(private handlers: StreamHandlers) {}

  async start(): Promise<void> {
    this.stopped = false
    this.handlers.onStatus('connecting')

    // Same-origin under the API base (e.g. wss://host/api/ws/stream) — the bare
    // /ws/stream path only existed behind nginx's extra /ws/ location; the demo
    // server (vg_serve) mounts the API solely under /api, so a root-path WS hit
    // the static-files mount and the connection always failed.
    const ws = new WebSocket(getWsUrl(`${API_BASE}/ws/stream`))
    this.ws = ws
    ws.binaryType = 'arraybuffer'

    ws.onopen = () => {
      this.handlers.onStatus('authenticating')
      ws.send(JSON.stringify({ token: getToken() }))
    }

    ws.onmessage = (msg) => {
      let data: Record<string, unknown>
      try {
        data = JSON.parse(msg.data as string)
      } catch {
        return
      }
      if (data.type === 'auth_ok') {
        void this.startMicrophone()
        return
      }
      this.handlers.onEvent(data as unknown as StreamEvent)
    }

    ws.onclose = (e) => {
      if (this.stopped) return
      if (e.code === 1013) {
        this.handlers.onStatus('busy')
        this.handlers.onError('All live-analysis slots are busy — try again in a minute.')
      } else if (e.code === 1008) {
        this.handlers.onStatus('error')
        this.handlers.onError('Stream rejected — log in again and retry.')
      } else {
        this.handlers.onStatus('closed')
      }
      this.teardownAudio()
    }

    ws.onerror = () => {
      if (this.stopped) return
      this.handlers.onStatus('error')
      this.handlers.onError('Could not reach the live-analysis stream.')
    }
  }

  private async startMicrophone(): Promise<void> {
    try {
      this.mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: false },
      })
    } catch {
      this.handlers.onError('Microphone access denied — allow the mic permission and retry.')
      this.stop()
      return
    }

    const ctx = new AudioContext()
    this.ctx = ctx
    const workletUrl = URL.createObjectURL(
      new Blob([WORKLET_SOURCE], { type: 'application/javascript' }),
    )
    try {
      await ctx.audioWorklet.addModule(workletUrl)
    } finally {
      URL.revokeObjectURL(workletUrl)
    }

    const source = ctx.createMediaStreamSource(this.mediaStream)
    const node = new AudioWorkletNode(ctx, 'vg-pcm-forwarder')
    node.port.onmessage = (e: MessageEvent<Float32Array>) => this.push(e.data, ctx.sampleRate)
    source.connect(node)
    // Worklets need a path to the destination to keep processing; route through
    // a zero-gain node so the user doesn't hear their own mic.
    const mute = ctx.createGain()
    mute.gain.value = 0
    node.connect(mute)
    mute.connect(ctx.destination)

    this.handlers.onStatus('live')
  }

  private push(frame: Float32Array, sampleRate: number): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return
    this.pending.push(frame)
    this.pendingSamples += frame.length
    // Flush once we hold ≥ CHUNK_SAMPLES at the *mic* rate (chunk shrinks on resample).
    if (this.pendingSamples < CHUNK_SAMPLES * (sampleRate / TARGET_RATE)) return

    const joined = new Float32Array(this.pendingSamples)
    let off = 0
    for (const f of this.pending) {
      joined.set(f, off)
      off += f.length
    }
    this.pending = []
    this.pendingSamples = 0

    const pcm = floatToInt16(resampleTo16k(joined, sampleRate))
    this.ws.send(pcm)  // Int16Array is a BufferSource; sent as a binary frame
  }

  stop(): void {
    this.stopped = true
    this.teardownAudio()
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) this.ws.close(1000)
    this.ws = null
    this.handlers.onStatus('idle')
  }

  private teardownAudio(): void {
    this.mediaStream?.getTracks().forEach((t) => t.stop())
    this.mediaStream = null
    void this.ctx?.close().catch(() => undefined)
    this.ctx = null
    this.pending = []
    this.pendingSamples = 0
  }
}
