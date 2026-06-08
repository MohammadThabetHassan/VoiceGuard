import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts'

// Final results — ASVspoof 2021 LA (eval partition), lower EER is better.
// XLS-R + AASIST (Kokoro-hardened) is the deployed production detector.
const EER_RESULTS = [
  { model: 'XLS-R + AASIST', eer: 2.61, prod: true },
  { model: 'Wav2Vec2-large', eer: 3.09, prod: false },
  { model: 'WavLM-base+', eer: 8.11, prod: false },
  { model: 'WavLM-large', eer: 9.2, prod: false },
  { model: 'XLS-R', eer: 10.47, prod: false },
  { model: 'AASIST', eer: 10.9, prod: false },
  { model: 'DSFNet-V2', eer: 12.67, prod: false },
]

// Out-of-distribution generalisation of the production model to unseen TTS
// families, plus the genuine-speech pass rate.
const OOD = [
  { system: 'IndexTTS2 (deepfake)', rate: 100, kind: 'detect' },
  { system: 'Kokoro-82M (deepfake)', rate: 93, kind: 'detect' },
  { system: 'Genuine human speech', rate: 90, kind: 'pass' },
]

const CONFUSION = { TP: 237, FN: 0, FP: 1, TN: 236 }

function Stat({ value, label, accent }: { value: string; label: string; accent: string }) {
  return (
    <div className="bg-gray-800 rounded-xl p-5 border border-gray-700">
      <p className={`text-3xl font-bold font-mono ${accent}`}>{value}</p>
      <p className="text-xs text-gray-400 mt-1">{label}</p>
    </div>
  )
}

export default function ResultsTab() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Model Performance</h2>
        <p className="text-sm text-gray-400">
          Detection evaluated on ASVspoof 2021 LA; robustness on held-out modern TTS systems.
        </p>
      </div>

      {/* Headline stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat value="2.61%" label="Best EER (XLS-R + AASIST)" accent="text-indigo-300" />
        <Stat value="100%" label="IndexTTS2 detection" accent="text-green-400" />
        <Stat value="93%" label="Kokoro detection" accent="text-green-400" />
        <Stat value="90%" label="Genuine-voice pass rate" accent="text-green-400" />
      </div>

      {/* EER leaderboard */}
      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="w-full text-sm">
          <thead className="bg-gray-800 text-gray-400">
            <tr>
              {['Model', 'EER ↓', 'Role'].map((h) => (
                <th key={h} className="px-4 py-3 text-left font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {EER_RESULTS.map((row) => (
              <tr
                key={row.model}
                className={`transition-colors ${
                  row.prod ? 'bg-indigo-950/40' : 'bg-gray-900 hover:bg-gray-800'
                }`}
              >
                <td className="px-4 py-3 font-medium text-white">{row.model}</td>
                <td className="px-4 py-3 font-mono text-indigo-300">{row.eer.toFixed(2)}%</td>
                <td className="px-4 py-3">
                  {row.prod ? (
                    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-indigo-900/50 text-indigo-300 border border-indigo-700">
                      ★ Deployed
                    </span>
                  ) : (
                    <span className="text-xs text-gray-500">baseline</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* EER chart */}
      <div className="bg-gray-800 rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">Equal Error Rate by model (lower is better)</p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={EER_RESULTS} margin={{ left: -20 }}>
            <XAxis
              dataKey="model"
              tick={{ fill: '#9ca3af', fontSize: 10 }}
              interval={0}
              angle={-20}
              textAnchor="end"
              height={50}
            />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 11 }} unit="%" />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
              labelStyle={{ color: '#f3f4f6' }}
              formatter={(v: number) => [`${v}%`, 'EER']}
            />
            <Bar dataKey="eer" radius={[4, 4, 0, 0]}>
              {EER_RESULTS.map((entry) => (
                <Cell key={entry.model} fill={entry.prod ? '#6366f1' : '#374151'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* OOD robustness */}
      <div className="bg-gray-800 rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">
          Robustness to unseen voices — production model (XLS-R + AASIST)
        </p>
        <div className="space-y-3">
          {OOD.map(({ system, rate, kind }) => (
            <div key={system} className="flex items-center gap-3">
              <span className="w-44 shrink-0 text-sm text-gray-300">{system}</span>
              <div className="flex-1 h-3 bg-gray-900 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${
                    kind === 'pass' ? 'bg-emerald-500' : 'bg-indigo-500'
                  }`}
                  style={{ width: `${rate}%` }}
                />
              </div>
              <span className="w-10 text-right text-sm font-mono text-gray-300">{rate}%</span>
            </div>
          ))}
        </div>
        <p className="mt-3 text-xs text-gray-600">
          Detection = % of deepfakes correctly flagged. Pass rate = % of genuine speech correctly
          accepted.
        </p>
      </div>

      {/* Confusion matrix — classical baseline */}
      <div className="bg-gray-800 rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">
          Confusion matrix — Enhanced + XGBoost classical baseline
        </p>
        <div className="grid grid-cols-2 gap-1 max-w-xs">
          {[
            { label: 'TP', value: CONFUSION.TP, color: 'bg-green-900 text-green-300' },
            { label: 'FN', value: CONFUSION.FN, color: 'bg-red-900/40 text-red-400' },
            { label: 'FP', value: CONFUSION.FP, color: 'bg-yellow-900/40 text-yellow-400' },
            { label: 'TN', value: CONFUSION.TN, color: 'bg-green-900 text-green-300' },
          ].map(({ label, value, color }) => (
            <div key={label} className={`${color} rounded-lg p-4 text-center`}>
              <p className="text-xs opacity-70">{label}</p>
              <p className="text-2xl font-bold font-mono">{value}</p>
            </div>
          ))}
        </div>
        <div className="mt-2 text-xs text-gray-600">Rows: Actual · Columns: Predicted</div>
      </div>
    </div>
  )
}
