import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
} from 'recharts'

const BASELINE_RESULTS = [
  { model: 'XGBoost', f1: 0.9979, eer: null, status: 'done' },
  { model: 'DSFNet', f1: null, eer: null, status: 'pending' },
  { model: 'Wav2Vec2', f1: null, eer: null, status: 'pending' },
]

function StatusBadge({ status }: { status: string }) {
  if (status === 'done')
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-green-900/40 text-green-400 border border-green-800">
        ✓ Done
      </span>
    )
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs bg-gray-800 text-gray-500 border border-gray-700">
      GPU pending
    </span>
  )
}

const CONFUSION_MOCK = {
  TP: 237, FN: 0, FP: 1, TN: 236,
}

export default function ResultsTab() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-xl font-semibold text-white mb-1">Model Comparison</h2>
        <p className="text-sm text-gray-400">
          Evaluated on ASVspoof 2021 LA + SM2026 benchmark dataset.
        </p>
      </div>

      {/* Results table */}
      <div className="overflow-x-auto rounded-xl border border-gray-700">
        <table className="w-full text-sm">
          <thead className="bg-gray-800 text-gray-400">
            <tr>
              {['Model', 'F1', 'EER', 'Status'].map((h) => (
                <th key={h} className="px-4 py-3 text-left font-medium">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {BASELINE_RESULTS.map((row) => (
              <tr key={row.model} className="bg-gray-900 hover:bg-gray-800 transition-colors">
                <td className="px-4 py-3 font-medium text-white">{row.model}</td>
                <td className="px-4 py-3 font-mono text-indigo-300">
                  {row.f1 !== null ? row.f1.toFixed(4) : '—'}
                </td>
                <td className="px-4 py-3 font-mono text-indigo-300">
                  {row.eer !== null ? `${(row.eer * 100).toFixed(2)}%` : '—'}
                </td>
                <td className="px-4 py-3">
                  <StatusBadge status={row.status} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* F1 chart (baseline only) */}
      <div className="bg-gray-800 rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">F1 Score Comparison</p>
        <ResponsiveContainer width="100%" height={180}>
          <BarChart data={BASELINE_RESULTS} margin={{ left: -20 }}>
            <XAxis dataKey="model" tick={{ fill: '#9ca3af', fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fill: '#9ca3af', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1f2937', border: '1px solid #374151' }}
              labelStyle={{ color: '#f3f4f6' }}
            />
            <Bar dataKey="f1" radius={[4, 4, 0, 0]}>
              {BASELINE_RESULTS.map((entry) => (
                <Cell
                  key={entry.model}
                  fill={entry.status === 'done' ? '#6366f1' : '#374151'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Confusion matrix */}
      <div className="bg-gray-800 rounded-xl p-6">
        <p className="text-sm text-gray-400 mb-4">
          Confusion Matrix — Enhanced+XGBoost (SM2026 Baseline)
        </p>
        <div className="grid grid-cols-2 gap-1 max-w-xs">
          {[
            { label: 'TP', value: CONFUSION_MOCK.TP, color: 'bg-green-900 text-green-300' },
            { label: 'FN', value: CONFUSION_MOCK.FN, color: 'bg-red-900/40 text-red-400' },
            { label: 'FP', value: CONFUSION_MOCK.FP, color: 'bg-yellow-900/40 text-yellow-400' },
            { label: 'TN', value: CONFUSION_MOCK.TN, color: 'bg-green-900 text-green-300' },
          ].map(({ label, value, color }) => (
            <div
              key={label}
              className={`${color} rounded-lg p-4 text-center`}
            >
              <p className="text-xs opacity-70">{label}</p>
              <p className="text-2xl font-bold font-mono">{value}</p>
            </div>
          ))}
        </div>
        <div className="mt-2 text-xs text-gray-600">
          Rows: Actual · Columns: Predicted
        </div>
      </div>

      {/* Detection history placeholder */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <p className="text-sm text-gray-400 mb-2">Detection History</p>
        <p className="text-xs text-gray-600">
          Upload audio files in the Detect tab to populate this table.
        </p>
      </div>
    </div>
  )
}
