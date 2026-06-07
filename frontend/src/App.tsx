import { useState } from 'react'
import DetectTab from './pages/DetectTab'
import GenerateTab from './pages/GenerateTab'
import ResultsTab from './pages/ResultsTab'
import ApiSettings from './components/ApiSettings'

type Tab = 'detect' | 'generate' | 'results'

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('detect')

  const tabs: { id: Tab; label: string }[] = [
    { id: 'detect', label: 'Detect' },
    { id: 'generate', label: 'Generate' },
    { id: 'results', label: 'Results' },
  ]

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* Header */}
      <header className="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div className="max-w-5xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">VG</span>
            </div>
            <span className="text-white font-semibold text-lg">VoiceGuard</span>
            <span className="text-gray-500 text-sm hidden sm:block">
              Real-time Voice Deepfake Detection
            </span>
          </div>
          <span className="text-xs text-gray-600 font-mono">v0.1.0</span>
        </div>
      </header>

      {/* Tab bar */}
      <nav className="bg-gray-900 border-b border-gray-800 px-6">
        <div className="max-w-5xl mx-auto flex gap-1">
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-6 py-3 text-sm font-medium transition-colors border-b-2 ${
                activeTab === tab.id
                  ? 'text-indigo-400 border-indigo-500'
                  : 'text-gray-400 border-transparent hover:text-gray-200'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </nav>

      {/* Content */}
      <main className="flex-1 px-6 py-8">
        <div className="max-w-5xl mx-auto">
          {activeTab === 'detect' && <DetectTab />}
          {activeTab === 'generate' && <GenerateTab />}
          {activeTab === 'results' && <ResultsTab />}
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 border-t border-gray-800 px-6 py-3">
        <div className="max-w-5xl mx-auto flex items-center justify-between text-xs text-gray-600">
          <span>Canadian University Dubai · GP2 2025/26</span>
          <span>Apache 2.0 License</span>
        </div>
      </footer>

      {/* Floating API settings (gear button, always visible) */}
      <ApiSettings />
    </div>
  )
}
