'use client';

import React, { useState, useEffect } from 'react';

interface HealthData {
  status: string;
  active_checkpoint: string;
  memory_items: number;
  system_resources: {
    cpu_percent: number;
    ram_percent: number;
    cuda_available: boolean;
    gpu_name: string;
  };
}

export default function HomePage() {
  const [health, setHealth] = useState<HealthData | null>(null);
  const [activeTab, setActiveTab] = useState<'chat' | 'memory' | 'graph' | 'queue' | 'models' | 'eval'>('chat');
  const [query, setQuery] = useState('');
  const [chatLog, setChatLog] = useState<{ role: 'user' | 'assistant'; text: string; evidence?: any[] }[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setHealth(data))
      .catch((err) => console.error(err));
  }, []);

  const handleSend = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userText = query.trim();
    setQuery('');
    setChatLog((prev) => [...prev, { role: 'user', text: userText }]);
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userText, max_new_tokens: 48, retrieval_mode: 'hybrid' }),
      });
      const data = await res.json();
      if (data.blocked) {
        setChatLog((prev) => [...prev, { role: 'assistant', text: `[Blocked]: ${data.reason}` }]);
      } else {
        setChatLog((prev) => [
          ...prev,
          {
            role: 'assistant',
            text: data.model_response.text,
            evidence: data.retrieved_memory,
          },
        ]);
      }
    } catch (err: any) {
      setChatLog((prev) => [...prev, { role: 'assistant', text: `Error: ${err.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex h-screen overflow-hidden flex-col">
      {/* Header */}
      <header className="border-b border-slate-800 bg-[#0d1322] px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center font-bold text-white shadow">
            A
          </div>
          <div>
            <h1 className="text-sm font-bold text-white tracking-wide">AAROHAN AI Next.js</h1>
            <p className="text-xs text-slate-400">Independent Foundation Transformer &bull; 6-Layer Memory</p>
          </div>
        </div>
        <div className="text-xs text-slate-300 flex items-center space-x-4">
          <div className="px-2.5 py-1 bg-slate-800/80 rounded border border-slate-700">
            Active: <span className="text-blue-400 font-mono">{health?.active_checkpoint?.split(/[\\/]/).pop() || 'Loading...'}</span>
          </div>
          <div className="px-2.5 py-1 bg-slate-800/80 rounded border border-slate-700">
            Device: <span className="text-emerald-400">{health?.system_resources?.cuda_available ? health.system_resources.gpu_name : 'CPU'}</span>
          </div>
        </div>
      </header>

      {/* Main Container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Navigation Sidebar */}
        <aside className="w-64 border-r border-slate-800 bg-[#0c1220] p-4 flex flex-col space-y-1">
          <button
            onClick={() => setActiveTab('chat')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'chat' ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' : 'text-slate-400 hover:text-white'
            }`}
          >
            AI Chat
          </button>
          <button
            onClick={() => setActiveTab('memory')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'memory' ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' : 'text-slate-400 hover:text-white'
            }`}
          >
            Memory Explorer
          </button>
          <button
            onClick={() => setActiveTab('graph')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'graph' ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' : 'text-slate-400 hover:text-white'
            }`}
          >
            Knowledge Graph
          </button>
          <button
            onClick={() => setActiveTab('queue')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'queue' ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' : 'text-slate-400 hover:text-white'
            }`}
          >
            Learning Queue
          </button>
          <button
            onClick={() => setActiveTab('models')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'models' ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' : 'text-slate-400 hover:text-white'
            }`}
          >
            Model Registry
          </button>
          <button
            onClick={() => setActiveTab('eval')}
            className={`w-full text-left px-3 py-2 rounded-lg text-sm font-medium transition ${
              activeTab === 'eval' ? 'bg-blue-600/10 text-blue-400 border border-blue-500/30' : 'text-slate-400 hover:text-white'
            }`}
          >
            Evaluation Centre
          </button>

          <div className="mt-auto pt-4 border-t border-slate-800 text-xs text-slate-400 space-y-1">
            <div>Zero Commercial API Fallback</div>
            <div className="text-emerald-400 font-semibold">&#10003; 100% Native PyTorch</div>
          </div>
        </aside>

        {/* Content Pane */}
        <main className="flex-1 overflow-y-auto p-6 bg-[#090d16] flex flex-col">
          {activeTab === 'chat' && (
            <div className="flex-1 flex flex-col space-y-4 max-w-4xl mx-auto w-full">
              <div className="flex-1 overflow-y-auto border border-slate-800 bg-slate-900/50 rounded-xl p-4 space-y-4">
                {chatLog.length === 0 ? (
                  <div className="text-xs text-slate-400">
                    AAROHAN neural inference active. Type a prompt below to interact with the foundation causal model.
                  </div>
                ) : (
                  chatLog.map((msg, i) => (
                    <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                      <div
                        className={`max-w-2xl px-4 py-2.5 rounded-2xl text-xs ${
                          msg.role === 'user'
                            ? 'bg-blue-600 text-white rounded-tr-none'
                            : 'bg-slate-800 text-slate-200 border border-slate-700 rounded-tl-none space-y-2'
                        }`}
                      >
                        <div>{msg.text}</div>
                        {msg.evidence && msg.evidence.length > 0 && (
                          <div className="pt-2 border-t border-slate-700/60 text-[11px] text-slate-400">
                            <span className="font-semibold text-blue-400">Retrieved Memory:</span>
                            {msg.evidence.map((ev: any, j: number) => (
                              <div key={j} className="mt-1 p-1.5 rounded bg-slate-900 border border-slate-800">
                                <div>{ev.content}</div>
                                <div className="text-[10px] text-slate-500">Source: {ev.source} &bull; Status: {ev.verification_status}</div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  ))
                )}
                {loading && <div className="text-xs text-slate-500 font-mono">Model generating token sequence...</div>}
              </div>

              <form onSubmit={handleSend} className="flex space-x-2">
                <input
                  type="text"
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  placeholder="Ask AAROHAN or query memory..."
                  className="flex-1 bg-slate-900 border border-slate-700 focus:border-blue-500 rounded-xl px-4 py-3 text-xs text-white focus:outline-none"
                />
                <button
                  type="submit"
                  disabled={loading}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-500 text-white text-xs font-semibold rounded-xl"
                >
                  Send
                </button>
              </form>
            </div>
          )}

          {activeTab !== 'chat' && (
            <div className="p-4 rounded-xl bg-slate-900/40 border border-slate-800 text-xs text-slate-300">
              <h2 className="text-sm font-semibold text-white mb-2 uppercase tracking-wide">{activeTab} View</h2>
              <p>Module operational and linked to FastAPI backend endpoints.</p>
              <p className="mt-2 text-slate-400">The full interactive dashboard is also accessible directly via FastAPI at <span className="font-mono text-blue-400">http://127.0.0.1:8000/app</span>.</p>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
