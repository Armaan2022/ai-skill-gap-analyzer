import { useState, useRef } from 'react';
import type { ChangeEvent } from 'react';
import { analyzeSkillGap, saveAnalysis } from './api';
import type { SkillGapResult } from './types';
import ResultsPanel from './components/ResultsPanel';
import HistoryPage from './components/HistoryPage';
import { extractTextFromPdf } from './pdfExtract';
import './App.css';

type Tab = 'analyze' | 'history';

export default function App() {
  const [tab, setTab] = useState<Tab>('analyze');

  const [resumeText, setResumeText] = useState('');
  const [jobText, setJobText] = useState('');
  const [jobTitle, setJobTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SkillGapResult | null>(null);
  const [saveState, setSaveState] = useState<'idle' | 'saving' | 'saved'>('idle');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const resultsRef = useRef<HTMLDivElement>(null);

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    e.target.value = '';

    if (file.type === 'application/pdf' || file.name.endsWith('.pdf')) {
      setPdfLoading(true);
      setError(null);
      try {
        const text = await extractTextFromPdf(file);
        if (!text.trim()) {
          setError('Could not extract text from this PDF. It may be scanned/image-based. Try copying the text manually.');
        } else {
          setResumeText(text);
        }
      } catch {
        setError('Failed to read the PDF file. Please try a different file or paste the text directly.');
      } finally {
        setPdfLoading(false);
      }
    } else {
      const reader = new FileReader();
      reader.onload = (ev) => setResumeText(ev.target?.result as string ?? '');
      reader.readAsText(file);
    }
  };

  const handleAnalyze = async () => {
    if (!resumeText.trim() || !jobText.trim()) {
      setError('Please provide both a resume and a job description.');
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    setSaveState('idle');
    try {
      const data = await analyzeSkillGap(resumeText, jobText);
      setResult(data);
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth' }), 100);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!result) return;
    setSaveState('saving');
    try {
      await saveAnalysis(
        resumeText,
        jobTitle.trim() || 'Untitled Job',
        jobText,
        result,
      );
      setSaveState('saved');
    } catch {
      setSaveState('idle');
      setError('Failed to save analysis. Please try again.');
    }
  };

  const handleReset = () => {
    setResumeText('');
    setJobText('');
    setJobTitle('');
    setResult(null);
    setError(null);
    setSaveState('idle');
  };

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="app__header">
        <div className="header__inner">
          <div className="header__brand">
            <h1 className="header__title">Skill Gap Analyzer</h1>
          </div>
          <p className="header__sub">Paste your resume and a job description to instantly see how well you match.</p>
        </div>
      </header>

      {/* ── Tabs ── */}
      <div className="tabs-bar">
        <div className="tabs-bar__inner">
          <button
            className={`tab-btn ${tab === 'analyze' ? 'tab-btn--active' : ''}`}
            onClick={() => setTab('analyze')}
            type="button"
          >
            Analyze
          </button>
          <button
            className={`tab-btn ${tab === 'history' ? 'tab-btn--active' : ''}`}
            onClick={() => setTab('history')}
            type="button"
          >
            History
          </button>
        </div>
      </div>

      <main className="app__main">
        {tab === 'history' ? (
          <HistoryPage />
        ) : (
          <>
            {/* ── Input section ── */}
            <section className="input-section">
              {/* Resume */}
              <div className="input-card">
                <div className="input-card__header">
                  <label className="input-card__label" htmlFor="resume-input">
                    Your Resume
                  </label>
                  <button
                    className="btn btn--ghost btn--sm"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={pdfLoading}
                    type="button"
                  >
                    {pdfLoading ? '⏳ Reading PDF…' : '📎 Upload file'}
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".txt,.md,.text,.pdf"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                  />
                </div>
                <textarea
                  id="resume-input"
                  className="input-card__textarea"
                  placeholder="Paste your resume text here, or upload a .txt / .pdf file…"
                  value={resumeText}
                  onChange={(e) => setResumeText(e.target.value)}
                  rows={12}
                />
                <p className="input-card__hint">{resumeText.trim().split(/\s+/).filter(Boolean).length} words</p>
              </div>

              {/* Job description */}
              <div className="input-card">
                <div className="input-card__header">
                  <label className="input-card__label" htmlFor="job-title-input">
                    Job Description
                  </label>
                </div>
                <input
                  id="job-title-input"
                  className="input-card__title-input"
                  placeholder="Job title (e.g. Senior ML Engineer)"
                  value={jobTitle}
                  onChange={(e) => setJobTitle(e.target.value)}
                />
                <textarea
                  id="job-input"
                  className="input-card__textarea"
                  placeholder="Paste the job description here…"
                  value={jobText}
                  onChange={(e) => setJobText(e.target.value)}
                  rows={11}
                />
                <p className="input-card__hint">{jobText.trim().split(/\s+/).filter(Boolean).length} words</p>
              </div>
            </section>

            {/* ── Actions ── */}
            <div className="action-row">
              {error && <p className="error-msg">{error}</p>}
              <div className="action-row__buttons">
                {result && (
                  <>
                    <button className="btn btn--ghost" onClick={handleReset} type="button">
                      Clear
                    </button>
                    <button
                      className="btn btn--ghost"
                      onClick={handleSave}
                      disabled={saveState !== 'idle'}
                      type="button"
                    >
                      {saveState === 'saving' ? 'Saving…' : saveState === 'saved' ? '✓ Saved' : 'Save Analysis'}
                    </button>
                  </>
                )}
                <button
                  className="btn btn--primary"
                  onClick={handleAnalyze}
                  disabled={loading}
                  type="button"
                >
                  {loading ? (
                    <span className="spinner-label">
                      <span className="spinner" />
                      Analyzing…
                    </span>
                  ) : (
                    'Analyze Match'
                  )}
                </button>
              </div>
            </div>

            {/* ── Results ── */}
            {result && (
              <div ref={resultsRef}>
                <ResultsPanel result={result} />
              </div>
            )}
          </>
        )}
      </main>

      <footer className="app__footer">
        <p>Powered by spaCy · Sentence Transformers · FastAPI</p>
      </footer>
    </div>
  );
}
