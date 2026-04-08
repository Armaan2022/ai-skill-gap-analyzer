import { useEffect, useState } from 'react';
import type { SavedAnalysis, SkillGapResult } from '../types';
import { listAnalyses, deleteAnalysis } from '../api';
import ResultsPanel from './ResultsPanel';

export default function HistoryPage() {
  const [analyses, setAnalyses] = useState<SavedAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<number | null>(null);
  const [deleting, setDeleting] = useState<number | null>(null);

  useEffect(() => {
    listAnalyses()
      .then(setAnalyses)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (id: number) => {
    setDeleting(id);
    try {
      await deleteAnalysis(id);
      setAnalyses((prev) => prev.filter((a) => a.id !== id));
      if (expanded === id) setExpanded(null);
    } catch {
      setError('Failed to delete analysis.');
    } finally {
      setDeleting(null);
    }
  };

  if (loading) return <p className="history-status">Loading history…</p>;
  if (error) return <p className="history-status history-status--error">{error}</p>;
  if (analyses.length === 0)
    return (
      <div className="history-empty">
        <p className="history-empty__title">No saved analyses yet.</p>
        <p className="history-empty__sub">
          Run an analysis and click <strong>Save Analysis</strong> to keep a record.
        </p>
      </div>
    );

  return (
    <div className="history">
      <p className="history__count">{analyses.length} saved {analyses.length === 1 ? 'analysis' : 'analyses'}</p>
      <div className="history-list">
        {analyses.map((a) => (
          <div key={a.id} className="history-card">
            <div className="history-card__header">
              <div className="history-card__meta">
                <span className="history-card__label">{a.label}</span>
                <span className="history-card__date">
                  {new Date(a.created_at).toLocaleDateString('en-US', {
                    month: 'short', day: 'numeric', year: 'numeric',
                  })}
                </span>
              </div>

              <div className="history-card__stats">
                <ScoreBadge score={a.result.weighted_score} />
                <span className="history-card__pill history-card__pill--matched">
                  {a.result.matched.length} matched
                </span>
                <span className="history-card__pill history-card__pill--missing">
                  {a.result.missing.length} missing
                </span>
              </div>

              <div className="history-card__actions">
                <button
                  className="btn btn--ghost btn--sm"
                  onClick={() => setExpanded(expanded === a.id ? null : a.id)}
                  type="button"
                >
                  {expanded === a.id ? 'Collapse' : 'View'}
                </button>
                <button
                  className="btn btn--ghost btn--sm btn--danger"
                  onClick={() => handleDelete(a.id)}
                  disabled={deleting === a.id}
                  type="button"
                >
                  {deleting === a.id ? '…' : 'Delete'}
                </button>
              </div>
            </div>

            {expanded === a.id && (
              <div className="history-card__results">
                <ResultsPanel result={a.result as SkillGapResult} />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoreBadge({ score }: { score: number }) {
  const color =
    score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--amber)' : 'var(--red)';
  return (
    <span className="score-badge" style={{ color, borderColor: color }}>
      {Math.round(score)}/100
    </span>
  );
}
