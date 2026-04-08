import { useState } from 'react';
import type { SkillGapResult } from '../types';
import ScoreGauge from './ScoreGauge';
import SkillChip from './SkillChip';
import CategoryBreakdown from './CategoryBreakdown';

interface Props {
  result: SkillGapResult;
}

export default function ResultsPanel({ result }: Props) {
  const { matched, missing, extra, weighted_score, explanation, details, job_skills } = result;
  const [tableCategory, setTableCategory] = useState('all');
  const [copied, setCopied] = useState(false);

  // Related skills from details
  const relatedDetails = details.filter((d) => d.status === 'related');
  const relatedSkills = relatedDetails.map((d) => ({
    skill: d.job_skill,
    similarity: d.similarity,
  }));

  // Missing skills sorted by weight descending (highest priority first)
  const weightBySkill = Object.fromEntries(details.map((d) => [d.job_skill, d.job_weight]));
  const missingSorted = [...missing].sort(
    (a, b) => (weightBySkill[b] ?? 1) - (weightBySkill[a] ?? 1)
  );

  // Category filter options for breakdown table
  const categories = ['all', ...Array.from(new Set(details.map((d) => d.category))).sort()];
  const filteredDetails =
    tableCategory === 'all' ? details : details.filter((d) => d.category === tableCategory);

  const handleCopy = () => {
    const lines = [
      `Skill Gap Analysis — Score: ${weighted_score}/100`,
      '',
      `Matched (${matched.length}): ${matched.join(', ') || 'none'}`,
      `Missing (${missing.length}): ${missingSorted.join(', ') || 'none'}`,
      `Bonus skills (${extra.length}): ${extra.join(', ') || 'none'}`,
      '',
      explanation,
    ];
    navigator.clipboard.writeText(lines.join('\n')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <section className="results">
      {/* ── Top row: gauge + stats ── */}
      <div className="results__top">
        <ScoreGauge score={weighted_score} />

        <div className="results__right">
          {/* Stat cards */}
          <div className="stat-cards">
            <StatCard
              label="Match Score"
              value={`${weighted_score}`}
              unit="/100"
              color={weighted_score >= 70 ? 'var(--green)' : weighted_score >= 40 ? 'var(--amber)' : 'var(--red)'}
            />
            <StatCard
              label="Matched"
              value={`${matched.length}`}
              unit={`/ ${job_skills.length}`}
              color="var(--green)"
            />
            <StatCard
              label="Missing"
              value={`${missing.length}`}
              unit={`/ ${job_skills.length}`}
              color="var(--red)"
            />
            <StatCard
              label="Bonus Skills"
              value={`${extra.length}`}
              unit="extra"
              color="var(--blue)"
            />
          </div>

          {/* Explanation */}
          <p className="results__explanation">{explanation}</p>

          {/* Copy button */}
          <button className="btn btn--ghost btn--sm copy-btn" onClick={handleCopy} type="button">
            {copied ? '✓ Copied' : '⎘ Copy Results'}
          </button>
        </div>
      </div>

      {/* ── Category breakdown ── */}
      <CategoryBreakdown details={details} />

      {/* ── Skill sections ── */}
      <div className="results__grid">
        <SkillSection
          title="Matched Skills"
          count={matched.length}
          accent="matched"
          empty="No skills directly matched."
        >
          {matched.map((s) => (
            <SkillChip key={s} skill={s} variant="matched" />
          ))}
        </SkillSection>

        <SkillSection
          title="Missing Skills"
          count={missingSorted.length}
          accent="missing"
          empty="No required skills missing — great!"
          hint="sorted by priority"
        >
          {missingSorted.map((s) => (
            <SkillChip key={s} skill={s} variant="missing" />
          ))}
        </SkillSection>

        {relatedSkills.length > 0 && (
          <SkillSection
            title="Partially Matched"
            count={relatedSkills.length}
            accent="related"
            empty=""
          >
            {relatedSkills.map(({ skill, similarity }) => (
              <SkillChip key={skill} skill={skill} variant="related" similarity={similarity} />
            ))}
          </SkillSection>
        )}

        {extra.length > 0 && (
          <SkillSection
            title="Bonus Skills"
            count={extra.length}
            accent="extra"
            empty=""
            hint="not in job description"
          >
            {extra.map((s) => (
              <SkillChip key={s} skill={s} variant="extra" />
            ))}
          </SkillSection>
        )}
      </div>

      {/* ── Detail breakdown table ── */}
      <details className="details-toggle">
        <summary>Full breakdown ({details.length} job skills)</summary>

        {/* Category filter */}
        <div className="table-filter">
          <label className="table-filter__label" htmlFor="cat-filter">Filter by category:</label>
          <select
            id="cat-filter"
            className="table-filter__select"
            value={tableCategory}
            onChange={(e) => setTableCategory(e.target.value)}
          >
            {categories.map((c) => (
              <option key={c} value={c}>
                {c === 'all' ? 'All categories' : c.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
          {tableCategory !== 'all' && (
            <button
              className="btn btn--ghost btn--sm"
              onClick={() => setTableCategory('all')}
              type="button"
            >
              Clear
            </button>
          )}
        </div>

        <div className="breakdown-table-wrap">
          <table className="breakdown-table">
            <thead>
              <tr>
                <th>Job Skill</th>
                <th>Category</th>
                <th>Status</th>
                <th>Best Match</th>
                <th>Similarity</th>
                <th>Weight</th>
              </tr>
            </thead>
            <tbody>
              {filteredDetails.map((d) => (
                <tr key={d.job_skill} className={`row--${d.status}`}>
                  <td>{d.job_skill}</td>
                  <td className="td--category">{d.category.replace(/_/g, ' ')}</td>
                  <td>
                    <span className={`chip chip--${d.status}`}>{d.status}</span>
                  </td>
                  <td>{d.best_match ?? '—'}</td>
                  <td>{d.similarity > 0 ? `${(d.similarity * 100).toFixed(0)}%` : '—'}</td>
                  <td>{d.job_weight.toFixed(1)}×</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </section>
  );
}

// ── helpers ────────────────────────────────────────────────────────────────

interface SectionProps {
  title: string;
  count: number;
  accent: 'matched' | 'missing' | 'extra' | 'related';
  empty: string;
  hint?: string;
  children: React.ReactNode;
}

function SkillSection({ title, count, accent, empty, hint, children }: SectionProps) {
  return (
    <div className={`skill-section skill-section--${accent}`}>
      <h3 className="skill-section__title">
        {title}
        <span className="skill-section__count">{count}</span>
        {hint && <span className="skill-section__hint">{hint}</span>}
      </h3>
      <div className="chip-list">
        {count === 0 ? <p className="skill-section__empty">{empty}</p> : children}
      </div>
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string;
  unit: string;
  color: string;
}

function StatCard({ label, value, unit, color }: StatCardProps) {
  return (
    <div className="stat-card">
      <p className="stat-card__label">{label}</p>
      <p className="stat-card__value">
        <span style={{ color }}>{value}</span>
        <span className="stat-card__unit">{unit}</span>
      </p>
    </div>
  );
}
