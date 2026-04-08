import type { SkillDetail } from '../types';

interface Props {
  details: SkillDetail[];
}

const CATEGORY_LABELS: Record<string, string> = {
  programming_languages: 'Programming Languages',
  machine_learning: 'Machine Learning',
  deep_learning: 'Deep Learning',
  cloud: 'Cloud',
  databases: 'Databases',
  devops: 'DevOps',
  frontend: 'Frontend',
  backend: 'Backend',
  data_engineering: 'Data Engineering',
  testing: 'Testing',
  security: 'Security',
  architecture: 'Architecture',
  analytics: 'Analytics',
  mobile: 'Mobile',
  soft_skills: 'Soft Skills',
  tools: 'Tools',
  default: 'Other',
};

export default function CategoryBreakdown({ details }: Props) {
  // Group by category
  const byCategory: Record<string, { total: number; matched: number }> = {};
  for (const d of details) {
    const cat = d.category || 'default';
    if (!byCategory[cat]) byCategory[cat] = { total: 0, matched: 0 };
    byCategory[cat].total++;
    if (d.status === 'matched') byCategory[cat].matched++;
  }

  const rows = Object.entries(byCategory)
    .filter(([, v]) => v.total > 0)
    .sort((a, b) => b[1].total - a[1].total);

  if (rows.length === 0) return null;

  return (
    <div className="category-breakdown">
      <h3 className="category-breakdown__title">Match by Category</h3>
      <div className="category-bars">
        {rows.map(([cat, { total, matched }]) => {
          const pct = Math.round((matched / total) * 100);
          const color =
            pct >= 70 ? 'var(--green)' : pct >= 40 ? 'var(--amber)' : 'var(--red)';
          return (
            <div key={cat} className="category-bar">
              <div className="category-bar__meta">
                <span className="category-bar__name">
                  {CATEGORY_LABELS[cat] ?? cat}
                </span>
                <span className="category-bar__count" style={{ color }}>
                  {matched}/{total}
                </span>
              </div>
              <div className="category-bar__track">
                <div
                  className="category-bar__fill"
                  style={{ width: `${pct}%`, background: color }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
