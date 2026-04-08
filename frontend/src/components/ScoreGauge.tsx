interface Props {
  score: number; // 0–100
}

export default function ScoreGauge({ score }: Props) {
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;

  const color =
    score >= 70 ? '#16a34a' : score >= 40 ? '#b45309' : '#dc2626';

  const label =
    score >= 70 ? 'Strong Match' : score >= 40 ? 'Partial Match' : 'Weak Match';

  return (
    <div className="score-gauge">
      <svg viewBox="0 0 120 120" width="150" height="150">
        {/* Track */}
        <circle
          cx="60" cy="60" r={radius}
          fill="none"
          stroke="#e0e0e0"
          strokeWidth="9"
        />
        {/* Progress */}
        <circle
          cx="60" cy="60" r={radius}
          fill="none"
          stroke={color}
          strokeWidth="9"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 60 60)"
          style={{ transition: 'stroke-dashoffset 0.7s ease' }}
        />
        <text x="60" y="56" textAnchor="middle" fontSize="22" fontWeight="700" fill="#1a1a1a">
          {Math.round(score)}
        </text>
        <text x="60" y="70" textAnchor="middle" fontSize="9" fill="#9a9a9a">
          / 100
        </text>
      </svg>
      <p className="score-label" style={{ color }}>{label}</p>
    </div>
  );
}
