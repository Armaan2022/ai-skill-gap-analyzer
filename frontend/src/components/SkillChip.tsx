type Variant = 'matched' | 'missing' | 'extra' | 'related';

interface Props {
  skill: string;
  variant: Variant;
  similarity?: number;
}

const VARIANT_CLASS: Record<Variant, string> = {
  matched: 'chip chip--matched',
  missing: 'chip chip--missing',
  extra:   'chip chip--extra',
  related: 'chip chip--related',
};

export default function SkillChip({ skill, variant, similarity }: Props) {
  return (
    <span className={VARIANT_CLASS[variant]} title={similarity != null ? `Similarity: ${(similarity * 100).toFixed(0)}%` : undefined}>
      {skill}
      {similarity != null && similarity < 1 && (
        <span className="chip__sim"> {(similarity * 100).toFixed(0)}%</span>
      )}
    </span>
  );
}
