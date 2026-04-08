export interface SkillDetail {
  job_skill: string;
  job_weight: number;
  category: string;
  status: 'matched' | 'related' | 'missing';
  best_match: string | null;
  similarity: number;
}

export interface SkillGapResult {
  resume_skills: string[];
  job_skills: string[];
  matched: string[];
  missing: string[];
  extra: string[];
  weighted_score: number;
  explanation: string;
  details: SkillDetail[];
}

export interface SavedAnalysis {
  id: number;
  label: string;
  resume_id: number | null;
  job_id: number | null;
  result: SkillGapResult;
  created_at: string;
}
