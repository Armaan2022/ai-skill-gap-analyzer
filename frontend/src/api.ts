import type { SkillGapResult, SavedAnalysis } from './types';

const BASE = '/api/v1';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    throw new Error(detail?.detail ?? `Request failed (${res.status})`);
  }
  return res.json();
}

export async function analyzeSkillGap(
  resumeText: string,
  jobText: string,
): Promise<SkillGapResult> {
  const res = await fetch(`${BASE}/skill-gap`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume_text: resumeText, job_text: jobText }),
  });
  return handleResponse<SkillGapResult>(res);
}

export async function saveAnalysis(
  resumeText: string,
  jobTitle: string,
  jobDescription: string,
  result: SkillGapResult,
  label?: string,
): Promise<SavedAnalysis> {
  const res = await fetch(`${BASE}/db/analyses`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      resume_text: resumeText,
      job_title: jobTitle,
      job_description: jobDescription,
      result,
      label,
    }),
  });
  return handleResponse<SavedAnalysis>(res);
}

export async function listAnalyses(): Promise<SavedAnalysis[]> {
  const res = await fetch(`${BASE}/db/analyses`);
  return handleResponse<SavedAnalysis[]>(res);
}

export async function deleteAnalysis(id: number): Promise<void> {
  const res = await fetch(`${BASE}/db/analyses/${id}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 204) {
    throw new Error(`Delete failed (${res.status})`);
  }
}
