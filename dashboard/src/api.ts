import type {
  AskResult,
  CorporaResponse,
  EntityProfile,
  Episode,
  EpisodeInsights,
  GraphData,
  QualityReport,
  ProcessUrlResult,
  Stats,
  Status,
  TimelineEntry,
  Topic
} from "./types";

async function getJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || response.statusText);
  }
  return payload as T;
}

function withParams(path: string, params: Record<string, string | number | undefined>) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") search.set(key, String(value));
  }
  const suffix = search.toString();
  return suffix ? `${path}?${suffix}` : path;
}

export const api = {
  corpora: () => getJson<CorporaResponse>("/api/corpora"),
  status: (corpus?: string) => getJson<Status>(withParams("/api/status", { corpus })),
  stats: (corpus?: string) => getJson<Stats>(withParams("/api/stats", { corpus })),
  episodes: (corpus?: string) => getJson<Episode[]>(withParams("/api/episodes", { corpus })),
  topics: (limit = 150, corpus?: string) => getJson<Topic[]>(withParams("/api/topics", { limit, corpus })),
  graph: (limit = 300, corpus?: string) => getJson<GraphData>(withParams("/api/graph", { limit, corpus })),
  quality: (corpus?: string) => getJson<QualityReport>(withParams("/api/quality", { corpus })),
  entityProfile: (name: string, limit = 25) =>
    getJson<EntityProfile>(`/api/entity-profile?name=${encodeURIComponent(name)}&limit=${limit}`),
  entityProfileForCorpus: (name: string, limit = 25, corpus?: string) =>
    getJson<EntityProfile>(withParams("/api/entity-profile", { name, limit, corpus })),
  timeline: (topic?: string, corpus?: string) => getJson<TimelineEntry[]>(withParams("/api/timeline", { limit: 220, topic, corpus })),
  episodeInsights: (episodeId: number) => getJson<EpisodeInsights>(`/api/episode-insights?episode_id=${episodeId}`),
  episodeInsightsForCorpus: (episodeId: number, corpus?: string) =>
    getJson<EpisodeInsights>(withParams("/api/episode-insights", { episode_id: episodeId, corpus })),
  ask: (question: string, limit = 5, corpus?: string, mode = "local") =>
    getJson<AskResult>(withParams("/api/ask", { q: question, limit, corpus, mode })),
  processUrl: (params: {
    url: string;
    corpus?: string;
    language?: string;
    whisper_model?: string;
    transcribe_seconds?: string;
    domain_profile?: string;
    force_index?: string;
  }) => getJson<ProcessUrlResult>(withParams("/api/process-url", params))
};
