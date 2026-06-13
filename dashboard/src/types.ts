export type Counts = {
  episodes: number;
  transcript_segments: number;
  chunks: number;
  entities: number;
  entity_relations: number;
};

export type Status = {
  stats: { counts: Counts };
  chunking: { strategy: string; max_words: number; overlap_words: number };
  recommendations: string[];
};

export type Stats = {
  counts: Counts;
  entity_types: Array<{ entity_type: string; count: number }>;
  richest_episodes: Array<{
    episode_id: number;
    title: string;
    unique_entities: number;
    mentions: number;
    chunks: number;
  }>;
};

export type Episode = {
  id: number;
  title: string;
  author?: string | null;
  language?: string | null;
  segment_count: number;
  corpus_id?: string;
  corpus_name?: string;
};

export type Topic = {
  name: string;
  entity_type: string;
  confidence: number;
  mentions: number;
  episodes: number;
};

export type GraphNode = {
  id: number | string;
  name: string;
  entity_type: string;
  mentions: number;
  episodes: number;
};

export type GraphEdge = {
  source: number | string;
  target: number | string;
  relation_type: string;
  weight: number;
};

export type GraphData = {
  nodes: GraphNode[];
  edges: GraphEdge[];
};

export type TimelineEntry = {
  episode_id: number;
  title: string;
  start_seconds: number | null;
  name: string;
  entity_type: string;
  text: string;
};

export type EntityProfile = {
  entity: Topic;
  mentions: Array<{ title: string; start_seconds: number | null; chunk_id: number; text: string }>;
  connections: Array<{ source: string; target: string; relation_type: string; weight: number }>;
};

export type EpisodeInsights = {
  episode: Episode;
  top_entities: Array<{ name: string; entity_type: string; mentions: number; chunks: number }>;
  entity_density: Array<{ chunk_id: number; start_seconds: number | null; unique_entities: number; text: string }>;
};

export type AskResult = {
  brief: string;
  mode?: "local" | "llm";
  local_brief?: string;
  llm_answer?: string;
};

export type ProcessUrlResult = {
  corpus: string;
  data_dir: string;
  source_url: string;
  ready: boolean;
  ingest: Array<{ status: string; title?: string | null; episode_id?: number | null; message?: string | null }>;
  entities?: { entities: number; mentions: number; relations: number } | null;
  index?: { enabled: boolean; collection: string; indexed_chunks?: number | null };
};

export type QualityReport = Record<string, Array<Record<string, unknown>>>;

export type CorpusConfig = {
  id: string;
  name: string;
  data_dir: string;
  description?: string | null;
  domain_profile?: string | null;
  qdrant_url?: string | null;
  tags?: string[];
};

export type CorporaResponse = {
  default: CorpusConfig;
  corpora: CorpusConfig[];
};
