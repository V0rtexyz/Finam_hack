export type DatasetInfo = {
  name: string;
  path: string;
  size_bytes: number;
  modified_at: string;
};

export type DatasetState = {
  name: string;
  path?: string | null;
  total_messages: number;
  updated_at?: string | null;
  params?: Record<string, unknown> | null;
};

export type ClusterSummary = {
  cluster: number;
  label: string;
  size: number;
  share: number;
  is_noise: boolean;
  top_terms: string[];
  priority_score: number;
};

export type TopicMapPoint = {
  message_id: string;
  cluster: number;
  cluster_label: string;
  is_noise: boolean;
  x: number;
  y: number;
  z?: number | null;
  sender?: string | null;
  date?: string | null;
  text?: string | null;
};

export type TopicTimelinePoint = {
  cluster: number;
  cluster_label: string;
  date: string;
  count: number;
  period_count: number;
};

export type QualityMetrics = {
  total_messages?: number;
  noise_messages?: number;
  noise_ratio?: number;
  unique_clusters?: number;
  avg_cluster_size?: number;
  largest_cluster_size?: number;
  cluster_size_distribution?: Record<string, number>;
  [key: string]: number | Record<string, number> | undefined;
};

export type AnalyzeResponse = {
  dataset: DatasetState;
  quality_metrics: QualityMetrics;
};

export type IntentMatchRow = {
  message_id: string;
  text: string;
  sender?: string | null;
  date?: string | null;
  intent_score: number;
};

export type IntentMatch = {
  intent: string;
  results: IntentMatchRow[];
};

export type IntentSuggestResponse = {
  suggested: string[];
};

export type IntentApplyResponse = {
  dataset: DatasetState;
  matches: IntentMatch[];
};

export type DraftMode = "article" | "social" | "market";

export type ContentDraftResponse = {
  topic: string;
  mode: DraftMode;
  draft: string;
  used_llm: boolean;
};


export type TelegramSyncResponse = {
  dataset: DatasetState;
  quality_metrics: QualityMetrics;
  records_fetched: number;
  channels: string[];
  start_date: string;
  end_date: string;
};

export type StateResponse = {
  generated_at: string;
  active_dataset: DatasetState;
  datasets: DatasetInfo[];
  quality_metrics: QualityMetrics;
  clusters: ClusterSummary[];
  topic_map: TopicMapPoint[];
  topic_timeline: TopicTimelinePoint[];
  intent_suggestions: string[];
};
