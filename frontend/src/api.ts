import type {
  AnalyzeResponse,
  DatasetInfo,
  StateResponse,
  IntentSuggestResponse,
  IntentApplyResponse,
  ContentDraftResponse,
  DraftMode,
  TelegramSyncResponse,
} from "./types";

const DEFAULT_API_BASE = "http://localhost:8000/api";
const apiBaseUrl = (import.meta.env.VITE_API_BASE_URL ?? DEFAULT_API_BASE).replace(/\/$/, "");

type AnalyzePayload = Partial<{
  n_components: number;
  eps: number | null;
  min_samples: number | null;
  metric: string;
  normalize: boolean;
  force: boolean;
}>;

type IntentSuggestPayload = {
  primary_query?: string;
  intents: string[];
  top_n?: number;
};

type IntentApplyPayload = {
  dataset?: string;
  intents: string[];
  top_k?: number;
  min_score?: number;
};

type TelegramSyncPayload = {
  start_date: string;
  end_date: string;
  cluster_limit: number;
  max_messages?: number;
  dataset_name?: string;
};

function cleanPayload<T extends Record<string, unknown>>(payload: T): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(payload).filter(([, value]) => value !== undefined)
  );
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const url = `${apiBaseUrl}${path}`;
  const response = await fetch(url, options);
  const contentType = response.headers.get("content-type");
  const isJson = contentType ? contentType.includes("application/json") : false;
  const payload = isJson ? await response.json() : null;
  if (!response.ok) {
    const detail = payload?.detail ?? payload?.message ?? response.statusText;
    throw new Error(detail || `Request to ${path} failed with status ${response.status}`);
  }
  return payload as T;
}

export async function fetchDatasets(): Promise<DatasetInfo[]> {
  return request<DatasetInfo[]>("/datasets");
}

export async function fetchState(): Promise<StateResponse> {
  return request<StateResponse>("/state");
}

export async function analyzeDataset(
  datasetName: string,
  payload: AnalyzePayload = {}
): Promise<AnalyzeResponse> {
  const body = JSON.stringify(cleanPayload(payload));
  return request<AnalyzeResponse>(`/datasets/${encodeURIComponent(datasetName)}/analyze`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });
}

export async function suggestIntents(payload: IntentSuggestPayload): Promise<IntentSuggestResponse> {
  const body = JSON.stringify(cleanPayload(payload));
  return request<IntentSuggestResponse>("/intents/suggest", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });
}

export async function applyIntents(payload: IntentApplyPayload): Promise<IntentApplyResponse> {
  const body = JSON.stringify(cleanPayload(payload));
  return request<IntentApplyResponse>("/intents/apply", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });
}

export async function syncTelegramDataset(payload: TelegramSyncPayload): Promise<TelegramSyncResponse> {
  const body = JSON.stringify(cleanPayload(payload));
  return request<TelegramSyncResponse>("/datasets/telegram-sync", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });
}

export async function generateContentDraft(topic: string, mode: DraftMode): Promise<ContentDraftResponse> {
  const body = JSON.stringify({ topic, mode });
  return request<ContentDraftResponse>("/content/generate", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body,
  });
}

