import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ColumnsType } from "antd/es/table";
import {
  Button,
  Card,
  Col,
  ConfigProvider,
  Divider,
  Empty,
  Input,
  Layout,
  Modal,
  Progress,
  Row,
  Select,
  Segmented,
  Space,
  Spin,
  Statistic,
  Switch,
  Table,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  message,
  theme,
} from "antd";
import {
  ClusterOutlined,
  FileTextOutlined,
  LineChartOutlined,
  MessageOutlined,
  ReloadOutlined,
  SmileOutlined,
  SoundOutlined,
} from "@ant-design/icons";
import { Line } from "@ant-design/plots";
import Plot from "react-plotly.js";
import type { Config as PlotlyConfig, Data as PlotlyData, Layout as PlotlyLayout } from "plotly.js";

import {
  analyzeDataset,
  applyIntents,
  fetchDatasets,
  fetchState,
  suggestIntents,
  generateContentDraft,
  syncTelegramDataset,
} from "./api";
import type {
  DatasetInfo,
  IntentApplyResponse,
  IntentMatch,
  IntentMatchRow,
  IntentSuggestResponse,
  QualityMetrics,
  StateResponse,
  TopicMapPoint,
  TopicTimelinePoint,
  ContentDraftResponse,
  DraftMode,
  ClusterSummary,
} from "./types";
import LandingScreen from "./LandingScreen";
import type { LandingFormValues } from "./LandingScreen";

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;

type Granularity = "Day" | "Week" | "Month";

const GRANULARITY_OPTIONS: Granularity[] = ["Day", "Week", "Month"];

const COLOR_PALETTE_LIGHT = [
  '#1C3144', '#365985', '#2A7F62', '#B49A5A', '#8C4668', '#D16A4C', '#5E548E', '#3F6F8C', '#7D916C', '#BA7C4E', '#597D9B', '#A56982', '#4E8C6A', '#9C5B57', '#E0B89C',
];

const COLOR_PALETTE_DARK = [
  '#7A93C9', '#66C4B3', '#F1C27C', '#F48A6A', '#C7A2D1', '#7ED4B6', '#F3A4B5', '#F5C15B', '#86B4E0', '#BA9AD6', '#F28D52', '#6FC6E2', '#E8C547', '#8FD3A0', '#F4AFA6',
];

const TOPIC_MAP_PALETTE_LIGHT = [
  '#0f4c5c', '#bf1363', '#f0a500', '#5c6784', '#69a297', '#8d6a9f', '#d9777d', '#4a7c59', '#c49c6a', '#2b4964', '#b85c38', '#6f7c92',
];

const TOPIC_MAP_PALETTE_DARK = [
  '#5dade2', '#f78fb3', '#f5c16c', '#7ab8a8', '#b79ddf', '#f28482', '#6ddccf', '#e0b1cb', '#aecf64', '#f39c12', '#b8b8ff', '#ffadad',
];

const DRAFT_MODE_META: Record<DraftMode, { title: string; tagColor: string; tagText: string }> = {
  article: { title: "Letter draft", tagColor: "blue", tagText: "Letter" },
  social: { title: "Social post draft", tagColor: "purple", tagText: "Social" },
  market: { title: "Market impact brief", tagColor: "volcano", tagText: "Market" },
};

const LANDING_REVEAL_DELAY = 160;

const formatDateTime = (value?: string | null): string => {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  try {
    return new Intl.DateTimeFormat(undefined, {
      dateStyle: "medium",
      timeStyle: "short",
    }).format(date);
  } catch {
    return value;
  }
};

const normalizeLandingDate = (value?: string | null): string | null => {
  if (!value) {
    return null;
  }
  const trimmed = value.trim();
  const dotFormat = trimmed.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (dotFormat) {
    const [, day, month, year] = dotFormat;
    return `${year}-${month}-${day}`;
  }
  const candidate = new Date(trimmed);
  if (!Number.isNaN(candidate.getTime())) {
    return candidate.toISOString().slice(0, 10);
  }
  return null;
};

const normalizeDatasetLabel = (name: string) => name.replace(/\.json$/i, "");

const getWeekStart = (date: Date): Date => {
  const copy = new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()))
  const day = copy.getUTCDay();
  const diff = (day + 6) % 7; // Monday as week start
  copy.setUTCDate(copy.getUTCDate() - diff);
  return copy;
};

const formatDateKey = (iso: string, granularity: Granularity): string => {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  if (granularity === "Day") {
    return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()))
      .toISOString()
      .slice(0, 10);
  }
  if (granularity === "Week") {
    return getWeekStart(date).toISOString().slice(0, 10);
  }
  // Month
  const month = (date.getUTCMonth() + 1).toString().padStart(2, "0");
  return `${date.getUTCFullYear()}-${month}-01`;
};

const getInitialThemePreference = () => {
  if (typeof window === "undefined") {
    return false;
  }
  const stored = window.localStorage.getItem("app-theme");
  if (stored) {
    return stored === "dark";
  }
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? false;
};

const App = () => {
  const [messageApi, contextHolder] = message.useMessage();
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [state, setState] = useState<StateResponse | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<string>();
  const [loading, setLoading] = useState<boolean>(true);
  const [processing, setProcessing] = useState<boolean>(false);
  const [isDarkMode, setIsDarkMode] = useState<boolean>(getInitialThemePreference);

  const [timelineSelection, setTimelineSelection] = useState<string[]>([]);
  const [timelineGranularity, setTimelineGranularity] = useState<Granularity>("Day");
  const timelineInitializedRef = useRef(false);

  const [intentText, setIntentText] = useState<string>("");
  const [primaryQuery, setPrimaryQuery] = useState<string>("");
  const [intentSelection, setIntentSelection] = useState<string[]>([]);
  const [intentResults, setIntentResults] = useState<IntentMatch[]>([]);
  const [intentSuggestions, setIntentSuggestions] = useState<string[]>([]);
  const [rankingLoading, setRankingLoading] = useState<boolean>(false);
  const [intentLoading, setIntentLoading] = useState<boolean>(false);
  const [draftTopic, setDraftTopic] = useState<string>("");
  const [draftResponse, setDraftResponse] = useState<ContentDraftResponse | null>(null);
  const [draftPendingMode, setDraftPendingMode] = useState<DraftMode | null>(null);
  const [clusterDraftPending, setClusterDraftPending] = useState<{ cluster: number; mode: DraftMode } | null>(null);
  const [clusterDraftResponse, setClusterDraftResponse] = useState<ContentDraftResponse | null>(null);
  const [clusterDraftTarget, setClusterDraftTarget] = useState<ClusterSummary | null>(null);
  const [clusterDraftModalOpen, setClusterDraftModalOpen] = useState<boolean>(false);
  const intentInitializedRef = useRef(false);
  const [landingVisible, setLandingVisible] = useState<boolean>(true);
  const [landingData, setLandingData] = useState<LandingFormValues | null>(null);
  const [landingMaskActive, setLandingMaskActive] = useState<boolean>(true);
  const landingRevealTimerRef = useRef<number | null>(null);
  const [landingSubmitting, setLandingSubmitting] = useState<boolean>(false);
  const [landingKey, setLandingKey] = useState<number>(0);

  const composeClusterTopic = useCallback((summary: ClusterSummary): string => {
    const label = summary.label?.trim() || `Cluster ${summary.cluster}`;
    const keywords = summary.top_terms?.length
      ? summary.top_terms.slice(0, 10).join(", ")
      : "no keywords captured";
    const sharePct = Math.max(0, Math.min(100, (summary.share ?? 0) * 100));
    return [
      `Cluster label: ${label}`,
      `Cluster id: ${summary.cluster}`,
      `Messages in cluster: ${summary.size}`,
      `Share of dataset: ${sharePct.toFixed(2)}%`,
      `Top keywords: ${keywords}`,
      "Use only the facts above; flag any data gaps instead of inventing details.",
    ].join("\n");
  }, []);

  const isClusterDraftLoading = useCallback(
    (clusterId: number, mode: DraftMode) =>
      clusterDraftPending?.cluster === clusterId && clusterDraftPending.mode === mode,
    [clusterDraftPending]
  );

  const draftTopicReady = useMemo(() => draftTopic.trim().length > 0, [draftTopic]);

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    document.body.classList.toggle("dark-mode", isDarkMode);
    if (typeof window !== "undefined") {
      window.localStorage.setItem("app-theme", isDarkMode ? "dark" : "light");
    }
  }, [isDarkMode]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) {
      return;
    }
    const media = window.matchMedia("(prefers-color-scheme: dark)");
    const handleChange = (event: MediaQueryListEvent) => {
      if (!window.localStorage.getItem("app-theme")) {
        setIsDarkMode(event.matches);
      }
    };
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", handleChange);
      return () => media.removeEventListener("change", handleChange);
    }
    media.addListener(handleChange);
    return () => media.removeListener(handleChange);
  }, []);

  useEffect(() => {
    if (!landingData) {
      return;
    }
    messageApi.success({
      content: `Поиск: ${landingData.startDate} - ${landingData.endDate}. Горячих новостей: ${landingData.hotNewsCount}`,
      key: "landing-summary",
      duration: 3,
    });
  }, [landingData, messageApi]);

  useEffect(() => {
    if (landingVisible) {
      setLandingMaskActive(true);
    }
  }, [landingVisible]);

  useEffect(() => () => {
    if (landingRevealTimerRef.current !== null) {
      window.clearTimeout(landingRevealTimerRef.current);
      landingRevealTimerRef.current = null;
    }
  }, []);

  const loadDatasets = useCallback(async () => {
    try {
      const list = await fetchDatasets();
      setDatasets(list);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Failed to load datasets";
      messageApi.error({ content: description, key: "datasets" });
    }
  }, [messageApi]);

  const loadState = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchState();
      setState(data);
      setSelectedDataset(data.active_dataset?.name);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Failed to load metrics";
      messageApi.error({ content: description, key: "state" });
    } finally {
      setLoading(false);
    }
  }, [messageApi]);

  useEffect(() => {
    void loadDatasets();
    void loadState();
  }, [loadDatasets, loadState]);

  const processDataset = useCallback(
    async (value: string, force: boolean) => {
      setProcessing(true);
      try {
        await analyzeDataset(value, force ? { force: true } : {});
        if (force) {
          messageApi.success({ content: `Dataset "${normalizeDatasetLabel(value)}" recomputed`, key: "analyze" });
        }
        await loadState();
      } catch (error) {
        const description = error instanceof Error ? error.message : "Failed to process dataset";
        messageApi.error({ content: description, key: "analyze" });
      } finally {
        setProcessing(false);
      }
    },
    [loadState, messageApi]
  );

  const handleLandingSubmit = useCallback(
    async (values: LandingFormValues) => {
      if (landingSubmitting) {
        return;
      }
      const startIso = normalizeLandingDate(values.startDate);
      const endIso = normalizeLandingDate(values.endDate);
      if (!startIso || !endIso) {
        messageApi.error({ content: "Неверный формат дат", key: "landing-error" });
        return;
      }
      setLandingSubmitting(true);
      try {
        const response = await syncTelegramDataset({
          start_date: startIso,
          end_date: endIso,
          cluster_limit: values.hotNewsCount,
        });
        setLandingData(values);
        setSelectedDataset(response.dataset.name);
        await loadDatasets();
        await loadState();
        messageApi.success({
          content: `Загружено ${response.records_fetched.toLocaleString()} сообщений из Telegram`,
          key: "telegram-sync",
        });
        if (landingRevealTimerRef.current !== null) {
          window.clearTimeout(landingRevealTimerRef.current);
          landingRevealTimerRef.current = null;
        }
        setLandingVisible(false);
        landingRevealTimerRef.current = window.setTimeout(() => {
          setLandingMaskActive(false);
          landingRevealTimerRef.current = null;
        }, LANDING_REVEAL_DELAY);
      } catch (error) {
        const detail = error instanceof Error ? error.message : "Не удалось обновить данные";
        messageApi.error({ content: detail, key: "telegram-sync" });
        if (landingRevealTimerRef.current !== null) {
          window.clearTimeout(landingRevealTimerRef.current);
          landingRevealTimerRef.current = null;
        }
        setLandingKey((prev) => prev + 1);
        setLandingVisible(true);
        setLandingMaskActive(true);
      } finally {
        setLandingSubmitting(false);
      }
    },
    [
      landingSubmitting,
      loadDatasets,
      loadState,
      messageApi,
      setLandingData,
      setLandingKey,
      setLandingMaskActive,
      setLandingVisible,
      setSelectedDataset,
    ]
  );

  const handleThemeToggle = useCallback((checked: boolean) => {
    setIsDarkMode(checked);
  }, []);

  const metrics: QualityMetrics = state?.quality_metrics ?? {};
  const totalMessages = metrics.total_messages ?? 0;
  const noiseMessages = metrics.noise_messages ?? 0;
  const noiseRatio = metrics.noise_ratio ?? (totalMessages ? noiseMessages / totalMessages : 0);
  const signalMessages = Math.max(totalMessages - noiseMessages, 0);
  const uniqueClusters = metrics.unique_clusters ?? (state?.clusters?.filter((cluster) => !cluster.is_noise).length ?? 0);
  const avgClusterSize = metrics.avg_cluster_size ?? 0;
  const largestClusterSize = metrics.largest_cluster_size ?? 0;
  const topicMapData = state?.topic_map ?? [];
  const timelineRaw = state?.topic_timeline ?? [];

  const timelineClusterOptions = useMemo(() => {
    if (!timelineRaw.length) {
      return [] as { value: string; label: string; total: number }[];
    }
    const totals = new Map<string, number>();
    const labels = new Map<string, string>();
    timelineRaw.forEach((point) => {
      const key = String(point.cluster);
      const cumulative = typeof point.count === "number" ? point.count : 0;
      const current = totals.get(key) ?? 0;
      totals.set(key, Math.max(current, cumulative));
      labels.set(key, point.cluster_label);
    });
    return Array.from(totals.entries())
      .map(([cluster, total]) => ({
        value: cluster,
        label: `${labels.get(cluster) ?? `Cluster ${cluster}`} (${total})`,
        total,
      }))
      .sort((a, b) => b.total - a.total);
  }, [timelineRaw]);

  useEffect(() => {
    if (!timelineInitializedRef.current && timelineClusterOptions.length) {
      const defaults = timelineClusterOptions.slice(0, 5).map((option) => option.value);
      setTimelineSelection(defaults);
      timelineInitializedRef.current = true;
    }
  }, [timelineClusterOptions]);

  const timelineChartData = useMemo(() => {
    if (!timelineRaw.length || !timelineSelection.length) {
      return [] as Array<{ date: Date; count: number; period_count: number; cluster_label: string }>;
    }
    const selectionSet = new Set(timelineSelection);
    const labelMap = new Map<string, string>();
    timelineRaw.forEach((point) => {
      labelMap.set(String(point.cluster), point.cluster_label);
    });

    const grouped = new Map<string, TopicTimelinePoint[]>();
    timelineRaw.forEach((point) => {
      const clusterKey = String(point.cluster);
      if (!selectionSet.has(clusterKey)) {
        return;
      }
      if (!grouped.has(clusterKey)) {
        grouped.set(clusterKey, []);
      }
      grouped.get(clusterKey)!.push(point);
    });

    const data: Array<{ date: Date; count: number; period_count: number; cluster_label: string }> = [];
    grouped.forEach((points, clusterKey) => {
      const label = labelMap.get(clusterKey) ?? `Cluster ${clusterKey}`;
      const sortedPoints = [...points].sort((a, b) => {
        const timeA = new Date(a.date).getTime();
        const timeB = new Date(b.date).getTime();
        return timeA - timeB;
      });

      const buckets = new Map<
        string,
        { cumulative: number; period: number; lastTimestamp: number }
      >();

      sortedPoints.forEach((point) => {
        const bucketKey = formatDateKey(point.date, timelineGranularity);
        const timestamp = new Date(point.date).getTime();
        const periodValue =
          typeof point.period_count === "number" ? point.period_count : point.count;
        const existing = buckets.get(bucketKey);
        if (!existing) {
          buckets.set(bucketKey, {
            cumulative: point.count,
            period: periodValue,
            lastTimestamp: timestamp,
          });
          return;
        }
        existing.period += periodValue;
        if (timestamp >= existing.lastTimestamp) {
          existing.cumulative = point.count;
          existing.lastTimestamp = timestamp;
        }
      });

      Array.from(buckets.entries())
        .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
        .forEach(([dateKey, entry]) => {
          const date = new Date(`${dateKey}T00:00:00Z`);
          data.push({
            date,
            count: entry.cumulative,
            period_count: entry.period,
            cluster_label: label,
          });
        });
    });

    return data.sort((a, b) => a.date.getTime() - b.date.getTime());
  }, [timelineRaw, timelineSelection, timelineGranularity]);

  const timelineColorMap = useMemo(() => {
    if (!timelineChartData.length) {
      return new Map<string, string>();
    }
    const palette = isDarkMode ? COLOR_PALETTE_DARK : COLOR_PALETTE_LIGHT;
    const uniqueLabels = Array.from(new Set(timelineChartData.map((item) => item.cluster_label)));
    const map = new Map<string, string>();
    uniqueLabels.forEach((label, index) => {
      map.set(label, palette[index % palette.length]);
    });
    return map;
  }, [timelineChartData, isDarkMode]);

  const timelineConfig = useMemo(
    () => {
      const palette = isDarkMode ? COLOR_PALETTE_DARK : COLOR_PALETTE_LIGHT;
      const colorEntries = Array.from(timelineColorMap.entries());
      const colorDomain = colorEntries.map(([label]) => label);
      const colorRange = colorEntries.map(([, color]) => color);

      return {
        data: timelineChartData,
        xField: "date",
        yField: "count",
        seriesField: "cluster_label",
        colorField: "cluster_label",
        color: colorRange.length ? colorRange : palette,
        scale: colorRange.length
          ? {
              color: {
                domain: colorDomain,
                range: colorRange,
              },
            }
          : undefined,
        smooth: true,
        legend: {
          color: {
            position: "top",
            itemLabelFill: isDarkMode ? "#f8fff4" : "#0f1f17",
            itemLabelFontWeight: 500,
          },
        },
        animation: { appear: { animation: "wave-in", duration: 600 } },
        tooltip: {
          shared: true,
          showMarkers: true,
          formatter: (datum: { cluster_label: string; count: number; period_count: number; date: Date }) => ({
            name: datum.cluster_label,
            value:
              datum.period_count && Number.isFinite(datum.period_count)
                ? `${datum.count} total (Δ${datum.period_count})`
                : `${datum.count} total`,
          }),
        },
        xAxis: {
          type: "time" as const,
          label: {
            formatter: (value: string) => {
              const date = new Date(value);
              return Number.isNaN(date.getTime())
                ? value
                : new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(date);
            },
          },
        },
      };
    },
    [timelineChartData, timelineColorMap, isDarkMode]
  );

  const topicMapHasZ = useMemo(
    () => topicMapData.some((point) => point.z !== undefined && point.z !== null),
    [topicMapData]
  );

  const topicMapColorMap = useMemo(() => {
    if (!topicMapData.length) {
      return new Map<string, string>();
    }
    const palette = isDarkMode ? TOPIC_MAP_PALETTE_DARK : TOPIC_MAP_PALETTE_LIGHT;
    const labelToCluster = new Map<string, number>();
    topicMapData.forEach((point) => {
      if (point.is_noise) {
        return;
      }
      const current = labelToCluster.get(point.cluster_label);
      if (current === undefined || point.cluster < current) {
        labelToCluster.set(point.cluster_label, point.cluster);
      }
    });
    const entries = Array.from(labelToCluster.entries()).sort((a, b) => a[1] - b[1]);
    const map = new Map<string, string>();
    entries.forEach(([label], index) => {
      map.set(label, palette[index % palette.length]);
    });
    return map;
  }, [topicMapData, isDarkMode]);

  const topicMapTraces = useMemo<PlotlyData[]>(() => {
    if (!topicMapData.length) {
      return [];
    }
    const palette = isDarkMode ? TOPIC_MAP_PALETTE_DARK : TOPIC_MAP_PALETTE_LIGHT;
    const noiseColor = isDarkMode ? "#9aa5b4" : "#cbd5e1";
    const noiseLineColor = isDarkMode ? "#cbd5e1" : "#94a3b8";

    const groups = new Map<string, TopicMapPoint[]>();
    topicMapData.forEach((point) => {
      const label = point.cluster_label || `Cluster ${point.cluster}`;
      if (!groups.has(label)) {
        groups.set(label, []);
      }
      groups.get(label)!.push(point);
    });

    const escapeHtml = (value: string) =>
      value.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    const normalizeWhitespace = (value: string) => value.replace(/\s+/g, " ").trim();

    const sortedGroups = Array.from(groups.entries()).sort((a, b) => {
      const pointsA = a[1];
      const pointsB = b[1];
      const clusterA = pointsA[0]?.cluster ?? 0;
      const clusterB = pointsB[0]?.cluster ?? 0;
      const isNoiseA = pointsA.every((point) => point.is_noise || point.cluster === -1);
      const isNoiseB = pointsB.every((point) => point.is_noise || point.cluster === -1);
      if (isNoiseA && !isNoiseB) {
        return 1;
      }
      if (!isNoiseA && isNoiseB) {
        return -1;
      }
      return clusterA - clusterB;
    });

    return sortedGroups.map(([label, points], index) => {
      const isNoiseCluster = points.every((point) => point.is_noise || point.cluster === -1);
      const color = isNoiseCluster
        ? noiseColor
        : topicMapColorMap.get(label) ?? palette[index % palette.length];

      const hoverTexts = points.map((point) => {
        const lines: string[] = [escapeHtml(label)];
        const meta: string[] = [];
        if (point.sender) {
          meta.push(escapeHtml(normalizeWhitespace(String(point.sender))));
        }
        if (point.date) {
          const formatted = formatDateTime(point.date);
          if (formatted && formatted !== "-") {
            meta.push(escapeHtml(normalizeWhitespace(formatted)));
          }
        }
        if (meta.length) {
          lines.push(meta.join(" | "));
        }
        if (point.text) {
          lines.push(escapeHtml(normalizeWhitespace(point.text)));
        }
        return lines.join("<br>");
      });

      return {
        type: "scatter3d" as const,
        mode: "markers" as const,
        name: label,
        x: points.map((point) => point.x),
        y: points.map((point) => point.y),
        z: points.map((point) => (topicMapHasZ ? point.z ?? 0 : 0)),
        text: hoverTexts,
        hovertemplate: "%{text}<extra></extra>",
        marker: {
          size: isNoiseCluster ? 3 : 5,
          opacity: isNoiseCluster ? 0.35 : 0.85,
          color,
          line: isNoiseCluster ? { width: 0.8, color: noiseLineColor } : undefined,
        },
      } as PlotlyData;
    });
  }, [topicMapData, topicMapColorMap, isDarkMode, topicMapHasZ]);

  const topicMapLayout = useMemo<Partial<PlotlyLayout>>(() => {
    const foreground = isDarkMode ? "#e2e8f0" : "#0f172a";
    const gridColor = isDarkMode ? "rgba(148, 163, 184, 0.25)" : "rgba(148, 163, 184, 0.35)";
    const buildAxis = (title: string) => ({
      title: { text: title, font: { color: foreground } },
      tickfont: { color: foreground },
      gridcolor: gridColor,
      zeroline: false,
      showbackground: false,
    });
    return {
      autosize: true,
      margin: { l: 0, r: 0, b: 0, t: 40, pad: 0 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      legend: {
        orientation: "h",
        x: 0,
        y: 1.1,
        bgcolor: "rgba(0,0,0,0)",
        font: { color: foreground },
      },
      scene: {
        bgcolor: "rgba(0,0,0,0)",
        dragmode: "orbit",
        xaxis: buildAxis("Component 1"),
        yaxis: buildAxis("Component 2"),
        zaxis: buildAxis(topicMapHasZ ? "Component 3" : "Component"),
        aspectmode: "cube",
      },
    };
  }, [isDarkMode, topicMapHasZ]);

  const topicMapPlotConfig = useMemo<Partial<PlotlyConfig>>(
    () => ({
      displaylogo: false,
      responsive: true,
      scrollZoom: true,
      modeBarButtonsToRemove: ["toImage"],
    }),
    []
  );

  const parsedIntents = useMemo(() => {
    return intentText
      .split(/\n+/)
      .map((item) => item.trim())
      .filter((item, index, arr) => item.length > 0 && arr.indexOf(item) === index);
  }, [intentText]);

  useEffect(() => {
    if (!intentInitializedRef.current && state?.intent_suggestions?.length) {
      setIntentSuggestions(state.intent_suggestions);
      setIntentText(state.intent_suggestions.join("\n"));
      setIntentSelection(state.intent_suggestions.slice(0, Math.min(3, state.intent_suggestions.length)));
      intentInitializedRef.current = true;
    }
  }, [state?.intent_suggestions]);

  useEffect(() => {
    const available = new Set(parsedIntents);
    setIntentSelection((prev) => prev.filter((intent) => available.has(intent)));
  }, [parsedIntents]);

  const handleSuggestIntents = useCallback(async () => {
    const intents = parsedIntents;
    if (!primaryQuery.trim() || !intents.length) {
      messageApi.info("Provide a query and at least one intent to get suggestions.");
      return;
    }
    setRankingLoading(true);
    try {
      const response: IntentSuggestResponse = await suggestIntents({
        primary_query: primaryQuery,
        intents,
        top_n: Math.min(5, intents.length),
      });
      if (response.suggested.length) {
        setIntentSelection(response.suggested.filter((intent) => intents.includes(intent)));
        messageApi.success("Intent suggestions updated.");
      } else {
        messageApi.info("No additional suggestions available.");
      }
    } catch (error) {
      const description = error instanceof Error ? error.message : "Failed to suggest intents";
      messageApi.error(description);
    } finally {
      setRankingLoading(false);
    }
  }, [messageApi, parsedIntents, primaryQuery]);

  const handleApplyIntents = useCallback(async () => {
    const intents = intentSelection.filter((intent) => parsedIntents.includes(intent));
    if (!intents.length) {
      messageApi.warning("Select at least one intent to apply.");
      return;
    }
    const dataset = selectedDataset ?? state?.active_dataset?.name;
    if (!dataset) {
      messageApi.error("No active dataset selected.");
      return;
    }
    setIntentLoading(true);
    try {
      const response: IntentApplyResponse = await applyIntents({
        dataset,
        intents,
      });
      setIntentResults(response.matches);
      if (response.matches.every((match) => match.results.length === 0)) {
        messageApi.warning("No matches found for the selected intents.");
      } else {
        messageApi.success("Intent results updated.");
      }
    } catch (error) {
      const description = error instanceof Error ? error.message : "Intent matching failed";
      messageApi.error(description);
      setIntentResults([]);
    } finally {
      setIntentLoading(false);
    }
  }, [intentSelection, messageApi, parsedIntents, selectedDataset, state?.active_dataset?.name]);

  const handleGenerateDraft = useCallback(async (mode: DraftMode) => {
    const topic = draftTopic.trim();
    if (!topic) {
      messageApi.warning("Введите тему в окно ниже.");
      return;
    }
    setDraftPendingMode(mode);
    try {
      const response = await generateContentDraft(topic, mode);
      setDraftResponse(response);
    } catch (error) {
      const description = error instanceof Error ? error.message : "Не удалось получить черновик.";
      messageApi.error({ content: description, key: "content-draft" });
    } finally {
      setDraftPendingMode(null);
    }
  }, [draftTopic, messageApi]);

  const handleClusterDraft = useCallback(
    async (summary: ClusterSummary, mode: DraftMode) => {
      if (summary.is_noise) {
        messageApi.info("Content generation is not available for the noise cluster.");
        return;
      }
      const topic = composeClusterTopic(summary);
      setClusterDraftPending({ cluster: summary.cluster, mode });
      setClusterDraftTarget(summary);
      setClusterDraftResponse(null);
      try {
        const response = await generateContentDraft(topic, mode);
        setClusterDraftResponse(response);
        setClusterDraftModalOpen(true);
        const meta = DRAFT_MODE_META[mode];
        messageApi.success(`${meta.tagText} draft ready for cluster ${summary.cluster}.`);
      } catch (error) {
        const description = error instanceof Error ? error.message : "Failed to generate cluster draft";
        messageApi.error({ content: description, key: `cluster-draft-${summary.cluster}` });
      } finally {
        setClusterDraftPending(null);
      }
    },
    [composeClusterTopic, messageApi]
  );

  const closeClusterDraftModal = useCallback(() => {
    setClusterDraftModalOpen(false);
  }, []);

  const intentResultColumns: ColumnsType<IntentMatchRow> = useMemo(
    () => [
      {
        title: "Date",
        dataIndex: "date",
        key: "date",
        width: 140,
        render: (value: string | null) => formatDateTime(value ?? undefined),
      },
      {
        title: "Sender",
        dataIndex: "sender",
        key: "sender",
        width: 160,
        render: (value: string | null) => value || "-",
      },
      {
        title: "Text",
        dataIndex: "text",
        key: "text",
        render: (value: string) => value,
      },
      {
        title: "Score",
        dataIndex: "intent_score",
        key: "intent_score",
        width: 120,
        align: "right",
        render: (value: number) => value.toFixed(3),
      },
    ],
    []
  );

  const clusterSummaryColumns: ColumnsType<ClusterSummary> = useMemo(
    () => [
      {
        title: "Cluster",
        dataIndex: "cluster",
        key: "cluster",
        render: (value: number, record) =>
          record.is_noise ? (
            <Tag color="default">Noise</Tag>
          ) : (
            <Tag color="blue">{`Cluster ${value}`}</Tag>
          ),
      },
      {
        title: "Label",
        dataIndex: "label",
        key: "label",
        render: (value: string) => value || "-",
      },
      {
        title: "Messages",
        dataIndex: "size",
        key: "size",
        align: "right",
        render: (value: number) => value.toLocaleString(),
      },
      {
        title: "Share",
        dataIndex: "share",
        key: "share",
        align: "right",
        render: (value: number) => `${(value * 100).toFixed(1)}%`,
      },
      {
        title: "Top keywords",
        dataIndex: "top_terms",
        key: "top_terms",
        render: (terms: string[]) =>
          terms && terms.length ? (
            <Space size={[4, 8]} wrap>
              {terms.map((term) => (
                <Tag key={term} color="geekblue">
                  {term}
                </Tag>
              ))}
            </Space>
          ) : (
            <Text type="secondary">-</Text>
          ),
      },
      {
        title: "Actions",
        key: "actions",
        align: "right",
        render: (_: unknown, record) =>
          record.is_noise ? (
            <Text type="secondary">-</Text>
          ) : (
            <Space size={8} wrap>
              <Tooltip title="Generate a formal letter draft">
                <Button
                  size="small"
                  icon={<FileTextOutlined />}
                  onClick={() => handleClusterDraft(record, "article")}
                  loading={isClusterDraftLoading(record.cluster, "article")}
                  disabled={processing}
                >
                  Letter
                </Button>
              </Tooltip>
              <Tooltip title="Generate an informal social post">
                <Button
                  size="small"
                  icon={<SmileOutlined />}
                  onClick={() => handleClusterDraft(record, "social")}
                  loading={isClusterDraftLoading(record.cluster, "social")}
                  disabled={processing}
                >
                  Social
                </Button>
              </Tooltip>
              <Tooltip title="Generate a market impact brief">
                <Button
                  size="small"
                  icon={<LineChartOutlined />}
                  onClick={() => handleClusterDraft(record, "market")}
                  loading={isClusterDraftLoading(record.cluster, "market")}
                  disabled={processing}
                >
                  Market
                </Button>
              </Tooltip>
            </Space>
          ),
      },
    ],
    [handleClusterDraft, isClusterDraftLoading, processing]
  );

  const draftResponseMeta = draftResponse ? DRAFT_MODE_META[draftResponse.mode] : null;
  const clusterDraftMeta = clusterDraftResponse ? DRAFT_MODE_META[clusterDraftResponse.mode] : null;

  const themeConfig = useMemo(
    () => {
      const primary = isDarkMode ? '#4cf8b4' : '#0b9d61';
      const surface = isDarkMode ? 'rgba(6, 20, 17, 0.92)' : '#ffffff';
      const baseBg = isDarkMode ? '#020604' : '#f3fbf6';
      const textBase = isDarkMode ? '#eafff4' : '#0f1f17';
      const borderColor = isDarkMode ? 'rgba(76, 248, 180, 0.22)' : 'rgba(15, 115, 84, 0.18)';
      const descriptionColor = isDarkMode ? '#71bca1' : '#4f6f60';

      return {
        algorithm: isDarkMode ? [theme.darkAlgorithm] : [theme.defaultAlgorithm],
        token: {
          colorPrimary: primary,
          colorInfo: primary,
          colorSuccess: primary,
          colorLink: primary,
          colorBorder: borderColor,
          colorBorderSecondary: borderColor,
          colorBgBase: baseBg,
          colorBgLayout: baseBg,
          colorBgContainer: surface,
          colorBgElevated: surface,
          colorTextBase: textBase,
          colorText: textBase,
          colorTextSecondary: descriptionColor,
          colorTextDescription: descriptionColor,
          fontFamily: "'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif",
          borderRadius: 16,
          controlHeight: 44,
        },
        components: {
          Layout: {
            bodyBg: baseBg,
            headerBg: 'transparent',
            headerPadding: '0 32px',
          },
          Card: {
            borderRadiusLG: 20,
            colorBgContainer: surface,
            boxShadowTertiary: isDarkMode
              ? '0 40px 120px rgba(0, 0, 0, 0.65), 0 18px 60px rgba(76, 248, 180, 0.18)'
              : '0 24px 72px rgba(14, 75, 51, 0.08)',
            headerBg: 'transparent',
          },
          Button: {
            controlHeight: 44,
            borderRadius: 999,
            colorPrimary: primary,
            colorPrimaryHover: isDarkMode ? '#5efac0' : '#12b37a',
            colorPrimaryActive: isDarkMode ? '#39e7a5' : '#0a8c56',
            colorPrimaryBorder: 'transparent',
          },
          Select: {
            borderRadius: 14,
            colorBgContainer: surface,
            colorBgElevated: surface,
            colorBorder: borderColor,
            controlHeight: 44,
          },
          Input: {
            borderRadius: 14,
            colorBgContainer: surface,
          },
          Segmented: {
            borderRadius: 999,
            itemSelectedBg: isDarkMode ? 'rgba(76, 248, 180, 0.2)' : 'rgba(35, 214, 151, 0.16)',
            itemSelectedColor: primary,
            itemHoverColor: primary,
          },
          Tabs: {
            inkBarColor: primary,
            itemSelectedColor: primary,
            itemHoverColor: primary,
            itemActiveColor: primary,
          },
          Table: {
            headerBg: isDarkMode ? 'rgba(9, 30, 23, 0.85)' : 'rgba(226, 255, 240, 0.85)',
            headerColor: textBase,
            borderColor: borderColor,
            headerSplitColor: borderColor,
            rowHoverBg: isDarkMode ? 'rgba(34, 197, 151, 0.12)' : 'rgba(35, 214, 151, 0.08)',
          },
          Tag: {
            colorText: primary,
            defaultBg: isDarkMode ? 'rgba(76, 248, 180, 0.16)' : 'rgba(35, 214, 151, 0.12)',
            colorBorderBg: isDarkMode ? 'rgba(76, 248, 180, 0.1)' : 'rgba(35, 214, 151, 0.08)',
          },
          Progress: {
            defaultColor: primary,
          },
        },
      };
    },
    [isDarkMode]
  );

  return (
    <ConfigProvider theme={themeConfig}>
      {landingVisible && (
        <LandingScreen
          key={landingKey}
          loading={landingSubmitting}
          onComplete={(values) => {
            void handleLandingSubmit(values);
          }}
        />
      )}
      <div className={landingMaskActive ? "app-shell app-shell--masked" : "app-shell"}>
        {contextHolder}
        <Layout>
          <Header className="app-header">
          <Row align="middle" justify="space-between" style={{ width: "100%" }}>
            <Col>
              <Title level={3} className="app-title">
                Radar Metrics Dashboard
              </Title>
            </Col>
            <Col>
              <Space size={12} align="center">
                <Text type="secondary">{isDarkMode ? "Dark" : "Light"} mode</Text>
                <Switch
                  checked={isDarkMode}
                  onChange={handleThemeToggle}
                  checkedChildren="Dark"
                  unCheckedChildren="Light"
                />
              </Space>
            </Col>
          </Row>
        </Header>
        <Content className="app-content">
          <Spin spinning={loading} tip="Loading data...">
            {state ? (
              <Space direction="vertical" size={24} style={{ width: "100%" }}>
                <Card bordered={false} className="metric-card">
                  <Row justify="space-between" align="middle" gutter={[16, 16]}>
                    <Col flex="auto">
                      <Space direction="vertical" size={4}>
                        <Text type="secondary">Active dataset</Text>
                        <Title level={4} style={{ margin: 0 }}>
                          {state.active_dataset?.name ? normalizeDatasetLabel(state.active_dataset.name) : "-"}
                        </Title>
                        <Space size={16} wrap>
                          <Text type="secondary">
                            {`Messages: ${totalMessages.toLocaleString()}`}
                          </Text>
                          <Text type="secondary">
                            {`Updated: ${formatDateTime(state.active_dataset?.updated_at)}`}
                          </Text>
                        </Space>
                      </Space>
                    </Col>
                    <Col>
                      <Space size={12} wrap>
                        <Select
                          className="dataset-select"
                          value={selectedDataset}
                          placeholder="Select dataset"
                          style={{ minWidth: 220 }}
                          options={datasets.map((dataset) => ({
                            label: normalizeDatasetLabel(dataset.name),
                            value: dataset.name,
                          }))}
                          onChange={(value) => {
                            setSelectedDataset(value);
                            void processDataset(value, false);
                          }}
                          loading={processing}
                          disabled={processing}
                        />
                        <Button
                          type="primary"
                          icon={<ReloadOutlined />}
                          onClick={() => selectedDataset && void processDataset(selectedDataset, true)}
                          loading={processing}
                          disabled={!selectedDataset}
                        >
                          Recompute
                        </Button>
                      </Space>
                    </Col>
                  </Row>
                </Card>

                <Row gutter={[16, 16]}>
                  <Col xs={24} sm={12} md={8}>
                    <Card bordered={false} className="metric-card">
                      <Space align="baseline" size={8}>
                        <MessageOutlined className="metric-icon" />
                        <Statistic title="Total messages" value={totalMessages.toLocaleString()} />
                      </Space>
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={8}>
                    <Card bordered={false} className="metric-card">
                      <Space align="baseline" size={8}>
                        <SoundOutlined className="metric-icon" />
                        <Statistic title="Noise" value={noiseMessages.toLocaleString()} />
                      </Space>
                    </Card>
                  </Col>
                  <Col xs={24} sm={12} md={8}>
                    <Card bordered={false} className="metric-card">
                      <Space align="baseline" size={8}>
                        <ClusterOutlined className="metric-icon" />
                        <Statistic title="Clusters" value={uniqueClusters} />
                      </Space>
                    </Card>
                  </Col>
                </Row>

                <Row gutter={[16, 16]}>
                  <Col xs={24} md={12}>
                    <Card bordered={false} className="metric-card emphasis">
                      <Space direction="vertical" size={12} style={{ width: "100%" }}>
                        <Text type="secondary">Signal messages</Text>
                        <Title level={2} style={{ margin: 0 }}>
                          {signalMessages.toLocaleString()}
                        </Title>
                        <Text type="secondary">clustered</Text>
                      </Space>
                    </Card>
                  </Col>
                  <Col xs={24} md={12}>
                    <Card bordered={false} className="metric-card">
                      <Space direction="vertical" size={12} style={{ width: "100%" }}>
                        <Text type="secondary">Noise share</Text>
                        <Progress
                          className="noise-progress"
                          percent={Number((noiseRatio * 100).toFixed(1))}
                          strokeWidth={12}
                          showInfo
                        />
                        <Text type="secondary">
                          Average cluster size: {avgClusterSize.toFixed(1)}
                        </Text>
                      </Space>
                    </Card>
                  </Col>
                </Row>

                <Card bordered={false} className="metric-card">
                  <Space direction="vertical" size={16} style={{ width: "100%" }}>
                    <Row justify="space-between" align="middle">
                      <Col>
                        <Title level={4} style={{ margin: 0 }}>
                          Topic activity timeline
                        </Title>
                        <Text type="secondary">Daily topic volume with adjustable granularity</Text>
                      </Col>
                      <Col>
                        <Space size={12} wrap>
                          <Select
                            mode="multiple"
                            allowClear
                            placeholder="Select topics"
                            style={{ minWidth: 240 }}
                            value={timelineSelection}
                            options={timelineClusterOptions.map(({ value, label }) => ({
                              value,
                              label,
                            }))}
                            onChange={(values) => setTimelineSelection(values)}
                            maxTagCount={3}
                          />
                          <Segmented<Granularity>
                            options={GRANULARITY_OPTIONS}
                            value={timelineGranularity}
                            onChange={(value) => setTimelineGranularity(value as Granularity)}
                          />
                        </Space>
                      </Col>
                    </Row>
                    <Divider style={{ margin: "8px 0" }} />
                    <div className="topic-map">
                      {timelineChartData.length ? (
                        <Line {...timelineConfig} />
                      ) : (
                        <Empty description="No timeline data" />
                      )}
                    </div>
                  </Space>
                </Card>

                <Card bordered={false} className="metric-card">
                  <Space direction="vertical" size={16} style={{ width: "100%" }}>
                    <Row justify="space-between" align="middle">
                      <Col>
                        <Title level={4} style={{ margin: 0 }}>
                          Topic map
                        </Title>
                        <Text type="secondary">Clusters after PCA projection</Text>
                      </Col>
                      <Col>
                        <Tag color="blue">{topicMapData.length.toLocaleString()} points</Tag>
                      </Col>
                    </Row>
                    <Divider style={{ margin: "8px 0" }} />
                    <div className="topic-map">
                      {topicMapData.length ? (
                        <Plot
                          data={topicMapTraces}
                          layout={topicMapLayout as PlotlyLayout}
                          config={topicMapPlotConfig as PlotlyConfig}
                          style={{ width: "100%", height: "100%", minHeight: 420 }}
                          useResizeHandler
                        />
                      ) : (
                        <Empty description="No points to display" />
                      )}
                    </div>
                  </Space>
                </Card>

                <Card bordered={false} className="metric-card">
                  <Space direction="vertical" size={12} style={{ width: "100%" }}>
                    <Row justify="space-between" align="middle">
                      <Col>
                        <Title level={4} style={{ margin: 0 }}>
                          Cluster summary
                        </Title>
                      </Col>
                      <Col>
                        <Text type="secondary">
                          Largest cluster: {largestClusterSize.toLocaleString()} messages
                        </Text>
                      </Col>
                    </Row>
                    <Divider style={{ margin: "8px 0" }} />
                    <Table
                      dataSource={state.clusters.map((cluster) => ({ key: cluster.cluster, ...cluster }))}
                      columns={clusterSummaryColumns}
                      pagination={{ pageSize: 8, hideOnSinglePage: true }}
                      scroll={{ x: true }}
                    />
                  </Space>
                </Card>

                <Card bordered={false} className="metric-card">
                  <Space direction="vertical" size={16} style={{ width: "100%" }}>
                    <Title level={4} style={{ margin: 0 }}>
                      Intent discovery
                    </Title>
                    <Text type="secondary">
                      Refine suggested intents, optionally add a primary research query and run semantic matching.
                    </Text>
                    <Input.TextArea
                      value={intentText}
                      onChange={(event) => setIntentText(event.target.value)}
                      autoSize={{ minRows: 6, maxRows: 10 }}
                      placeholder="Enter one intent per line"
                    />
                    <Row gutter={[12, 12]} align="middle">
                      <Col flex="auto">
                        <Input
                          value={primaryQuery}
                          onChange={(event) => setPrimaryQuery(event.target.value)}
                          placeholder="Primary research query (optional)"
                        />
                      </Col>
                      <Col>
                        <Button onClick={handleSuggestIntents} loading={rankingLoading} disabled={!parsedIntents.length}>
                          Suggest intents
                        </Button>
                      </Col>
                    </Row>
                    <Select
                      mode="multiple"
                      value={intentSelection}
                      placeholder="Select intents to apply"
                      options={parsedIntents.map((intent) => ({ label: intent, value: intent }))}
                      onChange={(values) => setIntentSelection(values)}
                      style={{ width: "100%" }}
                      maxTagCount={4}
                    />
                    <Space>
                      <Button
                        type="primary"
                        onClick={handleApplyIntents}
                        loading={intentLoading}
                        disabled={!intentSelection.length}
                      >
                        Apply intents
                      </Button>
                      {intentSuggestions.length && (
                        <Text type="secondary">
                          Suggested: {intentSuggestions.slice(0, 5).join(", ")}
                        </Text>
                      )}
                    </Space>
                  </Space>
                </Card>

                {intentResults.length > 0 && (
                  <Card bordered={false} className="metric-card">
                    <Space direction="vertical" size={12} style={{ width: "100%" }}>
                      <Title level={4} style={{ margin: 0 }}>
                        Intent results
                      </Title>
                      <Tabs
                        destroyInactiveTabPane
                        items={intentResults.map((match) => ({
                          key: match.intent,
                          label: `${match.intent} (${match.results.length})`,
                          children: (
                            <Table
                              rowKey="message_id"
                              dataSource={match.results.map((row, index) => ({ key: `${row.message_id}-${index}`, ...row }))}
                              columns={intentResultColumns}
                              pagination={{ pageSize: 10, hideOnSinglePage: true }}
                              size="small"
                            />
                          ),
                        }))}
                      />
                    </Space>
                  </Card>
                )}

                <Card bordered={false} className="metric-card">
                  <Space direction="vertical" size={16} style={{ width: "100%" }}>
                    <Title level={4} style={{ margin: 0 }}>
                      Генерация черновиков
                    </Title>
                    <Text type="secondary">
                      Введите тему в окно ниже и выберите формат, чтобы получить базовый текст.
                    </Text>
                    <Input.TextArea
                      value={draftTopic}
                      onChange={(event) => setDraftTopic(event.target.value)}
                      autoSize={{ minRows: 3, maxRows: 5 }}
                      placeholder="Опишите тему будущего материала"
                    />
                    <Row gutter={[12, 12]} align="middle">
                      <Col>
                        <Button
                          type="primary"
                          icon={<FileTextOutlined />}
                          onClick={() => handleGenerateDraft("article")}
                          disabled={!draftTopicReady}
                          loading={draftPendingMode === "article"}
                        >
                          Generate letter
                        </Button>
                      </Col>
                      <Col>
                        <Button
                          icon={<SmileOutlined />}
                          onClick={() => handleGenerateDraft("social")}
                          disabled={!draftTopicReady}
                          loading={draftPendingMode === "social"}
                        >
                          Social post
                        </Button>
                      </Col>
                      <Col>
                        <Button
                          icon={<LineChartOutlined />}
                          onClick={() => handleGenerateDraft("market")}
                          disabled={!draftTopicReady}
                          loading={draftPendingMode === "market"}
                        >
                          Market brief
                        </Button>
                      </Col>
                    </Row>
                    {draftResponse && draftResponseMeta && (
                      <Card type="inner" title={draftResponseMeta.title}>
                        <Space direction="vertical" size={8} style={{ width: "100%" }}>
                          <Row justify="space-between" align="middle">
                            <Col>
                              <Tag color={draftResponseMeta.tagColor}>{draftResponseMeta.tagText}</Tag>
                            </Col>
                            <Col>
                              <Tag color={draftResponse.used_llm ? "green" : "default"}>
                                {draftResponse.used_llm ? "LLM" : "Fallback"}
                              </Tag>
                            </Col>
                          </Row>
                          <Text strong>{draftResponse.topic}</Text>
                          <Typography.Paragraph style={{ whiteSpace: "pre-wrap", marginBottom: 0 }}>
                            {draftResponse.draft}
                          </Typography.Paragraph>
                        </Space>
                      </Card>
                    )}
                  </Space>
                </Card>

              </Space>
            ) : (
              <Empty description="No data" />
            )}
          </Spin>
        </Content>
        <Modal
          open={clusterDraftModalOpen}
          onCancel={closeClusterDraftModal}
          onOk={closeClusterDraftModal}
          okText="Close"
          cancelButtonProps={{ style: { display: 'none' } }}
          title={clusterDraftTarget ? `Cluster ${clusterDraftTarget.cluster}` : 'Cluster draft'}
          width={760}
        >
          {clusterDraftResponse && clusterDraftMeta ? (
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Row justify="space-between" align="middle">
                <Col>
                  <Tag color={clusterDraftMeta.tagColor}>{clusterDraftMeta.tagText}</Tag>
                </Col>
                <Col>
                  <Tag color={clusterDraftResponse.used_llm ? 'green' : 'default'}>
                    {clusterDraftResponse.used_llm ? 'LLM' : 'Fallback'}
                  </Tag>
                </Col>
              </Row>
              {clusterDraftTarget && (
                <Space direction="vertical" size={4}>
                  <Text strong>{clusterDraftTarget.label || `Cluster ${clusterDraftTarget.cluster}`}</Text>
                  <Text type="secondary">
                    Messages: {clusterDraftTarget.size.toLocaleString()} | Share: {(clusterDraftTarget.share * 100).toFixed(1)}%
                  </Text>
                </Space>
              )}
              <Typography.Paragraph style={{ whiteSpace: 'pre-wrap', marginBottom: 0 }}>
                {clusterDraftResponse.draft}
              </Typography.Paragraph>
            </Space>
          ) : (
            <Space align="center" style={{ width: '100%', justifyContent: 'center' }}>
              <Spin />
            </Space>
          )}
        </Modal>

        <Footer className="app-footer">
          <Text type="secondary">
            API: uvicorn api_server:app - Frontend: npm run dev (port 5173)
          </Text>
        </Footer>
        </Layout>
      </div>
    </ConfigProvider>
  );
};

export default App;



