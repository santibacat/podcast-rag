import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  BookOpen,
  Brain,
  CircleHelp,
  GitBranch,
  ListFilter,
  Network,
  RefreshCw,
  Search,
  Settings,
  ShieldCheck,
  Sparkles,
  UploadCloud
} from "lucide-react";
import { api } from "./api";
import type {
  CorpusConfig,
  EntityProfile,
  Episode,
  EpisodeInsights,
  GraphData,
  GraphNode,
  QualityReport,
  ProcessUrlResult,
  Stats,
  Status,
  TimelineEntry,
  Topic
} from "./types";
import "./styles.css";

type View = "overview" | "ingest" | "episodes" | "entities" | "graph" | "timeline" | "ask" | "quality" | "settings";
type Locale = "en" | "es";
type Translator = (key: string, values?: Record<string, string | number>) => string;

const views: Array<{ id: View; labelKey: string; icon: React.ReactNode }> = [
  { id: "overview", labelKey: "nav.overview", icon: <Activity size={17} /> },
  { id: "ingest", labelKey: "nav.ingest", icon: <UploadCloud size={17} /> },
  { id: "episodes", labelKey: "nav.episodes", icon: <BookOpen size={17} /> },
  { id: "entities", labelKey: "nav.entities", icon: <ListFilter size={17} /> },
  { id: "graph", labelKey: "nav.graph", icon: <Network size={17} /> },
  { id: "timeline", labelKey: "nav.timeline", icon: <GitBranch size={17} /> },
  { id: "ask", labelKey: "nav.ask", icon: <Brain size={17} /> },
  { id: "quality", labelKey: "nav.quality", icon: <ShieldCheck size={17} /> },
  { id: "settings", labelKey: "nav.settings", icon: <Settings size={17} /> }
];

const messages: Record<Locale, Record<string, string>> = {
  en: {
    "app.subtitle": "Corpus explorer, entity graph and agentic retrieval workspace.",
    "status.loading": "Loading corpus state...",
    "status.empty": "No corpus loaded",
    "status.summary": "{episodes} episodes | {entities} entities | {connections} connections",
    "nav.overview": "Overview",
    "nav.ingest": "Ingest",
    "nav.episodes": "Episodes",
    "nav.entities": "Entities",
    "nav.graph": "Graph",
    "nav.timeline": "Timeline",
    "nav.ask": "Ask",
    "nav.quality": "Quality",
    "nav.settings": "Settings",
    "locale.label": "Language",
    "locale.en": "English",
    "locale.es": "Spanish",
    "corpus.selector": "Corpus selector",
    "corpus.default": "Default corpus",
    "corpus.all": "All corpora",
    "action.refresh": "Refresh",
    "overview.entityTypeMix": "Entity Type Mix",
    "overview.richestEpisodes": "Most Informative Episodes",
    "overview.entityCloud": "Entity Cloud",
    "overview.nextActions": "Next Actions",
    "overview.types": "{count} types",
    "overview.indexed": "{count} indexed",
    "overview.mentions": "mentions",
    "overview.chunking": "{strategy} | max {max} words | overlap {overlap} words",
    "ingest.title": "Ingest and Index",
    "ingest.url": "Podcast page, media URL, YouTube video or playlist",
    "ingest.language": "Language",
    "ingest.model": "Whisper model",
    "ingest.seconds": "Limit seconds",
    "ingest.profile": "Domain profile",
    "ingest.process": "Process URL",
    "ingest.running": "processing...",
    "ingest.initial": "Choose a single corpus, paste a URL, and process it into a ready-to-query index.",
    "ingest.allWarning": "Choose a single corpus before ingesting.",
    "ingest.ready": "Ready: {count} item(s), {chunks} indexed chunks.",
    "chunking.timestamp-preserving segment accumulation": "timestamp-preserving segment accumulation",
    "metric.episodes": "episodes",
    "metric.segments": "segments",
    "metric.chunks": "chunks",
    "metric.entities": "entities",
    "metric.entity_mentions": "entity mentions",
    "metric.entity_relations": "entity relations",
    "chart.types": "types",
    "episodes.total": "{count} total",
    "episodes.loading": "Loading episode...",
    "episodes.entities": "{title} | Entities",
    "episodes.entityDensity": "Entity Density",
    "table.id": "ID",
    "table.title": "Title",
    "table.segments": "Segments",
    "table.author": "Author",
    "table.lang": "Lang",
    "table.corpus": "Corpus",
    "table.entity": "Entity",
    "table.type": "Type",
    "table.confidence": "Confidence",
    "table.mentions": "Mentions",
    "table.episodes": "Episodes",
    "action.inspect": "Inspect",
    "entities.cloudAside": "size = mentions",
    "entities.detectedTypes": "Detected Types",
    "entities.index": "Entity Index",
    "entities.rows": "{count} rows",
    "entities.profile": "{name} Profile",
    "entities.connections": "Connections",
    "graph.nodes": "nodes",
    "graph.edges": "edges",
    "graph.visible": "visible",
    "graph.filters": "Graph Filters",
    "graph.minWeight": "min weight",
    "graph.interactive": "Interactive Graph",
    "graph.visibleNodes": "{count} visible nodes",
    "graph.selectedNode": "Selected Node",
    "graph.emptySelection": "Click a node or connection to focus the graph.",
    "graph.nodeStats": "{mentions} mentions across {episodes} episodes",
    "graph.strongest": "Strongest Connections",
    "timeline.controls": "Timeline Controls",
    "timeline.allTopics": "All topics",
    "timeline.load": "Load timeline",
    "timeline.visual": "Visual Timeline",
    "timeline.entries": "{count} entries",
    "ask.title": "Ask the Corpus",
    "ask.thinking": "thinking...",
    "ask.mode": "agentic retrieval",
    "ask.localMode": "Local",
    "ask.llmMode": "LLM",
    "ask.localHelp": "Local retrieval tools only.",
    "ask.llmHelp": "Retrieval plus optional LLM synthesis.",
    "ask.placeholder": "What connects Pizarro with Peru?",
    "ask.initial": "Ask uses local agentic retrieval: topic inference, Qdrant retrieval, connections and evidence context.",
    "quality.issues": "{count} issues",
    "quality.noIssues": "No issues",
    "quality.issue": "Issue {number}",
    "settings.title": "Settings",
    "settings.display": "Display",
    "settings.language": "Dashboard language",
    "settings.languageHelp": "Changes apply immediately and are saved in this browser.",
    "settings.data": "Data",
    "settings.corpusHelp": "Choose a corpus from the top bar. Use All corpora for combined analysis.",
    "recommend.singleCorpus": "Use a single corpus for ingestion/indexing actions.",
    "recommend.allCorpus": "Use corpus=all for cross-podcast exploration.",
    "recommend.ingest": "Ingest transcripts or URLs before building retrieval indexes.",
    "recommend.rebuildEntities": "Run rebuild-entities with an appropriate domain profile.",
    "recommend.indexRetrieval": "Run index-retrieval before hybrid-search, retrieve, or ask.",
    "recommend.lowConfidence": "Inspect low-confidence entities and consider another domain profile.",
    "recommend.timestamps": "Some segments lack timestamps; timestamp-based navigation will be weaker.",
    "recommend.ready": "Corpus looks ready for dashboard exploration and agentic retrieval.",
    "type.person": "Person",
    "type.place": "Place",
    "type.event": "Event",
    "type.concept": "Concept",
    "type.organization": "Organization",
    "type.work": "Work",
    "type.date": "Date",
    "type.unknown": "Unknown"
  },
  es: {
    "app.subtitle": "Explorador de corpus, grafo de entidades y espacio de recuperacion agentica.",
    "status.loading": "Cargando estado del corpus...",
    "status.empty": "No hay corpus cargado",
    "status.summary": "{episodes} episodios | {entities} entidades | {connections} conexiones",
    "nav.overview": "Resumen",
    "nav.ingest": "Ingesta",
    "nav.episodes": "Episodios",
    "nav.entities": "Entidades",
    "nav.graph": "Grafo",
    "nav.timeline": "Linea temporal",
    "nav.ask": "Preguntar",
    "nav.quality": "Calidad",
    "nav.settings": "Ajustes",
    "locale.label": "Idioma",
    "locale.en": "Inglés",
    "locale.es": "Español",
    "corpus.selector": "Selector de corpus",
    "corpus.default": "Corpus por defecto",
    "corpus.all": "Todos los corpus",
    "action.refresh": "Actualizar",
    "overview.entityTypeMix": "Mezcla de tipos de entidad",
    "overview.richestEpisodes": "Episodios con más información",
    "overview.entityCloud": "Nube de entidades",
    "overview.nextActions": "Siguientes acciones",
    "overview.types": "{count} tipos",
    "overview.indexed": "{count} indexadas",
    "overview.mentions": "menciones",
    "overview.chunking": "{strategy} | máximo {max} palabras | solape {overlap} palabras",
    "ingest.title": "Ingestar e Indexar",
    "ingest.url": "Página de podcast, URL de audio/video, YouTube o playlist",
    "ingest.language": "Idioma",
    "ingest.model": "Modelo Whisper",
    "ingest.seconds": "Limitar segundos",
    "ingest.profile": "Perfil de dominio",
    "ingest.process": "Procesar URL",
    "ingest.running": "procesando...",
    "ingest.initial": "Elige un corpus concreto, pega una URL y procésala hasta dejarla lista para consultas.",
    "ingest.allWarning": "Elige un corpus concreto antes de ingestar.",
    "ingest.ready": "Listo: {count} item(s), {chunks} chunks indexados.",
    "chunking.timestamp-preserving segment accumulation": "acumulación de segmentos preservando marcas de tiempo",
    "metric.episodes": "episodios",
    "metric.segments": "segmentos",
    "metric.chunks": "chunks",
    "metric.entities": "entidades",
    "metric.entity_mentions": "menciones de entidades",
    "metric.entity_relations": "relaciones de entidades",
    "chart.types": "tipos",
    "episodes.total": "{count} total",
    "episodes.loading": "Cargando episodio...",
    "episodes.entities": "{title} | Entidades",
    "episodes.entityDensity": "Densidad de Entidades",
    "table.id": "ID",
    "table.title": "Título",
    "table.segments": "Segmentos",
    "table.author": "Autor",
    "table.lang": "Idioma",
    "table.corpus": "Corpus",
    "table.entity": "Entidad",
    "table.type": "Tipo",
    "table.confidence": "Confianza",
    "table.mentions": "Menciones",
    "table.episodes": "Episodios",
    "action.inspect": "Inspeccionar",
    "entities.cloudAside": "tamaño = menciones",
    "entities.detectedTypes": "Tipos detectados",
    "entities.index": "Índice de entidades",
    "entities.rows": "{count} filas",
    "entities.profile": "Perfil de {name}",
    "entities.connections": "Conexiones",
    "graph.nodes": "nodos",
    "graph.edges": "aristas",
    "graph.visible": "visible",
    "graph.filters": "Filtros del grafo",
    "graph.minWeight": "peso mínimo",
    "graph.interactive": "Grafo interactivo",
    "graph.visibleNodes": "{count} nodos visibles",
    "graph.selectedNode": "Nodo seleccionado",
    "graph.emptySelection": "Haz clic en un nodo o conexión para enfocar el grafo.",
    "graph.nodeStats": "{mentions} menciones en {episodes} episodios",
    "graph.strongest": "Conexiones más fuertes",
    "timeline.controls": "Controles de línea temporal",
    "timeline.allTopics": "Todos los temas",
    "timeline.load": "Cargar línea temporal",
    "timeline.visual": "Línea temporal visual",
    "timeline.entries": "{count} entradas",
    "ask.title": "Preguntar al Corpus",
    "ask.thinking": "pensando...",
    "ask.mode": "recuperación agéntica",
    "ask.localMode": "Local",
    "ask.llmMode": "LLM",
    "ask.localHelp": "Solo herramientas locales de recuperación.",
    "ask.llmHelp": "Recuperación más síntesis LLM opcional.",
    "ask.placeholder": "¿Qué conecta a Pizarro con Perú?",
    "ask.initial": "Preguntar usa recuperación agéntica local: inferencia de temas, recuperación Qdrant, conexiones y contexto de evidencia.",
    "quality.issues": "{count} incidencias",
    "quality.noIssues": "Sin incidencias",
    "quality.issue": "Incidencia {number}",
    "settings.title": "Ajustes",
    "settings.display": "Visualización",
    "settings.language": "Idioma del dashboard",
    "settings.languageHelp": "Los cambios se aplican al momento y se guardan en este navegador.",
    "settings.data": "Datos",
    "settings.corpusHelp": "Elige un corpus desde la barra superior. Usa Todos los corpus para análisis conjunto.",
    "recommend.singleCorpus": "Usa un corpus individual para acciones de ingesta e indexado.",
    "recommend.allCorpus": "Usa corpus=all para exploración cruzada entre podcasts.",
    "recommend.ingest": "Ingiere transcripciones o URLs antes de construir índices de recuperación.",
    "recommend.rebuildEntities": "Ejecuta rebuild-entities con un perfil de dominio adecuado.",
    "recommend.indexRetrieval": "Ejecuta index-retrieval antes de hybrid-search, retrieve o ask.",
    "recommend.lowConfidence": "Revisa entidades de baja confianza y considera otro perfil de dominio.",
    "recommend.timestamps": "Algunos segmentos no tienen marcas de tiempo; la navegación temporal será más débil.",
    "recommend.ready": "El corpus está listo para exploración en dashboard y recuperación agéntica.",
    "type.person": "Persona",
    "type.place": "Lugar",
    "type.event": "Evento",
    "type.concept": "Concepto",
    "type.organization": "Organización",
    "type.work": "Obra",
    "type.date": "Fecha",
    "type.unknown": "Desconocido"
  }
};

function makeTranslator(locale: Locale): Translator {
  return (key, values = {}) => {
    const template = messages[locale][key] || messages.en[key] || key;
    return template.replace(/\{(\w+)\}/g, (_, name: string) => String(values[name] ?? ""));
  };
}

function translateRecommendation(value: string, t: Translator) {
  const known: Record<string, string> = {
    "Use a single corpus for ingestion/indexing actions.": "recommend.singleCorpus",
    "Use corpus=all for cross-podcast exploration.": "recommend.allCorpus",
    "Ingest transcripts or URLs before building retrieval indexes.": "recommend.ingest",
    "Run rebuild-entities with an appropriate domain profile.": "recommend.rebuildEntities",
    "Run index-retrieval before hybrid-search, retrieve, or ask.": "recommend.indexRetrieval",
    "Inspect low-confidence entities and consider another domain profile.": "recommend.lowConfidence",
    "Some segments lack timestamps; timestamp-based navigation will be weaker.": "recommend.timestamps",
    "Corpus looks ready for dashboard exploration and agentic retrieval.": "recommend.ready"
  };
  const key = known[value];
  return key ? t(key) : value;
}

function typeLabel(type: string | undefined, t: Translator) {
  return t(`type.${String(type || "unknown").toLowerCase()}`);
}

function chunkingStrategyLabel(strategy: string, t: Translator) {
  return t(`chunking.${strategy}`);
}

const typeColors: Record<string, string> = {
  person: "#5f8e75",
  place: "#557f96",
  event: "#b58a4f",
  concept: "#8b75a4",
  organization: "#ad6d6d",
  work: "#7e8f69",
  date: "#ad6d6d",
  unknown: "#9aa79f"
};

function colorFor(type?: string) {
  return typeColors[String(type || "unknown").toLowerCase()] || typeColors.unknown;
}

function fmtTime(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--:--";
  const total = Math.max(0, Math.floor(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h
    ? `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function Tag({ type, children }: { type: string; children?: React.ReactNode }) {
  const label = children || type;
  return (
    <span className="tag" style={{ backgroundColor: `${colorFor(type)}1f`, color: colorFor(type) }}>
      {label}
    </span>
  );
}

function App() {
  const [active, setActive] = useState<View>("overview");
  const [status, setStatus] = useState<Status | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [graph, setGraph] = useState<GraphData>({ nodes: [], edges: [] });
  const [quality, setQuality] = useState<QualityReport>({});
  const [corpora, setCorpora] = useState<CorpusConfig[]>([]);
  const [activeCorpus, setActiveCorpus] = useState("default");
  const [locale, setLocale] = useState<Locale>(() => (localStorage.getItem("podcast-rag-locale") as Locale) || "en");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const t = useMemo(() => makeTranslator(locale), [locale]);

  function setAndStoreLocale(nextLocale: Locale) {
    setLocale(nextLocale);
    localStorage.setItem("podcast-rag-locale", nextLocale);
  }

  async function loadAll() {
    setLoading(true);
    setError(null);
    try {
      const [nextStatus, nextStats, nextEpisodes, nextTopics, nextGraph, nextQuality] = await Promise.all([
        api.status(activeCorpus),
        api.stats(activeCorpus),
        api.episodes(activeCorpus),
        api.topics(180, activeCorpus),
        api.graph(350, activeCorpus),
        api.quality(activeCorpus)
      ]);
      setStatus(nextStatus);
      setStats(nextStats);
      setEpisodes(nextEpisodes);
      setTopics(nextTopics);
      setGraph(nextGraph);
      setQuality(nextQuality);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void api.corpora().then((payload) => setCorpora([payload.default, ...payload.corpora]));
  }, []);

  useEffect(() => {
    void loadAll();
  }, [activeCorpus]);

  const activeTitle = t(views.find((view) => view.id === active)?.labelKey || "nav.overview");
  const statusLine = error
    ? error
    : loading
      ? t("status.loading")
      : stats
        ? t("status.summary", {
            episodes: stats.counts.episodes,
            entities: stats.counts.entities,
            connections: stats.counts.entity_relations
          })
        : t("status.empty");

  return (
    <div className="layout">
      <aside>
        <div className="brand">
          <Sparkles size={18} />
          <span>Podcast RAG</span>
        </div>
        <p className="subtle">{t("app.subtitle")}</p>
        <nav>
          {views.map((view) => (
            <button key={view.id} className={active === view.id ? "active" : ""} onClick={() => setActive(view.id)}>
              {view.icon}
              {t(view.labelKey)}
            </button>
          ))}
        </nav>
      </aside>

      <main>
        <header className="topbar">
          <div>
            <h1>{activeTitle}</h1>
            <div className={error ? "status-line error" : "status-line"}>{statusLine}</div>
          </div>
          <div className="topbar-actions">
            <select value={activeCorpus} onChange={(event) => setActiveCorpus(event.target.value)} title={t("corpus.selector")}>
              <option value="default">{t("corpus.default")}</option>
              {corpora.filter((corpus) => corpus.id !== "default").map((corpus) => (
                <option key={corpus.id} value={corpus.id}>
                  {corpus.name}
                </option>
              ))}
              {corpora.length > 1 && <option value="all">{t("corpus.all")}</option>}
            </select>
            <button className="btn secondary" onClick={loadAll}>
              <RefreshCw size={16} />
              {t("action.refresh")}
            </button>
          </div>
        </header>

        {active === "overview" && stats && status && <Overview stats={stats} status={status} topics={topics} t={t} />}
        {active === "ingest" && <Ingest corpus={activeCorpus} t={t} onDone={loadAll} />}
        {active === "episodes" && <Episodes episodes={episodes} corpus={activeCorpus} t={t} />}
        {active === "entities" && <Entities topics={topics} corpus={activeCorpus} t={t} />}
        {active === "graph" && <GraphExplorer graph={graph} t={t} />}
        {active === "timeline" && <Timeline topics={topics} corpus={activeCorpus} t={t} />}
        {active === "ask" && <Ask corpus={activeCorpus} t={t} />}
        {active === "quality" && <Quality quality={quality} t={t} />}
        {active === "settings" && <SettingsView locale={locale} setLocale={setAndStoreLocale} activeCorpus={activeCorpus} corpora={corpora} t={t} />}
      </main>
    </div>
  );
}

function Ingest({ corpus, t, onDone }: { corpus: string; t: Translator; onDone: () => Promise<void> }) {
  const [url, setUrl] = useState("");
  const [language, setLanguage] = useState("es");
  const [model, setModel] = useState("tiny");
  const [seconds, setSeconds] = useState("");
  const [profile, setProfile] = useState("generic_es");
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<string>(t("ingest.initial"));

  useEffect(() => {
    if (!url && (result === messages.en["ingest.initial"] || result === messages.es["ingest.initial"])) {
      setResult(t("ingest.initial"));
    }
  }, [t]);

  async function run() {
    if (!url.trim()) return;
    if (corpus === "all") {
      setResult(t("ingest.allWarning"));
      return;
    }
    setRunning(true);
    setResult(t("ingest.running"));
    try {
      const data: ProcessUrlResult = await api.processUrl({
        url: url.trim(),
        corpus,
        language,
        whisper_model: model,
        transcribe_seconds: seconds,
        domain_profile: profile
      });
      await onDone();
      const count = data.ingest.length;
      const chunks = data.index?.indexed_chunks ?? 0;
      const lines = [
        t("ingest.ready", { count, chunks }),
        ...data.ingest.map((item) => `${item.status}: ${item.title || data.source_url} ${item.message || ""}`.trim())
      ];
      setResult(lines.join("\n"));
    } catch (err) {
      setResult(err instanceof Error ? err.message : String(err));
    } finally {
      setRunning(false);
    }
  }

  return (
    <section className="view">
      <Panel title={t("ingest.title")} aside={running ? t("ingest.running") : corpus}>
        <div className="ingest-form">
          <input value={url} onChange={(event) => setUrl(event.target.value)} placeholder={t("ingest.url")} />
          <select value={language} onChange={(event) => setLanguage(event.target.value)} title={t("ingest.language")}>
            <option value="es">es</option>
            <option value="en">en</option>
            <option value="">auto</option>
          </select>
          <select value={model} onChange={(event) => setModel(event.target.value)} title={t("ingest.model")}>
            <option value="tiny">tiny</option>
            <option value="base">base</option>
            <option value="small">small</option>
          </select>
          <input value={seconds} onChange={(event) => setSeconds(event.target.value)} placeholder={t("ingest.seconds")} />
          <select value={profile} onChange={(event) => setProfile(event.target.value)} title={t("ingest.profile")}>
            <option value="generic_es">generic_es</option>
            <option value="history_es">history_es</option>
            <option value="generic_en">generic_en</option>
            <option value="custom">custom</option>
          </select>
          <button className="btn" onClick={run} disabled={running || !url.trim()}>
            {t("ingest.process")}
          </button>
        </div>
        <pre className="answer">{result}</pre>
      </Panel>
    </section>
  );
}

function Overview({ stats, status, topics, t }: { stats: Stats; status: Status; topics: Topic[]; t: Translator }) {
  const maxMetric = Math.max(...Object.values(stats.counts), 1);
  return (
    <section className="view">
      <div className="metric-grid">
        {Object.entries(stats.counts).map(([key, value], index) => (
          <Metric key={key} label={t(`metric.${key}`)} value={value} max={maxMetric} tone={index} />
        ))}
      </div>
      <div className="split">
        <Panel title={t("overview.entityTypeMix")} aside={t("overview.types", { count: stats.entity_types.length })}>
          <Donut rows={stats.entity_types.map((row) => ({ label: row.entity_type, value: row.count }))} t={t} />
        </Panel>
        <Panel title={t("overview.richestEpisodes")} aside={t("overview.mentions")}>
          <Bars rows={stats.richest_episodes.map((row) => ({ label: row.title, value: row.mentions }))} />
        </Panel>
      </div>
      <div className="split">
        <Panel title={t("overview.entityCloud")} aside={t("overview.indexed", { count: topics.length })}>
          <EntityCloud topics={topics.slice(0, 70)} t={t} />
        </Panel>
        <Panel title={t("overview.nextActions")}>
          <ul className="clean-list">
            {status.recommendations.map((item) => (
              <li key={item}>{translateRecommendation(item, t)}</li>
            ))}
          </ul>
          <p className="mini">
            {t("overview.chunking", {
              strategy: chunkingStrategyLabel(status.chunking.strategy, t),
              max: status.chunking.max_words,
              overlap: status.chunking.overlap_words
            })}
          </p>
        </Panel>
      </div>
    </section>
  );
}

function Metric({ label, value, max, tone }: { label: string; value: number; max: number; tone: number }) {
  const colors = ["#5f8e75", "#557f96", "#b58a4f", "#8b75a4", "#ad6d6d"];
  const width = Math.max(4, Math.round((value / max) * 100));
  return (
    <div className="panel metric" style={{ borderLeftColor: colors[tone % colors.length] }}>
      <span>{label}</span>
      <strong>{value}</strong>
      <div className="spark">
        <i style={{ width: `${width}%`, background: colors[tone % colors.length] }} />
      </div>
    </div>
  );
}

function Panel({ title, aside, children }: { title: string; aside?: string; children: React.ReactNode }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <h2>{title}</h2>
        {aside && <span>{aside}</span>}
      </div>
      {children}
    </section>
  );
}

function Bars({ rows }: { rows: Array<{ label: string; value: number }> }) {
  const max = Math.max(...rows.map((row) => row.value), 1);
  return (
    <div className="bars">
      {rows.slice(0, 10).map((row) => (
        <div className="bar-row" key={row.label}>
          <strong title={row.label}>{row.label}</strong>
          <div>
            <i style={{ width: `${Math.max(3, Math.round((row.value / max) * 100))}%` }} />
          </div>
          <span>{row.value}</span>
        </div>
      ))}
    </div>
  );
}

function Donut({ rows, t }: { rows: Array<{ label: string; value: number }>; t: Translator }) {
  const total = rows.reduce((sum, row) => sum + row.value, 0) || 1;
  let offset = 25;
  return (
    <div className="donut-layout">
      <svg viewBox="0 0 100 100" className="donut" role="img" aria-label={t("overview.entityTypeMix")}>
        {rows.map((row) => {
          const value = (row.value / total) * 100;
          const circle = (
            <circle
              key={row.label}
              r="36"
              cx="50"
              cy="50"
              fill="transparent"
              stroke={colorFor(row.label)}
              strokeWidth="15"
              strokeDasharray={`${value} ${100 - value}`}
              strokeDashoffset={offset}
            />
          );
          offset -= value;
          return circle;
        })}
        <circle r="24" cx="50" cy="50" fill="#fffefa" />
        <text x="50" y="48" textAnchor="middle" fontSize="10" fill="#6e7672">
          {t("chart.types")}
        </text>
        <text x="50" y="61" textAnchor="middle" fontSize="15" fontWeight="700" fill="#25302f">
          {rows.length}
        </text>
      </svg>
      <div className="legend">
        {rows.map((row) => (
          <div key={row.label}>
            <Tag type={row.label}>{typeLabel(row.label, t)}</Tag>
            <span>{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EntityCloud({ topics, onPick, t }: { topics: Topic[]; onPick?: (name: string) => void; t: Translator }) {
  const max = Math.max(...topics.map((topic) => topic.mentions), 1);
  return (
    <div className="cloud">
      {topics.map((topic) => {
        const size = 12 + Math.round((topic.mentions / max) * 24);
        return (
          <button
            key={topic.name}
            style={{ fontSize: size, color: colorFor(topic.entity_type) }}
            title={`${typeLabel(topic.entity_type, t)} | ${topic.mentions} ${t("overview.mentions")}`}
            onClick={() => onPick?.(topic.name)}
          >
            {topic.name}
          </button>
        );
      })}
    </div>
  );
}

function Episodes({ episodes, corpus, t }: { episodes: Episode[]; corpus: string; t: Translator }) {
  const [selected, setSelected] = useState<EpisodeInsights | null>(null);
  const [loading, setLoading] = useState(false);

  async function inspect(episode: Episode) {
    setLoading(true);
    setSelected(await api.episodeInsightsForCorpus(episode.id, episode.corpus_id || corpus));
    setLoading(false);
  }

  return (
    <section className="view">
      <Panel title={t("nav.episodes")} aside={t("episodes.total", { count: episodes.length })}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("table.id")}</th>
                <th>{t("table.title")}</th>
                <th>{t("table.segments")}</th>
                <th>{t("table.author")}</th>
                <th>{t("table.lang")}</th>
                <th>{t("table.corpus")}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {episodes.map((episode) => (
                <tr key={episode.id}>
                  <td>{episode.id}</td>
                  <td>{episode.title}</td>
                  <td>{episode.segment_count}</td>
                  <td>{episode.author || ""}</td>
                  <td>{episode.language || ""}</td>
                  <td>{episode.corpus_name || ""}</td>
                  <td>
                    <button className="btn secondary" onClick={() => inspect(episode)}>
                      {t("action.inspect")}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      {loading && <p className="mini">{t("episodes.loading")}</p>}
      {selected && (
        <div className="split">
          <Panel title={t("episodes.entities", { title: selected.episode.title })}>
            <Bars rows={selected.top_entities.map((row) => ({ label: row.name, value: row.mentions }))} />
          </Panel>
          <Panel title={t("episodes.entityDensity")}>
            <Bars rows={selected.entity_density.map((row) => ({ label: `chunk ${row.chunk_id}`, value: row.unique_entities }))} />
          </Panel>
        </div>
      )}
    </section>
  );
}

function Entities({ topics, corpus, t }: { topics: Topic[]; corpus: string; t: Translator }) {
  const [profile, setProfile] = useState<EntityProfile | null>(null);
  const typeRows = useMemo(() => {
    const counts = new Map<string, number>();
    topics.forEach((topic) => counts.set(topic.entity_type, (counts.get(topic.entity_type) || 0) + 1));
    return [...counts].map(([label, value]) => ({ label, value })).sort((a, b) => b.value - a.value);
  }, [topics]);

  async function openProfile(name: string) {
    setProfile(await api.entityProfileForCorpus(name, 25, corpus));
  }

  return (
    <section className="view">
      <div className="split">
        <Panel title={t("overview.entityCloud")} aside={t("entities.cloudAside")}>
          <EntityCloud topics={topics} onPick={openProfile} t={t} />
        </Panel>
        <Panel title={t("entities.detectedTypes")}>
          <Donut rows={typeRows} t={t} />
        </Panel>
      </div>
      <Panel title={t("entities.index")} aside={t("entities.rows", { count: topics.length })}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>{t("table.entity")}</th>
                <th>{t("table.type")}</th>
                <th>{t("table.confidence")}</th>
                <th>{t("table.mentions")}</th>
                <th>{t("table.episodes")}</th>
              </tr>
            </thead>
            <tbody>
              {topics.map((topic) => (
                <tr key={topic.name}>
                  <td>
                    <button className="link-button" onClick={() => openProfile(topic.name)}>
                      {topic.name}
                    </button>
                  </td>
                  <td>
                    <Tag type={topic.entity_type}>{typeLabel(topic.entity_type, t)}</Tag>
                  </td>
                  <td>{topic.confidence.toFixed(2)}</td>
                  <td>{topic.mentions}</td>
                  <td>{topic.episodes}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      {profile && (
        <div className="split">
          <Panel title={t("entities.profile", { name: profile.entity.name })} aside={typeLabel(profile.entity.entity_type, t)}>
            <ul className="clean-list">
              {profile.mentions.slice(0, 12).map((mention) => (
                <li key={`${mention.chunk_id}-${mention.start_seconds}`}>
                  <strong>{fmtTime(mention.start_seconds)}</strong> {mention.text.slice(0, 180)}
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title={t("entities.connections")}>
            <Bars rows={profile.connections.map((row) => ({ label: `${row.source} -> ${row.target}`, value: row.weight }))} />
          </Panel>
        </div>
      )}
    </section>
  );
}

function GraphExplorer({ graph, t }: { graph: GraphData; t: Translator }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [minWeight, setMinWeight] = useState(0);
  const [enabledTypes, setEnabledTypes] = useState<Set<string>>(new Set());

  const types = useMemo(() => [...new Set(graph.nodes.map((node) => node.entity_type || "unknown"))].sort(), [graph.nodes]);

  useEffect(() => {
    setEnabledTypes(new Set(types));
  }, [types]);

  const maxWeight = Math.max(...graph.edges.map((edge) => edge.weight), 1);
  const data = useMemo(() => {
    const nodeMap = new Map(graph.nodes.map((node) => [String(node.id), node]));
    const visibleNodes = graph.nodes.filter((node) => enabledTypes.has(node.entity_type || "unknown"));
    const visibleIds = new Set(visibleNodes.map((node) => String(node.id)));
    const visibleEdges = graph.edges
      .filter((edge) => visibleIds.has(String(edge.source)) && visibleIds.has(String(edge.target)) && edge.weight >= minWeight)
      .slice(0, 180);
    const connected = new Set(visibleEdges.flatMap((edge) => [String(edge.source), String(edge.target)]));
    return {
      nodes: visibleNodes.filter((node) => connected.has(String(node.id))).slice(0, 90),
      edges: visibleEdges,
      nodeMap
    };
  }, [enabledTypes, graph, minWeight]);

  const selectedNode = selected ? graph.nodes.find((node) => String(node.id) === selected) : null;
  const layout = useMemo(() => layoutNodes(data.nodes), [data.nodes]);

  return (
    <section className="view">
      <div className="metric-grid compact">
        <Metric label={t("graph.nodes")} value={graph.nodes.length} max={Math.max(graph.nodes.length, graph.edges.length, 1)} tone={0} />
        <Metric label={t("graph.edges")} value={graph.edges.length} max={Math.max(graph.nodes.length, graph.edges.length, 1)} tone={1} />
        <Metric label={t("graph.visible")} value={data.nodes.length} max={Math.max(graph.nodes.length, 1)} tone={2} />
      </div>
      <Panel title={t("graph.filters")}>
        <div className="filters">
          {types.map((type) => (
            <label key={type}>
              <input
                type="checkbox"
                checked={enabledTypes.has(type)}
                onChange={(event) => {
                  const next = new Set(enabledTypes);
                  if (event.target.checked) next.add(type);
                  else next.delete(type);
                  setEnabledTypes(next);
                }}
              />
              <Tag type={type}>{typeLabel(type, t)}</Tag>
            </label>
          ))}
          <label>
            {t("graph.minWeight")}
            <input
              type="range"
              min={0}
              max={maxWeight}
              step={Math.max(maxWeight / 100, 0.01)}
              value={minWeight}
              onChange={(event) => setMinWeight(Number(event.target.value))}
            />
            <span>{minWeight.toFixed(2)}</span>
          </label>
        </div>
      </Panel>
      <div className="split graph-split">
        <Panel title={t("graph.interactive")} aside={t("graph.visibleNodes", { count: data.nodes.length })}>
          <svg className="graph-canvas" viewBox="0 0 920 560" role="img" aria-label={t("graph.interactive")}>
            {data.edges.map((edge) => {
              const source = layout.get(String(edge.source));
              const target = layout.get(String(edge.target));
              if (!source || !target) return null;
              const active = selected && (String(edge.source) === selected || String(edge.target) === selected);
              return (
                <line
                  key={`${edge.source}-${edge.target}`}
                  className={active ? "graph-edge active" : selected ? "graph-edge dim" : "graph-edge"}
                  x1={source.x}
                  y1={source.y}
                  x2={target.x}
                  y2={target.y}
                  strokeWidth={Math.max(1, Math.min(7, edge.weight))}
                />
              );
            })}
            {data.nodes.map((node) => {
              const point = layout.get(String(node.id));
              if (!point) return null;
              const active = selected === String(node.id);
              const dim = selected && !isNeighbor(selected, node, data.edges);
              return (
                <g
                  key={node.id}
                  className={active ? "graph-node active" : dim ? "graph-node dim" : "graph-node"}
                  transform={`translate(${point.x},${point.y})`}
                  onClick={() => setSelected(active ? null : String(node.id))}
                >
                  <circle r={point.r} fill={colorFor(node.entity_type)} />
                  <text x={point.r + 6} y="4">
                    {node.name.slice(0, 24)}
                  </text>
                </g>
              );
            })}
          </svg>
        </Panel>
        <div className="stack">
          <Panel title={t("graph.selectedNode")}>
            {selectedNode ? (
              <div className="selected-node">
                <strong>{selectedNode.name}</strong>
                <Tag type={selectedNode.entity_type}>{typeLabel(selectedNode.entity_type, t)}</Tag>
                <p>{t("graph.nodeStats", { mentions: selectedNode.mentions ?? 0, episodes: selectedNode.episodes ?? 0 })}</p>
              </div>
            ) : (
              <p className="mini">{t("graph.emptySelection")}</p>
            )}
          </Panel>
          <Panel title={t("graph.strongest")}>
            <div className="connection-list">
              {data.edges.slice(0, 60).map((edge) => {
                const source = data.nodeMap.get(String(edge.source));
                const target = data.nodeMap.get(String(edge.target));
                const active = selected && (String(edge.source) === selected || String(edge.target) === selected);
                return (
                  <button
                    key={`${edge.source}-${edge.target}-${edge.relation_type}`}
                    className={active ? "connection-card active" : "connection-card"}
                    onClick={() => setSelected(String(edge.source))}
                  >
                    <strong>{source?.name || edge.source}</strong>
                    <span>{" -> "}</span>
                    <strong>{target?.name || edge.target}</strong>
                    <small>{edge.relation_type} | {edge.weight.toFixed(2)}</small>
                  </button>
                );
              })}
            </div>
          </Panel>
        </div>
      </div>
    </section>
  );
}

function layoutNodes(nodes: GraphNode[]) {
  const maxMentions = Math.max(...nodes.map((node) => node.mentions), 1);
  const map = new Map<string, { x: number; y: number; r: number }>();
  nodes.forEach((node, index) => {
    const ring = Math.floor(index / 18);
    const inRing = index % 18;
    const angle = (inRing / Math.min(18, Math.max(nodes.length, 1))) * Math.PI * 2 + ring * 0.31;
    const radius = Math.min(90 + ring * 76, 390);
    map.set(String(node.id), {
      x: 460 + Math.cos(angle) * radius,
      y: 280 + Math.sin(angle) * Math.min(radius, 238),
      r: 7 + Math.round((node.mentions / maxMentions) * 16)
    });
  });
  return map;
}

function isNeighbor(selected: string, node: GraphNode, edges: GraphData["edges"]) {
  const id = String(node.id);
  if (id === selected) return true;
  return edges.some((edge) => (String(edge.source) === selected && String(edge.target) === id) || (String(edge.target) === selected && String(edge.source) === id));
}

function Timeline({ topics, corpus, t }: { topics: Topic[]; corpus: string; t: Translator }) {
  const [topic, setTopic] = useState("");
  const [entries, setEntries] = useState<TimelineEntry[]>([]);

  async function load() {
    setEntries(await api.timeline(topic || undefined, corpus));
  }

  return (
    <section className="view">
      <Panel title={t("timeline.controls")}>
        <div className="toolbar">
          <select value={topic} onChange={(event) => setTopic(event.target.value)}>
            <option value="">{t("timeline.allTopics")}</option>
            {topics.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
          <button className="btn" onClick={load}>
            {t("timeline.load")}
          </button>
        </div>
      </Panel>
      <Panel title={t("timeline.visual")} aside={t("timeline.entries", { count: entries.length })}>
        <div className="timeline">
          {entries.slice(0, 100).map((entry) => (
            <article key={`${entry.episode_id}-${entry.name}-${entry.start_seconds}`} style={{ borderLeftColor: colorFor(entry.entity_type) }}>
              <time>{fmtTime(entry.start_seconds)}</time>
              <div>
                <strong>{entry.name}</strong> <Tag type={entry.entity_type}>{typeLabel(entry.entity_type, t)}</Tag>
                <p>{entry.text.slice(0, 220)}</p>
                <small>{entry.title}</small>
              </div>
            </article>
          ))}
        </div>
      </Panel>
    </section>
  );
}

function Ask({ corpus, t }: { corpus: string; t: Translator }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(t("ask.initial"));
  const [mode, setMode] = useState<"local" | "llm">("local");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!question && (answer === messages.en["ask.initial"] || answer === messages.es["ask.initial"])) {
      setAnswer(t("ask.initial"));
    }
  }, [t]);

  async function run() {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const result = await api.ask(question.trim(), 5, corpus, mode);
      setAnswer(result.brief);
    } catch (err) {
      setAnswer(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="view">
      <Panel title={t("ask.title")} aside={loading ? t("ask.thinking") : t("ask.mode")}>
        <div className="ask-box">
          <Search size={18} />
          <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder={t("ask.placeholder")} />
          <div className="mode-switch" aria-label={t("ask.mode")}>
            <button className={mode === "local" ? "active" : ""} onClick={() => setMode("local")} title={t("ask.localHelp")}>
              {t("ask.localMode")}
            </button>
            <button className={mode === "llm" ? "active" : ""} onClick={() => setMode("llm")} title={t("ask.llmHelp")}>
              {t("ask.llmMode")}
            </button>
          </div>
          <button className="btn" onClick={run}>
            {t("nav.ask")}
          </button>
        </div>
        <pre className="answer">{answer}</pre>
      </Panel>
    </section>
  );
}

function Quality({ quality, t }: { quality: QualityReport; t: Translator }) {
  const sections = Object.entries(quality);
  return (
    <section className="view">
      {sections.map(([name, rows]) => (
        <Panel key={name} title={name} aside={t("quality.issues", { count: rows.length })}>
          {rows.length === 0 ? (
            <p className="mini">{t("quality.noIssues")}</p>
          ) : (
            <div className="quality-list">
              {rows.slice(0, 40).map((row, index) => (
                <details key={index}>
                  <summary>
                    <CircleHelp size={15} />
                    {t("quality.issue", { number: index + 1 })}
                  </summary>
                  <pre>{JSON.stringify(row, null, 2)}</pre>
                </details>
              ))}
            </div>
          )}
        </Panel>
      ))}
    </section>
  );
}

function SettingsView({
  locale,
  setLocale,
  activeCorpus,
  corpora,
  t
}: {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  activeCorpus: string;
  corpora: CorpusConfig[];
  t: Translator;
}) {
  const activeCorpusName =
    activeCorpus === "all"
      ? t("corpus.all")
      : corpora.find((corpus) => corpus.id === activeCorpus)?.name || t("corpus.default");

  return (
    <section className="view">
      <Panel title={t("settings.display")}>
        <div className="settings-list">
          <label className="setting-row">
            <div>
              <strong>{t("settings.language")}</strong>
              <p>{t("settings.languageHelp")}</p>
            </div>
            <select value={locale} onChange={(event) => setLocale(event.target.value as Locale)}>
              <option value="en">{t("locale.en")}</option>
              <option value="es">{t("locale.es")}</option>
            </select>
          </label>
        </div>
      </Panel>
      <Panel title={t("settings.data")}>
        <div className="settings-list">
          <div className="setting-row">
            <div>
              <strong>{t("corpus.selector")}</strong>
              <p>{t("settings.corpusHelp")}</p>
            </div>
            <span className="setting-pill">{activeCorpusName}</span>
          </div>
        </div>
      </Panel>
    </section>
  );
}

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
