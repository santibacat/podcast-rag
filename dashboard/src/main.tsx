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
  ShieldCheck,
  Sparkles
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
  Stats,
  Status,
  TimelineEntry,
  Topic
} from "./types";
import "./styles.css";

type View = "overview" | "episodes" | "entities" | "graph" | "timeline" | "ask" | "quality";

const views: Array<{ id: View; label: string; icon: React.ReactNode }> = [
  { id: "overview", label: "Overview", icon: <Activity size={17} /> },
  { id: "episodes", label: "Episodes", icon: <BookOpen size={17} /> },
  { id: "entities", label: "Entities", icon: <ListFilter size={17} /> },
  { id: "graph", label: "Graph", icon: <Network size={17} /> },
  { id: "timeline", label: "Timeline", icon: <GitBranch size={17} /> },
  { id: "ask", label: "Ask", icon: <Brain size={17} /> },
  { id: "quality", label: "Quality", icon: <ShieldCheck size={17} /> }
];

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
  return (
    <span className="tag" style={{ backgroundColor: `${colorFor(type)}1f`, color: colorFor(type) }}>
      {children || type}
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
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

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

  const activeTitle = views.find((view) => view.id === active)?.label || "Dashboard";
  const statusLine = error
    ? error
    : loading
      ? "Loading corpus state..."
      : stats
        ? `${stats.counts.episodes} episodes | ${stats.counts.entities} entities | ${stats.counts.entity_relations} connections`
        : "No corpus loaded";

  return (
    <div className="layout">
      <aside>
        <div className="brand">
          <Sparkles size={18} />
          <span>Podcast RAG</span>
        </div>
        <p className="subtle">Corpus explorer, entity graph and agentic retrieval workspace.</p>
        <nav>
          {views.map((view) => (
            <button key={view.id} className={active === view.id ? "active" : ""} onClick={() => setActive(view.id)}>
              {view.icon}
              {view.label}
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
            <select value={activeCorpus} onChange={(event) => setActiveCorpus(event.target.value)} title="Corpus selector">
              <option value="default">Default corpus</option>
              {corpora.filter((corpus) => corpus.id !== "default").map((corpus) => (
                <option key={corpus.id} value={corpus.id}>
                  {corpus.name}
                </option>
              ))}
              {corpora.length > 1 && <option value="all">All corpora</option>}
            </select>
            <button className="btn secondary" onClick={loadAll}>
              <RefreshCw size={16} />
              Refresh
            </button>
          </div>
        </header>

        {active === "overview" && stats && status && <Overview stats={stats} status={status} topics={topics} />}
        {active === "episodes" && <Episodes episodes={episodes} corpus={activeCorpus} />}
        {active === "entities" && <Entities topics={topics} corpus={activeCorpus} />}
        {active === "graph" && <GraphExplorer graph={graph} />}
        {active === "timeline" && <Timeline topics={topics} corpus={activeCorpus} />}
        {active === "ask" && <Ask corpus={activeCorpus} />}
        {active === "quality" && <Quality quality={quality} />}
      </main>
    </div>
  );
}

function Overview({ stats, status, topics }: { stats: Stats; status: Status; topics: Topic[] }) {
  const maxMetric = Math.max(...Object.values(stats.counts), 1);
  return (
    <section className="view">
      <div className="metric-grid">
        {Object.entries(stats.counts).map(([key, value], index) => (
          <Metric key={key} label={key.replaceAll("_", " ")} value={value} max={maxMetric} tone={index} />
        ))}
      </div>
      <div className="split">
        <Panel title="Entity Type Mix" aside={`${stats.entity_types.length} types`}>
          <Donut rows={stats.entity_types.map((row) => ({ label: row.entity_type, value: row.count }))} />
        </Panel>
        <Panel title="Richest Episodes" aside="mentions">
          <Bars rows={stats.richest_episodes.map((row) => ({ label: row.title, value: row.mentions }))} />
        </Panel>
      </div>
      <div className="split">
        <Panel title="Entity Cloud" aside={`${topics.length} indexed`}>
          <EntityCloud topics={topics.slice(0, 70)} />
        </Panel>
        <Panel title="Next Actions">
          <ul className="clean-list">
            {status.recommendations.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
          <p className="mini">
            {status.chunking.strategy} | max {status.chunking.max_words} words | overlap {status.chunking.overlap_words}
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

function Donut({ rows }: { rows: Array<{ label: string; value: number }> }) {
  const total = rows.reduce((sum, row) => sum + row.value, 0) || 1;
  let offset = 25;
  return (
    <div className="donut-layout">
      <svg viewBox="0 0 100 100" className="donut" role="img" aria-label="Entity type distribution">
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
          types
        </text>
        <text x="50" y="61" textAnchor="middle" fontSize="15" fontWeight="700" fill="#25302f">
          {rows.length}
        </text>
      </svg>
      <div className="legend">
        {rows.map((row) => (
          <div key={row.label}>
            <Tag type={row.label} />
            <span>{row.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EntityCloud({ topics, onPick }: { topics: Topic[]; onPick?: (name: string) => void }) {
  const max = Math.max(...topics.map((topic) => topic.mentions), 1);
  return (
    <div className="cloud">
      {topics.map((topic) => {
        const size = 12 + Math.round((topic.mentions / max) * 24);
        return (
          <button
            key={topic.name}
            style={{ fontSize: size, color: colorFor(topic.entity_type) }}
            title={`${topic.entity_type} | ${topic.mentions} mentions`}
            onClick={() => onPick?.(topic.name)}
          >
            {topic.name}
          </button>
        );
      })}
    </div>
  );
}

function Episodes({ episodes, corpus }: { episodes: Episode[]; corpus: string }) {
  const [selected, setSelected] = useState<EpisodeInsights | null>(null);
  const [loading, setLoading] = useState(false);

  async function inspect(episode: Episode) {
    setLoading(true);
    setSelected(await api.episodeInsightsForCorpus(episode.id, episode.corpus_id || corpus));
    setLoading(false);
  }

  return (
    <section className="view">
      <Panel title="Episodes" aside={`${episodes.length} total`}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Title</th>
                <th>Segments</th>
                <th>Author</th>
                <th>Lang</th>
                <th>Corpus</th>
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
                      Inspect
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
      {loading && <p className="mini">Loading episode...</p>}
      {selected && (
        <div className="split">
          <Panel title={`${selected.episode.title} | Entities`}>
            <Bars rows={selected.top_entities.map((row) => ({ label: row.name, value: row.mentions }))} />
          </Panel>
          <Panel title="Entity Density">
            <Bars rows={selected.entity_density.map((row) => ({ label: `chunk ${row.chunk_id}`, value: row.unique_entities }))} />
          </Panel>
        </div>
      )}
    </section>
  );
}

function Entities({ topics, corpus }: { topics: Topic[]; corpus: string }) {
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
        <Panel title="Entity Cloud" aside="size = mentions">
          <EntityCloud topics={topics} onPick={openProfile} />
        </Panel>
        <Panel title="Detected Types">
          <Donut rows={typeRows} />
        </Panel>
      </div>
      <Panel title="Entity Index" aside={`${topics.length} rows`}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Entity</th>
                <th>Type</th>
                <th>Confidence</th>
                <th>Mentions</th>
                <th>Episodes</th>
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
                    <Tag type={topic.entity_type} />
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
          <Panel title={`${profile.entity.name} Profile`} aside={profile.entity.entity_type}>
            <ul className="clean-list">
              {profile.mentions.slice(0, 12).map((mention) => (
                <li key={`${mention.chunk_id}-${mention.start_seconds}`}>
                  <strong>{fmtTime(mention.start_seconds)}</strong> {mention.text.slice(0, 180)}
                </li>
              ))}
            </ul>
          </Panel>
          <Panel title="Connections">
            <Bars rows={profile.connections.map((row) => ({ label: `${row.source} -> ${row.target}`, value: row.weight }))} />
          </Panel>
        </div>
      )}
    </section>
  );
}

function GraphExplorer({ graph }: { graph: GraphData }) {
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
        <Metric label="nodes" value={graph.nodes.length} max={Math.max(graph.nodes.length, graph.edges.length, 1)} tone={0} />
        <Metric label="edges" value={graph.edges.length} max={Math.max(graph.nodes.length, graph.edges.length, 1)} tone={1} />
        <Metric label="visible" value={data.nodes.length} max={Math.max(graph.nodes.length, 1)} tone={2} />
      </div>
      <Panel title="Graph Filters">
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
              <Tag type={type} />
            </label>
          ))}
          <label>
            min weight
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
        <Panel title="Interactive Graph" aside={`${data.nodes.length} visible nodes`}>
          <svg className="graph-canvas" viewBox="0 0 920 560" role="img" aria-label="Interactive entity graph">
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
          <Panel title="Selected Node">
            {selectedNode ? (
              <div className="selected-node">
                <strong>{selectedNode.name}</strong>
                <Tag type={selectedNode.entity_type} />
                <p>{selectedNode.mentions ?? 0} mentions across {selectedNode.episodes ?? 0} episodes</p>
              </div>
            ) : (
              <p className="mini">Click a node or connection to focus the graph.</p>
            )}
          </Panel>
          <Panel title="Strongest Connections">
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

function Timeline({ topics, corpus }: { topics: Topic[]; corpus: string }) {
  const [topic, setTopic] = useState("");
  const [entries, setEntries] = useState<TimelineEntry[]>([]);

  async function load() {
    setEntries(await api.timeline(topic || undefined, corpus));
  }

  return (
    <section className="view">
      <Panel title="Timeline Controls">
        <div className="toolbar">
          <select value={topic} onChange={(event) => setTopic(event.target.value)}>
            <option value="">All topics</option>
            {topics.map((item) => (
              <option key={item.name} value={item.name}>
                {item.name}
              </option>
            ))}
          </select>
          <button className="btn" onClick={load}>
            Load timeline
          </button>
        </div>
      </Panel>
      <Panel title="Visual Timeline" aside={`${entries.length} entries`}>
        <div className="timeline">
          {entries.slice(0, 100).map((entry) => (
            <article key={`${entry.episode_id}-${entry.name}-${entry.start_seconds}`} style={{ borderLeftColor: colorFor(entry.entity_type) }}>
              <time>{fmtTime(entry.start_seconds)}</time>
              <div>
                <strong>{entry.name}</strong> <Tag type={entry.entity_type} />
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

function Ask({ corpus }: { corpus: string }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("Ask uses local agentic retrieval: topic inference, Qdrant retrieval, connections and evidence context.");
  const [loading, setLoading] = useState(false);

  async function run() {
    if (!question.trim()) return;
    setLoading(true);
    try {
      const result = await api.ask(question.trim(), 5, corpus);
      setAnswer(result.brief);
    } catch (err) {
      setAnswer(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="view">
      <Panel title="Ask the Corpus" aside={loading ? "thinking..." : "agentic retrieval"}>
        <div className="ask-box">
          <Search size={18} />
          <input value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="What connects Pizarro with Peru?" />
          <button className="btn" onClick={run}>
            Ask
          </button>
        </div>
        <pre className="answer">{answer}</pre>
      </Panel>
    </section>
  );
}

function Quality({ quality }: { quality: QualityReport }) {
  const sections = Object.entries(quality);
  return (
    <section className="view">
      {sections.map(([name, rows]) => (
        <Panel key={name} title={name} aside={`${rows.length} issues`}>
          {rows.length === 0 ? (
            <p className="mini">No issues</p>
          ) : (
            <div className="quality-list">
              {rows.slice(0, 40).map((row, index) => (
                <details key={index}>
                  <summary>
                    <CircleHelp size={15} />
                    Issue {index + 1}
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

createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
