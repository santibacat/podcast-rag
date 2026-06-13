from __future__ import annotations

import argparse
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from podcast_rag.agent_tools import AgentToolConfig, agentic_research
from podcast_rag.analytics import (
    corpus_stats,
    entity_profile,
    entity_timeline,
    episode_insights,
    graph_export,
    quality_report,
    system_status,
    topic_episode_matrix,
)
from podcast_rag.config import build_settings
from podcast_rag.corpora import corpus_to_dict, load_corpora, resolve_corpus_set
from podcast_rag.search import entity_connections, list_episodes, list_topics


def create_handler(data_dir: Path, qdrant_url: str | None = None):
    settings = build_settings(data_dir, qdrant_url=qdrant_url)
    static_dir = _dashboard_static_dir()

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path.startswith("/api/"):
                self._handle_api(parsed.path, parse_qs(parsed.query))
                return
            if self._send_static(parsed.path):
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")

        def log_message(self, format: str, *args: object) -> None:
            return

        def _handle_api(self, path: str, query: dict[str, list[str]]) -> None:
            try:
                payload = route_api(path, query, settings.data_dir, settings.qdrant_url)
            except LookupError as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.NOT_FOUND)
                return
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            self._send_json(payload)

        def _send_html(self, html: str) -> None:
            data = html.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _send_static(self, request_path: str) -> bool:
            if static_dir is None:
                if request_path in {"/", "/index.html"}:
                    self._send_html(BUILD_MISSING_HTML)
                    return True
                return False

            relative = "index.html" if request_path in {"/", "/index.html"} else request_path.lstrip("/")
            target = (static_dir / relative).resolve()
            if static_dir not in target.parents and target != static_dir:
                return False
            if not target.exists() or not target.is_file():
                target = static_dir / "index.html"
                if not target.exists():
                    return False

            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            data = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return True

    return DashboardHandler


def _dashboard_static_dir() -> Path | None:
    repo_root = Path(__file__).resolve().parents[2]
    static_dir = repo_root / "dashboard" / "dist"
    return static_dir if (static_dir / "index.html").exists() else None


def route_api(path: str, query: dict[str, list[str]], data_dir: Path, qdrant_url: str | None) -> object:
    if path == "/api/corpora":
        return {
            "default": corpus_to_dict(resolve_corpus_set(data_dir, "default")[0]),
            "corpora": [corpus_to_dict(corpus) for corpus in load_corpora(data_dir)],
        }

    corpus_selector = _str_query(query, "corpus")
    corpora = resolve_corpus_set(data_dir, corpus_selector)
    if len(corpora) > 1:
        return route_multi_corpus_api(path, query, corpora, qdrant_url)

    settings = build_settings(Path(corpora[0].data_dir), qdrant_url=qdrant_url or corpora[0].qdrant_url)
    db_path = settings.db_path
    if path == "/api/status":
        return system_status(db_path, settings.data_dir, settings.qdrant_url)
    if path == "/api/stats":
        return corpus_stats(db_path)
    if path == "/api/episodes":
        return list_episodes(db_path)
    if path == "/api/topics":
        return list_topics(db_path, limit=_int_query(query, "limit", 100))
    if path == "/api/connections":
        return entity_connections(db_path, name=_str_query(query, "topic"), limit=_int_query(query, "limit", 100))
    if path == "/api/entity-profile":
        name = _required_query(query, "name")
        return entity_profile(db_path, name=name, limit=_int_query(query, "limit", 20))
    if path == "/api/timeline":
        return entity_timeline(
            db_path,
            topic=_str_query(query, "topic"),
            episode_id=_optional_int_query(query, "episode_id"),
            limit=_int_query(query, "limit", 200),
        )
    if path == "/api/episode-insights":
        return episode_insights(db_path, episode_id=_int_query(query, "episode_id", 1), limit=_int_query(query, "limit", 30))
    if path == "/api/topic-matrix":
        return topic_episode_matrix(db_path, limit_entities=_int_query(query, "limit_entities", 50))
    if path == "/api/graph":
        return graph_export(db_path, min_weight=_float_query(query, "min_weight", 0.0), limit=_int_query(query, "limit", 500))
    if path == "/api/quality":
        return quality_report(db_path)
    if path == "/api/ask":
        question = _required_query(query, "q")
        return agentic_research(
            question=question,
            limit=_int_query(query, "limit", 5),
            config=AgentToolConfig(data_dir=settings.data_dir, qdrant_url=settings.qdrant_url),
        )
    raise LookupError(f"Unknown API route: {path}")


def route_multi_corpus_api(
    path: str,
    query: dict[str, list[str]],
    corpora: list[object],
    qdrant_url: str | None,
) -> object:
    if path == "/api/status":
        stats = _merge_stats([_corpus_stats(corpus) for corpus in corpora])
        return {
            "stats": stats,
            "chunking": system_status(Path(str(corpora[0].data_dir)) / "podcast_rag.sqlite3", Path(str(corpora[0].data_dir))).get("chunking"),
            "qdrant": {"mode": "multi-corpus"},
            "recommendations": ["Use a single corpus for ingestion/indexing actions.", "Use corpus=all for cross-podcast exploration."],
            "corpora": [corpus_to_dict(corpus) for corpus in corpora],
        }
    if path == "/api/stats":
        return _merge_stats([_corpus_stats(corpus) for corpus in corpora])
    if path == "/api/episodes":
        rows: list[dict[str, object]] = []
        for corpus in corpora:
            rows.extend(_tag_rows(list_episodes(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3"), corpus))
        return rows
    if path == "/api/topics":
        return _merge_topics(
            [
                _tag_rows(list_topics(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3", limit=_int_query(query, "limit", 1000)), corpus)
                for corpus in corpora
            ],
            limit=_int_query(query, "limit", 100),
        )
    if path == "/api/connections":
        return _merge_connections(
            [
                _tag_rows(
                    entity_connections(
                        Path(str(corpus.data_dir)) / "podcast_rag.sqlite3",
                        name=_str_query(query, "topic"),
                        limit=_int_query(query, "limit", 1000),
                    ),
                    corpus,
                )
                for corpus in corpora
            ],
            limit=_int_query(query, "limit", 100),
        )
    if path == "/api/graph":
        topics = _merge_topics(
            [_tag_rows(list_topics(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3", limit=1000), corpus) for corpus in corpora],
            limit=_int_query(query, "limit", 500),
        )
        connections = _merge_connections(
            [_tag_rows(entity_connections(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3", limit=2000), corpus) for corpus in corpora],
            limit=_int_query(query, "limit", 500),
        )
        return _graph_from_merged(topics, connections)
    if path == "/api/timeline":
        rows = []
        for corpus in corpora:
            rows.extend(
                _tag_rows(
                    entity_timeline(
                        Path(str(corpus.data_dir)) / "podcast_rag.sqlite3",
                        topic=_str_query(query, "topic"),
                        episode_id=_optional_int_query(query, "episode_id"),
                        limit=_int_query(query, "limit", 200),
                    ),
                    corpus,
                )
            )
        return rows[: _int_query(query, "limit", 200)]
    if path == "/api/entity-profile":
        name = _required_query(query, "name")
        profiles = []
        for corpus in corpora:
            try:
                profile = entity_profile(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3", name=name, limit=_int_query(query, "limit", 20))
            except LookupError:
                continue
            profiles.append(_tag_profile(profile, corpus))
        if not profiles:
            raise LookupError(f"Entity {name!r} does not exist")
        return _merge_entity_profiles(name, profiles)
    if path == "/api/quality":
        merged: dict[str, list[dict[str, object]]] = {}
        for corpus in corpora:
            report = quality_report(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3")
            for section, rows in report.items():
                merged.setdefault(section, []).extend(_tag_rows(rows, corpus))
        return merged
    if path == "/api/ask":
        question = _required_query(query, "q")
        results = []
        for corpus in corpora:
            settings = build_settings(Path(str(corpus.data_dir)), qdrant_url=qdrant_url or corpus.qdrant_url)
            try:
                result = agentic_research(
                    question=question,
                    limit=_int_query(query, "limit", 5),
                    config=AgentToolConfig(data_dir=settings.data_dir, qdrant_url=settings.qdrant_url),
                )
            except Exception as exc:
                result = {"brief": f"Failed for {corpus.id}: {exc}", "tool_calls": [], "evidence": []}
            result["corpus_id"] = corpus.id
            result["corpus_name"] = corpus.name
            results.append(result)
        return {
            "brief": "\n\n".join(f"[{item['corpus_name']}]\n{item['brief']}" for item in results),
            "corpora": results,
        }
    raise LookupError(f"Unknown multi-corpus API route: {path}")


def _corpus_stats(corpus: Any) -> dict[str, Any]:
    stats = corpus_stats(Path(str(corpus.data_dir)) / "podcast_rag.sqlite3")
    stats["corpus_id"] = corpus.id
    stats["corpus_name"] = corpus.name
    return stats


def _tag_rows(rows: list[dict[str, Any]], corpus: Any) -> list[dict[str, Any]]:
    tagged = []
    for row in rows:
        item = dict(row)
        item["corpus_id"] = corpus.id
        item["corpus_name"] = corpus.name
        tagged.append(item)
    return tagged


def _merge_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    entity_types: dict[str, int] = {}
    richest_episodes: list[dict[str, Any]] = []
    for stats in stats_list:
        for key, value in stats.get("counts", {}).items():
            counts[key] = counts.get(key, 0) + int(value or 0)
        for row in stats.get("entity_types", []):
            entity_type = str(row["entity_type"])
            entity_types[entity_type] = entity_types.get(entity_type, 0) + int(row["count"] or 0)
        richest_episodes.extend(_tag_rows(stats.get("richest_episodes", []), _StatsCorpus(stats)))
    return {
        "counts": counts,
        "entity_types": [
            {"entity_type": entity_type, "count": count}
            for entity_type, count in sorted(entity_types.items(), key=lambda item: (-item[1], item[0]))
        ],
        "richest_episodes": sorted(
            richest_episodes,
            key=lambda row: (int(row.get("unique_entities") or 0), int(row.get("mentions") or 0)),
            reverse=True,
        )[:10],
        "avg_entities_per_episode": counts.get("entities", 0) / counts.get("episodes", 1) if counts.get("episodes") else 0,
    }


class _StatsCorpus:
    def __init__(self, stats: dict[str, Any]) -> None:
        self.id = stats.get("corpus_id")
        self.name = stats.get("corpus_name")


def _merge_topics(topic_groups: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for group in topic_groups:
        for row in group:
            key = (str(row["name"]), str(row.get("entity_type") or "UNKNOWN"))
            item = merged.setdefault(
                key,
                {
                    "name": key[0],
                    "entity_type": key[1],
                    "confidence": 0.0,
                    "mentions": 0,
                    "episodes": 0,
                    "corpora": [],
                },
            )
            item["confidence"] = max(float(item["confidence"]), float(row.get("confidence") or 0))
            item["mentions"] += int(row.get("mentions") or 0)
            item["episodes"] += int(row.get("episodes") or 0)
            item["corpora"].append({"id": row.get("corpus_id"), "name": row.get("corpus_name")})
    return sorted(merged.values(), key=lambda row: (-int(row["mentions"]), str(row["name"])))[:limit]


def _merge_connections(connection_groups: list[list[dict[str, Any]]], limit: int) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for group in connection_groups:
        for row in group:
            source = str(row["source"])
            target = str(row["target"])
            ordered = tuple(sorted([source, target]))
            key = (ordered[0], ordered[1], str(row.get("relation_type") or "CO_OCCURS"))
            item = merged.setdefault(
                key,
                {
                    "source": key[0],
                    "target": key[1],
                    "source_type": row.get("source_type"),
                    "target_type": row.get("target_type"),
                    "relation_type": key[2],
                    "weight": 0.0,
                    "shared_chunks": 0,
                    "shared_episodes": 0,
                    "corpora": [],
                },
            )
            item["weight"] += float(row.get("weight") or 0)
            item["shared_chunks"] += int(row.get("shared_chunks") or 0)
            item["shared_episodes"] += int(row.get("shared_episodes") or 0)
            item["corpora"].append({"id": row.get("corpus_id"), "name": row.get("corpus_name")})
    return sorted(merged.values(), key=lambda row: (-float(row["weight"]), str(row["source"]), str(row["target"])))[:limit]


def _graph_from_merged(topics: list[dict[str, Any]], connections: list[dict[str, Any]]) -> dict[str, Any]:
    nodes = [
        {
            "id": row["name"],
            "name": row["name"],
            "entity_type": row["entity_type"],
            "mentions": row["mentions"],
            "episodes": row["episodes"],
            "corpora": row.get("corpora", []),
        }
        for row in topics
    ]
    node_names = {str(node["id"]) for node in nodes}
    edges = [
        {
            "source": row["source"],
            "target": row["target"],
            "relation_type": row["relation_type"],
            "weight": row["weight"],
            "shared_chunks": row["shared_chunks"],
            "shared_episodes": row["shared_episodes"],
            "corpora": row.get("corpora", []),
        }
        for row in connections
        if str(row["source"]) in node_names and str(row["target"]) in node_names
    ]
    return {"nodes": nodes, "edges": edges}


def _tag_profile(profile: dict[str, Any], corpus: Any) -> dict[str, Any]:
    item = dict(profile)
    item["corpus_id"] = corpus.id
    item["corpus_name"] = corpus.name
    item["mentions"] = _tag_rows(item.get("mentions", []), corpus)
    item["connections"] = _tag_rows(item.get("connections", []), corpus)
    item["episodes"] = _tag_rows(item.get("episodes", []), corpus)
    return item


def _merge_entity_profiles(name: str, profiles: list[dict[str, Any]]) -> dict[str, Any]:
    entity = dict(profiles[0]["entity"])
    entity["name"] = name
    entity["mentions"] = sum(int(profile["entity"].get("mentions") or 0) for profile in profiles if isinstance(profile.get("entity"), dict))
    mentions: list[dict[str, Any]] = []
    connections: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    for profile in profiles:
        mentions.extend(profile.get("mentions", []))
        connections.extend(profile.get("connections", []))
        episodes.extend(profile.get("episodes", []))
    return {"entity": entity, "mentions": mentions, "connections": connections, "episodes": episodes, "profiles": profiles}


def _required_query(query: dict[str, list[str]], key: str) -> str:
    value = _str_query(query, key)
    if not value:
        raise ValueError(f"Missing required query parameter: {key}")
    return value


def _str_query(query: dict[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = values[0].strip()
    return value or None


def _int_query(query: dict[str, list[str]], key: str, default: int) -> int:
    value = _str_query(query, key)
    return int(value) if value is not None else default


def _optional_int_query(query: dict[str, list[str]], key: str) -> int | None:
    value = _str_query(query, key)
    return int(value) if value is not None else None


def _float_query(query: dict[str, list[str]], key: str, default: float) -> float:
    value = _str_query(query, key)
    return float(value) if value is not None else default


def run_dashboard(host: str, port: int, data_dir: Path, qdrant_url: str | None = None) -> None:
    handler = create_handler(data_dir=data_dir, qdrant_url=qdrant_url)
    server = ThreadingHTTPServer((host, port), handler)
    print(f"Podcast RAG dashboard: http://{host}:{port}")
    print(f"Data dir: {data_dir}")
    if qdrant_url:
        print(f"Qdrant URL: {qdrant_url}")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Podcast RAG local dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--qdrant-url", default=None)
    args = parser.parse_args()
    run_dashboard(args.host, args.port, args.data_dir, args.qdrant_url)


BUILD_MISSING_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Podcast RAG Dashboard</title>
  <style>
    body { margin: 0; min-height: 100vh; display: grid; place-items: center; background: #f5f3ee; color: #24302f; font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(680px, calc(100vw - 32px)); border: 1px solid #ddd8cc; border-radius: 8px; background: #fffdfa; padding: 24px; box-shadow: 0 18px 40px rgba(48, 43, 34, 0.08); }
    code { background: #eeebe3; border-radius: 6px; padding: 2px 6px; }
    pre { overflow: auto; background: #eeebe3; border-radius: 8px; padding: 12px; }
  </style>
</head>
<body>
  <main>
    <h1>Podcast RAG Dashboard</h1>
    <p>The React dashboard has not been built yet.</p>
    <pre>cd dashboard
npm install
npm run build</pre>
    <p>Then run <code>uv run podcast-rag-dashboard --data-dir data --port 8765</code>.</p>
  </main>
</body>
</html>"""


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Podcast RAG Dashboard</title>
  <style>
    :root {
      --bg: #f7f5f0;
      --panel: #fffdf8;
      --panel-soft: #f0eee6;
      --ink: #25302f;
      --muted: #6e7672;
      --line: #ded9cd;
      --blue: #557f96;
      --green: #6d9278;
      --rose: #ad6d6d;
      --amber: #ba9055;
      --violet: #8d7aa8;
      --shadow: 0 16px 34px rgba(55, 48, 37, 0.09);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background:
        radial-gradient(circle at 20% 0%, rgba(109,146,120,.16), transparent 28%),
        radial-gradient(circle at 90% 12%, rgba(85,127,150,.12), transparent 26%),
        var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }
    button, input, select { font: inherit; }
    .layout {
      min-height: 100vh;
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr);
    }
    aside {
      background: #25302f;
      color: #f5f7f3;
      padding: 22px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
    }
    .brand { font-size: 18px; font-weight: 700; margin-bottom: 4px; }
    .subtle { color: #b8c5bf; font-size: 13px; line-height: 1.4; }
    nav { margin-top: 24px; display: grid; gap: 6px; }
    nav button {
      display: flex;
      align-items: center;
      gap: 10px;
      width: 100%;
      min-height: 38px;
      border: 0;
      border-radius: 8px;
      padding: 9px 10px;
      background: transparent;
      color: #dfe7e2;
      text-align: left;
      cursor: pointer;
    }
    nav button.active, nav button:hover { background: rgba(255,255,255,0.12); color: #fff; }
    main { padding: 22px; overflow: hidden; }
    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }
    h1 { font-size: 22px; margin: 0; }
    h2 { font-size: 16px; margin: 0 0 12px; }
    .status-line { color: var(--muted); font-size: 13px; }
    .grid { display: grid; gap: 14px; }
    .grid.cols-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
    .grid.cols-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .grid.cols-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 14px;
      min-width: 0;
    }
    .panel.flush { padding: 0; overflow: hidden; }
    .metric {
      min-height: 104px;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      border-left: 4px solid var(--green);
    }
    .metric strong { display: block; font-size: 26px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .metric .spark {
      height: 6px;
      border-radius: 999px;
      background: var(--panel-soft);
      overflow: hidden;
      margin-top: 12px;
    }
    .metric .spark i {
      display: block;
      height: 100%;
      width: var(--w, 35%);
      background: linear-gradient(90deg, var(--green), var(--blue));
    }
    .toolbar {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }
    input, select {
      min-height: 38px;
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      color: var(--ink);
      padding: 8px 10px;
      min-width: 220px;
    }
    .btn {
      min-height: 38px;
      border: 0;
      border-radius: 8px;
      background: var(--blue);
      color: #fff;
      padding: 8px 12px;
      cursor: pointer;
    }
    .btn.secondary { background: #dbe6df; color: var(--ink); }
    .table-wrap { overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { background: var(--panel-soft); color: #35413e; position: sticky; top: 0; z-index: 1; }
    tr:hover td { background: #fbfcfa; }
    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      padding: 2px 7px;
      border-radius: 8px;
      background: #e6eef2;
      color: #31566c;
      font-size: 12px;
      margin: 0 4px 4px 0;
    }
    .tag.person { background: #e7f0ea; color: #315e47; }
    .tag.place { background: #e5edf5; color: #315879; }
    .tag.event { background: #f4ece2; color: #7c5930; }
    .tag.concept { background: #eee8f3; color: #624f72; }
    .tag.date { background: #f1e7e7; color: #7a4747; }
    .view { display: none; }
    .view.active { display: grid; gap: 14px; }
    .brief {
      white-space: pre-wrap;
      line-height: 1.45;
      background: #fbfcfa;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 480px;
      overflow: auto;
    }
    .graph-list { display: grid; gap: 8px; max-height: 520px; overflow: auto; }
    .edge {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      padding: 9px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcfa;
      font-size: 13px;
    }
    .weight { color: var(--muted); font-variant-numeric: tabular-nums; }
    .muted { color: var(--muted); }
    .error { color: var(--rose); }
    .chart {
      width: 100%;
      min-height: 220px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffefa;
      overflow: hidden;
    }
    .chart svg { display: block; width: 100%; height: auto; }
    .bar-label { font-size: 12px; fill: var(--muted); }
    .bar-value { font-size: 12px; fill: var(--ink); font-weight: 700; }
    .chart-title {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
    }
    .chart-title small { color: var(--muted); }
    .cloud {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px 10px;
      min-height: 180px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffefa;
    }
    .cloud button {
      border: 0;
      background: transparent;
      color: var(--blue);
      cursor: pointer;
      padding: 2px 3px;
      line-height: 1.05;
    }
    .cloud button:hover { color: var(--rose); text-decoration: underline; }
    .filters {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .filters label {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 32px;
      padding: 5px 8px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffefa;
      font-size: 12px;
      color: var(--muted);
    }
    .filters input[type="checkbox"] { min-width: 0; min-height: 0; padding: 0; }
    .filters input[type="range"] { min-width: 160px; padding: 0; }
    .graph-board {
      position: relative;
      width: 100%;
      height: 560px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(rgba(37,48,47,.045) 1px, transparent 1px),
        linear-gradient(90deg, rgba(37,48,47,.045) 1px, transparent 1px),
        #fffefa;
      background-size: 28px 28px;
      overflow: hidden;
    }
    .graph-board svg { width: 100%; height: 100%; display: block; }
    .graph-node { cursor: pointer; transition: opacity .15s ease; }
    .graph-node text { font-size: 12px; fill: #283432; paint-order: stroke; stroke: #fffefa; stroke-width: 4px; stroke-linejoin: round; }
    .graph-edge { stroke: #9aa79f; stroke-opacity: .55; }
    .graph-edge.active { stroke: var(--amber); stroke-opacity: .95; }
    .graph-node.active circle { stroke: var(--rose); stroke-width: 3; }
    .graph-node.dim, .graph-edge.dim { opacity: .16; }
    .graph-side {
      display: grid;
      gap: 10px;
      align-content: start;
    }
    .connection-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px;
      background: #fffefa;
      cursor: pointer;
      font-size: 13px;
    }
    .connection-card:hover { border-color: var(--blue); }
    .connection-card.active { border-color: var(--rose); box-shadow: inset 3px 0 0 var(--rose); }
    .tooltip {
      position: absolute;
      pointer-events: none;
      display: none;
      max-width: 280px;
      padding: 8px 10px;
      border-radius: 8px;
      border: 1px solid var(--line);
      background: rgba(255,253,248,.96);
      box-shadow: var(--shadow);
      font-size: 12px;
      color: var(--ink);
    }
    .timeline-rail {
      display: grid;
      gap: 8px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fffefa;
      max-height: 520px;
      overflow: auto;
    }
    .timeline-item {
      display: grid;
      grid-template-columns: 72px 1fr;
      gap: 10px;
      align-items: start;
      border-left: 3px solid var(--green);
      padding: 8px 8px 8px 10px;
      background: #fbfaf5;
      border-radius: 0 8px 8px 0;
    }
    .time { font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px; }
    .density-row {
      display: grid;
      grid-template-columns: minmax(110px, 170px) minmax(0, 1fr) auto;
      gap: 10px;
      align-items: center;
      font-size: 13px;
      padding: 7px 0;
      border-bottom: 1px solid var(--line);
    }
    .density-row:last-child { border-bottom: 0; }
    .density-bar {
      height: 9px;
      border-radius: 999px;
      background: var(--panel-soft);
      overflow: hidden;
    }
    .density-bar i {
      display: block;
      width: var(--w, 0%);
      height: 100%;
      background: linear-gradient(90deg, var(--amber), var(--rose));
    }
    .mini {
      font-size: 12px;
      color: var(--muted);
      line-height: 1.35;
    }
    .split {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, .8fr);
      gap: 14px;
    }
    @media (max-width: 960px) {
      .layout { grid-template-columns: 1fr; }
      aside { position: relative; height: auto; }
      nav { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .grid.cols-4, .grid.cols-3, .grid.cols-2, .split { grid-template-columns: 1fr; }
      .topbar { align-items: flex-start; flex-direction: column; }
      .graph-board { height: 460px; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <aside>
      <div class="brand">Podcast RAG</div>
      <div class="subtle">Corpus explorer, entity graph and agentic retrieval dashboard.</div>
      <nav id="nav"></nav>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1 id="view-title">Overview</h1>
          <div class="status-line" id="status-line">Loading corpus state...</div>
        </div>
        <button class="btn secondary" id="refresh">Refresh</button>
      </div>

      <section class="view active" id="overview"></section>
      <section class="view" id="episodes"></section>
      <section class="view" id="entities"></section>
      <section class="view" id="graph"></section>
      <section class="view" id="timeline"></section>
      <section class="view" id="ask"></section>
      <section class="view" id="quality"></section>
    </main>
  </div>
  <script>
    const views = [
      ["overview", "Overview"],
      ["episodes", "Episodes"],
      ["entities", "Entities"],
      ["graph", "Graph"],
      ["timeline", "Timeline"],
      ["ask", "Ask"],
      ["quality", "Quality"]
    ];
    const state = { active: "overview", topics: [], episodes: [] };

    const esc = (value) => String(value ?? "").replace(/[&<>"']/g, c => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    }[c]));
    const api = async (path) => {
      const res = await fetch(path);
      const data = await res.json();
      if (!res.ok || data.error) throw new Error(data.error || res.statusText);
      return data;
    };
    const fmtTime = (seconds) => {
      if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "--:--";
      const total = Math.max(0, Math.floor(Number(seconds)));
      const h = Math.floor(total / 3600);
      const m = Math.floor((total % 3600) / 60);
      const s = total % 60;
      return h ? `${String(h).padStart(2,"0")}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}` : `${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}`;
    };
    const tag = (type, text) => `<span class="tag ${String(type || "").toLowerCase()}">${esc(text)}</span>`;
    const table = (headers, rows) => `
      <div class="table-wrap"><table><thead><tr>${headers.map(h => `<th>${esc(h)}</th>`).join("")}</tr></thead>
      <tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${cell}</td>`).join("")}</tr>`).join("") || `<tr><td colspan="${headers.length}" class="muted">No data</td></tr>`}</tbody></table></div>`;
    const panel = (title, body) => `<div class="panel"><h2>${esc(title)}</h2>${body}</div>`;
    const colorByType = {
      person: "#6d9278", place: "#557f96", event: "#ba9055", concept: "#8d7aa8",
      organization: "#ad6d6d", work: "#7e8f69", date: "#ad6d6d", unknown: "#9aa79f"
    };
    const typeColor = (type) => colorByType[String(type || "unknown").toLowerCase()] || colorByType.unknown;
    const slug = (value) => String(value ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "-");
    const pct = (value, max) => max ? Math.max(3, Math.round((Number(value || 0) / max) * 100)) : 0;
    const metricCard = (label, value, max, colorVar = "var(--green)") => `
      <div class="panel metric" style="border-left-color:${colorVar}">
        <span>${esc(label.replaceAll("_", " "))}</span>
        <strong>${esc(value)}</strong>
        <div class="spark"><i style="--w:${pct(value, max)}%"></i></div>
      </div>`;
    const horizontalBars = (rows, labelKey, valueKey, color = "var(--blue)") => {
      const max = Math.max(...rows.map(r => Number(r[valueKey] || 0)), 1);
      const bars = rows.slice(0, 10).map(r => {
        const label = r[labelKey] ?? "";
        const value = Number(r[valueKey] || 0);
        const width = pct(value, max);
        return `<div class="density-row"><strong title="${esc(label)}">${esc(String(label).slice(0, 34))}</strong><div class="density-bar"><i style="--w:${width}%;background:${color}"></i></div><span class="weight">${esc(value)}</span></div>`;
      }).join("") || '<div class="muted">No data</div>';
      return `<div>${bars}</div>`;
    };
    const typeDonut = (rows) => {
      const total = rows.reduce((sum, r) => sum + Number(r.count || 0), 0) || 1;
      let offset = 25;
      const rings = rows.map(r => {
        const value = (Number(r.count || 0) / total) * 100;
        const stroke = typeColor(r.entity_type);
        const circle = `<circle r="36" cx="50" cy="50" fill="transparent" stroke="${stroke}" stroke-width="15" stroke-dasharray="${value} ${100 - value}" stroke-dashoffset="${offset}" />`;
        offset -= value;
        return circle;
      }).join("");
      const legend = rows.map(r => `<div>${tag(r.entity_type, r.entity_type)} <span class="weight">${esc(r.count)}</span></div>`).join("");
      return `<div class="chart"><svg viewBox="0 0 100 100" role="img" aria-label="Entity type distribution">${rings}<circle r="24" cx="50" cy="50" fill="#fffefa" /><text x="50" y="47" text-anchor="middle" font-size="10" fill="#6e7672">types</text><text x="50" y="60" text-anchor="middle" font-size="15" font-weight="700" fill="#25302f">${rows.length}</text></svg></div><div class="mini" style="display:grid;gap:5px;margin-top:10px">${legend}</div>`;
    };
    const entityCloud = (topics, limit = 70) => {
      const max = Math.max(...topics.map(t => Number(t.mentions || 0)), 1);
      return `<div class="cloud">${topics.slice(0, limit).map(t => {
        const size = 12 + Math.round((Number(t.mentions || 0) / max) * 22);
        return `<button style="font-size:${size}px;color:${typeColor(t.entity_type)}" title="${esc(t.entity_type)} · ${esc(t.mentions)} mentions" onclick="loadEntityProfile('${esc(String(t.name)).replaceAll("'", "\\'")}')">${esc(t.name)}</button>`;
      }).join("") || '<span class="muted">No entities indexed yet</span>'}</div>`;
    };

    function initNav() {
      const nav = document.getElementById("nav");
      nav.innerHTML = views.map(([id, title]) => `<button data-view="${id}" class="${id === state.active ? "active" : ""}">${title}</button>`).join("");
      nav.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => setView(btn.dataset.view)));
      document.getElementById("refresh").addEventListener("click", loadAll);
    }

    function setView(id) {
      state.active = id;
      document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === id));
      document.querySelectorAll("nav button").forEach(el => el.classList.toggle("active", el.dataset.view === id));
      document.getElementById("view-title").textContent = views.find(v => v[0] === id)?.[1] || "Dashboard";
    }

    async function loadAll() {
      document.getElementById("status-line").textContent = "Loading...";
      try {
        const [status, stats, episodes, topics, graph, quality] = await Promise.all([
          api("/api/status"), api("/api/stats"), api("/api/episodes"), api("/api/topics?limit=150"),
          api("/api/graph?limit=200"), api("/api/quality")
        ]);
        state.topics = topics;
        state.episodes = episodes;
        renderOverview(status, stats);
        renderEpisodes(episodes);
        renderEntities(topics);
        renderGraph(graph);
        renderTimeline();
        renderAsk();
        renderQuality(quality);
        document.getElementById("status-line").textContent = `${stats.counts.episodes} episodes | ${stats.counts.entities} entities | ${stats.counts.entity_relations} connections`;
      } catch (err) {
        document.getElementById("status-line").innerHTML = `<span class="error">${esc(err.message)}</span>`;
      }
    }

    function renderOverview(status, stats) {
      const maxMetric = Math.max(...Object.values(stats.counts).map(Number), 1);
      const metricColors = ["var(--green)", "var(--blue)", "var(--amber)", "var(--violet)", "var(--rose)"];
      const metrics = Object.entries(stats.counts).map(([key, value], idx) => metricCard(key, value, maxMetric, metricColors[idx % metricColors.length])).join("");
      const recommendations = status.recommendations.map(item => `<li>${esc(item)}</li>`).join("");
      const types = table(["Type", "Count"], stats.entity_types.map(r => [tag(r.entity_type, r.entity_type), esc(r.count)]));
      const richest = table(["Episode", "Title", "Entities", "Mentions", "Chunks"], stats.richest_episodes.map(r => [esc(r.episode_id), esc(r.title), esc(r.unique_entities), esc(r.mentions), esc(r.chunks)]));
      document.getElementById("overview").innerHTML = `
        <div class="grid cols-4">${metrics}</div>
        <div class="grid cols-2">
          ${panel("Entity type mix", typeDonut(stats.entity_types))}
          ${panel("Richest episodes", `<div class="chart-title"><small>ranked by mentions</small><small>${stats.richest_episodes.length} episodes</small></div>${horizontalBars(stats.richest_episodes, "title", "mentions", "linear-gradient(90deg, var(--green), var(--blue))")}`)}
        </div>
        <div class="grid cols-2">
          ${panel("Recommended next actions", `<ul>${recommendations}</ul>`)}
          ${panel("Entity types", types)}
        </div>
        ${panel("Richest episodes", richest)}
        ${panel("Chunking", `<div class="muted">${esc(status.chunking.strategy)} | max ${esc(status.chunking.max_words)} words | overlap ${esc(status.chunking.overlap_words)}</div>`)}
      `;
    }

    function renderEpisodes(episodes) {
      const rows = episodes.map(e => [
        esc(e.id), esc(e.title), esc(e.segment_count), esc(e.author || ""), esc(e.language || ""),
        `<button class="btn secondary" onclick="loadEpisodeInsights(${Number(e.id)})">Inspect</button>`
      ]);
      document.getElementById("episodes").innerHTML = `${panel("Episodes", table(["ID", "Title", "Segments", "Author", "Lang", ""], rows))}<div id="episode-detail"></div>`;
    }

    async function loadEpisodeInsights(id) {
      const data = await api(`/api/episode-insights?episode_id=${id}`);
      const top = table(["Entity", "Type", "Mentions", "Chunks"], data.top_entities.map(r => [esc(r.name), tag(r.entity_type, r.entity_type), esc(r.mentions), esc(r.chunks)]));
      const density = table(["Chunk", "Time", "Entities", "Text"], data.entity_density.map(r => [esc(r.chunk_id), esc(fmtTime(r.start_seconds)), esc(r.unique_entities), esc(r.text).slice(0, 180)]));
      document.getElementById("episode-detail").innerHTML = panel(`Episode ${id}: ${data.episode.title}`, `<div class="grid cols-2">${top}${density}</div>`);
    }

    function renderEntities(topics) {
      const options = topics.map(t => `<option value="${esc(t.name)}">${esc(t.name)}</option>`).join("");
      const rows = topics.map(t => [
        `<button class="btn secondary" onclick="loadEntityProfile('${esc(String(t.name)).replaceAll("'", "\\'")}')">${esc(t.name)}</button>`,
        tag(t.entity_type, t.entity_type),
        Number(t.confidence || 0).toFixed(2),
        esc(t.mentions),
        esc(t.episodes)
      ]);
      const typeRows = Object.values(topics.reduce((acc, t) => {
        const key = t.entity_type || "unknown";
        acc[key] = acc[key] || { entity_type: key, count: 0 };
        acc[key].count += 1;
        return acc;
      }, {})).sort((a, b) => b.count - a.count);
      document.getElementById("entities").innerHTML = `
        <div class="toolbar"><select id="entity-select">${options}</select><button class="btn" onclick="loadSelectedEntity()">Open profile</button></div>
        <div class="grid cols-2">
          ${panel("Entity cloud", `<div class="chart-title"><small>size = mentions, color = inferred type</small><small>${topics.length} entities</small></div>${entityCloud(topics)}`)}
          ${panel("Detected types", typeDonut(typeRows))}
        </div>
        ${panel("Entity index", table(["Entity", "Type", "Confidence", "Mentions", "Episodes"], rows))}
        <div id="entity-profile"></div>
      `;
    }

    function loadSelectedEntity() {
      const value = document.getElementById("entity-select").value;
      if (value) loadEntityProfile(value);
    }

    async function loadEntityProfile(name) {
      const data = await api(`/api/entity-profile?name=${encodeURIComponent(name)}&limit=25`);
      const mentions = table(["Episode", "Time", "Chunk", "Text"], data.mentions.map(r => [esc(r.title), esc(fmtTime(r.start_seconds)), esc(r.chunk_id), esc(r.text).slice(0, 180)]));
      const connections = table(["Source", "Target", "Relation", "Weight"], data.connections.map(r => [esc(r.source), esc(r.target), esc(r.relation_type), Number(r.weight || 0).toFixed(2)]));
      document.getElementById("entity-profile").innerHTML = panel(`${data.entity.name} profile`, `
        <p>${tag(data.entity.entity_type, data.entity.entity_type)} <span class="muted">confidence ${Number(data.entity.confidence || 0).toFixed(2)}</span></p>
        <div class="grid cols-2">${mentions}${connections}</div>
      `);
    }

    function renderGraph(graph) {
      state.graph = graph;
      state.graphSelection = null;
      const types = [...new Set(graph.nodes.map(n => String(n.entity_type || "unknown").toLowerCase()))].sort();
      const filters = `
        <div class="filters">
          ${types.map(type => `<label>${tag(type, type)} <input type="checkbox" class="graph-type" value="${esc(type)}" checked onchange="drawGraph()"></label>`).join("")}
          <label>min weight <input id="graph-weight" type="range" min="0" max="100" value="0" oninput="drawGraph()"></label>
          <span class="weight" id="graph-weight-label">0.00</span>
        </div>`;
      document.getElementById("graph").innerHTML = `
        <div class="grid cols-3">
          ${metricCard("Nodes", graph.nodes.length, Math.max(graph.nodes.length, graph.edges.length, 1), "var(--green)")}
          ${metricCard("Edges", graph.edges.length, Math.max(graph.nodes.length, graph.edges.length, 1), "var(--blue)")}
          ${metricCard("Top edge", graph.edges[0] ? Number(graph.edges[0].weight).toFixed(2) : "0", graph.edges[0] ? Number(graph.edges[0].weight) : 1, "var(--amber)")}
        </div>
        ${panel("Graph filters", filters)}
        <div class="split">
          <div class="graph-board" id="graph-board"><div class="tooltip" id="graph-tip"></div></div>
          <div class="graph-side">${panel("Selected node", '<div id="graph-selection" class="muted">Click a node or connection to focus the graph.</div>')}${panel("Strongest connections", '<div class="graph-list" id="graph-list"></div>')}</div>
        </div>
      `;
      drawGraph();
    }

    function graphFilteredData() {
      const graph = state.graph || { nodes: [], edges: [] };
      const activeTypes = new Set([...document.querySelectorAll(".graph-type:checked")].map(el => el.value));
      const maxWeight = Math.max(...graph.edges.map(e => Number(e.weight || 0)), 1);
      const minWeight = (Number(document.getElementById("graph-weight")?.value || 0) / 100) * maxWeight;
      const nodes = graph.nodes.filter(n => activeTypes.has(String(n.entity_type || "unknown").toLowerCase()));
      const nodeIds = new Set(nodes.map(n => n.id));
      const edges = graph.edges.filter(e => nodeIds.has(e.source) && nodeIds.has(e.target) && Number(e.weight || 0) >= minWeight);
      const connected = new Set(edges.flatMap(e => [e.source, e.target]));
      const visibleNodes = nodes.filter(n => connected.has(n.id)).slice(0, 90);
      const visibleIds = new Set(visibleNodes.map(n => n.id));
      return { nodes: visibleNodes, edges: edges.filter(e => visibleIds.has(e.source) && visibleIds.has(e.target)).slice(0, 180), minWeight };
    }

    function drawGraph() {
      const board = document.getElementById("graph-board");
      if (!board) return;
      const { nodes, edges, minWeight } = graphFilteredData();
      const label = document.getElementById("graph-weight-label");
      if (label) label.textContent = Number(minWeight).toFixed(2);
      const idToNode = Object.fromEntries(nodes.map(n => [n.id, n]));
      const width = 900, height = 560, cx = width / 2, cy = height / 2;
      const maxMentions = Math.max(...nodes.map(n => Number(n.mentions || 0)), 1);
      nodes.forEach((n, idx) => {
        const ring = Math.floor(idx / 18);
        const inRing = idx % 18;
        const angle = (inRing / Math.min(18, nodes.length)) * Math.PI * 2 + (ring * 0.31);
        const radius = 82 + ring * 74;
        n.x = cx + Math.cos(angle) * Math.min(radius, 380);
        n.y = cy + Math.sin(angle) * Math.min(radius, 235);
        n.r = 7 + Math.round((Number(n.mentions || 0) / maxMentions) * 15);
      });
      const selected = state.graphSelection;
      const selectedEdges = new Set(edges.filter(e => selected && (String(e.source) === selected || String(e.target) === selected)).flatMap(e => [`${e.source}-${e.target}`, String(e.source), String(e.target)]));
      const edgeSvg = edges.map(e => {
        const s = idToNode[e.source], t = idToNode[e.target];
        if (!s || !t) return "";
        const active = selected && (e.source === selected || e.target === selected);
        const dim = selected && !active;
        return `<line class="graph-edge ${active ? "active" : ""} ${dim ? "dim" : ""}" x1="${s.x}" y1="${s.y}" x2="${t.x}" y2="${t.y}" stroke-width="${Math.max(1, Math.min(7, Number(e.weight || 1)))}" />`;
      }).join("");
      const nodeSvg = nodes.map(n => {
        const active = selected === String(n.id);
        const dim = selected && !selectedEdges.has(String(n.id));
        return `<g class="graph-node ${active ? "active" : ""} ${dim ? "dim" : ""}" data-node-id="${esc(n.id)}" data-node-name="${esc(n.name)}" data-node-type="${esc(n.entity_type)}" data-node-mentions="${esc(n.mentions || 0)}" transform="translate(${n.x},${n.y})"><circle r="${n.r}" fill="${typeColor(n.entity_type)}" opacity=".88"></circle><text x="${n.r + 5}" y="4">${esc(String(n.name).slice(0, 22))}</text></g>`;
      }).join("");
      board.innerHTML = `<div class="tooltip" id="graph-tip"></div><svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Interactive entity graph">${edgeSvg}${nodeSvg}</svg>`;
      drawGraphList(nodes, edges);
      bindGraphInteractions();
    }

    function drawGraphList(nodes, edges) {
      const idToNode = Object.fromEntries(nodes.map(n => [n.id, n]));
      const list = document.getElementById("graph-list");
      if (!list) return;
      list.innerHTML = edges.slice(0, 60).map(edge => {
        const source = idToNode[edge.source] || {};
        const target = idToNode[edge.target] || {};
        const active = state.graphSelection && (String(edge.source) === state.graphSelection || String(edge.target) === state.graphSelection);
        return `<div class="connection-card ${active ? "active" : ""}" data-node-id="${esc(edge.source)}"><strong>${esc(source.name || edge.source)}</strong> <span class="muted">-></span> <strong>${esc(target.name || edge.target)}</strong><div class="mini">${esc(edge.relation_type || "cooccurs")} · weight ${Number(edge.weight || 0).toFixed(2)}</div></div>`;
      }).join("") || '<div class="muted">No visible edges with the current filters.</div>';
    }

    function bindGraphInteractions() {
      document.querySelectorAll(".graph-node").forEach(node => {
        node.addEventListener("click", () => selectGraphNode(node.dataset.nodeId));
        node.addEventListener("mousemove", (event) => showGraphTip(event, node.dataset.nodeName, node.dataset.nodeType, node.dataset.nodeMentions));
        node.addEventListener("mouseleave", hideGraphTip);
      });
      document.querySelectorAll(".connection-card").forEach(card => {
        card.addEventListener("click", () => selectGraphNode(card.dataset.nodeId));
      });
    }

    function selectGraphNode(id) {
      id = String(id ?? "");
      state.graphSelection = state.graphSelection === id ? null : id;
      const node = (state.graph?.nodes || []).find(n => String(n.id) === state.graphSelection);
      const box = document.getElementById("graph-selection");
      if (box) {
        box.innerHTML = node ? `<strong>${esc(node.name)}</strong><div>${tag(node.entity_type, node.entity_type)}</div><div class="mini">${esc(node.mentions || 0)} mentions across ${esc(node.episodes || 0)} episodes</div>` : "Click a node or connection to focus the graph.";
      }
      drawGraph();
    }

    function showGraphTip(event, name, type, mentions) {
      const tip = document.getElementById("graph-tip");
      if (!tip) return;
      tip.style.display = "block";
      tip.style.left = `${event.offsetX + 12}px`;
      tip.style.top = `${event.offsetY + 12}px`;
      tip.innerHTML = `<strong>${esc(name)}</strong><div>${tag(type, type)}</div><div class="mini">${esc(mentions)} mentions</div>`;
    }

    function hideGraphTip() {
      const tip = document.getElementById("graph-tip");
      if (tip) tip.style.display = "none";
    }

    function renderTimeline() {
      const options = state.topics.map(t => `<option value="${esc(t.name)}">${esc(t.name)}</option>`).join("");
      document.getElementById("timeline").innerHTML = `
        <div class="toolbar"><select id="timeline-topic"><option value="">All topics</option>${options}</select><button class="btn" onclick="loadTimeline()">Load timeline</button></div>
        <div id="timeline-results">${panel("Timeline", '<div class="muted">Select a topic or load all timeline entries.</div>')}</div>
      `;
    }

    async function loadTimeline() {
      const topic = document.getElementById("timeline-topic").value;
      const data = await api(`/api/timeline?limit=200${topic ? `&topic=${encodeURIComponent(topic)}` : ""}`);
      const rows = data.map(r => [esc(r.episode_id), esc(r.title), esc(fmtTime(r.start_seconds)), esc(r.name), tag(r.entity_type, r.entity_type), esc(r.text).slice(0, 150)]);
      const visual = `<div class="timeline-rail">${data.slice(0, 80).map(r => `
        <div class="timeline-item" style="border-left-color:${typeColor(r.entity_type)}">
          <div class="time">${esc(fmtTime(r.start_seconds))}<br><span class="mini">ep ${esc(r.episode_id)}</span></div>
          <div><strong>${esc(r.name)}</strong> ${tag(r.entity_type, r.entity_type)}<div class="mini">${esc(r.title)}</div><div>${esc(String(r.text || "").slice(0, 190))}</div></div>
        </div>`).join("") || '<div class="muted">No timeline entries</div>'}</div>`;
      document.getElementById("timeline-results").innerHTML = `<div class="grid cols-2">${panel("Visual timeline", visual)}${panel("Timeline table", table(["Episode", "Title", "Time", "Topic", "Type", "Text"], rows))}</div>`;
    }

    function renderAsk() {
      document.getElementById("ask").innerHTML = `
        <div class="panel">
          <h2>Ask the corpus</h2>
          <div class="toolbar"><input id="ask-input" style="min-width:min(680px,100%)" placeholder="What connects Pizarro with Peru?"><button class="btn" onclick="runAsk()">Ask</button></div>
          <div id="ask-output" class="brief muted">Ask uses local agentic retrieval: topic inference, Qdrant retrieval, connections and evidence context.</div>
        </div>
      `;
    }

    async function runAsk() {
      const q = document.getElementById("ask-input").value.trim();
      if (!q) return;
      document.getElementById("ask-output").textContent = "Thinking with retrieval tools...";
      try {
        const data = await api(`/api/ask?q=${encodeURIComponent(q)}`);
        document.getElementById("ask-output").textContent = data.brief;
      } catch (err) {
        document.getElementById("ask-output").innerHTML = `<span class="error">${esc(err.message)}</span>`;
      }
    }

    function renderQuality(quality) {
      const sections = Object.entries(quality).map(([name, rows]) => {
        const body = rows.length ? table(Object.keys(rows[0]), rows.map(row => Object.values(row).map(v => esc(v).slice(0, 160)))) : '<div class="muted">No issues</div>';
        return panel(`${name} (${rows.length})`, body);
      }).join("");
      document.getElementById("quality").innerHTML = sections;
    }

    initNav();
    loadAll();
  </script>
</body>
</html>"""


if __name__ == "__main__":
    main()
