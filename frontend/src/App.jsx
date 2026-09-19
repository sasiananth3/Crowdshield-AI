import { useEffect, useRef, useState } from "react";
import {
  ShieldCheck,
  LayoutDashboard,
  Video,
  Activity,
  Bell,
  BarChart3,
  Database,
  Upload,
  Play,
  Square,
  Download,
  ChevronRight,
  CircleHelp,
  Cpu,
  Users,
  Layers,
  Clock3,
  Check,
  FileVideo,
  AlertTriangle,
  Menu,
  X,
} from "lucide-react";

const fmt = (n, digits = 0) =>
  n == null
    ? "—"
    : Number(n).toLocaleString(undefined, { maximumFractionDigits: digits });
const stamp = (seconds) =>
  `${Math.floor((seconds || 0) / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor((seconds || 0) % 60)
    .toString()
    .padStart(2, "0")}`;
const recordedAt = (value) =>
  value ? new Date(value).toLocaleString() : "Recorded time unavailable";
async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response
      .json()
      .catch(() => ({ detail: response.statusText }));
    throw new Error(
      typeof body.detail === "string"
        ? body.detail
        : JSON.stringify(body.detail),
    );
  }
  return response.json();
}
const post = (path, body) =>
  api(path, {
    method: "POST",
    ...(body
      ? {
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        }
      : {}),
  });

function Badge({ value }) {
  return (
    <span className={`badge ${(value || "").toLowerCase()}`}>
      {value || "Awaiting video"}
    </span>
  );
}

function Trend({ history }) {
  if (history.length < 2)
    return (
      <div className="chart-empty">
        <Activity size={27} />
        <span>Count history will appear during analysis.</span>
      </div>
    );
  const values = history.map((s) => s.detected_people);
  const max = Math.max(10, ...values);
  const first = history[0].timestamp;
  const span = Math.max(1, history.at(-1).timestamp - first);
  const points = history
    .map(
      (s) =>
        `${42 + ((s.timestamp - first) / span) * 520},${155 - (s.detected_people / max) * 125}`,
    )
    .join(" ");
  return (
    <svg
      viewBox="0 0 590 190"
      role="img"
      aria-label="Detected people over video time"
      className="trend-chart"
    >
      {[0, 0.5, 1].map((v) => (
        <g key={v}>
          <line
            x1="42"
            y1={155 - v * 125}
            x2="562"
            y2={155 - v * 125}
            stroke="#e7ecf2"
            strokeDasharray="3 4"
          />
          <text x="27" y={159 - v * 125} textAnchor="end">
            {Math.round(max * v)}
          </text>
        </g>
      ))}
      <polygon points={`42,155 ${points} 562,155`} fill="#eaf1ff" />
      <polyline
        points={points}
        fill="none"
        stroke="#3976d7"
        strokeWidth="2.5"
        strokeLinejoin="round"
      />
      <text x="42" y="180">
        {stamp(first)}
      </text>
      <text x="562" y="180" textAnchor="end">
        {stamp(history.at(-1).timestamp)}
      </text>
    </svg>
  );
}

export default function App() {
  const [tab, setTab] = useState("Overview"),
    [health, setHealth] = useState(null),
    [jobs, setJobs] = useState([]),
    [job, setJob] = useState(null);
  const [history, setHistory] = useState([]),
    [alerts, setAlerts] = useState([]),
    [allAlerts, setAllAlerts] = useState([]),
    [evaluation, setEvaluation] = useState({});
  const [capacity, setCapacity] = useState(""),
    [area, setArea] = useState(""),
    [zoneName, setZoneName] = useState("Observation zone");
  const [layout, setLayout] = useState("full"),
    [pose, setPose] = useState(false),
    [heatmap, setHeatmap] = useState(false),
    [forecastMode, setForecastMode] = useState("persistence");
  const [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [connected, setConnected] = useState(false),
    [mobile, setMobile] = useState(false),
    [alertQueue, setAlertQueue] = useState([]);
  const fileRef = useRef();
  const knownAlertIds = useRef(null);
  const jobId = job?.id;

  function syncGlobalAlerts(nextAlerts) {
    setAllAlerts(nextAlerts);
    if (knownAlertIds.current === null) {
      knownAlertIds.current = new Set(nextAlerts.map((alert) => alert.id));
      return;
    }
    const fresh = nextAlerts.filter(
      (alert) =>
        !alert.acknowledged && !knownAlertIds.current.has(alert.id),
    );
    nextAlerts.forEach((alert) => knownAlertIds.current.add(alert.id));
    if (!fresh.length) return;
    setAlertQueue((current) => {
      const queued = new Set(current.map((alert) => alert.id));
      // The API returns newest first. Queue oldest first so simultaneous alerts
      // are shown to the operator in the order in which they occurred.
      const additions = fresh
        .slice()
        .reverse()
        .filter((alert) => !queued.has(alert.id));
      return [...current, ...additions];
    });
  }

  async function acknowledgeAlert(id) {
    await post(`/api/alerts/${id}/acknowledge`);
    const requests = [api("/api/alerts")];
    if (jobId) requests.push(api(`/api/alerts?job_id=${jobId}`));
    const [globalAlerts, jobAlerts] = await Promise.all(requests);
    syncGlobalAlerts(globalAlerts);
    if (jobAlerts) setAlerts(jobAlerts);
    setAlertQueue((current) => current.filter((alert) => alert.id !== id));
  }

  function dismissAlertPopup() {
    setAlertQueue((current) => current.slice(1));
  }
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const [h, j, e] = await Promise.all([
          api("/api/health"),
          api("/api/jobs"),
          api("/api/evaluation"),
        ]);
        if (active) {
          setHealth(h);
          setJobs(j);
          setEvaluation(e);
          setConnected(true);
        }
      } catch (e) {
        if (active) {
          setConnected(false);
          setError(`Backend unavailable: ${e.message}`);
        }
      }
    };
    refresh();
    const timer = setInterval(refresh, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);
  useEffect(() => {
    if (!jobId) {
      setHistory([]);
      return;
    }
    let active = true;
    const update = async () => {
      try {
        const [j, h, a] = await Promise.all([
          api(`/api/jobs/${jobId}`),
          api(`/api/jobs/${jobId}/history`),
          api(`/api/alerts?job_id=${jobId}`),
        ]);
        if (active) {
          setJob(j);
          setHistory(h);
          setAlerts(a);
        }
      } catch (e) {
        if (active) setError(e.message);
      }
    };
    update();
    const timer = setInterval(update, 1500);
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    const socket = new WebSocket(
      `${protocol}//${location.host}/ws/jobs/${jobId}`,
    );
    socket.onmessage = (event) => {
      if (active) setJob(JSON.parse(event.data));
    };
    return () => {
      active = false;
      clearInterval(timer);
      socket.close();
    };
  }, [jobId]);
  useEffect(() => {
    let active = true;
    const refresh = () =>
      api("/api/alerts")
        .then((a) => {
          if (active) syncGlobalAlerts(a);
        })
        .catch((e) => {
          if (active) setError(e.message);
        });
    refresh();
    const timer = setInterval(refresh, 1500);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);
  async function perform(fn) {
    setBusy(true);
    setError("");
    try {
      await fn();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }
  function selectJob(j) {
    setJob(j);
    setHistory([]);
    setAlerts([]);
    setHeatmap(false);
    setTab("Overview");
  }
  async function upload(file) {
    if (!file) return;
    await perform(async () => {
      const form = new FormData();
      form.append("file", file);
      selectJob(await api("/api/videos", { method: "POST", body: form }));
    });
  }
  const latest = job?.latest,
    zones = latest?.zones || [],
    risk = zones.reduce(
      (a, z) => ((z.risk.score ?? -1) > (a.score ?? -1) ? z.risk : a),
      zones[0]?.risk || {},
    );
  const running = job?.status === "processing" || job?.status === "stopping";
  async function start() {
    await perform(async () => {
      if (
        capacity &&
        (!Number.isInteger(Number(capacity)) || Number(capacity) < 1)
      )
        throw new Error("Reference capacity must be a positive whole number.");
      const base = {
        capacity: capacity ? Number(capacity) : null,
        area_m2: area ? Number(area) : null,
      };
      const z =
        layout === "full"
          ? [
              {
                id: "main",
                name: zoneName,
                polygon: [
                  [0, 0],
                  [1, 0],
                  [1, 1],
                  [0, 1],
                ],
                ...base,
              },
            ]
          : [
              {
                id: "left",
                name: `${zoneName} · left`,
                polygon: [
                  [0, 0],
                  [0.5, 0],
                  [0.5, 1],
                  [0, 1],
                ],
                ...base,
              },
              {
                id: "right",
                name: `${zoneName} · right`,
                polygon: [
                  [0.5, 0],
                  [1, 0],
                  [1, 1],
                  [0.5, 1],
                ],
                ...base,
              },
            ];
      setJob(
        await post(`/api/jobs/${job.id}/start`, {
          zones: z,
          enable_pose: pose,
          sample_fps: 2,
          forecast_mode: forecastMode,
        }),
      );
    });
  }
  const nav = [
    ["Overview", LayoutDashboard],
    ["Analyses", Video],
    ["Alerts", Bell],
    ["Evaluation", BarChart3],
    ["Datasets", Database],
  ];
  return (
    <div className="app-shell">
      <aside className={mobile ? "sidebar mobile-open" : "sidebar"}>
        <div className="brand">
          <div className="brand-mark">
            <ShieldCheck size={25} />
          </div>
          <div>
            CrowdShield <span>AI</span>
            <small>MONITOR · ASSESS · RESPOND</small>
          </div>
        </div>
        <div className="workspace-label">OPERATIONS WORKSPACE</div>
        <nav>
          {nav.map(([label, Icon]) => (
            <button
              key={label}
              className={tab === label ? "nav-item active" : "nav-item"}
              onClick={() => {
                setTab(label);
                setMobile(false);
              }}
            >
              <Icon size={19} />
              {label}
              {label === "Alerts" &&
                allAlerts.filter((a) => !a.acknowledged).length > 0 && (
                  <span className="nav-count">
                    {allAlerts.filter((a) => !a.acknowledged).length}
                  </span>
                )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="local-label">
            <Cpu size={17} />
            <span>Local processing</span>
          </div>
          <p>
            Video stays on this computer.
            <br />
            No cloud inference service.
          </p>
          <div className="operator">
            <div className="avatar">CS</div>
            <div>
              Research workspace<small>Software prototype · v0.1</small>
            </div>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumb">
            <button
              className="icon-button mobile-toggle"
              aria-label="Toggle navigation"
              onClick={() => setMobile(!mobile)}
            >
              {mobile ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span>Command center</span>
            <ChevronRight size={15} />
            <strong>{tab}</strong>
          </div>
          <div className="connection">
            <span className={connected ? "status-dot online" : "status-dot"} />
            {connected ? "Backend connected" : "Backend offline"}
            <span className="divider" />
            <ShieldCheck size={17} />
            Local research
          </div>
        </header>
        <main>
          <div className="page-heading">
            <div>
              <div className="eyebrow">
                CROWD MONITORING & EARLY RISK RESEARCH
              </div>
              <h1>
                {tab === "Overview"
                  ? "Live monitoring"
                  : tab === "Analyses"
                    ? "Analysis history"
                    : tab === "Alerts"
                      ? "Alert center"
                      : tab === "Evaluation"
                        ? "Model evaluation"
                        : "Dataset registry"}
              </h1>
              <p>
                {tab === "Overview"
                  ? "Observe crowd counts, movement and experimental risk indicators."
                  : "Traceable data and results for your CrowdShield project."}
              </p>
            </div>
            <div className="heading-actions">
              <button
                className="button secondary"
                disabled={busy || !health?.demo_available}
                onClick={() =>
                  perform(async () => selectJob(await post("/api/demo")))
                }
              >
                <FileVideo size={17} />
                Load sample video
              </button>
              <button
                className="button primary"
                disabled={busy || !connected}
                onClick={() => fileRef.current.click()}
              >
                <Upload size={17} />
                {busy ? "Please wait…" : "Upload video"}
              </button>
              <input
                ref={fileRef}
                type="file"
                accept=".mp4,.avi,.mov,.mkv,.webm"
                hidden
                onChange={(e) => {
                  upload(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </div>
          </div>
          <div className="notice">
            <CircleHelp size={16} />
            <span>
              Research prototype. Scores are experimental indicators, not
              probabilities or certified safety limits. Human review is
              required.
            </span>
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <AlertTriangle size={17} />
              {error}
              <button aria-label="Dismiss error" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          {job?.error && tab === "Overview" && (
            <div className="error-banner" role="alert">
              Analysis stopped: {job.error}
            </div>
          )}
          {tab === "Overview" && (
            <>
              <div className="kpi-grid">
                {[
                  [
                    "Detected people",
                    fmt(latest?.detected_people),
                    "YOLOv8 · current frame",
                    Users,
                  ],
                  [
                    "Density-model count",
                    fmt(latest?.density_estimate, 1),
                    "Independent estimate · do not add to detections",
                    Layers,
                  ],
                  [
                    "Experimental risk",
                    risk.score == null ? "—" : `${fmt(risk.score, 1)}/100`,
                    risk.tier || "Set a zone capacity to assess risk",
                    ShieldCheck,
                  ],
                  [
                    "Inference latency",
                    latest ? `${fmt(latest.inference_ms)} ms` : "—",
                    "Measured per sampled frame",
                    Clock3,
                  ],
                ].map(([label, value, note, Icon]) => (
                  <section className="kpi" key={label}>
                    <div className="kpi-label">
                      {label}
                      <Icon size={18} />
                    </div>
                    <div className="kpi-value">{value}</div>
                    <small>{note}</small>
                  </section>
                ))}
              </div>
              <div className="monitor-grid">
                <section className="panel video-panel">
                  <div className="panel-heading">
                    <div>
                      <Video size={18} />
                      <h2>Video observation</h2>
                    </div>
                    <Badge value={job?.status} />
                  </div>
                  <div
                    className="video-stage"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={(e) => {
                      e.preventDefault();
                      if (!running) upload(e.dataTransfer.files?.[0]);
                    }}
                  >
                    {job ? (
                      <>
                        <img
                          src={`/api/jobs/${job.id}/frame?heatmap=${heatmap}&t=${latest?.timestamp ?? 0}`}
                          alt={
                            heatmap
                              ? "Experimental density-map overlay"
                              : "Video frame with person detections"
                          }
                        />
                        <div className="video-tag">
                          <span
                            className={
                              running ? "status-dot online" : "status-dot"
                            }
                          />
                          {job.name}
                        </div>
                        <div className="video-time">
                          {stamp(latest?.timestamp)} /{" "}
                          {stamp(job.duration_seconds)}
                        </div>
                      </>
                    ) : (
                      <div className="video-empty">
                        <div className="empty-icon">
                          <Video size={34} />
                        </div>
                        <h3>Your observation starts here</h3>
                        <p>
                          Drop a crowd video or load the academic sample.
                          <br />
                          No simulated counts are shown.
                        </p>
                        <button
                          className="button video-upload"
                          onClick={() => fileRef.current.click()}
                          disabled={!connected}
                        >
                          <Upload size={16} />
                          Select video file
                        </button>
                        <small>MP4, AVI, MOV, MKV, WebM · up to 250 MB</small>
                      </div>
                    )}
                  </div>
                  <div className="video-controls">
                    <div className="segmented">
                      <button
                        className={!heatmap ? "selected" : ""}
                        onClick={() => setHeatmap(false)}
                      >
                        Detections
                      </button>
                      <button
                        className={heatmap ? "selected" : ""}
                        disabled={latest?.density_estimate == null}
                        onClick={() => setHeatmap(true)}
                      >
                        Density map
                      </button>
                    </div>
                    <span>
                      {fmt(job?.processed_samples || 0)} samples processed
                    </span>
                  </div>
                  <div className="progress-track">
                    <div style={{ width: `${job?.progress || 0}%` }} />
                  </div>
                  <div className="video-footer">
                    <span>
                      <Cpu size={14} />
                      CPU inference · sampled at 2 fps
                    </span>
                    <span>
                      {running
                        ? "Processing recorded video"
                        : job?.status === "completed"
                          ? "Analysis complete"
                          : "Awaiting analysis"}
                    </span>
                  </div>
                </section>
                <section className="panel config-panel">
                  <div className="panel-heading">
                    <div>
                      <ShieldCheck size={18} />
                      <h2>Zone configuration</h2>
                    </div>
                  </div>
                  <div className="config-body">
                    <label>
                      Zone name
                      <input
                        value={zoneName}
                        maxLength={65}
                        disabled={running}
                        onChange={(e) => setZoneName(e.target.value)}
                      />
                    </label>
                    <label>
                      Observation layout
                      <select
                        value={layout}
                        disabled={running}
                        onChange={(e) => setLayout(e.target.value)}
                      >
                        <option value="full">Full frame · one zone</option>
                        <option value="split">
                          Split frame · left and right zones
                        </option>
                      </select>
                    </label>
                    <div className="input-pair">
                      <label>
                        Reference capacity
                        <input
                          type="number"
                          min="1"
                          step="1"
                          placeholder="Not set"
                          value={capacity}
                          disabled={running}
                          onChange={(e) => setCapacity(e.target.value)}
                        />
                      </label>
                      <label>
                        Area per zone (m²)
                        <input
                          type="number"
                          min="0.1"
                          step="any"
                          placeholder="Optional"
                          value={area}
                          disabled={running}
                          onChange={(e) => setArea(e.target.value)}
                        />
                      </label>
                    </div>
                    <p className="field-help">
                      Enter an independently chosen reference per zone. Leave
                      blank for count-only analysis. Physical density requires a
                      measured ground area.
                    </p>
                    <label>
                      Count forecasting
                      <select
                        value={forecastMode}
                        disabled={running}
                        onChange={(e) => setForecastMode(e.target.value)}
                      >
                        <option value="persistence">
                          Persistence · comparison baseline
                        </option>
                        <option value="linear">
                          Linear trend · experimental
                        </option>
                        <option
                          value="lstm"
                          disabled={!health?.models?.forecast}
                        >
                          Trained LSTM · experimental
                        </option>
                      </select>
                    </label>
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={pose}
                        disabled={running || !health?.models?.pose}
                        onChange={(e) => setPose(e.target.checked)}
                      />
                      Extract body landmarks <span>Optional</span>
                    </label>
                    <p className="field-help">
                      Landmarks are experimental; falling and pushing are not
                      classified.
                    </p>
                    {running ? (
                      <button
                        className="button stop full"
                        disabled={job.status === "stopping"}
                        onClick={() =>
                          perform(async () =>
                            setJob(await post(`/api/jobs/${job.id}/stop`)),
                          )
                        }
                      >
                        <Square size={15} />
                        {job.status === "stopping"
                          ? "Stopping…"
                          : "Stop analysis"}
                      </button>
                    ) : (
                      <button
                        className="button primary full"
                        disabled={
                          !job ||
                          job.status !== "ready" ||
                          busy ||
                          !health?.models?.detector
                        }
                        onClick={start}
                      >
                        <Play size={16} />
                        Start analysis
                      </button>
                    )}
                    {job && job.status !== "ready" && !running && (
                      <small className="rerun-note">
                        Upload or load the sample again for a new analysis.
                      </small>
                    )}
                  </div>
                </section>
              </div>
              <div className="lower-grid">
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <Activity size={18} />
                      <h2>Crowd count over time</h2>
                    </div>
                    <span className="chart-legend">
                      <i />
                      Detected people
                    </span>
                  </div>
                  <Trend history={history} />
                  <div className="panel-footnote">
                    Video time, not wall-clock time. A count forecast is not a
                    crowd-incident prediction.
                  </div>
                </section>
                <section className="panel">
                  <div className="panel-heading">
                    <div>
                      <Layers size={18} />
                      <h2>Zone assessment</h2>
                    </div>
                  </div>
                  <div className="zone-list">
                    {zones.length ? (
                      zones.map((z) => (
                        <div className="zone-row" key={z.id}>
                          <div className="zone-title">
                            <strong>{z.name}</strong>
                            <Badge value={z.risk.tier} />
                          </div>
                          <div className="zone-metrics">
                            <span>
                              <b>{z.count}</b> detected
                            </span>
                            <span>
                              Forecast: <b>{fmt(z.forecast.count, 1)}</b> /{" "}
                              {z.forecast.horizon_seconds}s
                            </span>
                          </div>
                          <div className="risk-track">
                            <div style={{ width: `${z.risk.score || 0}%` }} />
                          </div>
                          <p>{z.risk.reasons[0]}</p>
                          <small>
                            {z.forecast.method.replaceAll("_", " ")} ·{" "}
                            {z.forecast.status}
                          </small>
                          {z.people_per_m2 != null && (
                            <small>
                              Observed count / supplied area: {z.people_per_m2}{" "}
                              people/m²
                            </small>
                          )}
                        </div>
                      ))
                    ) : (
                      <div className="empty-text">
                        Choose a video and start analysis to see zone counts and
                        risk reasons.
                      </div>
                    )}
                  </div>
                </section>
              </div>
              <section className="panel alerts-panel">
                <div className="panel-heading">
                  <div>
                    <Bell size={18} />
                    <h2>Recent warnings</h2>
                  </div>
                  <button
                    className="text-button"
                    onClick={() => setTab("Alerts")}
                  >
                    View alert center
                    <ChevronRight size={15} />
                  </button>
                </div>
                <AlertList
                  alerts={alerts.slice(0, 3)}
                  onAck={(id) =>
                    perform(() => acknowledgeAlert(id))
                  }
                />
              </section>
              <div className="model-strip">
                {Object.entries(health?.models || {}).map(
                  ([key, available]) => (
                    <span key={key}>
                      <span
                        className={
                          available ? "status-dot online" : "status-dot"
                        }
                      />
                      {key} weights {available ? "present" : "missing"}
                    </span>
                  ),
                )}
                <span>
                  Pose output: {latest?.pose?.status || "not running"}
                </span>
              </div>
            </>
          )}
          {tab === "Analyses" && (
            <section className="panel">
              <div className="panel-heading">
                <h2>Recorded analysis sessions</h2>
                <span>{jobs.length} sessions</span>
              </div>
              {jobs.length ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Video</th>
                        <th>Created</th>
                        <th>Duration</th>
                        <th>Status</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {jobs.map((j) => (
                        <tr key={j.id}>
                          <td>
                            <strong>{j.name}</strong>
                          </td>
                          <td>{new Date(j.created_at).toLocaleString()}</td>
                          <td>{stamp(j.duration_seconds)}</td>
                          <td>
                            <Badge value={j.status} />
                          </td>
                          <td>
                            <button
                              className="text-button"
                              onClick={() => selectJob(j)}
                            >
                              Open
                              <ChevronRight size={15} />
                            </button>
                            {j.processed_samples > 0 && (
                              <a
                                className="export-link"
                                href={`/api/jobs/${j.id}/export`}
                              >
                                <Download size={14} />
                                CSV
                              </a>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <div className="empty-text">
                  No analyses yet. Upload a video to create one.
                </div>
              )}
            </section>
          )}
          {tab === "Alerts" && (
            <section className="panel">
              <div className="panel-heading">
                <h2>Warnings requiring human review</h2>
                <span>Persistence-filtered · experimental</span>
              </div>
              <AlertList
                alerts={allAlerts}
                onAck={(id) =>
                  perform(() => acknowledgeAlert(id))
                }
              />
            </section>
          )}
          {tab === "Evaluation" && (
            <div className="evaluation-grid">
              <section className="panel">
                <div className="panel-heading">
                  <h2>Density-model evaluation</h2>
                  <Badge
                    value={evaluation.density ? "Measured" : "Not trained"}
                  />
                </div>
                <div className="evaluation-body">
                  {evaluation.density ? (
                    <>
                      <h3>ShanghaiTech Part B</h3>
                      <p>
                        {evaluation.density.train_images} train ·{" "}
                        {evaluation.density.validation_images} validation ·{" "}
                        {evaluation.density.test_images} test images
                      </p>
                      <div className="metric-pair">
                        <div>
                          <span>Test MAE</span>
                          <strong>
                            {fmt(evaluation.density.held_out_test.mae, 2)}
                          </strong>
                          <small>people / image</small>
                        </div>
                        <div>
                          <span>Test RMSE</span>
                          <strong>
                            {fmt(evaluation.density.held_out_test.rmse, 2)}
                          </strong>
                          <small>people / image</small>
                        </div>
                      </div>
                      <p>
                        MobileNetV2 frozen features + trained dilated head.{" "}
                        {evaluation.density.epochs} epochs.
                      </p>
                      <div className="notice compact">
                        The comparison is a constant-count sanity baseline, not
                        a reproduction of the base papers. No superiority claim
                        is established.
                      </div>
                    </>
                  ) : (
                    <p>
                      Run the density training script to produce measured
                      results.
                    </p>
                  )}
                </div>
              </section>
              <section className="panel">
                <div className="panel-heading">
                  <h2>Count forecasting</h2>
                  <Badge
                    value={
                      evaluation.forecast ? "Experimental" : "Baseline only"
                    }
                  />
                </div>
                <div className="evaluation-body">
                  {evaluation.forecast ? (
                    <>
                      <h3>
                        {evaluation.forecast.metadata.horizon}-second count LSTM
                      </h3>
                      <p>
                        Chronological holdout · detector-derived pseudo-labels
                      </p>
                      <div className="metric-pair">
                        <div>
                          <span>LSTM MAE</span>
                          <strong>
                            {fmt(evaluation.forecast.lstm_test.mae, 2)}
                          </strong>
                          <small>predicted people</small>
                        </div>
                        <div>
                          <span>Persistence MAE</span>
                          <strong>
                            {fmt(
                              evaluation.forecast.persistence_baseline_test.mae,
                              2,
                            )}
                          </strong>
                          <small>comparison baseline</small>
                        </div>
                      </div>
                      <p>{evaluation.forecast.metadata.source}</p>
                      <div className="notice compact">
                        This is not an incident-risk evaluation. False-alarm
                        rate and early-warning lead time have not been
                        validated.
                      </div>
                    </>
                  ) : (
                    <p>
                      The app uses a labelled linear-trend baseline until a
                      trained count checkpoint is installed.
                    </p>
                  )}
                </div>
              </section>
            </div>
          )}
          {tab === "Datasets" && (
            <section className="panel">
              <div className="panel-heading">
                <h2>Sources and permitted uses</h2>
                <Database size={18} />
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Source</th>
                      <th>Use in this project</th>
                      <th>Important limitation</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <a
                          href="https://github.com/desenzhou/ShanghaiTechDataset"
                          target="_blank"
                          rel="noreferrer"
                        >
                          ShanghaiTech Part B ↗
                        </a>
                      </td>
                      <td>Density-model training and held-out testing</td>
                      <td>Still images; no temporal or incident labels</td>
                    </tr>
                    <tr>
                      <td>
                        <a
                          href="https://mha.cs.umn.edu/proj_events.shtml"
                          target="_blank"
                          rel="noreferrer"
                        >
                          UMN university demonstration ↗
                        </a>
                      </td>
                      <td>
                        Functional video test; detector-derived count sequences
                      </td>
                      <td>Not a verified labelled crowd-risk benchmark</td>
                    </tr>
                    <tr>
                      <td>
                        <a
                          href="https://cocodataset.org/"
                          target="_blank"
                          rel="noreferrer"
                        >
                          COCO ↗
                        </a>
                      </td>
                      <td>Source of pretrained YOLOv8 detector weights</td>
                      <td>No detector fine-tuning performed</td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <div className="dataset-note">
                <h3>Data handling</h3>
                <p>
                  Uploads, frames and SQLite records are stored locally in the
                  runtime folder. Raw datasets are excluded from the public
                  source repository. Respect original dataset terms and only
                  upload footage you are authorized to process.
                </p>
                <p>
                  UCF-QNRF, CrowdHuman and PETS2009 are optional future
                  benchmarks; this version does not claim training on them.
                </p>
              </div>
            </section>
          )}
          <footer>
            <span>
              CrowdShield AI · Computer vision and predictive analytics
            </span>
            <span>Research use only · Human-in-the-loop</span>
          </footer>
        </main>
      </div>
      {alertQueue.length > 0 && (
        <AlertPopup
          alert={alertQueue[0]}
          pending={alertQueue.length}
          busy={busy}
          onDismiss={dismissAlertPopup}
          onAcknowledge={() =>
            perform(() => acknowledgeAlert(alertQueue[0].id))
          }
          onViewHistory={() => {
            dismissAlertPopup();
            setTab("Alerts");
            setMobile(false);
          }}
        />
      )}
    </div>
  );
}

function AlertPopup({
  alert,
  pending,
  busy,
  onDismiss,
  onAcknowledge,
  onViewHistory,
}) {
  return (
    <aside
      className={`alert-popup ${alert.tier.toLowerCase()}`}
      role="alertdialog"
      aria-live="assertive"
      aria-label={`${alert.tier} warning for ${alert.zone}`}
    >
      <div className="alert-popup-heading">
        <div className="alert-popup-symbol">
          <AlertTriangle size={21} />
        </div>
        <div>
          <span>Abnormal condition warning</span>
          <strong>{alert.zone}</strong>
        </div>
        <Badge value={alert.tier} />
        <button
          className="alert-popup-close"
          aria-label="Dismiss warning popup"
          onClick={onDismiss}
        >
          <X size={17} />
        </button>
      </div>
      <p>{alert.reasons?.join(" · ")}</p>
      <small>{alert.action}</small>
      <div className="alert-popup-meta">
        <span>Video {stamp(alert.video_timestamp)}</span>
        <span>{recordedAt(alert.created_at)}</span>
        {pending > 1 && <span>{pending} warnings pending</span>}
      </div>
      <div className="alert-popup-actions">
        <button className="button secondary small" onClick={onViewHistory}>
          View alert history
        </button>
        <button
          className="button small"
          disabled={busy}
          onClick={onAcknowledge}
        >
          <Check size={14} />
          Acknowledge
        </button>
      </div>
      <div className="alert-popup-note">
        Experimental warning - operator review required
      </div>
    </aside>
  );
}

function AlertList({ alerts, onAck }) {
  return alerts.length ? (
    <div className="alert-list">
      {alerts.map((a) => (
        <div className="alert-row" key={a.id}>
          <div className={`alert-icon ${a.tier.toLowerCase()}`}>
            <AlertTriangle size={18} />
          </div>
          <div className="alert-copy">
            <strong>
              {a.zone}
              <Badge value={a.tier} />
            </strong>
            <p>{a.reasons?.join(" · ")}</p>
            <small>{a.action}</small>
          </div>
          <span className="alert-time">
            <span>Video {stamp(a.video_timestamp)}</span>
            <span>{recordedAt(a.created_at)}</span>
          </span>
          <button
            className="button secondary small"
            disabled={a.acknowledged}
            onClick={() => onAck(a.id)}
          >
            {a.acknowledged ? (
              <>
                <Check size={14} />
                Acknowledged
              </>
            ) : (
              "Acknowledge"
            )}
          </button>
        </div>
      ))}
    </div>
  ) : (
    <div className="no-alerts">
      <ShieldCheck size={22} />
      <div>
        <strong>No warnings recorded</strong>
        <p>This does not certify that the crowd is safe.</p>
      </div>
    </div>
  );
}
