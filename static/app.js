/*
 * ARM Advantage Studio
 *
 * Deliberately dependency-free: FastAPI serves this file from the same mounted
 * subpath as the API. Every URL is resolved against document.baseURI, so the UI
 * works at `/arm/advantage_annotation/` without assuming it owns the domain root.
 * The exact HTTP shapes used here are documented beside this file in
 * API_CONTRACT.md.
 */

"use strict";

const state = {
  user: null,
  datasets: [],
  dataset: null,
  queue: null,
  sample: null,
  currentCamera: null,
  history: [],
  saving: false,
  processingLabels: false,
  labelBuffer: [],
  lastLabelAction: null,
  episodeBoundaryReached: false,
  completionRequired: false,
  samplePrefetch: new Map(),
  recentLabels: new Map(),
  prefetchGeneration: 0,
  frameReady: Promise.resolve(),
  visualReady: true,
  heldLabelKey: null,
  heldLabelTimer: null,
  importTimer: null,
  importPulse: 12,
  importSource: "hf",
  exportTimer: null,
  exportPulse: 8,
  curve: null,
  curveFrame: 0,
  curveImageLoading: false,
  curveImageDesired: null,
  curveImageActiveKey: null,
};

const HOLD_INITIAL_DELAY_MS = 360;
const HOLD_PULSE_MS = 80;

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

class ApiError extends Error {
  constructor(status, message, payload) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

function relativeUrl(path) {
  const safePath = String(path).replace(/^\/+/, "");
  return new URL(safePath, document.baseURI).toString();
}

async function api(path, options = {}) {
  const init = {
    method: options.method || "GET",
    credentials: "same-origin",
    headers: { Accept: "application/json", ...(options.headers || {}) },
  };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }

  let response;
  try {
    response = await fetch(relativeUrl(path), init);
  } catch (error) {
    throw new ApiError(0, "The annotation service is unreachable.", { cause: error });
  }

  const contentType = response.headers.get("content-type") || "";
  let payload = null;
  if (response.status !== 204) {
    payload = contentType.includes("application/json")
      ? await response.json().catch(() => null)
      : await response.text().catch(() => null);
  }

  if (!response.ok) {
    const detail =
      typeof payload?.detail === "string"
        ? payload.detail
        : payload?.detail?.message ||
          payload?.error?.message ||
          (typeof payload === "string" ? payload : `Request failed (${response.status}).`);
    if (response.status === 401 && !options.suppressAuthDialog) showLogin();
    throw new ApiError(response.status, detail, payload);
  }
  return payload;
}

function formatInteger(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "—";
  return new Intl.NumberFormat().format(Number(value));
}

function formatTime(seconds) {
  if (!Number.isFinite(Number(seconds))) return "—";
  const value = Math.max(0, Number(seconds));
  const minutes = Math.floor(value / 60);
  const remainder = value - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${remainder.toFixed(2).padStart(5, "0")}`;
}

function initials(name) {
  return (
    String(name || "A")
      .split(/[\s._-]+/)
      .filter(Boolean)
      .slice(0, 2)
      .map((piece) => piece[0])
      .join("") || "A"
  );
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label;
    button.disabled = true;
  } else {
    if (button.dataset.label) button.textContent = button.dataset.label;
    delete button.dataset.label;
    button.disabled = false;
  }
}

function setError(element, message) {
  element.textContent = message || "";
  element.classList.toggle("is-hidden", !message);
}

function toast(title, message, kind = "success", timeout = 3800) {
  const item = document.createElement("div");
  item.className = `toast ${kind}`;
  const copy = document.createElement("div");
  const heading = document.createElement("strong");
  const body = document.createElement("span");
  heading.textContent = title;
  body.textContent = message;
  copy.append(heading, body);
  item.append(copy);
  $("#toast-region").append(item);
  window.setTimeout(() => item.remove(), timeout);
}

function openDialog(id) {
  const dialog = document.getElementById(id);
  if (!dialog?.open) dialog?.showModal();
}

function closeDialog(id) {
  const dialog = document.getElementById(id);
  if (dialog?.open) dialog.close();
}

function showLogin() {
  $("#login-close").classList.toggle("is-hidden", !state.user);
  openDialog("login-dialog");
  window.setTimeout(() => $("#login-name")?.focus(), 50);
}

function renderIdentity() {
  const name = state.user?.name || "Sign in";
  $("#identity-name").textContent = name;
  $("#avatar").textContent = initials(name);
  $("#login-close").classList.toggle("is-hidden", !state.user);
}

async function bootstrap() {
  bindEvents();
  renderEmpty("Bring in a dataset to begin", "Import a Hugging Face LeRobot v3 dataset to start.");
  try {
    const me = await api("api/me", { suppressAuthDialog: true });
    if (!me?.authenticated) {
      showLogin();
      return;
    }
    state.user = me;
    renderIdentity();
    await loadDatasets();
  } catch (error) {
    renderFatal(error);
  }
}

function bindEvents() {
  $("#login-form").addEventListener("submit", onLogin);
  $("#login-close").addEventListener("click", () => closeDialog("login-dialog"));
  $("#identity-button").addEventListener("click", onIdentityClick);

  $("#open-import").addEventListener("click", () => openDialog("import-dialog"));
  $("#open-help").addEventListener("click", () => openDialog("help-dialog"));
  $("#open-export").addEventListener("click", showExportDialog);

  $$("[data-close-dialog]").forEach((button) => {
    button.addEventListener("click", () => closeDialog(button.dataset.closeDialog));
  });

  $("#import-form").addEventListener("submit", onImport);
  $("#import-source-hf").addEventListener("click", () => setImportSource("hf"));
  $("#import-source-zip").addEventListener("click", () => setImportSource("zip"));
  $("#export-form").addEventListener("submit", onExport);
  $("#dataset-select").addEventListener("change", (event) => selectDataset(Number(event.target.value)));
  $("#annotation-mode").addEventListener("change", onAnnotationModeChange);
  $("#curve-scrubber").addEventListener("input", (event) => selectCurveFrame(Number(event.target.value)));
  $("#progress-curve").addEventListener("click", onCurveTimelineClick);
  $$('[data-curve-change]').forEach((button) => button.addEventListener("click", () => addRelativeCurvePoint(button.dataset.curveChange)));
  $("#curve-set-value").addEventListener("click", () => setCurvePoint(Number($("#curve-value").value)));
  $("#curve-delete-point").addEventListener("click", deleteCurvePoint);
  $("#curve-save").addEventListener("click", saveCurve);
  $("#curve-prev").addEventListener("click", () => loadCurveEpisode(state.curve.episode_index - 1));
  $("#curve-next").addEventListener("click", () => loadCurveEpisode(state.curve.episode_index + 1));
  $("#camera-select").addEventListener("change", (event) => {
    state.currentCamera = event.target.value;
    state.prefetchGeneration += 1;
    renderFrames();
  });
  $("#save-settings").addEventListener("click", onSaveSettings);

  $$(".label-button").forEach((button) => {
    button.addEventListener("click", () => saveLabel(Number(button.dataset.label)));
  });
  $("#undo-label").addEventListener("click", undoLastLabel);
  $("#mark-complete").addEventListener("click", () => saveCompletion("marked"));
  $("#never-completes").addEventListener("click", () => saveCompletion("never"));
  $("#clear-completion").addEventListener("click", () => {
    toast("Completion is audited", "Replace the answer with another state; deletion is not exposed.", "warn");
  });
  $("#previous-sample").addEventListener("click", goPrevious);
  $("#next-sample").addEventListener("click", goNext);

  document.addEventListener("keydown", onKeydown);
  document.addEventListener("keyup", onKeyup);
  window.addEventListener("blur", () => stopHeldLabel());
  $$(".modal").forEach((dialog) => {
    dialog.addEventListener("click", (event) => {
      if (event.target === dialog && (dialog.id !== "login-dialog" || state.user)) dialog.close();
    });
  });
  $("#login-dialog").addEventListener("cancel", (event) => {
    if (!state.user) event.preventDefault();
  });
  window.addEventListener("resize", () => {
    alignFrameStrip();
    if (state.curve) drawProgressCurve();
  });
  window.addEventListener("beforeunload", (event) => {
    if (!state.saving && !state.processingLabels && !state.labelBuffer.length) return;
    event.preventDefault();
    event.returnValue = "";
  });
}

async function onLogin(event) {
  event.preventDefault();
  const button = $("#login-form button[type=submit]");
  const name = $("#login-name").value.trim();
  const password = $("#login-password").value;
  setError($("#login-error"), "");
  setBusy(button, true, "Signing in…");
  try {
    const user = await api("api/login", {
      method: "POST",
      body: { name, password },
      suppressAuthDialog: true,
    });
    state.user = { authenticated: true, name: user.name };
    renderIdentity();
    $("#login-password").value = "";
    closeDialog("login-dialog");
    toast("Signed in", `Labels will be attributed to ${user.name}.`);
    await loadDatasets();
  } catch (error) {
    setError($("#login-error"), error.message);
  } finally {
    setBusy(button, false);
  }
}

async function onIdentityClick() {
  if (!state.user) {
    showLogin();
    return;
  }
  const confirmed = window.confirm(`Sign out ${state.user.name}?`);
  if (!confirmed) return;
  try {
    await api("api/logout", { method: "POST" });
  } catch {
    // Clear the local session even if the service vanished after the user chose sign out.
  }
  clearTimeout(state.importTimer);
  clearTimeout(state.exportTimer);
  state.user = null;
  state.dataset = null;
  state.datasets = [];
  renderIdentity();
  renderDatasetSelector();
  renderDataset(null);
  showLogin();
}

async function loadDatasets(preferredId = null) {
  const payload = await api("api/datasets");
  state.datasets = Array.isArray(payload) ? payload : payload?.datasets || [];
  renderDatasetSelector();
  if (!state.datasets.length) {
    state.dataset = null;
    renderDataset(null);
    return;
  }
  const remembered = Number(localStorage.getItem("arm-annotation-dataset"));
  const candidate =
    preferredId ||
    (state.dataset && state.datasets.some((item) => item.id === state.dataset.id)
      ? state.dataset.id
      : null) ||
    (state.datasets.some((item) => item.id === remembered) ? remembered : null) ||
    state.datasets.find((item) => item.status === "ready")?.id ||
    state.datasets[0].id;
  await selectDataset(candidate, { refresh: false });
}

function renderDatasetSelector() {
  const select = $("#dataset-select");
  select.replaceChildren();
  if (!state.datasets.length) {
    select.append(new Option("No datasets yet", ""));
    select.disabled = true;
    return;
  }
  state.datasets.forEach((dataset) => {
    const suffix = dataset.status === "ready" ? "" : ` · ${dataset.status}`;
    select.append(new Option(`${dataset.title || dataset.repo_id}${suffix}`, String(dataset.id)));
  });
  select.disabled = false;
  if (state.dataset) select.value = String(state.dataset.id);
}

async function selectDataset(id, { refresh = true } = {}) {
  if (!id) return;
  stopHeldLabel();
  clearTimeout(state.importTimer);
  state.history = [];
  state.sample = null;
  state.queue = null;
  state.labelBuffer = [];
  state.lastLabelAction = null;
  state.episodeBoundaryReached = false;
  state.completionRequired = false;
  state.samplePrefetch.clear();
  state.recentLabels.clear();
  state.prefetchGeneration += 1;
  state.visualReady = true;
  state.curve = null;
  if (refresh) {
    state.dataset = await api(`api/datasets/${id}/status`);
  } else {
    state.dataset = state.datasets.find((item) => item.id === id) || (await api(`api/datasets/${id}/status`));
  }
  localStorage.setItem("arm-annotation-dataset", String(id));
  $("#dataset-select").value = String(id);
  renderDataset(state.dataset);
  if (state.dataset.status === "ready") {
    if (state.dataset.annotation_mode === "curve") await loadCurveEpisode(0);
    else await loadCurrentQueue();
  } else if (state.dataset.status === "importing") {
    pollDataset(id);
  }
}

function renderDataset(dataset) {
  const hasDataset = Boolean(dataset);
  const datasetName = hasDataset ? dataset.title || dataset.repo_id : "Nothing imported";
  $("#dataset-name").textContent = datasetName;
  $("#dataset-name").title = datasetName;
  $("#dataset-repo").textContent = hasDataset
    ? `${dataset.repo_id || dataset.source_url}@${dataset.revision || "main"}${dataset.subpath ? `/${dataset.subpath}` : ""}`
    : "Import a Hugging Face LeRobot v3 dataset.";

  const status = hasDataset ? dataset.status : "empty";
  const statusNode = $("#dataset-status");
  statusNode.className = `status-badge ${
    status === "ready" ? "ready" : status === "importing" ? "importing" : status === "failed" ? "failed" : "neutral"
  }`;
  statusNode.lastChild.textContent =
    status === "ready" ? " Ready" : status === "importing" ? " Importing" : status === "failed" ? " Failed" : " Empty";

  $("#dataset-episodes").textContent = hasDataset ? formatInteger(dataset.total_episodes) : "—";
  $("#dataset-frames").textContent = hasDataset ? formatInteger(dataset.total_frames) : "—";
  $("#dataset-fps").textContent = dataset?.fps ? `${Number(dataset.fps).toFixed(2)} fps` : "—";
  const version = dataset?.info?.codebase_version || dataset?.info?.version || "v3";
  $("#dataset-version").textContent = String(version);

  const importing = status === "importing";
  $("#import-progress").classList.toggle("is-hidden", !importing);
  if (importing) renderImportProgress();

  renderSettings(dataset);
  renderCoverage(dataset?.coverage);
  $("#annotation-mode").disabled = dataset?.status !== "ready";
  $("#annotation-mode").value = dataset?.annotation_mode || "direct";

  const exportButton = $("#open-export");
  exportButton.disabled = !dataset?.coverage?.export_ready;
  if (!hasDataset) {
    renderEmpty("Bring in a dataset to begin", "Import a Hugging Face LeRobot v3 dataset to start.");
  } else if (status === "importing") {
    renderEmpty("Import in progress", "Metadata, data shards, and the selected camera video are being prepared.");
  } else if (status === "failed") {
    renderEmpty("Import needs attention", dataset.error || "The import worker could not prepare this dataset.");
  }
}

function renderImportProgress() {
  state.importPulse = Math.min(88, Math.max(14, state.importPulse + 7));
  $("#import-phase").textContent = state.importPulse < 42 ? "Validating LeRobot v3" : "Downloading and indexing";
  $("#import-percent").textContent = `${state.importPulse}%`;
  $("#import-progress-bar").style.width = `${state.importPulse}%`;
  $("#import-progress-track").setAttribute("aria-valuenow", String(state.importPulse));
  $("#import-message").textContent =
    state.importPulse < 42
      ? "Reading metadata and selecting the requested camera…"
      : "Import runs in the background and survives this page being closed.";
}

async function pollDataset(id) {
  clearTimeout(state.importTimer);
  state.importTimer = window.setTimeout(async () => {
    try {
      const dataset = await api(`api/datasets/${id}/status`);
      if (!state.dataset || state.dataset.id !== id) return;
      state.dataset = dataset;
      renderDataset(dataset);
      if (dataset.status === "importing") {
        pollDataset(id);
      } else {
        await loadDatasets(id);
        if (dataset.status === "ready") toast("Dataset ready", `${dataset.title} is ready for annotation.`);
        else toast("Import failed", dataset.error || "Review the dataset reference and try again.", "error", 7000);
      }
    } catch (error) {
      toast("Status check failed", error.message, "error");
      pollDataset(id);
    }
  }, 1400);
}

function videoFeatures(dataset) {
  const features = dataset?.info?.features || {};
  return Object.entries(features)
    .filter(([key, value]) => {
      const dtype = String(value?.dtype || "").toLowerCase();
      return dtype === "video" || key.includes("images") || key.includes("image");
    })
    .map(([key]) => key);
}

function renderSettings(dataset) {
  const ready = dataset?.status === "ready";
  const selected = dataset?.camera_keys || [];
  // Only downloaded cameras can produce frame images. `info.features` may name
  // other video streams whose MP4s were intentionally excluded at import.
  const available = [...new Set(selected.length ? selected : videoFeatures(dataset))];
  const cameraSelect = $("#camera-select");
  cameraSelect.replaceChildren();
  if (!available.length) {
    cameraSelect.append(new Option("No imported camera", ""));
  } else {
    available.forEach((key) => cameraSelect.append(new Option(key, key)));
  }
  state.currentCamera =
    selected.includes(state.currentCamera) || available.includes(state.currentCamera)
      ? state.currentCamera
      : selected[0] || available[0] || null;
  cameraSelect.value = state.currentCamera || "";
  cameraSelect.disabled = !ready || !available.length;

  const chips = $("#camera-chips");
  chips.replaceChildren();
  if (!selected.length) {
    const empty = document.createElement("span");
    empty.className = "empty-chip";
    empty.textContent = "No camera selected";
    chips.append(empty);
  } else {
    selected.forEach((key) => {
      const chip = document.createElement("span");
      chip.className = "camera-chip";
      chip.textContent = key;
      chip.title = key;
      chips.append(chip);
    });
  }

  $("#delta-input").value = dataset?.delta_seconds ?? "1.0";
  $("#delta-input").disabled = !ready;
  $("#save-settings").disabled = !ready;
  const hasLabels = Number(dataset?.coverage?.labeled_samples || 0) > 0;
  $("#grid-lock").textContent = hasLabels ? "Reset required" : "Configurable";
  $("#grid-lock").classList.toggle("locked", hasLabels);
  $("#delta-resolution").textContent = dataset?.delta_frames
    ? `At ${Number(dataset.fps).toFixed(2)} fps: Δ = ${dataset.delta_frames} frames.`
    : "Δ resolves per episode from its frame rate.";
}

function renderCoverage(coverage = {}) {
  const curveMode = state.dataset?.annotation_mode === "curve";
  const labeled = Number(curveMode ? coverage?.curve_completed_episodes : coverage?.labeled_samples || 0);
  const total = Number(curveMode ? coverage?.total_episodes : coverage?.total_samples || 0);
  const percent = Number(curveMode ? coverage?.curve_percent : coverage?.percent ?? (total ? (labeled / total) * 100 : 0));
  const completed = Number(curveMode ? coverage?.curve_completed_episodes : coverage?.completed_episodes || 0);
  const episodes = Number(coverage?.total_episodes || 0);
  $("#coverage-labeled").textContent = formatInteger(labeled);
  $("#coverage-total").textContent = formatInteger(total);
  $("#coverage-percent").textContent = `${Math.round(percent)}%`;
  $("#coverage-ring").style.setProperty("--coverage", String(Math.max(0, Math.min(100, percent))));
  $("#coverage-ring-label").textContent = curveMode ? "curves" : "labeled";
  $("#coverage-unit").textContent = curveMode ? "episode curves saved" : "advantage transitions";
  $("#completion-answered").textContent = formatInteger(completed);
  $("#completion-pending").textContent = formatInteger(Math.max(0, episodes - completed));
  $("#completion-answered-label").textContent = curveMode ? "curves ready" : "completion answers";
  $("#completion-pending-label").textContent = curveMode ? "episodes remaining" : "pending";
  if (state.dataset) state.dataset.coverage = { ...state.dataset.coverage, ...coverage };
  $("#open-export").disabled = !coverage?.export_ready;
}

async function onAnnotationModeChange(event) {
  if (!state.dataset) return;
  $("#annotation-view").classList.add("is-hidden");
  $("#curve-view").classList.add("is-hidden");
  try {
    state.dataset = await api(`api/datasets/${state.dataset.id}`, {
      method: "PATCH",
      body: { annotation_mode: event.target.value },
    });
    renderDataset(state.dataset);
    if (state.dataset.annotation_mode === "curve") await loadCurveEpisode(0);
    else await loadCurrentQueue();
  } catch (error) {
    toast("Could not change mode", error.message, "error");
    renderDataset(state.dataset);
    if (state.dataset.annotation_mode === "curve") await loadCurveEpisode(0);
    else await loadCurrentQueue();
  }
}

async function loadCurveEpisode(episodeIndex) {
  const total = Number(state.dataset?.total_episodes || 0);
  if (episodeIndex < 0 || episodeIndex >= total) return;
  try {
    const curve = await api(`api/datasets/${state.dataset.id}/episodes/${episodeIndex}/progress-curve`);
    curve.persisted = curve.points.length >= 2;
    if (!curve.points.length) {
      curve.points = [
        { frame: 0, value: 0 },
        { frame: curve.episode_length - 1, value: 1 },
      ];
    }
    state.curve = curve;
    state.curveFrame = 0;
    $("#empty-workspace").classList.add("is-hidden");
    $("#annotation-view").classList.add("is-hidden");
    $("#curve-view").classList.remove("is-hidden");
    renderCurveEditor();
  } catch (error) {
    toast("Could not load curve", error.message, "error");
  }
}

function curveValueAt(frame) {
  const points = [...state.curve.points].sort((a, b) => a.frame - b.frame);
  const exact = points.find((point) => point.frame === frame);
  if (exact) return Number(exact.value);
  const rightIndex = points.findIndex((point) => point.frame > frame);
  if (rightIndex <= 0) return Number(points[Math.max(0, rightIndex)]?.value || 0);
  const left = points[rightIndex - 1];
  const right = points[rightIndex];
  const ratio = (frame - left.frame) / (right.frame - left.frame);
  return Number(left.value) + ratio * (Number(right.value) - Number(left.value));
}

function renderCurveEditor() {
  const curve = state.curve;
  if (!curve) return;
  const total = Number(state.dataset.total_episodes || 0);
  $("#curve-episode-position").textContent = `Episode ${curve.episode_index + 1} / ${total}`;
  $("#curve-task").textContent = curve.task || "Untitled task";
  $("#curve-meta").textContent = `${formatInteger(curve.episode_length)} frames · ${Number(curve.fps).toFixed(2)} fps · linear interpolation`;
  $("#curve-save-state").textContent = curve.persisted
    ? "Saved · linear interpolation active"
    : "Not saved · click Save curve to complete this episode";
  $("#curve-save").childNodes[0].textContent = curve.persisted ? "Save changes " : "Save curve ";
  $("#curve-prev").disabled = curve.episode_index <= 0;
  $("#curve-next").disabled = curve.episode_index >= total - 1;
  const scrubber = $("#curve-scrubber");
  scrubber.max = String(curve.episode_length - 1);
  scrubber.value = String(state.curveFrame);
  const value = curveValueAt(state.curveFrame);
  $("#curve-value").value = value.toFixed(2);
  $("#curve-frame-label").textContent = `Frame ${formatInteger(state.curveFrame)} · progress ${value.toFixed(3)}`;
  $("#curve-time-label").textContent = formatTime(state.curveFrame / curve.fps);
  $("#curve-delete-point").disabled = state.curveFrame === 0 || state.curveFrame === curve.episode_length - 1 || !curve.points.some((p) => p.frame === state.curveFrame);
  drawProgressCurve();
  scheduleCurveImage();
}

function drawProgressCurve() {
  const svg = $("#progress-curve");
  svg.replaceChildren();
  const length = Math.max(1, state.curve.episode_length - 1);
  const ns = "http://www.w3.org/2000/svg";
  const bounds = svg.getBoundingClientRect();
  const scaleX = Math.max(bounds.width / 1000, 0.001);
  const scaleY = Math.max(bounds.height / 220, 0.001);
  for (let step = 0; step <= 4; step += 1) {
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", "0"); line.setAttribute("x2", "1000");
    line.setAttribute("y1", String(200 - step * 45)); line.setAttribute("y2", String(200 - step * 45));
    line.setAttribute("class", "curve-grid"); svg.append(line);
  }
  const polyline = document.createElementNS(ns, "polyline");
  polyline.setAttribute("points", [...state.curve.points].sort((a,b) => a.frame-b.frame).map((p) => `${(p.frame/length)*1000},${200-Number(p.value)*180}`).join(" "));
  polyline.setAttribute("class", "curve-line"); svg.append(polyline);
  state.curve.points.forEach((point) => {
    const marker = document.createElementNS(ns, "ellipse");
    const radius = point.frame === state.curveFrame ? 6 : 4.5;
    marker.setAttribute("cx", String((point.frame / length) * 1000));
    marker.setAttribute("cy", String(200 - Number(point.value) * 180));
    marker.setAttribute("rx", String(radius / scaleX));
    marker.setAttribute("ry", String(radius / scaleY));
    marker.setAttribute("class", point.frame === state.curveFrame ? "curve-point selected" : "curve-point");
    svg.append(marker);
  });
  const cursor = document.createElementNS(ns, "line");
  cursor.setAttribute("x1", String((state.curveFrame / length) * 1000)); cursor.setAttribute("x2", String((state.curveFrame / length) * 1000));
  cursor.setAttribute("y1", "10"); cursor.setAttribute("y2", "210"); cursor.setAttribute("class", "curve-cursor"); svg.append(cursor);
}

function onCurveTimelineClick(event) {
  const rect = event.currentTarget.getBoundingClientRect();
  const frame = Math.round(((event.clientX - rect.left) / rect.width) * (state.curve.episode_length - 1));
  selectCurveFrame(frame);
}

function selectCurveFrame(frame) {
  state.curveFrame = Math.max(0, Math.min(state.curve.episode_length - 1, Math.round(frame)));
  renderCurveEditor();
}

function scheduleCurveImage() {
  const camera = state.currentCamera || state.curve.camera_keys[0];
  const frame = state.curveFrame;
  state.curveImageDesired = {
    key: `${state.dataset.id}:${state.curve.episode_index}:${camera}:${frame}`,
    frame,
    episodeIndex: state.curve.episode_index,
    camera,
  };
  pumpCurveImage();
}

function pumpCurveImage() {
  if (state.curveImageLoading || !state.curveImageDesired) return;
  const request = state.curveImageDesired;
  if (request.key === state.curveImageActiveKey) return;
  state.curveImageLoading = true;
  state.curveImageActiveKey = request.key;
  const visible = $("#curve-image");
  const skeleton = $("#curve-image-skeleton");
  if (!visible.classList.contains("loaded")) skeleton.classList.remove("is-hidden");
  const source = relativeUrl(
    `api/datasets/${state.dataset.id}/episodes/${request.episodeIndex}/frames/${request.frame}` +
      `?camera=${encodeURIComponent(request.camera)}`,
  );
  const loader = new Image();
  const settle = () => {
    state.curveImageLoading = false;
    if (state.curveImageDesired?.key !== request.key) pumpCurveImage();
  };
  loader.onload = () => {
    // Keep the previous frame visible while decoding, then swap atomically like a video player.
    const sameEpisode = state.curve?.episode_index === request.episodeIndex;
    const sameCamera = (state.currentCamera || state.curve?.camera_keys?.[0]) === request.camera;
    if (sameEpisode && sameCamera) {
      visible.src = source;
      visible.classList.add("loaded");
      skeleton.classList.add("is-hidden");
    }
    settle();
  };
  loader.onerror = settle;
  loader.src = source;
}

function addRelativeCurvePoint(direction) {
  const previous = [...state.curve.points]
    .filter((point) => point.frame < state.curveFrame)
    .sort((a, b) => b.frame - a.frame)[0];
  const baseline = previous ? Number(previous.value) : curveValueAt(state.curveFrame);
  setCurvePoint(direction === "up" ? baseline + 0.1 : direction === "down" ? baseline - 0.1 : baseline);
}

async function setCurvePoint(value) {
  value = Math.max(0, Math.min(1, Number(value)));
  if (!Number.isFinite(value)) return;
  const existing = state.curve.points.find((point) => point.frame === state.curveFrame);
  if (existing) existing.value = value;
  else state.curve.points.push({ frame: state.curveFrame, value });
  await saveCurve();
}

async function deleteCurvePoint() {
  state.curve.points = state.curve.points.filter((point) => point.frame !== state.curveFrame);
  await saveCurve();
}

async function saveCurve() {
  $("#curve-save-state").textContent = "Saving…";
  try {
    const result = await api(`api/datasets/${state.dataset.id}/episodes/${state.curve.episode_index}/progress-curve`, {
      method: "PUT", body: { points: state.curve.points.map(({ frame, value }) => ({ frame, value })) },
    });
    state.curve.points = result.points;
    state.curve.persisted = true;
    state.dataset.coverage = result.coverage;
    renderCoverage(result.coverage);
    $("#curve-save-state").textContent = "Saved · linear interpolation active";
    renderCurveEditor();
  } catch (error) {
    $("#curve-save-state").textContent = "Save failed";
    toast("Could not save curve", error.message, "error");
  }
}

function setImportSource(source) {
  state.importSource = source;
  const local = source === "zip";
  $("#import-hf-field").classList.toggle("is-hidden", local);
  $("#import-zip-field").classList.toggle("is-hidden", !local);
  $("#import-revision-field").classList.toggle("is-hidden", local);
  $("#import-subdirectory-field").classList.toggle("is-hidden", local);
  $("#import-source").required = !local;
  $("#import-file").required = local;
  $("#import-source-hf").classList.toggle("is-active", !local);
  $("#import-source-zip").classList.toggle("is-active", local);
}

async function uploadLocalDataset(file, cameraKeys, deltaSeconds) {
  const query = new URLSearchParams({ filename: file.name, delta_seconds: String(deltaSeconds) });
  cameraKeys.forEach((key) => query.append("camera_keys", key));
  const response = await fetch(relativeUrl(`api/datasets/upload?${query}`), {
    method: "POST",
    credentials: "same-origin",
    headers: { Accept: "application/json", "Content-Type": "application/zip" },
    body: file,
  });
  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    if (response.status === 401) showLogin();
    throw new ApiError(response.status, payload?.detail || `Upload failed (${response.status}).`, payload);
  }
  return payload;
}

async function onImport(event) {
  event.preventDefault();
  const button = $("#import-submit");
  const cameraKeys = $("#import-camera")
    .value.split(",")
    .map((item) => item.trim())
    .filter(Boolean);
  const deltaSeconds = Number($("#import-delta").value);
  const body = {
    source_url: $("#import-source").value.trim(),
    revision: $("#import-revision").value.trim() || null,
    subpath: $("#import-subdirectory").value.trim() || null,
    camera_keys: cameraKeys,
    delta_seconds: deltaSeconds,
  };
  setError($("#import-error"), "");
  setBusy(button, true, "Starting import…");
  try {
    const created = state.importSource === "zip"
      ? await uploadLocalDataset($("#import-file").files[0], cameraKeys, deltaSeconds)
      : await api("api/datasets", { method: "POST", body });
    closeDialog("import-dialog");
    $("#import-form").reset();
    $("#import-delta").value = "1.0";
    setImportSource("hf");
    state.importPulse = 12;
    toast("Import started", "The dataset is being validated and indexed in the background.");
    await loadDatasets(created.id);
  } catch (error) {
    setError($("#import-error"), error.message);
  } finally {
    setBusy(button, false);
  }
}

async function onSaveSettings() {
  if (!state.dataset) return;
  const delta = Number($("#delta-input").value);
  if (!Number.isFinite(delta) || delta <= 0) {
    toast("Invalid frame delta", "Enter a positive interval in seconds.", "error");
    return;
  }
  const button = $("#save-settings");
  setBusy(button, true, "Saving…");
  const body = { delta_seconds: delta };
  try {
    let updated;
    try {
      updated = await api(`api/datasets/${state.dataset.id}`, { method: "PATCH", body });
    } catch (error) {
      if (error.status !== 409) throw error;
      const reset = window.confirm(
        "Changing Δ changes every annotation sample. Reset all labels and completion answers for this dataset?",
      );
      if (!reset) return;
      updated = await api(`api/datasets/${state.dataset.id}`, {
        method: "PATCH",
        body: { ...body, reset_annotations: true },
      });
    }
    state.dataset = updated;
    state.history = [];
    state.labelBuffer = [];
    state.lastLabelAction = null;
    state.episodeBoundaryReached = false;
    state.samplePrefetch.clear();
    state.recentLabels.clear();
    state.prefetchGeneration += 1;
    renderDataset(updated);
    toast("Observation setup saved", `Δ now resolves to ${updated.delta_frames} frames.`);
    await loadCurrentQueue();
  } catch (error) {
    toast("Could not save setup", error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

async function loadCurrentQueue() {
  if (!state.dataset || state.dataset.status !== "ready") return;
  try {
    const item = await api(`api/datasets/${state.dataset.id}/queue/current`);
    await followQueueItem(item);
  } catch (error) {
    renderFatal(error);
  }
}

async function followQueueItem(item, { recordHistory = true } = {}) {
  state.queue = item;
  if (item?.kind === "sample") {
    state.completionRequired = false;
    state.episodeBoundaryReached = false;
    showSample(item, { recordHistory });
    return;
  }
  if (item?.kind === "completion") {
    state.completionRequired = true;
    state.episodeBoundaryReached = true;
    try {
      const sample = await loadSample(item.episode_index, item.suggested_frame);
      showSample(sample, { recordHistory });
      $("#completion-copy").textContent =
        "This episode's transition grid is labeled. Record the first completed frame or explicitly mark non-completion.";
      $("#completion-panel").scrollIntoView({ behavior: "smooth", block: "center" });
      toast(
        "Completion answer needed",
        `Review episode ${Number(item.episode_index) + 1} before continuing.`,
        "warn",
      );
    } catch (error) {
      // Very short episodes may have no valid five-frame sample. Keep completion writable.
      showCompletionOnly(item);
    }
    return;
  }
  if (item?.kind === "done") {
    state.sample = null;
    state.labelBuffer = [];
    state.episodeBoundaryReached = true;
    renderEmpty(
      "Annotation queue complete",
      "Every required advantage transition and episode completion answer is present. The dataset is ready to export.",
      "Export for FluxVLA",
      showExportDialog,
    );
    const fresh = await api(`api/datasets/${state.dataset.id}/status`);
    state.dataset = fresh;
    renderDataset(fresh);
    return;
  }
  if (item?.kind === "waiting") {
    renderEmpty(
      item.status === "failed" ? "Dataset import failed" : "Dataset is still preparing",
      item.error || "The import worker has not finished indexing the dataset.",
    );
  }
}

function coordinate(sample) {
  return sample
    ? { episode_index: sample.episode_index, target_frame: sample.target_frame }
    : null;
}

function sameCoordinate(a, b) {
  return a && b && a.episode_index === b.episode_index && a.target_frame === b.target_frame;
}

function sampleCacheKey(episodeIndex, targetFrame) {
  return `${Number(episodeIndex)}:${Number(targetFrame)}`;
}

function rememberLabel(episodeIndex, targetFrame, value) {
  state.recentLabels.set(sampleCacheKey(episodeIndex, targetFrame), value);
}

function reconcileSample(sample) {
  if (!sample) return sample;
  sample.transition_labels = (sample.transition_labels || [null, null, null, null]).map(
    (saved, index) => {
      const target = sample.frame_indices?.[index + 1];
      const key = sampleCacheKey(sample.episode_index, target);
      return state.recentLabels.has(key) ? state.recentLabels.get(key) : saved;
    },
  );
  const currentKey = sampleCacheKey(sample.episode_index, sample.target_frame);
  if (state.recentLabels.has(currentKey)) sample.label = state.recentLabels.get(currentKey);
  if (state.dataset?.coverage) sample.coverage = state.dataset.coverage;
  return sample;
}

async function loadSample(episodeIndex, targetFrame) {
  const key = sampleCacheKey(episodeIndex, targetFrame);
  const prefetched = state.samplePrefetch.get(key);
  state.samplePrefetch.delete(key);
  let sample = prefetched ? await prefetched : null;
  if (!sample) {
    sample = await api(
      `api/datasets/${state.dataset.id}/episodes/${episodeIndex}/samples/${targetFrame}`,
    );
  }
  return reconcileSample(sample);
}

function prefetchSample(sample) {
  const nextTarget = Number(sample.target_frame) + Number(sample.delta_frames);
  if (nextTarget > episodeEndTarget(sample)) return;
  const key = sampleCacheKey(sample.episode_index, nextTarget);
  if (state.samplePrefetch.has(key)) return;
  const pending = api(
    `api/datasets/${sample.dataset_id}/episodes/${sample.episode_index}/samples/${nextTarget}`,
  ).catch(() => null);
  state.samplePrefetch.set(key, pending);
  while (state.samplePrefetch.size > 6) {
    state.samplePrefetch.delete(state.samplePrefetch.keys().next().value);
  }
}

async function prefetchFutureFrames(sample, camera, generation) {
  if (!camera) return;
  const pending = [];
  for (let step = 1; step <= 4; step += 1) {
    if (generation !== state.prefetchGeneration) break;
    const frame = Number(sample.target_frame) + step * Number(sample.delta_frames);
    if (frame >= Number(sample.episode_length)) break;
    const url = relativeUrl(
      `api/datasets/${sample.dataset_id}/episodes/${sample.episode_index}/frames/${frame}` +
        `?camera=${encodeURIComponent(camera)}`,
    );
    pending.push(
      fetch(url, {
        credentials: "same-origin",
        cache: "force-cache",
      })
        .then((response) => (response.ok ? response.blob() : null))
        .catch(() => null),
    );
  }
  await Promise.all(pending);
}

async function waitForVisibleSample() {
  const expected = coordinate(state.sample);
  if (state.visualReady) return true;
  await Promise.race([
    state.frameReady,
    new Promise((resolve) => window.setTimeout(resolve, 20000)),
  ]);
  if (!sameCoordinate(expected, coordinate(state.sample)) || !state.visualReady) return false;
  await new Promise((resolve) => {
    window.requestAnimationFrame(() => window.requestAnimationFrame(resolve));
  });
  return true;
}

function showSample(sample, { recordHistory = true } = {}) {
  const before = coordinate(state.sample);
  const after = coordinate(sample);
  if (recordHistory && before && !sameCoordinate(before, after)) state.history.push(before);
  state.sample = reconcileSample(sample);
  state.queue = sample;
  renderSample();
}

function showCompletionOnly(item) {
  const delta = Number(state.dataset?.delta_frames || 1);
  state.sample = {
    kind: "completion",
    dataset_id: state.dataset.id,
    episode_index: item.episode_index,
    episode_length: item.suggested_frame + 1,
    task: "Episode completion review",
    target_frame: item.suggested_frame,
    delta_frames: delta,
    delta_seconds: Number(state.dataset.delta_seconds || 1),
    realized_delta_seconds: Number(state.dataset.delta_seconds || 1),
    fps: Number(state.dataset.fps || 1),
    frame_indices: [],
    frame_rows: [],
    camera_keys: state.dataset.camera_keys || [],
    label: null,
    completion: null,
    coverage: state.dataset.coverage,
  };
  renderSample();
  setDecisionEnabled(false);
  $("#completion-copy").textContent =
    "This short episode has no complete five-frame window. Record its completion outcome to continue.";
}

function renderSample() {
  const sample = state.sample;
  if (!sample) return;
  $("#empty-workspace").classList.add("is-hidden");
  $("#curve-view").classList.add("is-hidden");
  $("#annotation-view").classList.remove("is-hidden");

  const dataset = state.dataset;
  const episodeTotal = Number(dataset?.total_episodes || 0);
  $("#episode-position").textContent = `Episode ${Number(sample.episode_index) + 1} / ${episodeTotal || "—"}`;
  const sampleOrdinal = Math.max(
    1,
    Math.round((sample.target_frame - 4 * sample.delta_frames) / sample.delta_frames) + 1,
  );
  const sampleTotal =
    sample.episode_length > 4 * sample.delta_frames
      ? Math.ceil((sample.episode_length - 4 * sample.delta_frames) / sample.delta_frames)
      : 0;
  $("#sample-position").textContent = `Transition ${sampleOrdinal} / ${sampleTotal || "—"}`;
  $("#task-title").textContent = sample.task || "Untitled task";
  $("#episode-meta").textContent =
    `${formatInteger(sample.episode_length)} frames · ${Number(sample.fps).toFixed(2)} fps · ` +
    `${state.currentCamera || sample.camera_keys?.[0] || "camera unavailable"}`;
  $("#target-frame").textContent = `Frame ${formatInteger(sample.target_frame)}`;
  $("#target-time").textContent = `${formatTime(sample.target_frame / sample.fps)} into episode`;
  $("#window-delta").textContent =
    `Δ = ${Number(sample.delta_seconds).toFixed(2)} s · ${sample.delta_frames} frames`;
  $("#complete-frame-label").textContent = formatInteger(sample.target_frame);

  renderFrames();
  renderLabel();
  renderCompletion();
  renderNavigation();
  renderCoverage(sample.coverage);
  setDecisionEnabled(
    sampleNeedsAdvantage(sample) &&
      !state.completionRequired &&
      !state.episodeBoundaryReached,
  );
}

function renderFrames() {
  const sample = state.sample;
  if (!sample) return;
  const renderedCoordinate = coordinate(sample);
  const camera = state.currentCamera || sample.camera_keys?.[0];
  const row = sample.frame_rows?.find((entry) => entry.camera === camera) || sample.frame_rows?.[0];
  const images = row?.images || [];
  const labels = ["t − 4Δ", "t − 3Δ", "t − 2Δ", "t − Δ", "t"];
  const frameLoads = [];
  state.visualReady = false;

  $$(".frame-card").forEach((card, index) => {
    const frame = sample.frame_indices?.[index];
    const image = $("img", card);
    const skeleton = $(".frame-skeleton", card);
    $(".frame-order", card).textContent = labels[index];
    $("footer strong", card).textContent = Number.isFinite(frame) ? `Frame ${formatInteger(frame)}` : "Unavailable";
    $("footer span", card).textContent = Number.isFinite(frame) ? formatTime(frame / sample.fps) : "—";
    card.classList.remove("image-error");
    image.classList.remove("loaded");
    image.removeAttribute("src");
    image.alt = Number.isFinite(frame)
      ? `${camera || "Selected camera"} at frame ${frame}, ${labels[index]}`
      : "Frame unavailable";
    skeleton.classList.remove("is-hidden");
    frameLoads.push(
      new Promise((resolve) => {
        let settled = false;
        const settle = () => {
          if (settled) return;
          settled = true;
          resolve();
        };
        if (!images[index]) {
          card.classList.add("image-error");
          settle();
          return;
        }
        const source = relativeUrl(images[index]);
        image.onload = () => {
          if (image.src !== source) return;
          image.classList.add("loaded");
          skeleton.classList.add("is-hidden");
          settle();
        };
        image.onerror = () => {
          if (image.src !== source) return;
          card.classList.add("image-error");
          skeleton.classList.remove("is-hidden");
          settle();
        };
        image.src = source;
        window.queueMicrotask(() => {
          if (image.src === source && image.complete && image.naturalWidth > 0) image.onload();
        });
      }),
    );
  });

  state.frameReady = Promise.all(frameLoads).then(() => {
    if (!sameCoordinate(renderedCoordinate, coordinate(state.sample))) return false;
    state.visualReady = true;
    if (state.labelBuffer.length && !state.processingLabels) drainLabelBuffer();
    return true;
  });
  renderTransitionLabels();
  prefetchSample(sample);
  const generation = ++state.prefetchGeneration;
  prefetchFutureFrames(sample, camera, generation);
  // On narrow screens, keep the emphasized t − Δ → t pair in view by default.
  // The earlier causal frames remain immediately reachable by scrolling left.
  window.requestAnimationFrame(alignFrameStrip);
}

function renderTransitionLabels() {
  const labels = state.sample?.transition_labels || [];
  $$(".frame-connector").forEach((connector, index) => {
    const saved = labels[index];
    const value = saved ? Number(saved.label) : null;
    const hasLabel = saved && [-1, 0, 1].includes(value);
    connector.classList.remove(
      "has-label",
      "label-regressive",
      "label-stagnant",
      "label-progressive",
    );
    const badge = $(".transition-badge", connector);
    if (hasLabel) {
      connector.classList.add(
        "has-label",
        value === -1 ? "label-regressive" : value === 0 ? "label-stagnant" : "label-progressive",
      );
      badge.textContent = value === -1 ? "−1" : value === 0 ? "0" : "+1";
      connector.title =
        `${value === -1 ? "Regressive" : value === 0 ? "Stagnant" : "Progressive"} label` +
        ` saved by ${saved.annotator || "an annotator"}`;
    } else {
      badge.textContent = index === 3 ? "label Δ" : "";
      connector.title = index === 3 ? "Label this final transition" : "Transition not yet annotated";
    }
  });
}

function alignFrameStrip() {
  const shell = $(".frame-strip-shell");
  if (!shell || $("#annotation-view").classList.contains("is-hidden")) return;
  window.requestAnimationFrame(() => {
    shell.scrollLeft = shell.scrollWidth > shell.clientWidth ? shell.scrollWidth - shell.clientWidth : 0;
  });
}

function setDecisionEnabled(enabled) {
  $$(".label-button").forEach((button) => {
    button.disabled = !enabled || state.saving || state.processingLabels;
  });
}

function episodeEndTarget(sample) {
  if (!sample) return 0;
  if (
    sample.completion?.state === "marked" &&
    Number.isFinite(Number(sample.completion.frame))
  ) {
    return Number(sample.completion.frame);
  }
  if (sample.completion_only) return Number(sample.target_frame);
  const delta = Number(sample.delta_frames);
  const first = 4 * delta;
  const lastFrame = Number(sample.episode_length) - 1;
  if (lastFrame < first) return lastFrame;
  return first + Math.floor((lastFrame - first) / delta) * delta;
}

function isEpisodeEnd(sample) {
  return Boolean(sample) && Number(sample.target_frame) >= episodeEndTarget(sample);
}

function sampleNeedsAdvantage(sample) {
  if (!sample || sample.kind === "completion" || sample.completion_only) return false;
  return !(
    sample.completion?.state === "marked" &&
    Number(sample.target_frame) > Number(sample.completion.frame)
  );
}

function canAdvanceEpisode(sample) {
  if (!sample || !isEpisodeEnd(sample) || !sample.completion) return false;
  return !sampleNeedsAdvantage(sample) || Boolean(sample.label);
}

function hasLaterEpisode(sample) {
  return (
    Boolean(sample) &&
    Number(sample.episode_index) + 1 < Number(state.dataset?.total_episodes || 0)
  );
}

function renderLabel() {
  const label = state.sample?.label?.label;
  $$(".label-button").forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.label) === Number(label));
  });
  const hasLabel = [-1, 0, 1].includes(Number(label));
  $("#label-status-dot").className = `mini-status ${hasLabel ? "saved" : "unsaved"}`;
  $("#navigation-status").textContent = hasLabel
    ? `Saved as ${label === 1 ? "Progressive" : label === 0 ? "Stagnant" : "Regressive"}`
    : "Current transition is unlabeled";
  const saveState = $("#save-state");
  if (!state.saving) {
    saveState.className = hasLabel ? "saved" : "";
    saveState.textContent = hasLabel
      ? `Autosaved by ${state.sample.label.annotator || "an annotator"} · revision ${state.sample.label.revision || 1}`
      : "Autosave is on · choose a label to save and advance.";
  }
}

function renderCompletion() {
  const completion = state.sample?.completion;
  const marked = completion?.state === "marked";
  const never = completion?.state === "never";
  $("#mark-complete").classList.toggle("selected", marked);
  $("#never-completes").classList.toggle("selected", never);
  $("#clear-completion").classList.add("is-hidden");
  $("#completion-copy").textContent = marked
    ? `Completion is marked at frame ${completion.frame}. Progress is held at 1.0 from there.`
    : never
      ? "This episode is explicitly marked as never completing."
      : state.completionRequired
        ? "All required transitions are labeled. A completion answer is required to finish this episode."
        : "Mark the first completed frame when it appears, or explicitly record that it never completes.";
}

function renderNavigation() {
  const sample = state.sample;
  const earliest = sample ? 4 * sample.delta_frames : 0;
  const atEpisodeEnd = isEpisodeEnd(sample);
  const canContinueFromCompletedEnd = canAdvanceEpisode(sample);
  const canMoveToLaterEpisode =
    canContinueFromCompletedEnd && hasLaterEpisode(sample);
  $("#previous-sample").disabled =
    state.saving ||
    (!state.history.length && (!sample || sample.target_frame - sample.delta_frames < earliest));
  $("#undo-label").disabled =
    state.saving || state.processingLabels || !state.lastLabelAction;
  $("#next-sample").disabled =
    state.saving ||
    !sample ||
    (state.episodeBoundaryReached && !canMoveToLaterEpisode) ||
    (atEpisodeEnd && !canMoveToLaterEpisode);
  $("#next-sample-label").textContent =
    canMoveToLaterEpisode
      ? "Next episode"
      : atEpisodeEnd && canContinueFromCompletedEnd
        ? "Last episode"
        : atEpisodeEnd && sample?.completion && sampleNeedsAdvantage(sample) && !sample?.label
          ? "Label transition to continue"
          : state.episodeBoundaryReached || atEpisodeEnd
            ? "End of episode"
            : state.sample?.label
              ? "Next"
              : "Skip to next";
}

function saveLabel(label) {
  const sample = state.sample;
  if (
    !sampleNeedsAdvantage(sample) ||
    state.completionRequired ||
    state.episodeBoundaryReached
  ) {
    return;
  }
  state.labelBuffer.push(label);
  renderBufferedInput();
  drainLabelBuffer();
}

function renderBufferedInput() {
  if (!state.processingLabels && !state.labelBuffer.length) return;
  const queued = state.labelBuffer.length;
  $("#save-state").className = "saving";
  $("#save-state").textContent = !state.visualReady
    ? `Loading frames · ${queued} choice${queued === 1 ? "" : "s"} buffered`
    : queued
      ? `Autosaving continuously · ${queued} choice${queued === 1 ? "" : "s"} queued`
      : "Autosaving label and provenance…";
}

async function drainLabelBuffer() {
  if (state.processingLabels) return;
  state.processingLabels = true;
  try {
    while (state.labelBuffer.length) {
      if (
        !state.sample ||
        state.completionRequired ||
        state.episodeBoundaryReached ||
        !sampleNeedsAdvantage(state.sample)
      ) {
        const dropped = state.labelBuffer.splice(0).length;
        if (dropped) {
          toast(
            "Episode boundary reached",
            `${dropped} buffered choice${dropped === 1 ? " was" : "s were"} not applied to another episode.`,
            "warn",
          );
        }
        break;
      }
      if (!(await waitForVisibleSample())) {
        renderBufferedInput();
        break;
      }
      const label = state.labelBuffer.shift();
      renderBufferedInput();
      const canContinue = await persistLabel(label);
      if (!canContinue) {
        state.labelBuffer = [];
        break;
      }
    }
  } finally {
    state.processingLabels = false;
    state.saving = false;
    setDecisionEnabled(
      sampleNeedsAdvantage(state.sample) &&
        !state.completionRequired &&
        !state.episodeBoundaryReached,
    );
    renderLabel();
    if (state.labelBuffer.length) renderBufferedInput();
    renderNavigation();
  }
}

async function persistLabel(label) {
  const sample = state.sample;
  const origin = coordinate(sample);
  const previous = sample.label;
  const previousTransitions = [...(sample.transition_labels || [])];
  state.saving = true;
  setDecisionEnabled(false);
  renderBufferedInput();
  $$(".label-button").forEach((button) => {
    button.classList.toggle("selected", Number(button.dataset.label) === label);
  });
  try {
    const result = await api(
      `api/datasets/${sample.dataset_id}/episodes/${sample.episode_index}/samples/${sample.target_frame}/label`,
      { method: "PUT", body: { label } },
    );
    sample.label = {
      label,
      annotator: state.user?.name,
      revision: result.revision,
      updated_at: new Date().toISOString(),
    };
    sample.transition_labels = [...previousTransitions];
    while (sample.transition_labels.length < 4) sample.transition_labels.push(null);
    sample.transition_labels[3] = {
      ...sample.label,
      start_frame: sample.target_frame - sample.delta_frames,
      target_frame: sample.target_frame,
    };
    rememberLabel(sample.episode_index, sample.target_frame, sample.transition_labels[3]);
    state.lastLabelAction = origin;
    renderCoverage(result.coverage);
    $("#save-state").className = "saved";
    $("#save-state").textContent = `Autosaved · revision ${result.revision}`;
    renderTransitionLabels();
    renderLabel();
    return await followQueueHint(result.next, { origin, stopAtEpisodeBoundary: true });
  } catch (error) {
    sample.label = previous;
    sample.transition_labels = previousTransitions;
    $("#save-state").className = "error";
    $("#save-state").textContent = error.message;
    renderTransitionLabels();
    renderLabel();
    toast("Label was not saved", error.message, "error");
    return false;
  } finally {
    state.saving = false;
    setDecisionEnabled(
      sampleNeedsAdvantage(state.sample) &&
        !state.completionRequired &&
        !state.episodeBoundaryReached,
    );
    renderNavigation();
  }
}

async function followQueueHint(hint, { origin = null, stopAtEpisodeBoundary = false } = {}) {
  if (!hint || !state.dataset) return false;
  if (hint.kind === "sample") {
    const sameEpisode = origin && Number(hint.episode_index) === Number(origin.episode_index);
    const wouldWrap =
      sameEpisode && Number(hint.target_frame) < Number(origin.target_frame);
    if (stopAtEpisodeBoundary && wouldWrap) {
      reachEpisodeBoundary(
        "End of demonstration reached. Some earlier transitions are still unlabeled; use Back to review them.",
      );
      return false;
    }
    const sample = await loadSample(hint.episode_index, hint.target_frame);
    if (origin && !sameEpisode) stopHeldLabel();
    state.completionRequired = false;
    state.episodeBoundaryReached = false;
    showSample(sample);
    return !origin || Number(hint.episode_index) === Number(origin.episode_index);
  }
  await followQueueItem(hint);
  return false;
}

function reachEpisodeBoundary(message) {
  state.episodeBoundaryReached = true;
  state.completionRequired = !state.sample?.completion;
  state.labelBuffer = [];
  setDecisionEnabled(false);
  renderCompletion();
  if (message) $("#completion-copy").textContent = message;
  renderNavigation();
  $("#completion-panel").scrollIntoView({ behavior: "smooth", block: "center" });
  toast("End of demonstration", message || "This episode will not wrap to its beginning.", "warn");
}

async function saveCompletion(completionState) {
  const sample = state.sample;
  if (!sample || state.saving || state.processingLabels) return;
  if (
    sample.completion?.state === completionState &&
    (completionState === "never" || Number(sample.completion.frame) === Number(sample.target_frame))
  ) {
    toast("Already autosaved", "This completion answer is already stored.");
    return;
  }
  const origin = coordinate(sample);
  state.labelBuffer = [];
  const body =
    completionState === "marked"
      ? { state: "marked", frame: sample.target_frame }
      : { state: "never", frame: null };
  state.saving = true;
  $("#mark-complete").disabled = true;
  $("#never-completes").disabled = true;
  try {
    const result = await api(
      `api/datasets/${sample.dataset_id}/episodes/${sample.episode_index}/completion`,
      { method: "PUT", body },
    );
    sample.completion = {
      ...body,
      annotator: state.user?.name,
      updated_at: new Date().toISOString(),
    };
    renderCompletion();
    renderCoverage(result.coverage);
    toast(
      completionState === "marked" ? "Completion frame saved" : "Non-completion saved",
      completionState === "marked"
        ? `Episode ${Number(sample.episode_index) + 1} completes from frame ${sample.target_frame}.`
        : `Episode ${Number(sample.episode_index) + 1} is recorded as never completing.`,
    );
    await followQueueHint(result.next, { origin, stopAtEpisodeBoundary: true });
  } catch (error) {
    toast("Completion was not saved", error.message, "error");
  } finally {
    state.saving = false;
    $("#mark-complete").disabled = false;
    $("#never-completes").disabled = false;
    setDecisionEnabled(
      sampleNeedsAdvantage(state.sample) &&
        !state.completionRequired &&
        !state.episodeBoundaryReached,
    );
    renderNavigation();
  }
}

async function undoLastLabel() {
  const action = state.lastLabelAction;
  if (!action || state.saving || state.processingLabels || !state.dataset) return;
  state.saving = true;
  state.labelBuffer = [];
  $("#undo-label").disabled = true;
  try {
    const result = await api(
      `api/datasets/${state.dataset.id}/episodes/${action.episode_index}/samples/${action.target_frame}/label/undo`,
      { method: "POST" },
    );
    state.lastLabelAction = null;
    state.completionRequired = false;
    state.episodeBoundaryReached = false;
    rememberLabel(
      action.episode_index,
      action.target_frame,
      result.label === null
        ? null
        : {
            label: result.label,
            annotator: state.user?.name,
            revision: result.revision,
            target_frame: action.target_frame,
          },
    );
    renderCoverage(result.coverage);
    await fetchSample(action, { recordHistory: false });
    toast(
      result.label === null ? "Label rolled back" : "Previous label restored",
      result.label === null
        ? "The transition is unlabeled again; the removed value remains in the audit history."
        : `The transition was restored to ${result.label > 0 ? "+1" : result.label}.`,
    );
  } catch (error) {
    if (error.status === 409) state.lastLabelAction = null;
    toast("Could not undo label", error.message, "error");
  } finally {
    state.saving = false;
    setDecisionEnabled(
      sampleNeedsAdvantage(state.sample) &&
        !state.completionRequired &&
        !state.episodeBoundaryReached,
    );
    renderNavigation();
  }
}

async function fetchSample(coordinateValue, { recordHistory = true } = {}) {
  if (!state.dataset || !coordinateValue) return;
  const sample = await loadSample(
    coordinateValue.episode_index,
    coordinateValue.target_frame,
  );
  state.completionRequired = false;
  state.episodeBoundaryReached = false;
  showSample(sample, { recordHistory });
}

async function goPrevious() {
  if (!state.sample || state.saving || state.processingLabels) return;
  try {
    if (state.history.length) {
      const previous = state.history.pop();
      await fetchSample(previous, { recordHistory: false });
      return;
    }
    const target = state.sample.target_frame - state.sample.delta_frames;
    if (target >= 4 * state.sample.delta_frames) {
      await fetchSample({ episode_index: state.sample.episode_index, target_frame: target }, { recordHistory: false });
    }
  } catch (error) {
    toast("Previous sample unavailable", error.message, "error");
  }
}

async function goNext() {
  if (!state.sample || state.saving || state.processingLabels) return;
  try {
    const sample = state.sample;
    const target = state.sample.target_frame + state.sample.delta_frames;
    if (isEpisodeEnd(sample)) {
      if (canAdvanceEpisode(sample)) {
        if (hasLaterEpisode(sample)) {
          await advanceToNextEpisode();
        } else {
          toast("Last episode", "There is no later episode; the dataset will not loop.", "warn");
        }
        return;
      }
      if (sample.completion && sampleNeedsAdvantage(sample) && !sample.label) {
        toast(
          "Label required",
          "Label this final transition before moving to the next episode.",
          "warn",
        );
        return;
      }
      reachEpisodeBoundary(
        sample.completion
          ? "End of demonstration reached. Use Back to review earlier transitions."
          : "End of demonstration reached. Record whether the episode completes; it will not wrap to the beginning.",
      );
      return;
    }
    await fetchSample({ episode_index: state.sample.episode_index, target_frame: target });
  } catch (error) {
    toast("Next sample unavailable", error.message, "error");
  }
}

async function advanceToNextEpisode() {
  const sample = state.sample;
  if (!sample || !state.dataset) return;
  const item = await api(
    `api/datasets/${state.dataset.id}/episodes/${sample.episode_index}/next`,
  );
  if (item?.kind === "sample") {
    stopHeldLabel();
    state.completionRequired = false;
    state.episodeBoundaryReached = false;
    showSample(item);
    toast(
      "Next episode",
      `Episode ${Number(item.episode_index) + 1} is ready for annotation.`,
    );
    return;
  }
  reachEpisodeBoundary("This is the final episode; the dataset will not loop.");
}

function stopHeldLabel(key = null) {
  if (key && state.heldLabelKey !== key) return;
  if (state.heldLabelTimer !== null) window.clearTimeout(state.heldLabelTimer);
  state.heldLabelTimer = null;
  state.heldLabelKey = null;
}

function heldLabelCanQueue() {
  return (
    Boolean(state.sample) &&
    sampleNeedsAdvantage(state.sample) &&
    !state.completionRequired &&
    !state.episodeBoundaryReached &&
    (!state.saving || state.processingLabels) &&
    !$$(".modal[open]").length
  );
}

function scheduleHeldLabel(key, label, delay) {
  state.heldLabelTimer = window.setTimeout(() => {
    if (state.heldLabelKey !== key) return;
    if (heldLabelCanQueue() && state.labelBuffer.length < 1) saveLabel(label);
    scheduleHeldLabel(key, label, HOLD_PULSE_MS);
  }, delay);
}

function beginHeldLabel(key, label) {
  if (state.heldLabelKey === key) return;
  stopHeldLabel();
  state.heldLabelKey = key;
  saveLabel(label);
  scheduleHeldLabel(key, label, HOLD_INITIAL_DELAY_MS);
}

function onKeydown(event) {
  if (event.defaultPrevented) return;
  const target = event.target;
  if (
    target instanceof HTMLInputElement ||
    target instanceof HTMLSelectElement ||
    target instanceof HTMLTextAreaElement ||
    target?.isContentEditable
  ) {
    return;
  }
  if ($$(".modal[open]").length) return;
  const key = event.key.toLowerCase();
  if (state.dataset?.annotation_mode === "curve" && state.curve) {
    handleCurveKeydown(event, key);
    return;
  }
  if (key === "z") {
    event.preventDefault();
    if (!event.repeat) beginHeldLabel(key, -1);
  } else if (key === "x") {
    event.preventDefault();
    if (!event.repeat) beginHeldLabel(key, 0);
  } else if (key === "c") {
    event.preventDefault();
    if (!event.repeat) beginHeldLabel(key, 1);
  } else if (event.repeat) {
    return;
  } else if (key === "f" && event.shiftKey) {
    event.preventDefault();
    saveCompletion("never");
  } else if (key === "f") {
    event.preventDefault();
    saveCompletion("marked");
  } else if (key === "b") {
    event.preventDefault();
    goPrevious();
  } else if (key === "u") {
    event.preventDefault();
    undoLastLabel();
  } else if (key === "n") {
    event.preventDefault();
    goNext();
  } else if (key === "?") {
    event.preventDefault();
    openDialog("help-dialog");
  }
}

function handleCurveKeydown(event, key) {
  if (key === "?") {
    event.preventDefault();
    openDialog("help-dialog");
    return;
  }
  if (key === "arrowleft" || key === "arrowright") {
    event.preventDefault();
    const direction = key === "arrowleft" ? -1 : 1;
    const heldStep = event.repeat ? 4 : 1;
    const step = event.shiftKey ? heldStep * 10 : heldStep;
    selectCurveFrame(state.curveFrame + direction * step);
    return;
  }
  if (event.repeat) return;
  if (key === "z" || key === "x" || key === "c") {
    event.preventDefault();
    addRelativeCurvePoint(key === "z" ? "down" : key === "x" ? "same" : "up");
  } else if (key === "s") {
    event.preventDefault();
    saveCurve();
  } else if (key === "u") {
    event.preventDefault();
    if (!$("#curve-delete-point").disabled) deleteCurvePoint();
  } else if (key === "b") {
    event.preventDefault();
    loadCurveEpisode(state.curve.episode_index - 1);
  } else if (key === "n") {
    event.preventDefault();
    loadCurveEpisode(state.curve.episode_index + 1);
  }
}

function onKeyup(event) {
  const key = event.key.toLowerCase();
  if (key === "z" || key === "x" || key === "c") stopHeldLabel(key);
}

function renderEmpty(title, copy, actionLabel = null, action = null) {
  $("#annotation-view").classList.add("is-hidden");
  $("#curve-view").classList.add("is-hidden");
  const empty = $("#empty-workspace");
  empty.classList.remove("is-hidden");
  $("#workspace-title").textContent = title;
  const paragraph = $("#workspace-title").nextElementSibling;
  paragraph.textContent = copy;
  const button = $("#empty-import");
  if (actionLabel && action) {
    button.textContent = actionLabel;
    button.onclick = action;
  } else {
    button.textContent = "Import a dataset";
    button.onclick = () => openDialog("import-dialog");
  }
}

function renderFatal(error) {
  renderEmpty("The workspace could not load", error?.message || "An unexpected error occurred.");
  toast("Workspace error", error?.message || "Unexpected error.", "error", 7000);
}

function showExportDialog() {
  if (!state.dataset) return;
  const coverage = state.dataset.coverage || {};
  const ready = Boolean(coverage.export_ready);
  const card = $("#export-readiness");
  card.className = `readiness-card ${ready ? "ready" : "blocked"}`;
  $(".readiness-icon", card).textContent = ready ? "✓" : "!";
  $("#readiness-title").textContent = ready ? "Ready for a FluxVLA release" : "Annotation is incomplete";
  $("#readiness-copy").textContent = ready
    ? `${formatInteger(coverage.labeled_samples)} required transitions and ${formatInteger(
        coverage.completed_episodes,
      )} completion answers are present.`
    : `${formatInteger(coverage.labeled_samples)} of ${formatInteger(
        coverage.total_samples,
      )} transitions; ${formatInteger(coverage.completed_episodes)} of ${formatInteger(
        coverage.total_episodes,
      )} completion answers.`;
  $("#export-submit").disabled = !ready;
  $("#export-progress").classList.add("is-hidden");
  setError($("#export-error"), "");
  openDialog("export-dialog");
}

async function onExport(event) {
  event.preventDefault();
  if (!state.dataset) return;
  const button = $("#export-submit");
  const videoMode = new FormData($("#export-form")).get("video_mode");
  setError($("#export-error"), "");
  setBusy(button, true, "Queueing export…");
  try {
    const created = await api(`api/datasets/${state.dataset.id}/exports`, {
      method: "POST",
      body: { video_mode: videoMode },
    });
    $("#export-progress").classList.remove("is-hidden");
    state.exportPulse = 8;
    renderExportProgress("queued");
    pollExport(created.id);
  } catch (error) {
    setError($("#export-error"), error.message);
    setBusy(button, false);
  }
}

function renderExportProgress(status, message = null) {
  if (status === "ready") state.exportPulse = 100;
  else state.exportPulse = Math.min(92, state.exportPulse + (status === "running" ? 14 : 7));
  $("#export-phase").textContent =
    status === "ready" ? "Export ready" : status === "running" ? "Writing LeRobot v3 release" : "Waiting for worker";
  $("#export-percent").textContent = `${state.exportPulse}%`;
  $("#export-progress-bar").style.width = `${state.exportPulse}%`;
  $("#export-message").textContent =
    message ||
    (status === "ready"
      ? "Progress, raw pair labels, and the reconstruction manifest passed export verification."
      : "The immutable source snapshot is never modified.");
}

function pollExport(id) {
  clearTimeout(state.exportTimer);
  state.exportTimer = window.setTimeout(async () => {
    try {
      const job = await api(`api/exports/${id}`);
      renderExportProgress(job.status);
      if (job.status === "ready") {
        setBusy($("#export-submit"), false);
        $("#export-submit").disabled = true;
        const output = job.output_path || job.manifest?.output_root || "the configured export directory";
        $("#export-message").textContent = `Release written to ${output}`;
        toast("FluxVLA export ready", output, "success", 7000);
        return;
      }
      if (job.status === "failed") {
        setBusy($("#export-submit"), false);
        setError($("#export-error"), job.error || "The export worker failed.");
        renderExportProgress("failed", "No source dataset files were modified.");
        return;
      }
      pollExport(id);
    } catch (error) {
      setError($("#export-error"), error.message);
      setBusy($("#export-submit"), false);
    }
  }, 1200);
}

window.addEventListener("DOMContentLoaded", bootstrap);
