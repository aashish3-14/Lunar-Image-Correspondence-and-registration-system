const state = { images: [], byPath: new Map(), result: null };
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

function formatValue(value, suffix = '') {
  if (value === null || value === undefined || value === '') return 'N/A';
  if (typeof value === 'number') return `${Number.isInteger(value) ? value.toLocaleString() : value.toFixed(3)}${suffix}`;
  return `${value}${suffix}`;
}

function formatPercent(value) {
  return value === null || value === undefined ? 'N/A' : `${(Number(value) * 100).toFixed(1)}%`;
}

function setPreview(kind, item) {
  const image = $(`#${kind}-preview`);
  const empty = $(`#${kind}-empty`);
  const meta = $(`#${kind}-meta`);
  if (!item) {
    image.classList.remove('ready');
    empty.style.display = 'grid';
    meta.textContent = '--';
    return;
  }
  image.src = `${item.preview_url}?v=${Date.now()}`;
  image.onload = () => { image.classList.add('ready'); empty.style.display = 'none'; };
  image.onerror = () => { image.classList.remove('ready'); empty.style.display = 'grid'; };
  meta.textContent = `${item.dataset.toUpperCase()}  /  ${item.width ? `${item.width} × ${item.height}` : 'DIMENSIONS N/A'}  /  ${item.type.toUpperCase()}`;
}

function populateSelect(select, selectedPath) {
  select.innerHTML = '';
  state.images.forEach((item) => {
    const option = document.createElement('option');
    option.value = item.path;
    option.textContent = `${item.name}  ·  ${item.dataset}`;
    option.selected = item.path === selectedPath;
    select.appendChild(option);
  });
}

function selectedItem(kind) {
  return state.byPath.get($(`#${kind}-select`).value);
}

function renderDatasetTable() {
  const table = $('#dataset-table');
  table.innerHTML = '<div class="dataset-row header"><span>FILE / DATASET</span><span>TYPE</span><span>WIDTH</span><span>HEIGHT</span></div>';
  state.images.forEach((item) => {
    const row = document.createElement('div');
    row.className = 'dataset-row';
    row.innerHTML = `<strong>${item.name}<br><small>${item.path}</small></strong><span>${item.type.toUpperCase()}</span><span>${item.width ?? 'N/A'}</span><span>${item.height ?? 'N/A'}</span>`;
    table.appendChild(row);
  });
}

function updateSelection(kind) {
  setPreview(kind, selectedItem(kind));
}

function resetOutput() {
  state.result = null;
  $('#result-badge').textContent = 'AWAITING RUN';
  $('#result-badge').className = 'result-badge';
  $('#quality-mark').textContent = '--';
  $('#quality-state').textContent = 'RUN THE ENGINE TO SEE THE DECISION';
  $('#quality-state').className = 'quality-state';
  $('#quality-reason').textContent = 'The dashboard reports the existing LUCAS quality gate without altering its decision.';
  ['registered-image', 'matches-image', 'inliers-image'].forEach((id) => $(`#${id}`).classList.remove('ready'));
  $$('[data-metric]').forEach((element) => { element.textContent = '--'; });
}

function renderResult(payload) {
  state.result = payload;
  const metrics = payload.metrics || {};
  const fields = {
    matches: formatValue(metrics.matches),
    inliers: formatValue(metrics.inliers),
    inlier_ratio: formatPercent(metrics.inlier_ratio),
    rmse: formatValue(metrics.rmse),
    spatial_coverage: formatPercent(metrics.spatial_coverage),
    geometry_model: payload.geometry_model || 'N/A',
    median_error: formatValue(metrics.median_error),
    max_error: formatValue(metrics.max_error),
    spatial_uniformity: formatPercent(metrics.spatial_uniformity),
  };
  Object.entries(fields).forEach(([key, value]) => $$(`[data-metric="${key}"]`).forEach((element) => { element.textContent = value; }));

  const accepted = payload.quality?.success === true;
  const badge = $('#result-badge');
  badge.textContent = accepted ? 'REGISTRATION ACCEPTED' : 'REGISTRATION REJECTED';
  badge.className = `result-badge ${accepted ? 'accepted' : 'rejected'}`;
  $('#quality-mark').textContent = accepted ? 'PASS' : 'REJECT';
  $('#quality-state').textContent = accepted ? 'REGISTRATION ACCEPTED' : 'REGISTRATION REJECTED';
  $('#quality-state').className = `quality-state ${accepted ? 'accepted' : 'rejected'}`;
  $('#quality-reason').textContent = payload.quality?.reason || 'No quality reason returned.';
  const outputMap = { registered_image: 'registered-image', match_visualization: 'matches-image', inlier_visualization: 'inliers-image' };
  Object.entries(outputMap).forEach(([key, id]) => {
    const image = $(`#${id}`);
    const url = payload.outputs?.[key];
    image.classList.remove('ready');
    if (url) { image.src = `${url}?v=${Date.now()}`; image.onload = () => image.classList.add('ready'); }
  });
  $('#run-id').textContent = payload.run_id || '--';
}

async function loadDataset() {
  const response = await fetch('/api/dataset');
  if (!response.ok) throw new Error('Dataset inventory unavailable.');
  const dataset = await response.json();
  state.images = dataset.images || [];
  state.byPath = new Map(state.images.map((item) => [item.path, item]));
  $('#dataset-count').textContent = state.images.length;
  populateSelect($('#source-select'), dataset.default_source || state.images[0]?.path);
  populateSelect($('#reference-select'), dataset.default_reference || state.images[1]?.path || state.images[0]?.path);
  updateSelection('source');
  updateSelection('reference');
  renderDatasetTable();
  if (!state.images.length) showError('No readable dataset images were discovered under data/.');
}

function showError(message) { $('#error-banner').textContent = message; $('#error-banner').classList.add('visible'); }
function clearError() { $('#error-banner').classList.remove('visible'); }

async function registerImages() {
  clearError();
  const source = selectedItem('source');
  const reference = selectedItem('reference');
  if (!source || !reference) { showError('Select both a source and a reference image before registering.'); return; }
  const button = $('#register-button');
  button.disabled = true;
  $('#pipeline-status').textContent = 'LUCAS ENGINE PROCESSING';
  button.innerHTML = '<span class="button-symbol">◌</span> PROCESSING <span class="button-arrow">...</span>';
  resetOutput();
  try {
    const form = new FormData();
    form.append('source_path', source.path);
    form.append('reference_path', reference.path);
    const response = await fetch('/api/register', { method: 'POST', body: form });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'Registration failed.');
    renderResult(payload);
  } catch (error) {
    showError(error.message || 'Backend unavailable.');
  } finally {
    button.disabled = false;
    button.innerHTML = '<span class="button-symbol">◎</span> REGISTER IMAGES <span class="button-arrow">↗</span>';
    $('#pipeline-status').textContent = 'READY FOR ANALYSIS';
  }
}

function setupNavigation() {
  $$('.nav-item').forEach((button) => button.addEventListener('click', () => {
    $$('.nav-item').forEach((item) => item.classList.remove('active'));
    $$('.view').forEach((view) => view.classList.remove('active-view'));
    button.classList.add('active');
    $(`#${button.dataset.view}-view`).classList.add('active-view');
  }));
}

$('#source-select').addEventListener('change', () => updateSelection('source'));
$('#reference-select').addEventListener('change', () => updateSelection('reference'));
$('#register-button').addEventListener('click', registerImages);
setupNavigation();
loadDataset().catch((error) => showError(error.message));
