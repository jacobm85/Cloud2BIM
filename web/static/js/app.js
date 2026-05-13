// ── State ─────────────────────────────────────────────────────────────────
const state = {
  currentStep: 1,
  uploadId: null,
  networkPath: null,
  sourceType: 'upload',
  jobId: null,
  jobStatus: null,
};
window._state = state; // expose for console debugging

const CHUNK_SIZE = 10 * 1024 * 1024; // 10 MB

// ── DOM helpers ───────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
function fmt_bytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b / 1048576).toFixed(1) + ' MB';
  return (b / 1073741824).toFixed(2) + ' GB';
}

// ── Step navigation ───────────────────────────────────────────────────────
function goTo(n) {
  console.log('[goTo]', n);
  state.currentStep = n;

  // Show/hide panels by ID — no index assumptions
  for (let i = 1; i <= 4; i++) {
    const panel = document.getElementById('step-' + i);
    if (panel) panel.style.display = (i === n) ? 'block' : 'none';
  }

  // Update stepper indicators
  $$('.step').forEach(el => {
    const s = parseInt(el.dataset.step, 10);
    el.classList.toggle('active', s === n);
    el.classList.toggle('done', s < n);
  });

  if (n === 4) setupResults();
}
window.goTo = goTo; // used by inline onclick in HTML

// ── Reset state for new job ───────────────────────────────────────────────
function resetJob() {
  state.jobId = null;
  state.jobStatus = null;
  state.uploadId = null;
  state.networkPath = null;

  clearUploadError();
  document.getElementById('upload-progress').classList.remove('visible');
  document.getElementById('upload-fill').style.width = '0%';
  document.getElementById('upload-name').textContent = '—';
  document.getElementById('upload-label').textContent = '0 B';
  document.getElementById('drop-zone').querySelector('strong').textContent = 'Dra och släpp din fil här';
  document.getElementById('drop-zone').querySelector('p').textContent =
    'eller klicka för att bläddra — stöd för .xyz, .e57, .las, .laz';
  document.getElementById('btn-next-1').disabled = true;

  document.getElementById('log-console').innerHTML =
    '<div class="log-line" style="color:var(--text-dim)">Loggar visas här när processen startar…</div>';
  document.getElementById('btn-run').disabled = false;
  document.getElementById('btn-back-3').disabled = false;
  setBadge('pending');

  document.getElementById('btn-download').style.display = 'none';
  const viewBtn = document.getElementById('btn-open-viewer');
  if (viewBtn) viewBtn.style.display = 'none';
  document.getElementById('result-stats').innerHTML = '';
}
window.newFile = function() { resetJob(); goTo(1); };

// Stepper nav — allow going back to completed steps
$$('.step').forEach(el => {
  el.addEventListener('click', () => {
    const s = parseInt(el.dataset.step, 10);
    if (s < state.currentStep) goTo(s);
  });
});

// ── Upload error helpers ──────────────────────────────────────────────────
function showUploadError(msg) {
  const el = document.getElementById('upload-error');
  if (!el) { alert(msg); return; }
  el.textContent = msg;
  el.style.display = 'block';
}
function clearUploadError() {
  const el = document.getElementById('upload-error');
  if (el) { el.textContent = ''; el.style.display = 'none'; }
}

// ── Step 1: Upload ────────────────────────────────────────────────────────
$$('.upload-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.upload-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.sourceType = tab.dataset.tab;
    document.getElementById('upload-panel').style.display =
      tab.dataset.tab === 'upload' ? 'block' : 'none';
    document.getElementById('network-panel').style.display =
      tab.dataset.tab === 'network' ? 'block' : 'none';
    if (tab.dataset.tab === 'network') loadDrives();
  });
});

const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) startUpload(file);
});
fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) startUpload(fileInput.files[0]);
});

async function startUpload(file) {
  clearUploadError();
  state.uploadId = null;
  document.getElementById('btn-next-1').disabled = true;

  const progressWrap = document.getElementById('upload-progress');
  progressWrap.classList.add('visible');
  const fill = document.getElementById('upload-fill');
  const label = document.getElementById('upload-label');
  document.getElementById('upload-name').textContent = file.name;
  fill.style.width = '0%';
  label.textContent = '0 B / ' + fmt_bytes(file.size);

  // Init
  let upload_id;
  try {
    const initRes = await fetch('/api/upload/init', {
      method: 'POST',
      body: new URLSearchParams({ filename: file.name, total_size: file.size }),
    });
    if (!initRes.ok) throw new Error('HTTP ' + initRes.status + ': ' + await initRes.text());
    const data = await initRes.json();
    if (!data.upload_id) throw new Error('Inget upload_id i svar: ' + JSON.stringify(data));
    upload_id = data.upload_id;
    console.log('[upload] init OK, id =', upload_id);
  } catch (err) {
    console.error('[upload init]', err);
    showUploadError('Init misslyckades: ' + err.message);
    progressWrap.classList.remove('visible');
    return;
  }

  // Chunks
  let offset = 0;
  try {
    while (offset < file.size) {
      const chunk = file.slice(offset, offset + CHUNK_SIZE);
      const fd = new FormData();
      fd.append('offset', offset);
      fd.append('chunk', chunk, file.name);
      const chunkRes = await fetch('/api/upload/' + upload_id + '/chunk', { method: 'POST', body: fd });
      if (!chunkRes.ok) throw new Error('Chunk HTTP ' + chunkRes.status + ': ' + await chunkRes.text());
      offset += CHUNK_SIZE;
      const pct = Math.min(100, Math.round((offset / file.size) * 100));
      fill.style.width = pct + '%';
      label.textContent = fmt_bytes(Math.min(offset, file.size)) + ' / ' + fmt_bytes(file.size);
    }
  } catch (err) {
    console.error('[upload chunk]', err);
    showUploadError('Uppladdning avbruten: ' + err.message);
    return;
  }

  // Done
  state.uploadId = upload_id;
  fill.style.width = '100%';
  label.textContent = 'Uppladdning klar ✓';
  dropZone.querySelector('strong').textContent = file.name;
  dropZone.querySelector('p').textContent = fmt_bytes(file.size) + ' — klar';
  // Format auto-detected on backend — no manual toggle needed

  document.getElementById('btn-next-1').disabled = false;
  console.log('[upload] KLAR — state.uploadId =', state.uploadId);
}

// Next button — uses onclick in HTML, but also wired here as fallback
function onNext1() {
  console.log('[next1] clicked, uploadId =', state.uploadId, 'sourceType =', state.sourceType);
  if (state.sourceType === 'upload' && !state.uploadId) {
    showUploadError('Filen är inte uppladdad än.');
    return;
  }
  if (state.sourceType === 'network' && !state.networkPath) {
    showUploadError('Välj en fil från nätverksdisken.');
    return;
  }
  goTo(2);
}
window.onNext1 = onNext1;

// ── Network browser ───────────────────────────────────────────────────────
let browserHistory = [];

async function loadDrives() {
  const res = await fetch('/api/browse');
  const data = await res.json();
  const panel = document.getElementById('network-panel');
  panel.innerHTML = '';
  if (!data.drives || data.drives.length === 0) {
    panel.innerHTML = '<p class="no-drives">Inga nätverksdiskar konfigurerade i web_config.yaml</p>';
    return;
  }
  const grid = document.createElement('div');
  grid.className = 'drives-grid';
  data.drives.forEach(drive => {
    const card = document.createElement('div');
    card.className = 'drive-card';
    card.innerHTML = '<div class="drive-icon">🗂️</div><div>' + drive.name + '</div>';
    card.addEventListener('click', () => browseDir(drive.path));
    grid.appendChild(card);
  });
  panel.appendChild(grid);
}

async function browseDir(path) {
  const res = await fetch('/api/browse?path=' + encodeURIComponent(path));
  if (!res.ok) { alert('Kunde inte läsa katalogen'); return; }
  const data = await res.json();
  browserHistory.push(path);
  const panel = document.getElementById('network-panel');
  panel.innerHTML = '';
  if (browserHistory.length > 1) {
    const back = document.createElement('button');
    back.type = 'button';
    back.className = 'btn btn-outline';
    back.textContent = '← Tillbaka';
    back.style.marginBottom = '10px';
    back.addEventListener('click', () => {
      browserHistory.pop();
      const prev = browserHistory.pop();
      if (prev) browseDir(prev); else loadDrives();
    });
    panel.appendChild(back);
  }
  const pathEl = document.createElement('div');
  pathEl.className = 'browser-path';
  pathEl.textContent = data.current;
  panel.appendChild(pathEl);
  const list = document.createElement('div');
  list.className = 'browser-list';
  data.items.forEach(item => {
    const row = document.createElement('div');
    row.className = 'browser-item';
    row.innerHTML = '<span class="item-icon">' + (item.type === 'dir' ? '📁' : '📄') + '</span>' +
      '<span>' + item.name + '</span>' +
      '<span class="item-size">' + (item.type === 'file' ? fmt_bytes(item.size) : '') + '</span>';
    if (item.type === 'dir') {
      row.addEventListener('click', () => browseDir(item.path));
    } else {
      row.addEventListener('click', () => {
        $$('.browser-item').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        state.networkPath = item.path;
        document.getElementById('btn-next-1').disabled = false;
      });
    }
    list.appendChild(row);
  });
  panel.appendChild(list);
}

// ── Step 2 ────────────────────────────────────────────────────────────────
$$('.collapsible-header').forEach(header => {
  header.addEventListener('click', () => {
    header.classList.toggle('open');
    header.nextElementSibling.classList.toggle('open');
  });
});

// ── Step 3: Run ───────────────────────────────────────────────────────────
document.getElementById('btn-run').addEventListener('click', async () => {
  document.getElementById('btn-run').disabled = true;
  document.getElementById('btn-back-3').disabled = true;
  const logEl = document.getElementById('log-console');
  logEl.innerHTML = '';

  function appendLog(text) {
    const line = document.createElement('div');
    line.className = 'log-line';
    if (text.startsWith('---') || text.startsWith('===')) line.classList.add('section');
    else if (/error|exception/i.test(text)) line.classList.add('error');
    else if (/saved|complete|done/i.test(text)) line.classList.add('success');
    line.textContent = text;
    logEl.appendChild(line);
    logEl.scrollTop = logEl.scrollHeight;
  }

  const cfg = collectConfig();
  const res = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  if (!res.ok) {
    const err = await res.json();
    appendLog('[ERROR] ' + (err.detail || 'Failed to create job'));
    document.getElementById('btn-run').disabled = false;
    return;
  }
  const { job_id } = await res.json();
  state.jobId = job_id;
  appendLog('[Job ' + job_id.slice(0, 8) + '] Startar pipeline…');
  setBadge('running');

  const sse = new EventSource('/api/jobs/' + job_id + '/logs');
  sse.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.line !== undefined) appendLog(msg.line);
    if (msg.done) {
      sse.close();
      state.jobStatus = msg.status;
      setBadge(msg.status);
      if (msg.status === 'completed') {
        appendLog('✓ IFC-modell sparad.');
        setTimeout(() => goTo(4), 800);
      } else {
        appendLog('[ERROR] Pipeline misslyckades.');
        document.getElementById('btn-run').disabled = false;
        document.getElementById('btn-back-3').disabled = false;
      }
    }
  };
  sse.onerror = () => {
    sse.close();
    appendLog('[ERROR] Tappad anslutning.');
    document.getElementById('btn-run').disabled = false;
  };
});

function setBadge(status) {
  const badge = document.getElementById('status-badge');
  badge.className = 'status-badge ' + status;
  badge.textContent = { pending: 'Väntar', running: 'Kör…', completed: 'Klar', failed: 'Misslyckad' }[status] || status;
}

// ── Step 4: Results ───────────────────────────────────────────────────────
async function setupResults() {
  const dlBtn = document.getElementById('btn-download');
  dlBtn.href = '/api/jobs/' + state.jobId + '/download';
  dlBtn.style.display = 'inline-flex';
  const viewBtn = document.getElementById('btn-open-viewer');
  if (viewBtn) viewBtn.style.display = 'inline-flex';
  const job = await fetch('/api/jobs/' + state.jobId).then(r => r.json());
  renderStats(parseStats(job.log_lines || []));
  if (typeof window.loadViewer === 'function') window.loadViewer(state.jobId);
}

function parseStats(lines) {
  const s = { walls: 0, slabs: 0, windows: 0, doors: 0, storeys: 0 };
  lines.forEach(line => {
    let m;
    // "Creating hull for slab no. 2 of 3" → slabs = 3
    if ((m = line.match(/slab no\.\s*\d+\s+of\s+(\d+)/i)))
      s.slabs = Math.max(s.slabs, +m[1]);
    // "Wall 5:" — one line printed per wall
    if ((m = line.match(/^\s*Wall\s+(\d+)\s*:/)))
      s.walls = Math.max(s.walls, +m[1]);
    // "Opening (window):" and "Opening (door):" — one per opening
    if (/^\s*Opening\s*\(window\)/i.test(line)) s.windows++;
    if (/^\s*Opening\s*\(door\)/i.test(line)) s.doors++;
  });
  s.storeys = s.slabs > 1 ? s.slabs - 1 : Math.min(s.slabs, 1);
  return s;
}

function renderStats(s) {
  document.getElementById('result-stats').innerHTML =
    [['Våningar', s.storeys], ['Väggar', s.walls], ['Bjälklag', s.slabs],
     ['Fönster', s.windows], ['Dörrar', s.doors]]
    .map(([l, n]) => '<div class="stat-box"><div class="stat-num">' + n +
      '</div><div class="stat-label">' + l + '</div></div>').join('');
}

// ── Config collector ──────────────────────────────────────────────────────
function collectConfig() {
  const v = id => (document.getElementById(id) || {}).value || '';
  const n = id => parseFloat(v(id)) || 0;
  const b = id => !!(document.getElementById(id) || {}).checked;
  return {
    upload_id: state.sourceType === 'upload' ? state.uploadId : null,
    network_path: state.sourceType === 'network' ? state.networkPath : null,
    e57_input: false,  // auto-detected on backend from file extension
    exterior_scan: b('exterior-scan'),
    dilute: b('dilute'), dilution_factor: parseInt(v('dilution-factor')) || 10,
    pc_resolution: n('pc-resolution'), grid_coefficient: parseInt(v('grid-coefficient')) || 5,
    bfs_thickness: n('bfs-thickness'), tfs_thickness: n('tfs-thickness'),
    min_wall_length: n('min-wall-length'), min_wall_thickness: n('min-wall-thickness'),
    max_wall_thickness: n('max-wall-thickness'), exterior_walls_thickness: n('ext-wall-thickness'),
    ifc_project_name: v('ifc-project-name'), ifc_project_long_name: v('ifc-project-long-name'),
    ifc_project_version: v('ifc-project-version'), ifc_author_name: v('ifc-author-name'),
    ifc_author_surname: v('ifc-author-surname'), ifc_author_organization: v('ifc-author-org'),
    ifc_building_name: v('ifc-building-name'), ifc_building_type: v('ifc-building-type'),
    ifc_building_phase: v('ifc-building-phase'),
    ifc_site_latitude: [0, 0, 0], ifc_site_longitude: [0, 0, 0],
    ifc_site_elevation: n('ifc-elevation'), material_for_objects: v('material'),
  };
}

// ── Init ──────────────────────────────────────────────────────────────────
goTo(1);
