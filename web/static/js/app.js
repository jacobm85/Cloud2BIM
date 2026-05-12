// ── State ─────────────────────────────────────────────────────────────────
const state = {
  currentStep: 1,
  uploadId: null,
  networkPath: null,
  sourceType: 'upload',   // 'upload' | 'network'
  jobId: null,
  jobStatus: null,
};

const CHUNK_SIZE = 10 * 1024 * 1024;  // 10 MB

// ── DOM helpers ───────────────────────────────────────────────────────────
const $ = (sel, ctx = document) => ctx.querySelector(sel);
const $$ = (sel, ctx = document) => [...ctx.querySelectorAll(sel)];
function show(el) { el && el.classList.remove('hidden'); el && (el.style.display = ''); }
function hide(el) { el && (el.style.display = 'none'); }
function fmt_bytes(b) {
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
  if (b < 1073741824) return (b / 1048576).toFixed(1) + ' MB';
  return (b / 1073741824).toFixed(2) + ' GB';
}

// ── Stepper ───────────────────────────────────────────────────────────────
function goTo(step) {
  state.currentStep = step;
  $$('.step').forEach((el, i) => {
    const s = i + 1;
    el.classList.toggle('active', s === step);
    el.classList.toggle('done', s < step);
  });
  $$('.step-panel').forEach((el, i) => {
    el.style.display = (i + 1 === step) ? '' : 'none';
  });
}

$$('.step').forEach((el, i) => {
  el.addEventListener('click', () => {
    const s = i + 1;
    // Only allow going back to completed steps
    if (s < state.currentStep) goTo(s);
  });
});

// ── Step 1: Upload / Network ───────────────────────────────────────────────
$$('.upload-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    $$('.upload-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.sourceType = tab.dataset.tab;
    $('#upload-panel').style.display = tab.dataset.tab === 'upload' ? '' : 'none';
    $('#network-panel').style.display = tab.dataset.tab === 'network' ? '' : 'none';
    if (tab.dataset.tab === 'network') loadDrives();
  });
});

// File drop zone
const dropZone = $('#drop-zone');
const fileInput = $('#file-input');

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
  const initRes = await fetch('/api/upload/init', {
    method: 'POST',
    body: new URLSearchParams({ filename: file.name, total_size: file.size }),
  });
  const { upload_id } = await initRes.json();
  state.uploadId = upload_id;

  const progressWrap = $('#upload-progress');
  progressWrap.classList.add('visible');
  const fill = $('#upload-fill');
  const label = $('#upload-label');
  const nameEl = $('#upload-name');
  nameEl.textContent = file.name;

  let offset = 0;
  while (offset < file.size) {
    const chunk = file.slice(offset, offset + CHUNK_SIZE);
    const fd = new FormData();
    fd.append('offset', offset);
    fd.append('chunk', chunk, file.name);

    await fetch(`/api/upload/${upload_id}/chunk`, { method: 'POST', body: fd });
    offset += CHUNK_SIZE;
    const pct = Math.min(100, Math.round((offset / file.size) * 100));
    fill.style.width = pct + '%';
    label.textContent = `${fmt_bytes(Math.min(offset, file.size))} / ${fmt_bytes(file.size)}`;
  }

  fill.style.width = '100%';
  label.textContent = 'Upload complete';
  dropZone.querySelector('strong').textContent = file.name;
  dropZone.querySelector('p').textContent = fmt_bytes(file.size) + ' — ready';

  // Auto-detect e57
  if (file.name.toLowerCase().endsWith('.e57')) {
    $('#e57-input').checked = true;
  }

  $('#btn-next-1').disabled = false;
}

$('#btn-next-1').addEventListener('click', () => {
  if (state.sourceType === 'upload' && !state.uploadId) return;
  if (state.sourceType === 'network' && !state.networkPath) return;
  goTo(2);
});

// ── Network browser ────────────────────────────────────────────────────────
let browserHistory = [];

async function loadDrives() {
  const res = await fetch('/api/browse');
  const data = await res.json();
  const panel = $('#network-panel');
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
    card.innerHTML = `<div class="drive-icon">🗂️</div><div>${drive.name}</div><div style="font-size:11px;color:var(--text-dim);margin-top:3px">${drive.path}</div>`;
    card.addEventListener('click', () => browseDir(drive.path));
    grid.appendChild(card);
  });
  panel.appendChild(grid);
}

async function browseDir(path) {
  const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
  if (!res.ok) { alert('Kunde inte läsa katalogen'); return; }
  const data = await res.json();
  browserHistory.push(path);

  const panel = $('#network-panel');
  panel.innerHTML = '';

  if (browserHistory.length > 1) {
    const back = document.createElement('button');
    back.className = 'btn btn-outline';
    back.innerHTML = '← Tillbaka';
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
    const icon = item.type === 'dir' ? '📁' : '📄';
    const sizeStr = item.type === 'file' ? fmt_bytes(item.size) : '';
    row.innerHTML = `<span class="item-icon">${icon}</span><span>${item.name}</span><span class="item-size">${sizeStr}</span>`;

    if (item.type === 'dir') {
      row.addEventListener('click', () => browseDir(item.path));
    } else {
      row.addEventListener('click', () => {
        $$('.browser-item').forEach(r => r.classList.remove('selected'));
        row.classList.add('selected');
        state.networkPath = item.path;
        if (item.name.toLowerCase().endsWith('.e57')) $('#e57-input').checked = true;
        $('#btn-next-1').disabled = false;
        pathEl.textContent = '✓ Vald: ' + item.name;
      });
    }
    list.appendChild(row);
  });
  panel.appendChild(list);
}

// ── Step 2: Configuration ──────────────────────────────────────────────────
$$('.collapsible-header').forEach(header => {
  header.addEventListener('click', () => {
    header.classList.toggle('open');
    const body = header.nextElementSibling;
    body.classList.toggle('open');
  });
});

$('#btn-back-2').addEventListener('click', () => goTo(1));
$('#btn-next-2').addEventListener('click', () => goTo(3));

// ── Step 3: Run ────────────────────────────────────────────────────────────
$('#btn-back-3').addEventListener('click', () => goTo(2));

$('#btn-run').addEventListener('click', async () => {
  $('#btn-run').disabled = true;
  $('#btn-back-3').disabled = true;

  const logEl = $('#log-console');
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

  // Build config payload
  const cfg = collectConfig();

  // Create job
  const res = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  if (!res.ok) {
    const err = await res.json();
    appendLog('[ERROR] ' + (err.detail || 'Failed to create job'));
    $('#btn-run').disabled = false;
    return;
  }
  const { job_id } = await res.json();
  state.jobId = job_id;

  appendLog(`[Job ${job_id.slice(0, 8)}] Startar pipeline…`);
  setBadge('running');

  // Stream logs via SSE
  const sse = new EventSource(`/api/jobs/${job_id}/logs`);
  sse.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.line !== undefined) appendLog(msg.line);
    if (msg.done) {
      sse.close();
      state.jobStatus = msg.status;
      setBadge(msg.status);
      if (msg.status === 'completed') {
        appendLog('\n✓ IFC-modell sparad.');
        setTimeout(() => goTo(4), 800);
      } else {
        appendLog('[ERROR] Pipeline misslyckades. Se logg ovan.');
        $('#btn-run').disabled = false;
        $('#btn-back-3').disabled = false;
      }
    }
  };
  sse.onerror = () => {
    sse.close();
    appendLog('[ERROR] Tappad anslutning.');
    $('#btn-run').disabled = false;
  };
});

function setBadge(status) {
  const badge = $('#status-badge');
  badge.className = `status-badge ${status}`;
  const labels = { pending: 'Väntar', running: 'Kör…', completed: 'Klar', failed: 'Misslyckad' };
  badge.textContent = labels[status] || status;
}

// ── Step 4: Results ────────────────────────────────────────────────────────
async function setupResults() {
  // Download button
  const dlBtn = $('#btn-download');
  dlBtn.href = `/api/jobs/${state.jobId}/download`;
  dlBtn.style.display = 'inline-flex';

  // Parse stats from logs
  const job = await fetch(`/api/jobs/${state.jobId}`).then(r => r.json());
  const stats = parseStats(job.log_lines || []);
  renderStats(stats);

  // Load viewer (defined in viewer.js ES module, exposed on window)
  if (typeof window.loadViewer === 'function') {
    window.loadViewer(state.jobId);
  }
}

function parseStats(lines) {
  const stats = { walls: 0, slabs: 0, windows: 0, doors: 0, storeys: 0 };
  lines.forEach(line => {
    let m;
    if ((m = line.match(/(\d+)\s+slab/i))) stats.slabs = +m[1];
    if ((m = line.match(/Wall\s+(\d+)/i))) stats.walls = Math.max(stats.walls, +m[1]);
    if ((m = line.match(/W(\d+)/))) stats.windows = Math.max(stats.windows, +m[1]);
    if ((m = line.match(/D(\d+)/))) stats.doors = Math.max(stats.doors, +m[1]);
    if ((m = line.match(/Floor\s+[\d.]+\s+m/gi))) stats.storeys = m.length;
  });
  // Count slabs as storeys if not detected
  if (stats.storeys === 0 && stats.slabs > 0) stats.storeys = stats.slabs - 1;
  return stats;
}

function renderStats(stats) {
  const box = $('#result-stats');
  const items = [
    { num: stats.storeys, label: 'Våningar' },
    { num: stats.walls,   label: 'Väggar' },
    { num: stats.slabs,   label: 'Bjälklag' },
    { num: stats.windows, label: 'Fönster' },
    { num: stats.doors,   label: 'Dörrar' },
  ];
  box.innerHTML = items.map(i =>
    `<div class="stat-box"><div class="stat-num">${i.num}</div><div class="stat-label">${i.label}</div></div>`
  ).join('');
}

// ── Config collector ──────────────────────────────────────────────────────
function collectConfig() {
  const g = id => document.getElementById(id);
  const v = id => g(id)?.value ?? '';
  const n = id => parseFloat(v(id)) || 0;
  const b = id => g(id)?.checked ?? false;

  return {
    upload_id: state.sourceType === 'upload' ? state.uploadId : null,
    network_path: state.sourceType === 'network' ? state.networkPath : null,
    e57_input: b('e57-input'),
    exterior_scan: b('exterior-scan'),
    dilute: b('dilute'),
    dilution_factor: parseInt(v('dilution-factor')) || 10,
    pc_resolution: n('pc-resolution'),
    grid_coefficient: parseInt(v('grid-coefficient')) || 5,
    bfs_thickness: n('bfs-thickness'),
    tfs_thickness: n('tfs-thickness'),
    min_wall_length: n('min-wall-length'),
    min_wall_thickness: n('min-wall-thickness'),
    max_wall_thickness: n('max-wall-thickness'),
    exterior_walls_thickness: n('ext-wall-thickness'),
    ifc_project_name: v('ifc-project-name'),
    ifc_project_long_name: v('ifc-project-long-name'),
    ifc_project_version: v('ifc-project-version'),
    ifc_author_name: v('ifc-author-name'),
    ifc_author_surname: v('ifc-author-surname'),
    ifc_author_organization: v('ifc-author-org'),
    ifc_building_name: v('ifc-building-name'),
    ifc_building_type: v('ifc-building-type'),
    ifc_building_phase: v('ifc-building-phase'),
    ifc_site_latitude: [0, 0, 0],
    ifc_site_longitude: [0, 0, 0],
    ifc_site_elevation: n('ifc-elevation'),
    material_for_objects: v('material'),
  };
}

// ── Init ──────────────────────────────────────────────────────────────────
// Expose goTo globally so inline onclick handlers and viewer.js can call it
window.goTo = function(step) {
  goTo(step);
  if (step === 4) setupResults();
};

goTo(1);
$('#btn-next-1').disabled = true;
