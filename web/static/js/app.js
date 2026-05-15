// ── State ─────────────────────────────────────────────────────────────────
const state = {
  currentStep: 1,
  uploadId: null,
  networkPath: null,
  sourceJobId: null,
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

  if (n === 3) {
    // Show the right run panel depending on selected mode
    const mode = (document.querySelector('input[name="run-mode"]:checked') || {}).value || 'full';
    document.getElementById('run-full-panel').style.display = mode === 'full' ? 'block' : 'none';
    document.getElementById('run-wizard-panel').style.display = mode === 'stepwise' ? 'block' : 'none';
  }

  if (n === 4) setupResults();
}
window.goTo = goTo; // used by inline onclick in HTML

// ── Reset state for new job ───────────────────────────────────────────────
function resetJob() {
  state.jobId = null;
  state.jobStatus = null;
  state.uploadId = null;
  state.networkPath = null;
  state.sourceJobId = null;

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
    document.getElementById('reuse-panel').style.display =
      tab.dataset.tab === 'reuse' ? 'block' : 'none';
    if (tab.dataset.tab === 'network') loadDrives();
    if (tab.dataset.tab === 'reuse') loadReusableJobs();
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
  if (state.sourceType === 'upload' && !state.uploadId) {
    showUploadError('Filen är inte uppladdad än.'); return;
  }
  if (state.sourceType === 'network' && !state.networkPath) {
    showUploadError('Välj en fil från nätverksdisken.'); return;
  }
  if (state.sourceType === 'reuse' && !state.sourceJobId) {
    showUploadError('Välj ett tidigare jobb.'); return;
  }
  goTo(2);
}

async function loadReusableJobs() {
  const list = document.getElementById('reuse-list');
  list.innerHTML = '<div style="color:var(--text-dim);font-size:13px">Laddar…</div>';
  try {
    const res = await fetch('/api/jobs/reusable');
    const jobs = await res.json();
    if (!jobs.length) {
      list.innerHTML = '<div style="color:var(--text-dim);font-size:13px">Inga tidigare jobb med konverterade punktmoln hittades.</div>';
      return;
    }
    list.innerHTML = '';
    jobs.forEach(job => {
      const row = document.createElement('div');
      row.className = 'browser-item';
      row.style.cssText = 'cursor:pointer;padding:8px 10px;border-radius:6px;margin-bottom:4px;display:flex;justify-content:space-between;align-items:center;background:var(--surface2)';
      const date = job.created_at ? new Date(job.created_at).toLocaleString('sv-SE') : '—';
      row.innerHTML = `
        <div style="flex:1;min-width:0" class="reuse-select-area">
          <div style="font-weight:600;font-size:13px">${job.original_filename}</div>
          <div style="font-size:11px;color:var(--text-dim)">${date} &nbsp;·&nbsp; ${job.xyz_size_mb} MB XYZ &nbsp;·&nbsp; ${job.job_id.slice(0,8)}…</div>
        </div>
        <button class="btn-delete-job" title="Ta bort jobb och alla filer"
          style="margin-left:10px;background:none;border:none;cursor:pointer;color:var(--text-dim);font-size:16px;padding:4px 6px;border-radius:4px;flex-shrink:0">🗑</button>`;
      row.querySelector('.reuse-select-area').addEventListener('click', () => {
        $$('#reuse-list .browser-item').forEach(r => {
          r.classList.remove('selected');
          r.style.background = 'var(--surface2)';
          r.style.color = '';
        });
        row.classList.add('selected');
        row.style.background = 'var(--accent, #4f6ef7)';
        row.style.color = 'white';
        state.sourceJobId = job.job_id;
        document.getElementById('btn-next-1').disabled = false;
      });
      row.querySelector('.btn-delete-job').addEventListener('click', async (e) => {
        e.stopPropagation();
        if (!confirm(`Ta bort jobbet och alla dess filer?\n${job.original_filename}`)) return;
        try {
          const res = await fetch(`/api/jobs/${job.job_id}`, { method: 'DELETE' });
          if (!res.ok) throw new Error(await res.text());
          row.remove();
          if (state.sourceJobId === job.job_id) {
            state.sourceJobId = null;
            document.getElementById('btn-next-1').disabled = true;
          }
        } catch (err) {
          alert('Kunde inte ta bort jobbet: ' + err.message);
        }
      });
      list.appendChild(row);
    });
  } catch (e) {
    list.innerHTML = '<div style="color:var(--danger)">Kunde inte ladda tidigare jobb.</div>';
  }
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
  try {
    const stats = await fetch('/api/jobs/' + state.jobId + '/stats').then(r => r.json());
    renderStats(stats);
  } catch (e) {
    renderStats({ storeys: 0, walls: 0, slabs: 0, openings: 0, roofs: 0 });
  }
  renderResultDxfButtons();
  if (typeof window.loadViewer === 'function') window.loadViewer(state.jobId);
}

function renderStats(s) {
  const rows = [
    ['Våningar', s.storeys ?? 0], ['Väggar', s.walls ?? 0],
    ['Bjälklag', s.slabs ?? 0], ['Öppningar', s.openings ?? 0],
    ['Pelare', s.columns ?? 0], ['Trappor', s.stairs ?? 0],
    ['Tak', s.roofs ?? 0],
  ];
  document.getElementById('result-stats').innerHTML = rows
    .map(([l, n]) => '<div class="stat-box"><div class="stat-num">' + n +
      '</div><div class="stat-label">' + l + '</div></div>').join('');
}

async function renderResultDxfButtons() {
  const wrap = document.getElementById('result-dxf-buttons');
  if (!wrap) return;
  wrap.innerHTML = '';
  try {
    const res = await fetch('/api/jobs/' + state.jobId + '/dxf/storeys');
    if (!res.ok) return;
    const { storeys } = await res.json();
    if (!storeys) return;
    const buttons = [];
    for (let i = 0; i < storeys; i++) {
      buttons.push(`<a class="btn btn-outline" href="/api/jobs/${state.jobId}/dxf/${i}" download>⬇ Plan våning ${i} (.dxf)</a>`);
    }
    wrap.innerHTML = buttons.join('');
  } catch (e) { /* DXF is optional */ }
}

// ── Config collector ──────────────────────────────────────────────────────
function collectConfig() {
  const v = id => (document.getElementById(id) || {}).value || '';
  const n = id => parseFloat(v(id)) || 0;
  const b = id => !!(document.getElementById(id) || {}).checked;
  const mode = (document.querySelector('input[name="run-mode"]:checked') || {}).value || 'full';
  const algorithm = (document.querySelector('input[name="algorithm"]:checked') || {}).value || 'v1';
  return {
    upload_id: state.sourceType === 'upload' ? state.uploadId : null,
    network_path: state.sourceType === 'network' ? state.networkPath : null,
    source_job_id: state.sourceType === 'reuse' ? state.sourceJobId : null,
    e57_input: false,  // auto-detected on backend from file extension
    mode,
    algorithm,
    seg_enabled: b('seg-enabled'),
    seg_backend: v('seg-backend') || 'ptv3',
    seg_weights: v('seg-weights') || null,
    slabs_enabled: b('slabs-enabled'),
    walls_enabled: b('walls-enabled'),
    openings_enabled: b('openings-enabled'),
    columns_enabled: b('columns-enabled'),
    stairs_enabled: b('stairs-enabled'),
    roofs_enabled: b('roofs-enabled'),
    exterior_scan: b('exterior-scan'),
    dilute: b('dilute'), dilution_factor: parseInt(v('dilution-factor')) || 10,
    pc_resolution: n('pc-resolution'), grid_coefficient: parseInt(v('grid-coefficient')) || 5,
    bfs_thickness: n('bfs-thickness'), tfs_thickness: n('tfs-thickness'),
    max_slab_thickness: n('max-slab-thickness') || 0.5,
    slab_peak_height_ratio: n('slab-peak-ratio') || 0.25,
    slab_z_step: n('slab-z-step') || 0.15,
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

// ── Wizard mode (stepwise pipeline) ───────────────────────────────────────
const WIZARD_STAGES = ['prepare', 'segment', 'slabs', 'walls', 'openings', 'columns', 'stairs', 'roofs', 'ifc'];

const STAGE_INFO = {
  prepare: {
    title: 'Förberedelse + crop',
    desc: 'Läser punktmolnet, glesar och centrerar koordinater. Här kan du också rita en polygon i planöversikten för att beskära punktmolnet — bara punkter inom polygonen följer med till resten av pipelinen.',
  },
  segment: {
    title: 'Semantisk segmentering',
    desc: 'Klassificerar varje punkt (golv, vägg, tak, möbler, …). Om ML-segmentering är avstängd märks alla punkter som "unknown" och påverkar inte vägg/öppningsdetekteringen.',
  },
  slabs: {
    title: 'Bjälklag',
    desc: 'Bygger ett Z-histogram över punktmolnet och hittar horisontella ytor som toppar. Toppar inom max_slab_thickness paras som botten+topp; övriga blir egna bjälklag.',
  },
  walls: {
    title: 'Väggar',
    desc: 'För varje våning tas ett horisontellt snitt 130–160 cm över golvet och 2D-histogrammet ger väggsegment. Du kan välja Z-snittet manuellt här om bjälklagsdetektionen blev fel.',
  },
  openings: {
    title: 'Öppningar',
    desc: 'Hittar fönster och dörrar i varje vägg baserat på lokala håligheter och semantiska etiketter.',
  },
  columns: {
    title: 'Pelare',
    desc: 'Hittar fristående vertikala pelare i 2D-occupancy-histogrammet — små isolerade kluster som spänner hela våningshöjden och inte ligger på en vägg.',
  },
  stairs: {
    title: 'Trappor',
    desc: 'Letar efter serier av tätt liggande horisontella Z-toppar (steg) mellan två bjälklagsnivåer och grupperar dem i trapplöp.',
  },
  roofs: {
    title: 'Tak',
    desc: 'RANSAC-planpassning för sneda tak. Hoppas över om "Sneda tak" är avstängt.',
  },
  ifc: {
    title: 'IFC-export',
    desc: 'Bygger IFC-modellen och genererar planlösningspreview.',
  },
};

const wizard = {
  jobId: null,
  pollTimer: null,
  sse: null,
  slabsData: null,
  slabCount: 0,
  bands: [],        // [{z_min, z_max} | null] per storey — used for wall detection
  bands_lower: [],  // [{z_min, z_max} | null] per storey — low diagnostic preview
};

function wizardLog(text) {
  const el = document.getElementById('wizard-log');
  const line = document.createElement('div');
  line.className = 'log-line';
  if (/error|exception/i.test(text)) line.classList.add('error');
  else if (/saved|complete|done/i.test(text)) line.classList.add('success');
  line.textContent = text;
  el.appendChild(line);
  el.scrollTop = el.scrollHeight;
}

function wizardSetBadge(status) {
  const badge = document.getElementById('wizard-status-badge');
  badge.className = 'status-badge ' + status;
  badge.textContent = { pending: 'Väntar', running: 'Kör…', completed: 'Klar', failed: 'Misslyckad', awaiting: 'Väntar på dig' }[status] || status;
}

function wizardUpdateStageList(completed, current, failed) {
  const items = document.querySelectorAll('#wizard-stages li');
  items.forEach(li => {
    const s = li.dataset.stage;
    li.classList.remove('active', 'done', 'failed');
    const tick = li.querySelector('.stage-tick');
    if (completed.includes(s)) {
      li.classList.add('done');
      tick.textContent = '✓';
    } else if (s === current) {
      li.classList.add('active');
      tick.textContent = '⋯';
    } else if (s === failed) {
      li.classList.add('failed');
      tick.textContent = '✗';
    } else {
      tick.textContent = '·';
    }
  });
}

async function wizardStart() {
  document.getElementById('btn-wizard-start').disabled = true;
  document.getElementById('btn-back-3-wiz').disabled = true;
  document.getElementById('wizard-log').innerHTML = '';
  wizardSetBadge('running');

  const cfg = collectConfig();
  cfg.mode = 'stepwise';
  const res = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(cfg),
  });
  if (!res.ok) {
    wizardLog('[ERROR] Kunde inte starta jobb');
    wizardSetBadge('failed');
    return;
  }
  const data = await res.json();
  wizard.jobId = data.job_id;
  state.jobId = data.job_id;
  wizardLog(`[Job ${data.job_id.slice(0, 8)}] Wizard startad`);

  wizardStreamLogs();
  wizardPollState();
}

function wizardStreamLogs() {
  if (wizard.sse) { try { wizard.sse.close(); } catch (e) {} }
  const sse = new EventSource('/api/jobs/' + wizard.jobId + '/logs');
  sse.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.line !== undefined) wizardLog(msg.line);
    if (msg.done) {
      sse.close();
      wizard.sse = null;
    }
  };
  sse.onerror = () => { try { sse.close(); } catch (e) {} wizard.sse = null; };
  wizard.sse = sse;
}

async function wizardPollState() {
  if (wizard.pollTimer) clearInterval(wizard.pollTimer);
  let lastStatus = null;
  let lastStage = null;

  const tick = async () => {
    try {
      const res = await fetch('/api/jobs/' + wizard.jobId + '/state');
      if (!res.ok) return;
      const st = await res.json();
      const completed = st.completed_stages || [];
      wizardUpdateStageList(completed, st.current_stage, null);
      wizardSetBadge(st.status === 'running' ? 'running' : (st.status === 'failed' ? 'failed' : (st.status === 'completed' ? 'awaiting' : st.status)));

      // Detect a stage finishing so we can render its review screen
      const justFinished = lastStatus === 'running' && st.status !== 'running';
      if (justFinished) {
        // Re-attach logs in case the stream closed
        if (!wizard.sse) wizardStreamLogs();
        const nextReview = completed[completed.length - 1];
        if (st.status === 'failed') {
          renderWizardStageReview(nextReview || lastStage, true);
        } else {
          renderWizardStageReview(nextReview, false);
        }
      }
      lastStatus = st.status;
      lastStage = st.current_stage;
    } catch (e) { /* keep polling */ }
  };
  wizard.pollTimer = setInterval(tick, 1500);
  tick();
}

function renderWizardStageReview(stage, failed) {
  const detail = document.getElementById('wizard-stage-detail');
  const info = STAGE_INFO[stage] || { title: stage, desc: '' };
  detail.innerHTML = `
    <div class="stage-panel">
      <h3>${failed ? '✗' : '✓'} ${info.title}</h3>
      <div class="stage-help">${info.desc}</div>
      <div id="stage-extra"></div>
      <div class="btn-row">
        <button class="btn btn-outline" id="btn-stage-redo">Kör om detta steg</button>
        <button class="btn btn-primary" id="btn-stage-continue" ${failed ? 'disabled' : ''}>Fortsätt →</button>
      </div>
    </div>`;
  const next = WIZARD_STAGES[WIZARD_STAGES.indexOf(stage) + 1];
  document.getElementById('btn-stage-continue').onclick = () => wizardRunStage(next || stage);
  document.getElementById('btn-stage-redo').onclick = () => wizardOpenStageOverrides(stage);

  // Stage-specific extras
  if (stage === 'prepare') renderPrepareReview();
  else if (stage === 'slabs') renderSlabsReview();
  else if (stage === 'walls') renderWallsReview();
  else if (stage === 'ifc') renderIfcReview();
}

async function renderPrepareReview() {
  const extra = document.getElementById('stage-extra');
  extra.innerHTML = '<div style="padding:30px;text-align:center;color:var(--text-dim)">Renderar planöversikt…</div>';
  let meta;
  try {
    const res = await fetch('/api/jobs/' + wizard.jobId + '/topdown');
    if (!res.ok) throw new Error('HTTP ' + res.status);
    meta = await res.json();
  } catch (e) {
    extra.innerHTML = '<div class="alert alert-danger">Kunde inte rendera översikt: ' + e.message + '</div>';
    return;
  }
  extra.innerHTML = `
    <div style="margin-bottom:14px">
      <div style="font-weight:600;margin-bottom:4px">Beskär punktmoln (valfritt)</div>
      <div style="font-size:12px;color:var(--text-dim);margin-bottom:10px">
        ${meta.point_count.toLocaleString()} punkter just nu. Klicka i bilden för att lägga
        till polygonpunkter; dubbelklicka eller "Tillämpa crop" för att stänga polygonen och
        bara behålla punkter innanför. Hoppa över helt om hela skanningen ska bearbetas.
      </div>
      <div id="crop-wrap" style="position:relative;display:inline-block;background:#0f1117;border:1px solid var(--border);border-radius:8px;overflow:hidden;max-width:100%">
        <img id="topdown-img" src="${meta.image_url}" alt="Top-down" style="display:block;max-width:100%;user-select:none;-webkit-user-drag:none">
        <canvas id="topdown-canvas" style="position:absolute;left:0;top:0;cursor:crosshair"></canvas>
      </div>
      <div class="btn-row" style="margin-top:10px;gap:8px;flex-wrap:wrap">
        <button class="btn btn-outline" id="btn-crop-undo">Ångra punkt</button>
        <button class="btn btn-outline" id="btn-crop-clear">Rensa</button>
        <button class="btn btn-primary" id="btn-crop-apply" disabled>Tillämpa crop</button>
        <span id="crop-status" style="font-size:12px;color:var(--text-dim);align-self:center;margin-left:6px"></span>
      </div>
    </div>`;
  setupCropTool(meta.bounds);
}

function setupCropTool(bounds) {
  const img = document.getElementById('topdown-img');
  const canvas = document.getElementById('topdown-canvas');
  const status = document.getElementById('crop-status');
  const pts = [];

  function sync() {
    canvas.width = img.naturalWidth || img.clientWidth;
    canvas.height = img.naturalHeight || img.clientHeight;
    canvas.style.width = img.clientWidth + 'px';
    canvas.style.height = img.clientHeight + 'px';
    redraw();
  }
  img.addEventListener('load', sync);
  if (img.complete) sync();
  window.addEventListener('resize', sync);

  function pixelToWorld(px, py) {
    const [xmin, ymin, xmax, ymax] = bounds;
    const wx = xmin + (px / canvas.width) * (xmax - xmin);
    // PNG y is top-down; world y is bottom-up
    const wy = ymax - (py / canvas.height) * (ymax - ymin);
    return [wx, wy];
  }

  function redraw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const applyBtn = document.getElementById('btn-crop-apply');
    if (applyBtn) applyBtn.disabled = pts.length < 3;
    if (status) status.textContent = pts.length + ' punkt' + (pts.length === 1 ? '' : 'er');
    if (pts.length === 0) return;
    ctx.strokeStyle = '#4f8ef7';
    ctx.fillStyle = 'rgba(79,142,247,0.18)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (pts.length >= 3) { ctx.closePath(); ctx.fill(); }
    ctx.stroke();
    for (const p of pts) {
      ctx.beginPath();
      ctx.arc(p[0], p[1], 4, 0, Math.PI * 2);
      ctx.fillStyle = '#fff';
      ctx.fill();
      ctx.strokeStyle = '#4f8ef7';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  canvas.addEventListener('click', e => {
    const rect = canvas.getBoundingClientRect();
    const px = (e.clientX - rect.left) * (canvas.width / rect.width);
    const py = (e.clientY - rect.top) * (canvas.height / rect.height);
    pts.push([px, py]);
    redraw();
  });
  canvas.addEventListener('dblclick', e => {
    e.preventDefault();
    if (pts.length >= 3) applyCrop();
  });

  document.getElementById('btn-crop-clear').onclick = () => { pts.length = 0; redraw(); };
  document.getElementById('btn-crop-undo').onclick = () => { pts.pop(); redraw(); };
  document.getElementById('btn-crop-apply').onclick = applyCrop;

  async function applyCrop() {
    if (pts.length < 3) return;
    const polygon = pts.map(p => pixelToWorld(p[0], p[1]));
    status.style.color = 'var(--text-dim)';
    status.textContent = 'Beskär…';
    try {
      const res = await fetch('/api/jobs/' + wizard.jobId + '/crop', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ polygon }),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || ('HTTP ' + res.status));
      }
      const r = await res.json();
      status.style.color = 'var(--success)';
      status.textContent = `✓ ${r.after.toLocaleString()} av ${r.before.toLocaleString()} punkter kvar (${Math.round(r.kept_fraction * 100)}%)`;
      // Re-render with fresh top-down so the user sees the cropped cloud
      setTimeout(renderPrepareReview, 400);
    } catch (e) {
      status.style.color = 'var(--danger)';
      status.textContent = '✗ ' + e.message;
    }
  }
}

async function renderSlabsReview() {
  const extra = document.getElementById('stage-extra');
  extra.innerHTML = '<div class="stage-preview-row"><div style="text-align:center;padding:40px;color:var(--text-dim)">Laddar Z-histogram…</div></div>';
  try {
    const data = await fetch('/api/jobs/' + wizard.jobId + '/slabs').then(r => r.json());
    wizard.slabsData = data;
    wizard.slabCount = data.slabs.length;
    // Initialize default bands: floor+1.30 to floor+1.60 per storey (walls)
    // and floor+0.30 to floor+0.35 per storey (low diagnostic preview).
    wizard.bands = [];
    wizard.bands_lower = [];
    for (let i = 0; i < data.slabs.length - 1; i++) {
      const floor = data.slabs[i].top_z;
      wizard.bands.push({ z_min: floor + 1.30, z_max: floor + 1.60 });
      wizard.bands_lower.push({ z_min: floor + 0.30, z_max: floor + 0.35 });
    }
    const rows = data.slabs.map((s, i) => `
      <tr>
        <td><input type="checkbox" class="slab-keep" data-idx="${i}" checked></td>
        <td>${i}</td>
        <td><input type="number" class="slab-bottom" data-idx="${i}" step="0.01"
                   value="${s.bottom_z.toFixed(3)}" style="width:90px;font-size:12px"></td>
        <td><input type="number" class="slab-thick" data-idx="${i}" step="0.005" min="0.01"
                   value="${s.thickness.toFixed(3)}" style="width:80px;font-size:12px"></td>
        <td align="right" class="slab-top" data-idx="${i}">${s.top_z.toFixed(3)}</td>
      </tr>`).join('');
    extra.innerHTML = `
      <div class="stage-preview-row">
        <div>
          <img id="slabs-z-hist" src="/api/jobs/${wizard.jobId}/z_histogram.png?t=${Date.now()}" alt="Z-histogram">
        </div>
        <div style="padding:12px">
          <div style="font-weight:600;margin-bottom:8px">Identifierade bjälklag (${data.slabs.length})</div>
          <table style="width:100%;font-size:12px;border-collapse:collapse">
            <thead><tr style="color:var(--text-dim)">
              <th align="left">Behåll</th><th align="left">#</th>
              <th align="left">Botten (m)</th><th align="left">Tjocklek (m)</th>
              <th align="right">Topp (m)</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>
          <div style="margin-top:10px;font-size:11px;color:var(--text-dim)">
            ${data.peak_z.length} Z-toppar hittades. Justera botten/tjocklek vid behov,
            avmarkera bjälklag som inte ska räknas, och klicka "Tillämpa".
            Du måste behålla minst 2 bjälklag (golv + tak) för att väggdetekteringen ska kunna köra.
          </div>
          <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap">
            <button class="btn btn-outline" id="btn-apply-slab-select">Tillämpa</button>
            <span id="slab-select-status" style="font-size:12px;color:var(--text-dim);align-self:center"></span>
          </div>
        </div>
      </div>`;
    document.getElementById('btn-apply-slab-select').onclick = applySlabSelection;
    extra.querySelectorAll('.slab-bottom, .slab-thick').forEach(inp => {
      inp.addEventListener('input', () => {
        const idx = parseInt(inp.dataset.idx, 10);
        const b = parseFloat(extra.querySelector(`.slab-bottom[data-idx="${idx}"]`).value) || 0;
        const t = parseFloat(extra.querySelector(`.slab-thick[data-idx="${idx}"]`).value) || 0;
        const topEl = extra.querySelector(`.slab-top[data-idx="${idx}"]`);
        if (topEl) topEl.textContent = (b + t).toFixed(3);
      });
    });
  } catch (e) {
    extra.innerHTML = '<div class="alert alert-danger">Kunde inte ladda bjälklagsdata: ' + e.message + '</div>';
  }
}

async function applySlabSelection() {
  const checks = document.querySelectorAll('.slab-keep');
  const keep = Array.from(checks).filter(c => c.checked).map(c => parseInt(c.dataset.idx, 10));
  const status = document.getElementById('slab-select-status');
  if (keep.length < 2) {
    status.style.color = 'var(--danger)';
    status.textContent = '✗ Behåll minst 2 bjälklag.';
    return;
  }
  status.style.color = 'var(--text-dim)';
  status.textContent = 'Uppdaterar…';
  try {
    // Apply per-slab bottom/thickness edits first (across all slabs, not
    // just the kept ones — keeps indices stable for the next call).
    const edits = Array.from(document.querySelectorAll('.slab-bottom')).map(inp => {
      const idx = parseInt(inp.dataset.idx, 10);
      const bottomEl = document.querySelector(`.slab-bottom[data-idx="${idx}"]`);
      const thickEl  = document.querySelector(`.slab-thick[data-idx="${idx}"]`);
      return {
        idx,
        bottom_z: bottomEl ? parseFloat(bottomEl.value) : null,
        thickness: thickEl ? parseFloat(thickEl.value) : null,
      };
    });
    await fetch('/api/jobs/' + wizard.jobId + '/slabs/edit', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ edits }),
    });
    const res = await fetch('/api/jobs/' + wizard.jobId + '/slabs/select', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ keep_indices: keep }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const r = await res.json();
    status.style.color = 'var(--success)';
    status.textContent = `✓ ${r.total_after} av ${r.total_before} bjälklag kvar.`;
    // Re-render review so the table + histogram + bands reflect the new state
    wizard.slabsData = null;
    wizard.bands = [];
    setTimeout(renderSlabsReview, 300);
  } catch (e) {
    status.style.color = 'var(--danger)';
    status.textContent = '✗ ' + e.message;
  }
}

async function renderWallsReview() {
  const extra = document.getElementById('stage-extra');
  // Make sure we have slab data — needed for default bands
  if (!wizard.slabsData) {
    try {
      wizard.slabsData = await fetch('/api/jobs/' + wizard.jobId + '/slabs').then(r => r.json());
      wizard.slabCount = wizard.slabsData.slabs.length;
      if (wizard.bands.length === 0) {
        for (let i = 0; i < wizard.slabsData.slabs.length - 1; i++) {
          const floor = wizard.slabsData.slabs[i].top_z;
          wizard.bands.push({ z_min: floor + 1.30, z_max: floor + 1.60 });
          wizard.bands_lower.push({ z_min: floor + 0.30, z_max: floor + 0.35 });
        }
      }
    } catch (e) { /* ignore — handled below */ }
  }
  // Backfill lower bands if the user advanced through slabs before this
  // feature existed.
  while (wizard.bands_lower.length < wizard.bands.length) {
    const i = wizard.bands_lower.length;
    const floor = wizard.slabsData ? wizard.slabsData.slabs[i].top_z : (wizard.bands[i].z_min - 1.30);
    wizard.bands_lower.push({ z_min: floor + 0.30, z_max: floor + 0.35 });
  }
  const numStoreys = Math.max(0, wizard.slabCount - 1);
  extra.innerHTML = `
    <div class="stage-preview-row" style="margin-bottom:12px">
      <div style="flex:0 0 320px">
        <img id="walls-z-hist" src="/api/jobs/${wizard.jobId}/z_histogram.png?t=${Date.now()}" alt="Z-histogram">
      </div>
      <div style="padding:12px;font-size:12px;line-height:1.55">
        <div style="font-weight:600;font-size:13px;margin-bottom:6px">Två horisontella tvärsnitt per våning</div>
        <p style="color:var(--text-dim);margin-bottom:8px">
          <strong>Vägg-snitt:</strong> ca 130–160 cm över golvet — används för väggdetektering.<br>
          <strong>Lågsnitt:</strong> 30–35 cm över golvet — under fönsterhöjd, hjälper dig
          se vilka "väggar" som egentligen är fönster (fönster försvinner i lågsnittet).
        </p>
        <div style="color:var(--text-dim)">${numStoreys} våning${numStoreys === 1 ? '' : 'ar'} hittade.</div>
      </div>
    </div>
    <div id="storey-bands"></div>
    <div id="walls-dxf-section" style="margin-top:14px"></div>`;
  renderWallsDxfButtons();
  const list = document.getElementById('storey-bands');
  if (numStoreys === 0) {
    list.innerHTML = '<div class="alert alert-danger">Inga våningar — bjälklagsdetektionen gav färre än 2 bjälklag. Kör om "Bjälklag"-steget med andra inställningar.</div>';
    return;
  }
  for (let i = 0; i < numStoreys; i++) {
    const band = wizard.bands[i] || { z_min: 0, z_max: 1 };
    const bandLo = wizard.bands_lower[i] || { z_min: 0, z_max: 0.05 };
    const row = document.createElement('div');
    row.className = 'storey-band';
    row.innerHTML = `
      <label>Våning ${i}</label>
      <div style="display:grid;grid-template-columns:auto auto auto auto auto auto;gap:8px;align-items:end;flex:1">
        <div style="grid-column:1 / span 2;font-size:11px;color:var(--text-dim);font-weight:600;margin-top:2px">Vägg-snitt</div>
        <div style="grid-column:3 / span 2;font-size:11px;color:var(--text-dim);font-weight:600;margin-top:2px">Lågsnitt (fönsterkoll)</div>
        <div></div>
        <div><label style="display:block;font-size:11px">Z min</label>
          <input type="number" step="0.05" class="band-min" value="${band.z_min.toFixed(2)}" data-storey="${i}" style="width:70px"></div>
        <div><label style="display:block;font-size:11px">Z max</label>
          <input type="number" step="0.05" class="band-max" value="${band.z_max.toFixed(2)}" data-storey="${i}" style="width:70px"></div>
        <div><label style="display:block;font-size:11px">Z min</label>
          <input type="number" step="0.05" class="band-lo-min" value="${bandLo.z_min.toFixed(2)}" data-storey="${i}" style="width:70px"></div>
        <div><label style="display:block;font-size:11px">Z max</label>
          <input type="number" step="0.05" class="band-lo-max" value="${bandLo.z_max.toFixed(2)}" data-storey="${i}" style="width:70px"></div>
        <button class="btn btn-outline" data-storey="${i}" data-action="preview">Uppdatera</button>
      </div>`;
    list.appendChild(row);
    // Two preview images side by side per storey
    const previewBox = document.createElement('div');
    previewBox.id = 'cross-section-preview-row-' + i;
    previewBox.style.display = 'grid';
    previewBox.style.gridTemplateColumns = '1fr 1fr';
    previewBox.style.gap = '8px';
    previewBox.style.marginTop = '4px';
    previewBox.innerHTML = `
      <div id="cross-section-preview-${i}-upper"></div>
      <div id="cross-section-preview-${i}-lower"></div>`;
    list.appendChild(previewBox);
    renderCrossSection(i, band.z_min, band.z_max, 'upper');
    renderCrossSection(i, bandLo.z_min, bandLo.z_max, 'lower');
  }
  const debouncers = {};
  let histDebouncer = null;
  list.addEventListener('input', e => {
    const storey = parseInt(e.target.dataset.storey, 10);
    if (Number.isNaN(storey)) return;
    let touched = null;
    if (e.target.classList.contains('band-min')) { wizard.bands[storey].z_min = parseFloat(e.target.value); touched = 'upper'; }
    else if (e.target.classList.contains('band-max')) { wizard.bands[storey].z_max = parseFloat(e.target.value); touched = 'upper'; }
    else if (e.target.classList.contains('band-lo-min')) { wizard.bands_lower[storey].z_min = parseFloat(e.target.value); touched = 'lower'; }
    else if (e.target.classList.contains('band-lo-max')) { wizard.bands_lower[storey].z_max = parseFloat(e.target.value); touched = 'lower'; }
    const key = `${storey}-${touched}`;
    if (debouncers[key]) clearTimeout(debouncers[key]);
    debouncers[key] = setTimeout(() => {
      const src = touched === 'lower' ? wizard.bands_lower[storey] : wizard.bands[storey];
      renderCrossSection(storey, src.z_min, src.z_max, touched);
    }, 600);
    if (histDebouncer) clearTimeout(histDebouncer);
    histDebouncer = setTimeout(refreshHistogramWithBands, 500);
  });
  list.addEventListener('click', async e => {
    if (e.target.dataset.action !== 'preview') return;
    const storey = parseInt(e.target.dataset.storey, 10);
    const band = wizard.bands[storey];
    const bandLo = wizard.bands_lower[storey];
    await Promise.all([
      renderCrossSection(storey, band.z_min, band.z_max, 'upper'),
      renderCrossSection(storey, bandLo.z_min, bandLo.z_max, 'lower'),
    ]);
    refreshHistogramWithBands();
  });
}

async function renderWallsDxfButtons() {
  const wrap = document.getElementById('walls-dxf-section');
  if (!wrap) return;
  try {
    const res = await fetch('/api/jobs/' + wizard.jobId + '/dxf/storeys');
    if (!res.ok) { wrap.innerHTML = ''; return; }
    const { storeys } = await res.json();
    if (!storeys) { wrap.innerHTML = ''; return; }
    const buttons = [];
    for (let i = 0; i < storeys; i++) {
      buttons.push(`<a class="btn btn-outline" href="/api/jobs/${wizard.jobId}/dxf/${i}" download>⬇ Plan våning ${i} (.dxf)</a>`);
    }
    wrap.innerHTML = `
      <div style="padding:12px;background:var(--surface2);border-radius:8px">
        <div style="font-weight:600;margin-bottom:6px">Exportera 2D-plan (DXF) — väggar + bjälklagskontur</div>
        <div style="font-size:12px;color:var(--text-dim);margin-bottom:10px">
          En fil per våning. Öppningar/pelare/trappor läggs till om de detekterats senare.
        </div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">${buttons.join('')}</div>
      </div>`;
  } catch (e) { wrap.innerHTML = ''; }
}

async function refreshHistogramWithBands() {
  const img = document.getElementById('walls-z-hist');
  if (!img) return;
  try {
    const bands = wizard.bands.map(b => b ? [b.z_min, b.z_max] : null);
    const bands_lower = wizard.bands_lower.map(b => b ? [b.z_min, b.z_max] : null);
    const res = await fetch('/api/jobs/' + wizard.jobId + '/z_histogram.png', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bands, bands_lower }),
    });
    if (!res.ok) return;
    const blob = await res.blob();
    img.src = URL.createObjectURL(blob);
  } catch (e) { /* histogram update is cosmetic */ }
}

async function renderCrossSection(storey, zMin, zMax, kind) {
  // kind = 'upper' (wall section) or 'lower' (window-check section).
  // Backward compat: when omitted, falls back to the legacy element id.
  const id = kind ? `cross-section-preview-${storey}-${kind}` : `cross-section-preview-${storey}`;
  const preview = document.getElementById(id);
  if (!preview) return;
  const label = kind === 'lower' ? 'Lågsnitt' : (kind === 'upper' ? 'Vägg-snitt' : 'Snitt');
  preview.innerHTML = '<div style="padding:30px;text-align:center;color:var(--text-dim);font-size:12px">Renderar ' + label.toLowerCase() + '…</div>';
  try {
    const res = await fetch('/api/jobs/' + wizard.jobId + '/cross_section_preview', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ storey_idx: storey, z_min: zMin, z_max: zMax }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    preview.innerHTML = `
      <div>
        <div style="padding:6px 10px;background:var(--surface2);border-bottom:1px solid var(--border);font-size:11px;color:var(--text-dim)">
          ${label} våning ${storey}: Z = ${zMin.toFixed(2)}–${zMax.toFixed(2)} m
        </div>
        <img src="${url}" alt="${label} våning ${storey}" style="width:100%">
      </div>`;
  } catch (e) {
    preview.innerHTML = '<div class="alert alert-danger">Kunde inte rendera snittet: ' + e.message + '</div>';
  }
}

function renderIfcReview() {
  const extra = document.getElementById('stage-extra');
  extra.innerHTML = `
    <div class="alert alert-success">IFC och planlösningspreview genererade. Klicka "Fortsätt →" för att gå till resultat-sidan.</div>
    <div class="stage-preview-row">
      <div><img src="/api/jobs/${wizard.jobId}/preview?t=${Date.now()}" alt="Planlösning"></div>
    </div>
    <div id="dxf-export-row" style="margin-top:14px;padding:12px;background:var(--surface2);border-radius:8px">
      <div style="font-weight:600;margin-bottom:6px">Exportera 2D-plan (DXF)</div>
      <div style="font-size:12px;color:var(--text-dim);margin-bottom:10px">
        En fil per våning. Lager: SLAB_OUTLINE, WALLS (väggens ytterkant), WALL_AXIS,
        WINDOWS, DOORS, COLUMNS, STAIRS.
      </div>
      <div id="dxf-buttons" style="display:flex;gap:8px;flex-wrap:wrap"></div>
    </div>`;
  document.getElementById('btn-stage-continue').textContent = 'Visa resultat →';
  document.getElementById('btn-stage-continue').onclick = () => goTo(4);
  renderDxfButtons();
}

async function renderDxfButtons() {
  const wrap = document.getElementById('dxf-buttons');
  if (!wrap) return;
  try {
    const res = await fetch('/api/jobs/' + wizard.jobId + '/dxf/storeys');
    if (!res.ok) {
      wrap.innerHTML = '<span style="font-size:12px;color:var(--text-dim)">Inga våningar att exportera.</span>';
      return;
    }
    const { storeys } = await res.json();
    if (!storeys) {
      wrap.innerHTML = '<span style="font-size:12px;color:var(--text-dim)">Inga våningar att exportera.</span>';
      return;
    }
    const buttons = [];
    for (let i = 0; i < storeys; i++) {
      buttons.push(`<a class="btn btn-outline" href="/api/jobs/${wizard.jobId}/dxf/${i}" download>⬇ Våning ${i} (.dxf)</a>`);
    }
    wrap.innerHTML = buttons.join('');
  } catch (e) {
    wrap.innerHTML = '<span style="font-size:12px;color:var(--danger)">Kunde inte hämta DXF-info: ' + e.message + '</span>';
  }
}

function wizardOpenStageOverrides(stage) {
  const extra = document.getElementById('stage-extra');
  let formHtml = '';
  if (stage === 'slabs') {
    const v = id => (document.getElementById(id) || {}).value || '';
    formHtml = `
      <div class="form-grid" style="margin-top:12px">
        <div class="form-group">
          <label>Max slab-tjocklek (m) — toppar närmare än detta paras</label>
          <input type="number" id="ovr-max-slab" value="0.5" step="0.05">
        </div>
        <div class="form-group">
          <label>Peak höjd-tröskel (0–1)</label>
          <input type="number" id="ovr-peak-ratio" value="0.25" step="0.05" min="0.05" max="1">
        </div>
        <div class="form-group">
          <label>Z-steg i histogram (m)</label>
          <input type="number" id="ovr-z-step" value="0.15" step="0.05" min="0.05">
        </div>
        <div class="form-group">
          <label>Golvbjälklag-tjocklek (m, default)</label>
          <input type="number" id="ovr-bfs" value="${v('bfs-thickness') || '0.3'}" step="0.05">
        </div>
        <div class="form-group">
          <label>Takbjälklag-tjocklek (m, default)</label>
          <input type="number" id="ovr-tfs" value="${v('tfs-thickness') || '0.4'}" step="0.05">
        </div>
      </div>`;
  } else if (stage === 'walls') {
    formHtml = `
      <div class="form-grid" style="margin-top:12px">
        <div class="form-group"><label>Min vägglängd (m)</label><input type="number" id="ovr-min-wl" value="0.10" step="0.05"></div>
        <div class="form-group"><label>Min väggtjocklek (m)</label><input type="number" id="ovr-min-wt" value="0.05" step="0.01"></div>
        <div class="form-group"><label>Max väggtjocklek (m)</label><input type="number" id="ovr-max-wt" value="0.75" step="0.05"></div>
        <div class="form-group"><label>Yttervägg-tjocklek (m)</label><input type="number" id="ovr-ext-wt" value="0.3" step="0.05"></div>
        <div class="form-group"><label>Max väggar per våning (cap)</label><input type="number" id="ovr-max-walls" value="300" min="1"></div>
      </div>
      <div style="margin-top:10px;padding:10px;background:var(--surface);border-radius:6px;font-size:12px">
        <label style="display:flex;gap:8px;align-items:center">
          <input type="checkbox" id="ovr-lower-support">
          <span>
            <strong>Filtrera bort väggar utan stöd i lågsnittet</strong> —
            droppar väggar som detekteras i vägg-snittet men saknar punkter
            i lågsnittet (30–35 cm). Tar typiskt bort fönster som tolkats
            som väggar.
          </span>
        </label>
        <div style="margin-top:8px;padding-left:24px">
          <label style="font-size:11px;color:var(--text-dim)">Min andel av väggens längd som måste finnas i lågsnittet (0–1):</label>
          <input type="number" id="ovr-lower-frac" value="0.30" step="0.05" min="0" max="1" style="width:80px">
        </div>
      </div>
      <div style="font-size:11px;color:var(--text-dim);margin-top:8px">
        Eventuella Z-band du satt ovan inkluderas automatiskt vid rerun. Höj
        "Max väggar per våning" om planlösningen har många rumsindelningar.
      </div>`;
  } else {
    formHtml = '<div class="alert alert-info">Inga parametrar att justera för det här steget. Klicka "Kör om" för att köra det igen som det är.</div>';
  }
  extra.innerHTML = `
    <div style="background:var(--surface2);padding:14px;border-radius:8px;margin-bottom:12px">
      <div style="font-weight:600;margin-bottom:6px">Nya inställningar för "${STAGE_INFO[stage].title}"</div>
      ${formHtml}
      <div class="btn-row" style="margin-top:14px">
        <button class="btn btn-outline" id="btn-cancel-ovr">Avbryt</button>
        <button class="btn btn-primary" id="btn-confirm-ovr">▶ Kör om med dessa inställningar</button>
      </div>
    </div>`;
  document.getElementById('btn-cancel-ovr').onclick = () => renderWizardStageReview(stage, false);
  document.getElementById('btn-confirm-ovr').onclick = () => wizardRunStage(stage);
}

async function wizardRunStage(stage) {
  const overrides = { stage };
  const num = id => {
    const el = document.getElementById(id);
    if (!el || el.value === '') return null;
    const v = parseFloat(el.value);
    return Number.isNaN(v) ? null : v;
  };
  // Slab overrides
  const sBfs = num('ovr-bfs'); if (sBfs !== null) overrides.bfs_thickness = sBfs;
  const sTfs = num('ovr-tfs'); if (sTfs !== null) overrides.tfs_thickness = sTfs;
  const sMax = num('ovr-max-slab'); if (sMax !== null) overrides.max_slab_thickness = sMax;
  const sPeak = num('ovr-peak-ratio'); if (sPeak !== null) overrides.slab_peak_height_ratio = sPeak;
  const sZ = num('ovr-z-step'); if (sZ !== null) overrides.slab_z_step = sZ;
  // Wall overrides
  const wMinL = num('ovr-min-wl'); if (wMinL !== null) overrides.min_wall_length = wMinL;
  const wMinT = num('ovr-min-wt'); if (wMinT !== null) overrides.min_wall_thickness = wMinT;
  const wMaxT = num('ovr-max-wt'); if (wMaxT !== null) overrides.max_wall_thickness = wMaxT;
  const wExtT = num('ovr-ext-wt'); if (wExtT !== null) overrides.exterior_walls_thickness = wExtT;
  const wMaxN = num('ovr-max-walls'); if (wMaxN !== null) overrides.max_walls_per_storey = Math.round(wMaxN);
  // Cross-section bands (for walls stage)
  if (stage === 'walls' && wizard.bands.length > 0) {
    overrides.cross_section_bands = wizard.bands.map(b => b ? [b.z_min, b.z_max] : null);
    overrides.cross_section_bands_lower = wizard.bands_lower.map(b => b ? [b.z_min, b.z_max] : null);
  }
  // Low-section support filter toggle
  if (stage === 'walls') {
    const lowEl = document.getElementById('ovr-lower-support');
    if (lowEl) overrides.require_lower_support = !!lowEl.checked;
    const lowFrac = num('ovr-lower-frac');
    if (lowFrac !== null) overrides.lower_support_fraction = lowFrac;
  }

  document.getElementById('wizard-stage-detail').innerHTML =
    '<div class="alert alert-info">Kör steg "' + STAGE_INFO[stage].title + '"…</div>';
  wizardSetBadge('running');

  const res = await fetch('/api/jobs/' + wizard.jobId + '/run_stage', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(overrides),
  });
  if (!res.ok) {
    wizardLog('[ERROR] Kunde inte starta steget');
    return;
  }
  if (!wizard.sse) wizardStreamLogs();
}

document.getElementById('btn-wizard-start').addEventListener('click', wizardStart);
// Note: the .collapsible-header global handler (registered in Step 2 block)
// already wires the "Loggar" toggle. A second listener here would double-
// toggle and leave the panel in its starting state.

// ── Init ──────────────────────────────────────────────────────────────────
goTo(1);
