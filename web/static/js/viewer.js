/**
 * IFC viewer using xeokit-sdk.
 * Assets are downloaded at Docker build time into /static/vendor/.
 * web-ifc-api.js is an ES module — load via dynamic import(), not <script>.
 */

const XEOKIT_PATH    = "/static/vendor/xeokit-sdk.es.js";
const WEBIFC_API     = "/static/vendor/web-ifc-api.js";
const WEBIFC_WASM    = "/static/vendor/";

let viewer = null;

export async function loadViewer(jobId) {
  const loadingEl = document.getElementById('viewer-loading');

  function showError(msg) {
    if (loadingEl) {
      loadingEl.innerHTML = `
        <div style="color:var(--danger);text-align:center;padding:24px;max-width:420px">
          <div style="font-size:24px;margin-bottom:8px">⚠️</div>
          <div style="font-weight:600;margin-bottom:6px">Viewer kunde inte laddas</div>
          <div style="font-size:12px;color:var(--text-dim);word-break:break-all">${msg}</div>
          <div style="margin-top:12px;font-size:12px">
            Ladda ned IFC-filen och öppna i t.ex. BIMvision, FreeCAD eller Revit.
          </div>
        </div>`;
    }
  }

  // ── Load xeokit ──────────────────────────────────────────────────────────
  let xeokit;
  try {
    xeokit = await import(XEOKIT_PATH);
  } catch (e) {
    console.error('xeokit load failed:', e);
    showError('xeokit: ' + (e.message || e));
    return;
  }

  // ── Load web-ifc as ES module ────────────────────────────────────────────
  let WebIFC;
  try {
    const mod = await import(WEBIFC_API);
    // CJS modules imported dynamically expose exports under .default
    WebIFC = mod.default || mod;
    console.log('[viewer] WebIFC keys:', Object.keys(WebIFC).slice(0, 8));
  } catch (e) {
    console.error('web-ifc load failed:', e);
    showError('web-ifc: ' + (e.message || e));
    return;
  }

  const { Viewer, WebIFCLoaderPlugin, AmbientLight, DirLight } = xeokit;

  viewer = new Viewer({ canvasId: "viewer-canvas", transparent: true });
  viewer.camera.eye  = [10, 10, 10];
  viewer.camera.look = [0, 0, 0];
  viewer.camera.up   = [0, 1, 0];

  new AmbientLight(viewer.scene, { color: [1,1,1], intensity: 0.3 });
  new DirLight(viewer.scene, { dir: [-0.5,-0.8,-0.4], color: [1,1,0.95], intensity: 0.7, space: "world" });
  new DirLight(viewer.scene, { dir: [0.5,0.2,0.8],   color: [0.8,0.9,1], intensity: 0.3, space: "world" });

  let ifcLoader;
  try {
    ifcLoader = new WebIFCLoaderPlugin(viewer, {
      WebIFC,
      wasmPath: WEBIFC_WASM,
    });
  } catch (e) {
    console.error('WebIFCLoaderPlugin init failed:', e);
    showError('WebIFCLoaderPlugin: ' + (e.message || e));
    return;
  }

  if (loadingEl) {
    loadingEl.innerHTML = '<div class="spinner"></div><div>Laddar IFC-modell…</div>';
  }

  const model = ifcLoader.load({
    id: "bim-model",
    src: `/api/jobs/${jobId}/download`,
    loadMetadata: true,
    edges: true,
  });

  model.on("loaded", () => {
    if (loadingEl) loadingEl.style.display = 'none';
    viewer.cameraFlight.flyTo(model);
  });

  model.on("error", err => {
    console.error("IFC load error:", err);
    showError('IFC parsing: ' + err);
  });

  document.getElementById('btn-fit')?.addEventListener('click', () =>
    viewer.cameraFlight.flyTo(model));
  document.getElementById('btn-reset-view')?.addEventListener('click', () =>
    viewer.scene.setObjectsVisible(viewer.scene.objectIds, true));
  document.getElementById('btn-section')?.addEventListener('click', () => {
    const sp = viewer.scene.sectionPlanes;
    if (sp["section-x"]) sp["section-x"].destroy();
    else viewer.scene.createSectionPlane({ id: "section-x", pos: [0,0,0], dir: [-1,0,0] });
  });
}
