/**
 * IFC viewer using xeokit-sdk.
 * Assets are downloaded at Docker build time and served locally from /static/vendor/.
 */

const XEOKIT_PATH = "/static/vendor/xeokit-sdk.es.js";
const WEBIFC_WASM_PATH = "/static/vendor/";

let viewer = null;

export async function loadViewer(jobId) {
  const container = document.getElementById('viewer-container');
  if (!container) return;

  const loadingEl = document.getElementById('viewer-loading');

  let xeokit;
  try {
    xeokit = await import(XEOKIT_PATH);
  } catch (e) {
    console.error('xeokit load failed:', e);
    if (loadingEl) {
      loadingEl.innerHTML = `
        <div style="color:var(--danger);text-align:center;padding:24px;max-width:400px">
          <div style="font-size:24px;margin-bottom:8px">⚠️</div>
          <div style="font-weight:600;margin-bottom:6px">Viewer kunde inte laddas</div>
          <div style="font-size:12px;color:var(--text-dim)">${e.message || e}</div>
          <div style="margin-top:12px;font-size:12px">
            Ladda ned IFC-filen och öppna i t.ex. Revit, FreeCAD eller BIMvision.
          </div>
        </div>`;
    }
    return;
  }

  const { Viewer, WebIFCLoaderPlugin, AmbientLight, DirLight } = xeokit;

  viewer = new Viewer({
    canvasId: "viewer-canvas",
    transparent: true,
  });

  viewer.camera.eye  = [10, 10, 10];
  viewer.camera.look = [0, 0, 0];
  viewer.camera.up   = [0, 1, 0];

  new AmbientLight(viewer.scene, { color: [1, 1, 1], intensity: 0.3 });
  new DirLight(viewer.scene,     { dir: [-0.5, -0.8, -0.4], color: [1, 1, 0.95], intensity: 0.7, space: "world" });
  new DirLight(viewer.scene,     { dir: [0.5, 0.2, 0.8],    color: [0.8, 0.9, 1], intensity: 0.3, space: "world" });

  // web-ifc-api.js is loaded as a plain <script> tag — access via window.WebIFC
  const WebIFC = window.WebIFC;
  if (!WebIFC) {
    throw new Error("window.WebIFC ej tillgänglig — web-ifc-api.js laddades inte");
  }

  const ifcLoader = new WebIFCLoaderPlugin(viewer, {
    WebIFC,
    wasmPath: WEBIFC_WASM_PATH,
  });

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
    if (loadingEl) {
      loadingEl.innerHTML =
        `<div style="color:var(--danger)">Fel vid laddning av IFC: ${err}</div>`;
    }
  });

  // Toolbar
  document.getElementById('btn-fit')?.addEventListener('click', () => {
    viewer.cameraFlight.flyTo(model);
  });
  document.getElementById('btn-reset-view')?.addEventListener('click', () => {
    viewer.scene.setObjectsVisible(viewer.scene.objectIds, true);
  });
  document.getElementById('btn-section')?.addEventListener('click', () => {
    const existing = viewer.scene.sectionPlanes["section-x"];
    if (existing) {
      existing.destroy();
    } else {
      viewer.scene.createSectionPlane({ id: "section-x", pos: [0, 0, 0], dir: [-1, 0, 0] });
    }
  });
}
