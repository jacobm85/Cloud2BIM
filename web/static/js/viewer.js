/**
 * IFC viewer using xeokit-sdk (ES module from CDN).
 * CDN requires internet access. For offline/intranet-only deployments,
 * download the bundles and adjust the import paths below.
 */

const XEOKIT_CDN = "https://cdn.jsdelivr.net/npm/@xeokit/xeokit-sdk@2.6.33/dist/xeokit-sdk.es.js";
const WEBIFC_CDN_WASM = "https://cdn.jsdelivr.net/npm/web-ifc@0.0.51/";

let viewer = null;

export async function loadViewer(jobId) {
  const container = document.getElementById('viewer-container');
  if (!container) return;

  const loadingEl = document.getElementById('viewer-loading');

  // Dynamically import xeokit so the rest of the page loads without it
  let xeokit;
  try {
    xeokit = await import(XEOKIT_CDN);
  } catch (e) {
    if (loadingEl) {
      loadingEl.innerHTML = `
        <div style="color:var(--danger);text-align:center;padding:24px">
          <div style="font-size:24px;margin-bottom:8px">⚠️</div>
          <div>Kunde inte ladda IFC-viewer.<br>
          Kontrollera internetåtkomst eller installera lokalt.</div>
        </div>`;
    }
    console.error('xeokit load failed', e);
    return;
  }

  const { Viewer, WebIFCLoaderPlugin, AmbientLight, DirLight, NavCubePlugin } = xeokit;

  viewer = new Viewer({
    canvasId: "viewer-canvas",
    transparent: true,
  });

  viewer.camera.eye = [10, 10, 10];
  viewer.camera.look = [0, 0, 0];
  viewer.camera.up = [0, 1, 0];

  new AmbientLight(viewer.scene, { color: [1, 1, 1], intensity: 0.3 });
  new DirLight(viewer.scene, { dir: [-0.5, -0.8, -0.4], color: [1, 1, 0.95], intensity: 0.7 });
  new DirLight(viewer.scene, { dir: [0.5, 0.2, 0.8],   color: [0.8, 0.9, 1],  intensity: 0.3 });

  if (NavCubePlugin) {
    new NavCubePlugin(viewer, { canvasId: "viewer-canvas", visible: true, size: 120 });
  }

  const ifcLoader = new WebIFCLoaderPlugin(viewer, {
    wasmPath: WEBIFC_CDN_WASM,
  });

  if (loadingEl) loadingEl.innerHTML = `<div class="spinner"></div><div>Laddar IFC-modell…</div>`;

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
    console.error("IFC load error", err);
    if (loadingEl) {
      loadingEl.innerHTML = `<div style="color:var(--danger)">Fel vid laddning av IFC: ${err}</div>`;
    }
  });

  // Toolbar buttons
  document.getElementById('btn-fit')?.addEventListener('click', () => {
    viewer.cameraFlight.flyTo(model);
  });
  document.getElementById('btn-reset-view')?.addEventListener('click', () => {
    viewer.scene.setObjectsVisible(viewer.scene.objectIds, true);
  });
  document.getElementById('btn-section')?.addEventListener('click', () => {
    // Toggle a simple X section plane
    const sp = viewer.scene.sectionPlanes;
    const existing = Object.values(sp).find(p => p.id === 'section-x');
    if (existing) {
      existing.destroy();
    } else {
      viewer.scene.createSectionPlane({
        id: "section-x",
        pos: [0, 0, 0],
        dir: [-1, 0, 0],
      });
    }
  });
}
