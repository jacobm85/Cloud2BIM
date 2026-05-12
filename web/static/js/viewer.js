/**
 * Results viewer — shows matplotlib floor plan preview from backend.
 * Exported as ES module so it can be imported in index.html.
 */

export async function loadViewer(jobId) {
  const loading = document.getElementById('preview-loading');
  const img     = document.getElementById('preview-img');
  if (!img) return;

  const url = `/api/jobs/${jobId}/preview`;

  // Poll up to 30 s for the preview PNG (pipeline generates it last)
  for (let i = 0; i < 30; i++) {
    const res = await fetch(url, { method: 'HEAD' });
    if (res.ok) {
      img.src = url + '?t=' + Date.now();  // cache-bust
      img.style.display = 'block';
      if (loading) loading.style.display = 'none';
      return;
    }
    await new Promise(r => setTimeout(r, 1000));
  }

  if (loading) loading.textContent = 'Förhandsgranskning ej tillgänglig.';
}
