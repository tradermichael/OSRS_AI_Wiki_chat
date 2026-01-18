const statusEl = document.getElementById('status');

function setStatus(text) {
  statusEl.textContent = text;
}

(async () => {
  const params = new URLSearchParams(window.location.search);
  const token = params.get('token'); // PayPal order id

  if (!token) {
    setStatus('Missing PayPal token in URL.');
    return;
  }

  setStatus('Capturing order ' + token + ' …');

  try {
    const r = await fetch('/api/paypal/capture-order?order_id=' + encodeURIComponent(token), {
      method: 'POST'
    });

    const data = await r.json().catch(() => ({}));

    if (!r.ok) {
      throw new Error(data.detail || `HTTP ${r.status}`);
    }

    setStatus(`Captured: ${data.status}\nOrder: ${data.order_id}\nPayer: ${data.payer_email || ''}`);
  } catch (err) {
    setStatus('Capture failed: ' + (err.message || String(err)));
  }
})();
