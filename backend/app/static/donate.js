const form = document.getElementById('donateForm');
const amountEl = document.getElementById('amount');
const noteEl = document.getElementById('note');

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const amount = (amountEl.value || '').trim();
  const note = (noteEl.value || '').trim();

  const r = await fetch('/api/paypal/create-order', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount_usd: amount, note: note || null })
  });

  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    alert(err.detail || `HTTP ${r.status}`);
    return;
  }

  const data = await r.json();
  window.location.href = data.approve_url;
});
