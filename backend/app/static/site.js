async function refreshGoldTotal() {
  const el = document.getElementById('goldTotal');
  if (!el) return;

  try {
    const r = await fetch('/api/gold/total');
    if (!r.ok) return;
    const data = await r.json();
    const n = Number(data.total_gold || 0);
    el.textContent = n.toLocaleString();
  } catch {
    // ignore
  }
}

async function donateGold(amountGold) {
  const r = await fetch('/api/gold/donate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ amount_gold: amountGold })
  });

  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }

  const data = await r.json();
  await refreshGoldTotal();
  return data;
}

window.refreshGoldTotal = refreshGoldTotal;
window.donateGold = donateGold;

document.addEventListener('DOMContentLoaded', () => {
  refreshGoldTotal();
});
