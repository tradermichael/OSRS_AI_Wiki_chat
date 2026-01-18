const form = document.getElementById('goldDonateForm');
const amountEl = document.getElementById('goldAmount');
const statusEl = document.getElementById('goldDonateStatus');

function setStatus(text) {
  if (!statusEl) return;
  statusEl.textContent = text;
}

function parseGold(value) {
  const cleaned = String(value || '').replace(/,/g, '').trim();
  const n = Number(cleaned);
  if (!Number.isFinite(n) || n <= 0) return null;
  return Math.floor(n);
}

document.querySelectorAll('[data-gold]').forEach((btn) => {
  btn.addEventListener('click', () => {
    amountEl.value = String(btn.getAttribute('data-gold') || '');
  });
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  const amountGold = parseGold(amountEl.value);
  if (!amountGold) {
    setStatus('Enter a valid gold amount.');
    return;
  }

  setStatus('Depositing gold into the coffer…');
  try {
    const data = await window.donateGold(amountGold);
    setStatus(`Thank you! Total donated: ${Number(data.total_gold || 0).toLocaleString()} gp`);
  } catch (err) {
    setStatus(err.message || String(err));
  }
});
