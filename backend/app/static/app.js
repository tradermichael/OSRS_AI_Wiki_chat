const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('message');

function addMessage(role, text) {
  const el = document.createElement('div');
  el.className = 'msg';
  el.innerHTML = `<div class="role">${role}</div><div class="text"></div>`;
  el.querySelector('.text').textContent = text;
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

addMessage('system', 'Ask a question. If you have not ingested content yet, answers may be empty.');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  addMessage('you', msg);

  try {
    const r = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: msg })
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const data = await r.json();
    addMessage('bot', data.answer || '(no answer)');

    if (data.sources && data.sources.length) {
      const srcText = data.sources.map((s, i) => `Source ${i+1}: ${s.title || ''} ${s.url}`).join('\n');
      addMessage('sources', srcText);
    }
  } catch (err) {
    addMessage('error', err.message || String(err));
  }
});
