const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('message');

const BOT_NAME = 'Wise Old AI';
const BOT_AVATAR_SRC = '/static/Wise_Old_Man_chathead.png';

function createRow(kind) {
  const row = document.createElement('div');
  row.className = 'msg-row' + (kind === 'me' ? ' me' : '');

  const avatar = document.createElement('div');
  avatar.className = 'avatar';
  if (kind === 'bot') {
    avatar.innerHTML = `<img src="${BOT_AVATAR_SRC}" alt="${BOT_NAME}" />`;
  } else {
    // Simple silhouette for the user
    avatar.innerHTML = `<svg viewBox="0 0 24 24" width="42" height="42" aria-hidden="true" style="opacity:0.9;display:block;">
      <path fill="rgba(243,210,122,0.85)" d="M12 12a5 5 0 1 0-5-5 5 5 0 0 0 5 5zm0 2c-4.42 0-8 2.24-8 5v1h16v-1c0-2.76-3.58-5-8-5z"/>
    </svg>`;
  }

  const bubble = document.createElement('div');
  bubble.className = 'msg' + (kind === 'me' ? ' me' : '');
  bubble.innerHTML = `<div class="role"></div><div class="text"></div>`;

  row.appendChild(avatar);
  row.appendChild(bubble);
  return { row, bubble };
}

function addMessage(kind, roleLabel, text) {
  const { row, bubble } = createRow(kind);
  bubble.querySelector('.role').textContent = roleLabel;
  bubble.querySelector('.text').textContent = text;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function addTyping() {
  const { row, bubble } = createRow('bot');
  bubble.querySelector('.role').textContent = BOT_NAME;
  bubble.querySelector('.text').innerHTML = `
    <span class="typing" aria-label="${BOT_NAME} is responding">
      <span class="orb" aria-hidden="true"></span>
      <span>Conjuring…</span>
      <span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>
    </span>
  `;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

addMessage('bot', BOT_NAME, 'Ask me anything about OSRS.');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  addMessage('me', 'You', msg);

  const typingBubble = addTyping();

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
    typingBubble.querySelector('.text').textContent = data.answer || '(no answer)';

    if (data.sources && data.sources.length) {
      const srcText = data.sources.map((s, i) => `Source ${i+1}: ${s.title || ''} ${s.url}`).join('\n');
      addMessage('bot', 'Sources', srcText);
    }
  } catch (err) {
    typingBubble.querySelector('.text').textContent = err.message || String(err);
    typingBubble.querySelector('.role').textContent = 'Error';
  }
});
