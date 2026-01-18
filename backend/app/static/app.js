const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('message');
const historyListEl = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistory');
const viewBannerEl = document.getElementById('viewBanner');
const viewBannerTextEl = document.getElementById('viewBannerText');
const resumeChatBtn = document.getElementById('resumeChat');

const BOT_NAME = 'Wise Old AI';
const BOT_AVATAR_SRC = '/static/Wise_Old_Man_chathead.png';

let liveChatSnapshot = null;
let viewingHistoryId = null;

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

function setComposerEnabled(enabled) {
  input.disabled = !enabled;
  const btn = form.querySelector('button[type="submit"]') || form.querySelector('button');
  if (btn) btn.disabled = !enabled;
}

function enterHistoryView(item) {
  if (!liveChatSnapshot) {
    liveChatSnapshot = {
      html: messagesEl.innerHTML,
      scrollTop: messagesEl.scrollTop,
    };
  }

  viewingHistoryId = item.id;
  messagesEl.innerHTML = '';

  const dt = new Date(item.created_at);
  const ts = Number.isNaN(dt.getTime()) ? item.created_at : dt.toLocaleString();
  if (viewBannerTextEl) viewBannerTextEl.textContent = `Viewing #${item.id} • ${ts}`;
  if (viewBannerEl) viewBannerEl.hidden = false;
  setComposerEnabled(false);

  addMessage('me', 'You', item.user_message);
  addMessage('bot', BOT_NAME, item.bot_answer || '(no answer)');

  if (item.sources && item.sources.length) {
    const srcText = item.sources
      .map((s, i) => `Source ${i + 1}: ${s.title || ''} ${s.url || ''}`.trim())
      .join('\n');
    addMessage('bot', 'Sources', srcText);
  }
}

function exitHistoryView() {
  viewingHistoryId = null;
  if (viewBannerEl) viewBannerEl.hidden = true;
  setComposerEnabled(true);
  input.focus();

  if (liveChatSnapshot) {
    messagesEl.innerHTML = liveChatSnapshot.html;
    messagesEl.scrollTop = liveChatSnapshot.scrollTop || messagesEl.scrollHeight;
  } else {
    messagesEl.innerHTML = '';
    addMessage('bot', BOT_NAME, 'Ask me anything about OSRS.');
  }
}

function renderHistory(items) {
  if (!historyListEl) return;
  historyListEl.innerHTML = '';

  if (!items || !items.length) {
    const empty = document.createElement('div');
    empty.className = 'history-meta';
    empty.textContent = 'No public chat history yet.';
    historyListEl.appendChild(empty);
    return;
  }

  for (const it of items) {
    const row = document.createElement('div');
    row.className = 'history-item';
    row.dataset.id = String(it.id);

    const meta = document.createElement('div');
    meta.className = 'history-meta';
    const dt = new Date(it.created_at);
    const ts = Number.isNaN(dt.getTime()) ? it.created_at : dt.toLocaleString();
    meta.textContent = `#${it.id} • ${ts}`;

    const q = document.createElement('div');
    q.className = 'history-q';
    const snippet = String(it.user_message || '').replace(/\s+/g, ' ').trim();
    q.textContent = snippet.length > 80 ? `${snippet.slice(0, 80)}…` : snippet;

    row.appendChild(meta);
    row.appendChild(q);

    row.addEventListener('click', async () => {
      try {
        const r = await fetch(`/api/history/${encodeURIComponent(it.id)}`);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const item = await r.json();
        enterHistoryView(item);
      } catch {
        // fallback: use list item if detail fails
        enterHistoryView(it);
      }
    });

    historyListEl.appendChild(row);
  }
}

async function loadHistory() {
  if (!historyListEl) return;
  try {
    const r = await fetch('/api/history?limit=50&offset=0');
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const data = await r.json();
    renderHistory((data && data.items) || []);
  } catch {
    // Don't break the chat UI if history is unavailable.
  }
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

if (refreshHistoryBtn) {
  refreshHistoryBtn.addEventListener('click', () => {
    loadHistory();
  });
}

if (resumeChatBtn) {
  resumeChatBtn.addEventListener('click', () => {
    exitHistoryView();
  });
}

// Load initial history once the page is ready.
document.addEventListener('DOMContentLoaded', () => {
  loadHistory();
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (viewingHistoryId) {
    exitHistoryView();
    return;
  }
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
  } finally {
    // Since every chat is publicly stored, refresh the sidebar.
    loadHistory();
  }
});
