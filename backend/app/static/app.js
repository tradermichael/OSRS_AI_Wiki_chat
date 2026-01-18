const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('message');
const historyListEl = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistory');
const toggleHistoryBtn = document.getElementById('toggleHistory');
const appLayoutEl = document.querySelector('.app-layout');
const viewBannerEl = document.getElementById('viewBanner');
const viewBannerTextEl = document.getElementById('viewBannerText');
const resumeChatBtn = document.getElementById('resumeChat');

const wikiPreviewEl = document.getElementById('wikiPreview');
const wikiPreviewTitleEl = document.getElementById('wikiPreviewTitle');
const wikiPreviewOpenEl = document.getElementById('wikiPreviewOpen');
const wikiPreviewCloseEl = document.getElementById('wikiPreviewClose');
const wikiPreviewThumbEl = document.getElementById('wikiPreviewThumb');
const wikiPreviewExtractEl = document.getElementById('wikiPreviewExtract');

const BOT_NAME = 'Wise Old AI';
const BOT_AVATAR_SRC = '/static/Wise_Old_Man_chathead.png';

let liveChatSnapshot = null;
let viewingHistoryId = null;

const SESSION_ID_KEY = 'osrs_session_id';
const LIVE_CHAT_KEY = 'osrs_live_chat_v1';
const HISTORY_SIDEBAR_KEY = 'osrs_history_sidebar_open';

function getSessionId() {
  let sid = localStorage.getItem(SESSION_ID_KEY);
  if (!sid) {
    sid = (crypto && crypto.randomUUID) ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2);
    localStorage.setItem(SESSION_ID_KEY, sid);
  }
  return sid;
}

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

function renderCitations(bubble, sources) {
  if (!sources || !sources.length) return;

  // De-dupe by URL (history entries may contain older duplicates).
  const seen = new Set();
  const deduped = [];
  for (const s of sources) {
    const url = (s && s.url) ? String(s.url) : '';
    if (!url) continue;
    if (seen.has(url)) continue;
    seen.add(url);
    deduped.push(s);
    if (deduped.length >= 6) break;
  }

  const box = document.createElement('div');
  box.className = 'citations';
  box.textContent = 'Sources: ';

  const thumbs = [];

  deduped.forEach((s, idx) => {
    const a = document.createElement('a');
    a.href = s.url;
    a.dataset.citationUrl = String(s.url);
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    const title = (s.title || s.url || '').trim();
    a.textContent = `[${idx + 1}] ${title || 'link'}`;
    box.appendChild(a);
    if (idx < deduped.length - 1) box.appendChild(document.createTextNode(' • '));

    const thumbUrl = (s && s.thumbnail_url) ? String(s.thumbnail_url) : '';
    if (thumbUrl) {
      thumbs.push({ url: s.url, thumb: thumbUrl, title: title || 'wiki image' });
    }
  });

  bubble.appendChild(box);

  if (thumbs.length) {
    const t = document.createElement('div');
    t.className = 'citation-thumbs';
    thumbs.slice(0, 3).forEach((it) => {
      const link = document.createElement('a');
      link.href = it.url;
      link.dataset.citationUrl = String(it.url);
      link.target = '_blank';
      link.rel = 'noopener noreferrer';

      const img = document.createElement('img');
      img.loading = 'lazy';
      img.alt = it.title;
      img.src = it.thumb;
      link.appendChild(img);
      t.appendChild(link);
    });
    bubble.appendChild(t);
  }
}

function setWikiPreviewVisible(visible) {
  if (!wikiPreviewEl) return;
  wikiPreviewEl.hidden = !visible;
}

function setWikiPreviewLoading(url) {
  if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = 'Loading…';
  if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = url || '#';
  if (wikiPreviewExtractEl) wikiPreviewExtractEl.textContent = 'Fetching preview from the wiki…';
  if (wikiPreviewThumbEl) {
    wikiPreviewThumbEl.hidden = true;
    wikiPreviewThumbEl.src = '';
  }
}

async function openWikiPreview(url) {
  const u = String(url || '').trim();
  if (!u) return;

  setWikiPreviewVisible(true);
  setWikiPreviewLoading(u);

  try {
    const r = await fetch(`/api/wiki/preview?url=${encodeURIComponent(u)}`);
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();

    if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = data.title || 'Wiki Preview';
    if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = data.url || u;
    if (wikiPreviewExtractEl) wikiPreviewExtractEl.textContent = data.extract || '(No preview text available.)';

    const thumb = data.thumbnail_url;
    if (wikiPreviewThumbEl) {
      if (thumb) {
        wikiPreviewThumbEl.src = thumb;
        wikiPreviewThumbEl.hidden = false;
      } else {
        wikiPreviewThumbEl.hidden = true;
        wikiPreviewThumbEl.src = '';
      }
    }
  } catch {
    // If preview fails, fall back to opening the source.
    if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = 'Preview unavailable';
    if (wikiPreviewExtractEl) wikiPreviewExtractEl.textContent = 'Could not fetch a preview. Use “Open in new tab”.';
    if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = u;
  }
}

async function sendFeedback({ historyId, rating }) {
  const r = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ history_id: String(historyId), rating: Number(rating), session_id: getSessionId() })
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }
  return await r.json();
}

function attachFeedbackControls(bubble, historyId) {
  if (!historyId) return;

  const wrap = document.createElement('div');
  wrap.className = 'feedback';

  const up = document.createElement('button');
  up.type = 'button';
  up.textContent = 'Helpful';

  const down = document.createElement('button');
  down.type = 'button';
  down.textContent = 'Not helpful';

  const status = document.createElement('span');
  status.className = 'status';
  status.textContent = '';

  const onClick = async (rating) => {
    try {
      up.disabled = true;
      down.disabled = true;
      status.textContent = 'Thanks — saved.';
      const data = await sendFeedback({ historyId, rating });
      if (data && data.summary) {
        status.textContent = `Saved. Up ${data.summary.thumbs_up || 0} / Down ${data.summary.thumbs_down || 0}`;
      }
    } catch {
      status.textContent = 'Could not save feedback.';
      up.disabled = false;
      down.disabled = false;
    }
  };

  up.addEventListener('click', () => onClick(1));
  down.addEventListener('click', () => onClick(-1));

  wrap.appendChild(up);
  wrap.appendChild(down);
  wrap.appendChild(status);
  bubble.appendChild(wrap);
}

function addBotAnswer(answerText, sources, historyId) {
  const { row, bubble } = createRow('bot');
  bubble.querySelector('.role').textContent = BOT_NAME;
  bubble.querySelector('.text').textContent = answerText;
  renderCitations(bubble, sources);
  attachFeedbackControls(bubble, historyId);
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function saveLiveChat() {
  try {
    const rows = Array.from(messagesEl.querySelectorAll('.msg-row'));
    const items = rows.slice(-50).map((r) => {
      const me = r.classList.contains('me');
      const bubble = r.querySelector('.msg');
      const role = bubble?.querySelector('.role')?.textContent || '';
      const text = bubble?.querySelector('.text')?.textContent || '';
      const cites = Array.from(bubble?.querySelectorAll('.citations a') || []).map((a) => ({
        title: a.textContent,
        url: a.getAttribute('href') || ''
      }));
      return { kind: me ? 'me' : 'bot', role, text, sources: cites };
    });
    localStorage.setItem(LIVE_CHAT_KEY, JSON.stringify(items));
  } catch {
    // ignore
  }
}

function loadLiveChat() {
  try {
    const raw = localStorage.getItem(LIVE_CHAT_KEY);
    if (!raw) return false;
    const items = JSON.parse(raw);
    if (!Array.isArray(items) || !items.length) return false;

    messagesEl.innerHTML = '';
    for (const it of items) {
      if (it.kind === 'me') {
        addMessage('me', 'You', it.text || '');
      } else {
        // Stored citations may have "[1] Title" in title; keep as-is.
        const sources = (it.sources || []).map((s) => ({ title: s.title || '', url: s.url || '' }));
        addBotAnswer(it.text || '', sources, null);
      }
    }
    return true;
  } catch {
    return false;
  }
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
  addBotAnswer(item.bot_answer || '(no answer)', item.sources || [], item.id);
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

// Restore live chat if available
if (messagesEl && loadLiveChat()) {
  // already restored
}

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

if (wikiPreviewCloseEl) {
  wikiPreviewCloseEl.addEventListener('click', () => {
    setWikiPreviewVisible(false);
  });
}

// Click citations to open an inline preview (normal click).
// Use Ctrl/Shift/Alt/Meta click to open the link normally.
if (messagesEl) {
  messagesEl.addEventListener('click', (e) => {
    const a = e.target && e.target.closest ? e.target.closest('.citations a') : null;
    if (!a) return;

    const url = a.dataset ? a.dataset.citationUrl : null;
    if (!url) return;

    if (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    openWikiPreview(url);
  });
}

// Load initial history once the page is ready.
document.addEventListener('DOMContentLoaded', () => {
  // Sidebar toggle state
  const open = localStorage.getItem(HISTORY_SIDEBAR_KEY);
  if (appLayoutEl && open === '0') {
    appLayoutEl.classList.add('sidebar-collapsed');
  }
  loadHistory();
});

if (toggleHistoryBtn && appLayoutEl) {
  toggleHistoryBtn.addEventListener('click', () => {
    const collapsed = appLayoutEl.classList.toggle('sidebar-collapsed');
    localStorage.setItem(HISTORY_SIDEBAR_KEY, collapsed ? '0' : '1');
  });
}

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
      body: JSON.stringify({ message: msg, session_id: getSessionId() })
    });

    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }

    const data = await r.json();

    // Replace typing bubble contents with the real answer + citations + feedback.
    const answerText = data.answer || '(no answer)';
    const historyId = data.history_id || null;
    typingBubble.querySelector('.text').textContent = answerText;
    typingBubble.dataset.historyId = historyId ? String(historyId) : '';

    if (data.sources && data.sources.length) {
      renderCitations(typingBubble, data.sources);
    }
    if (historyId) {
      attachFeedbackControls(typingBubble, historyId);
    }

    saveLiveChat();
  } catch (err) {
    typingBubble.querySelector('.text').textContent = err.message || String(err);
    typingBubble.querySelector('.role').textContent = 'Error';
  } finally {
    // Since every chat is publicly stored, refresh the sidebar.
    loadHistory();
  }
});
