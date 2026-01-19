const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('message');
const historyListEl = document.getElementById('historyList');
const refreshHistoryBtn = document.getElementById('refreshHistory');
const toggleHistoryBtn = document.getElementById('toggleHistory');
const fullscreenToggleBtn = document.getElementById('fullscreenToggle');
const appLayoutEl = document.querySelector('.app-layout');
const chatEl = document.querySelector('main.chat');
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

function isFullscreen() {
  return Boolean(document.fullscreenElement);
}

function getFullscreenTargetEl() {
  // Prefer fullscreening the chat interface only (not the whole page).
  return chatEl || appLayoutEl || document.documentElement;
}

function setFullscreenBtnState() {
  if (!fullscreenToggleBtn) return;
  const enabled = Boolean(document.fullscreenEnabled);
  if (!enabled) {
    fullscreenToggleBtn.hidden = true;
    return;
  }

  const on = isFullscreen();
  const label = on ? 'Exit fullscreen' : 'Enter fullscreen';
  fullscreenToggleBtn.title = label;
  fullscreenToggleBtn.setAttribute('aria-label', label);
}

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
    avatar.innerHTML = `<img class="bot-avatar" src="${BOT_AVATAR_SRC}" alt="${BOT_NAME}" />`;
  } else {
    avatar.innerHTML = `<img class="user-avatar" src="/static/default_man.png" alt="You" />`;
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

function renderVideos(bubble, videos) {
  if (!videos || !videos.length) return;

  function youtubeVideoId(url) {
    const u = String(url || '').trim();
    if (!u) return null;
    try {
      const parsed = new URL(u);
      const host = (parsed.hostname || '').toLowerCase();
      if (host === 'youtu.be') {
        const id = (parsed.pathname || '').replace('/', '').trim();
        return id || null;
      }
      if (host.endsWith('youtube.com')) {
        const id = parsed.searchParams.get('v');
        return (id && String(id).trim()) ? String(id).trim() : null;
      }
    } catch {
      // ignore
    }
    // last-chance regex
    const m = u.match(/[?&]v=([^&]+)/i);
    return m ? String(m[1] || '').trim() || null : null;
  }

  const box = document.createElement('div');
  box.className = 'videos';

  const head = document.createElement('div');
  head.className = 'videos-head';
  head.textContent = 'Quest videos: ';
  box.appendChild(head);

  const list = document.createElement('div');
  list.className = 'videos-list';

  (videos || []).slice(0, 3).forEach((v) => {
    if (!v || !v.url) return;
    const item = document.createElement('div');
    item.className = 'video';

    const vid = youtubeVideoId(v.url);
    if (vid) {
      const player = document.createElement('div');
      player.className = 'video-player';

      const thumbBtn = document.createElement('button');
      thumbBtn.type = 'button';
      thumbBtn.className = 'video-thumb';
      thumbBtn.setAttribute('aria-label', 'Play video');

      const img = document.createElement('img');
      img.loading = 'lazy';
      img.alt = String(v.title || 'YouTube video');
      img.src = `https://i.ytimg.com/vi/${encodeURIComponent(vid)}/hqdefault.jpg`;
      thumbBtn.appendChild(img);

      const play = document.createElement('div');
      play.className = 'video-play';
      play.setAttribute('aria-hidden', 'true');
      play.textContent = '▶';
      thumbBtn.appendChild(play);

      thumbBtn.addEventListener('click', () => {
        const iframe = document.createElement('iframe');
        iframe.className = 'video-iframe';
        iframe.loading = 'lazy';
        iframe.allowFullscreen = true;
        iframe.referrerPolicy = 'strict-origin-when-cross-origin';
        iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
        iframe.src = `https://www.youtube.com/embed/${encodeURIComponent(vid)}?autoplay=1`;
        iframe.title = String(v.title || 'YouTube video');

        player.replaceChildren(iframe);
      });

      player.appendChild(thumbBtn);
      item.appendChild(player);
    }

    const a = document.createElement('a');
    a.className = 'video-title';
    a.href = String(v.url);
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = String(v.title || v.url);
    item.appendChild(a);

    const meta = document.createElement('div');
    meta.className = 'video-meta';
    const channel = (v.channel ? String(v.channel) : '').trim();
    meta.textContent = channel ? `YouTube • ${channel}` : 'YouTube';
    item.appendChild(meta);

    const summary = document.createElement('div');
    summary.className = 'video-summary';
    summary.textContent = String(v.summary || '').trim();
    item.appendChild(summary);

    list.appendChild(item);
  });

  box.appendChild(list);
  bubble.appendChild(box);
}

function setWikiPreviewVisible(visible) {
  if (!wikiPreviewEl) return;
  wikiPreviewEl.hidden = !visible;
}

function setWikiPreviewLoading(url) {
  if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = 'Loading...';
  if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = url || '#';
  if (wikiPreviewExtractEl) wikiPreviewExtractEl.textContent = 'Fetching preview from the wiki...';
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

function addBotAnswer(answerText, sources, historyId, videos) {
  const { row, bubble } = createRow('bot');
  bubble.querySelector('.role').textContent = BOT_NAME;
  bubble.querySelector('.text').textContent = answerText;
  renderCitations(bubble, sources);
  renderVideos(bubble, videos);
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
      const vids = Array.from(bubble?.querySelectorAll('.videos .video') || []).map((v) => {
        const a = v.querySelector('a.video-title');
        const title = a?.textContent || '';
        const url = a?.getAttribute('href') || '';
        const channelText = v.querySelector('.video-meta')?.textContent || '';
        const channel = channelText.replace(/^YouTube\s*•\s*/i, '').trim() || null;
        const summary = v.querySelector('.video-summary')?.textContent || '';
        return { title, url, channel, summary };
      });
      return { kind: me ? 'me' : 'bot', role, text, sources: cites, videos: vids };
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
        addBotAnswer(it.text || '', sources, null, it.videos || []);
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
  addBotAnswer(item.bot_answer || '(no answer)', item.sources || [], item.id, item.videos || []);
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
    q.textContent = snippet.length > 80 ? `${snippet.slice(0, 80)}...` : snippet;

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
      <span class="typing-status">Conjuring...</span>
      <span class="dots" aria-hidden="true"><span></span><span></span><span></span></span>
    </span>
  `;
  messagesEl.appendChild(row);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return bubble;
}

function setTypingStatus(bubble, text) {
  if (!bubble) return;
  const el = bubble.querySelector('.typing-status');
  if (el) el.textContent = String(text || '').trim() || 'Conjuring...';
}

function renderActions(bubble, actions) {
  if (!bubble) return;
  if (!actions || !actions.length) return;

  const box = document.createElement('div');
  box.className = 'actions';

  const head = document.createElement('div');
  head.className = 'actions-head';
  head.textContent = 'What I did: ';
  box.appendChild(head);

  const list = document.createElement('ul');
  list.className = 'actions-list';

  (actions || []).slice(0, 8).forEach((a) => {
    const li = document.createElement('li');
    li.textContent = String(a || '').trim();
    if (!li.textContent) return;
    list.appendChild(li);
  });

  if (list.children.length) {
    box.appendChild(list);
    bubble.appendChild(box);
  }
}

function renderWebResults(bubble, webQuery, webResults) {
  if (!bubble) return;
  if (!webResults || !webResults.length) return;

  const box = document.createElement('div');
  box.className = 'web-results';

  const head = document.createElement('div');
  head.className = 'web-results-head';
  head.textContent = webQuery ? `Web leads (Google CSE): ${webQuery}` : 'Web leads (Google CSE):';
  box.appendChild(head);

  const list = document.createElement('div');
  list.className = 'web-results-list';

  (webResults || []).slice(0, 5).forEach((r) => {
    if (!r || !r.url) return;
    const item = document.createElement('div');
    item.className = 'web-result';

    const a = document.createElement('a');
    a.className = 'web-result-title';
    a.href = String(r.url);
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = String(r.title || r.url);
    item.appendChild(a);

    if (r.snippet) {
      const sn = document.createElement('div');
      sn.className = 'web-result-snippet';
      sn.textContent = String(r.snippet);
      item.appendChild(sn);
    }

    list.appendChild(item);
  });

  if (list.children.length) {
    box.appendChild(list);
    bubble.appendChild(box);
  }
}

async function postChatStream(payload, onEvent) {
  const r = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  });

  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${r.status}`);
  }

  if (!r.body || !r.body.getReader) {
    throw new Error('Streaming not supported by this browser.');
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    while (true) {
      const idx = buf.indexOf('\n');
      if (idx < 0) break;
      const line = buf.slice(0, idx).trim();
      buf = buf.slice(idx + 1);
      if (!line) continue;
      let ev;
      try {
        ev = JSON.parse(line);
      } catch {
        continue;
      }
      onEvent && onEvent(ev);
    }
  }

  const tail = buf.trim();
  if (tail) {
    try {
      onEvent && onEvent(JSON.parse(tail));
    } catch {
      // ignore
    }
  }
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
  setFullscreenBtnState();
  loadHistory();
});

if (fullscreenToggleBtn) {
  setFullscreenBtnState();

  fullscreenToggleBtn.addEventListener('click', async () => {
    try {
      if (!document.fullscreenEnabled) return;
      if (isFullscreen()) {
        await document.exitFullscreen();
      } else {
        // Fullscreen the chat interface (not the entire page UI).
        const target = getFullscreenTargetEl();
        if (target && target.requestFullscreen) {
          await target.requestFullscreen();
        }
      }
    } catch {
      // ignore
    } finally {
      setFullscreenBtnState();
    }
  });

  document.addEventListener('fullscreenchange', () => {
    setFullscreenBtnState();
  });
}

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
  setTypingStatus(typingBubble, 'Consulting my shelves...');

  // Flavor fallback in case the stream takes a moment to yield the first status.
  const flavor = [
    'Scouring the shelves of my tomes...',
    'Unlocking the magical trunks of knowledge...',
    'Consulting the enchanted index...',
    'Brushing dust off old scrolls...',
    'Pinning citations to the parchment...'
  ];
  let flavorI = 0;
  let gotLiveStatus = false;
  const flavorTimer = setInterval(() => {
    if (gotLiveStatus) return;
    flavorI = (flavorI + 1) % flavor.length;
    setTypingStatus(typingBubble, flavor[flavorI]);
  }, 1300);

  try {
    const payload = { message: msg, session_id: getSessionId() };

    let finalData = null;
    try {
      await postChatStream(payload, (ev) => {
        if (!ev || typeof ev !== 'object') return;
        if (ev.type === 'status') {
          gotLiveStatus = true;
          setTypingStatus(typingBubble, ev.text || 'Conjuring...');
          return;
        }
        if (ev.type === 'final') {
          finalData = ev.data || null;
          return;
        }
        if (ev.type === 'error') {
          throw new Error(ev.detail || `HTTP ${ev.status || 500}`);
        }
      });
    } catch (streamErr) {
      // Streaming is best-effort; fall back to the normal JSON endpoint.
      const r = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!r.ok) {
        const err = await r.json().catch(() => ({}));
        throw new Error(err.detail || streamErr.message || `HTTP ${r.status}`);
      }
      finalData = await r.json();
    }

    const data = finalData || {};

    // Replace typing bubble contents with the real answer + citations + feedback.
    const answerText = data.answer || '(no answer)';
    const historyId = data.history_id || null;
    typingBubble.querySelector('.text').textContent = answerText;
    typingBubble.dataset.historyId = historyId ? String(historyId) : '';

    if (data.sources && data.sources.length) {
      renderCitations(typingBubble, data.sources);
    }
    if (data.videos && data.videos.length) {
      renderVideos(typingBubble, data.videos);
    }
    if (data.actions && data.actions.length) {
      renderActions(typingBubble, data.actions);
    }
    if (data.web_results && data.web_results.length) {
      renderWebResults(typingBubble, data.web_query, data.web_results);
    }
    if (historyId) {
      attachFeedbackControls(typingBubble, historyId);
    }

    saveLiveChat();
  } catch (err) {
    typingBubble.querySelector('.text').textContent = err.message || String(err);
    typingBubble.querySelector('.role').textContent = 'Error';
  } finally {
    clearInterval(flavorTimer);
    // Since every chat is publicly stored, refresh the sidebar.
    loadHistory();
  }
});
