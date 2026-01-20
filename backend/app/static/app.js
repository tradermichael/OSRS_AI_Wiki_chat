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

const voiceToggleBtn = document.getElementById('voiceToggle');
const voiceStatusEl = document.getElementById('voiceStatus');

const voiceChatToggleBtn = document.getElementById('voiceChatToggle');
const voiceChatPanelEl = document.getElementById('voiceChatPanel');
const voiceChatStateEl = document.getElementById('voiceChatState');
const voiceChatDisconnectBtn = document.getElementById('voiceChatDisconnect');
const pushToTalkBtn = document.getElementById('pushToTalk');

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

let voice = null;
let liveVoice = null;

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

function renderMarkdownToFragment(md) {
  const src = String(md || '');
  const frag = document.createDocumentFragment();
  const lines = src.replace(/\r\n/g, '\n').split('\n');

  function appendInline(parent, text) {
    const s = String(text || '');
    // Very small markdown subset: **bold**, *italic*, `code`.
    // No HTML is interpreted; we create DOM nodes directly.
    const re = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
    let last = 0;
    let m;
    while ((m = re.exec(s)) !== null) {
      if (m.index > last) parent.appendChild(document.createTextNode(s.slice(last, m.index)));
      const token = m[0] || '';
      if (token.startsWith('**') && token.endsWith('**') && token.length >= 4) {
        const el = document.createElement('strong');
        el.textContent = token.slice(2, -2);
        parent.appendChild(el);
      } else if (token.startsWith('`') && token.endsWith('`') && token.length >= 2) {
        const el = document.createElement('code');
        el.textContent = token.slice(1, -1);
        parent.appendChild(el);
      } else if (token.startsWith('*') && token.endsWith('*') && token.length >= 2) {
        const el = document.createElement('em');
        el.textContent = token.slice(1, -1);
        parent.appendChild(el);
      } else {
        parent.appendChild(document.createTextNode(token));
      }
      last = m.index + token.length;
    }
    if (last < s.length) parent.appendChild(document.createTextNode(s.slice(last)));
  }

  function appendParagraph(blockLines) {
    const p = document.createElement('p');
    blockLines.forEach((ln, idx) => {
      if (idx) p.appendChild(document.createElement('br'));
      appendInline(p, ln);
    });
    frag.appendChild(p);
  }

  let i = 0;
  while (i < lines.length) {
    // Skip blank lines.
    while (i < lines.length && !String(lines[i] || '').trim()) i++;
    if (i >= lines.length) break;

    // Unordered list block.
    if (/^\s*[*\-+]\s+/.test(lines[i])) {
      const ul = document.createElement('ul');
      while (i < lines.length && /^\s*[*\-+]\s+/.test(lines[i])) {
        const li = document.createElement('li');
        const content = String(lines[i] || '').replace(/^\s*[*\-+]\s+/, '');
        appendInline(li, content);
        ul.appendChild(li);
        i++;
      }
      frag.appendChild(ul);
      continue;
    }

    // Paragraph block (until blank line or list).
    const block = [];
    while (i < lines.length && String(lines[i] || '').trim() && !/^\s*[*\-+]\s+/.test(lines[i])) {
      block.push(String(lines[i] || ''));
      i++;
    }
    if (block.length) appendParagraph(block);
  }

  return frag;
}

function setBubbleText(bubble, text, { markdown } = { markdown: false }) {
  if (!bubble) return;
  const textEl = bubble.querySelector('.text');
  if (!textEl) return;
  const raw = String(text || '');
  // Preserve raw text/markdown so saveLiveChat() can persist formatting.
  if (textEl.dataset) textEl.dataset.raw = raw;

  if (!markdown) {
    textEl.textContent = raw;
    return;
  }

  textEl.replaceChildren(renderMarkdownToFragment(raw));
}

function addMessage(kind, roleLabel, text) {
  const { row, bubble } = createRow(kind);
  bubble.querySelector('.role').textContent = roleLabel;
  setBubbleText(bubble, text, { markdown: kind === 'bot' });
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
  head.textContent = 'Videos: ';
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

  // When closing, stop any embedded media.
  if (!visible) {
    if (wikiPreviewExtractEl) wikiPreviewExtractEl.replaceChildren();
    if (wikiPreviewThumbEl) {
      wikiPreviewThumbEl.hidden = true;
      wikiPreviewThumbEl.src = '';
    }
    wikiPreviewEl.classList.remove('wiki-preview--has-thumb');
    if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = 'Wiki Preview';
  }
}

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
  const m = u.match(/[?&]v=([^&]+)/i);
  return m ? String(m[1] || '').trim() || null : null;
}

function isRedditUrl(url) {
  const u = String(url || '').trim();
  if (!u) return false;
  try {
    const parsed = new URL(u);
    const host = (parsed.hostname || '').toLowerCase();
    return host === 'reddit.com' || host.endsWith('.reddit.com');
  } catch {
    return false;
  }
}

function openYouTubePreview(url, title) {
  const u = String(url || '').trim();
  if (!u) return;
  const vid = youtubeVideoId(u);
  if (!vid) {
    openWikiPreview(u);
    return;
  }

  setWikiPreviewVisible(true);
  if (wikiPreviewEl) wikiPreviewEl.classList.remove('wiki-preview--has-thumb');
  if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = (String(title || '').trim() || 'YouTube');
  if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = u;
  if (wikiPreviewThumbEl) {
    wikiPreviewThumbEl.hidden = true;
    wikiPreviewThumbEl.src = '';
  }

  if (wikiPreviewExtractEl) {
    wikiPreviewExtractEl.replaceChildren();
    const player = document.createElement('div');
    player.className = 'video-player';

    const iframe = document.createElement('iframe');
    iframe.className = 'video-iframe';
    iframe.loading = 'lazy';
    iframe.allowFullscreen = true;
    iframe.referrerPolicy = 'strict-origin-when-cross-origin';
    iframe.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share';
    iframe.src = `https://www.youtube.com/embed/${encodeURIComponent(vid)}?autoplay=1`;
    iframe.title = String(title || 'YouTube video');

    player.appendChild(iframe);
    wikiPreviewExtractEl.appendChild(player);
  }
}

function setWikiPreviewLoading(url) {
  if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = 'Loading...';
  if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = url || '#';
  if (wikiPreviewExtractEl) wikiPreviewExtractEl.textContent = 'Fetching preview from the wiki...';
  if (wikiPreviewThumbEl) {
    wikiPreviewThumbEl.hidden = true;
    wikiPreviewThumbEl.src = '';
  }
  if (wikiPreviewEl) wikiPreviewEl.classList.remove('wiki-preview--has-thumb');
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
        if (wikiPreviewEl) wikiPreviewEl.classList.add('wiki-preview--has-thumb');
      } else {
        wikiPreviewThumbEl.hidden = true;
        wikiPreviewThumbEl.src = '';
        if (wikiPreviewEl) wikiPreviewEl.classList.remove('wiki-preview--has-thumb');
      }
    }
  } catch {
    // If preview fails, fall back to opening the source.
    if (wikiPreviewTitleEl) wikiPreviewTitleEl.textContent = 'Preview unavailable';
    if (wikiPreviewExtractEl) wikiPreviewExtractEl.textContent = 'Could not fetch a preview. Use “Open in new tab”.';
    if (wikiPreviewOpenEl) wikiPreviewOpenEl.href = u;
    if (wikiPreviewEl) wikiPreviewEl.classList.remove('wiki-preview--has-thumb');
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
  setBubbleText(bubble, answerText, { markdown: true });
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
      const textEl = bubble?.querySelector('.text');
      const text = (textEl && textEl.dataset && textEl.dataset.raw != null) ? String(textEl.dataset.raw) : (textEl?.textContent || '');
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

  if (voiceToggleBtn) voiceToggleBtn.disabled = !enabled;
  if (voiceChatToggleBtn) voiceChatToggleBtn.disabled = !enabled;
  if (pushToTalkBtn) pushToTalkBtn.disabled = !enabled;
  if (!enabled && voice && typeof voice.stop === 'function') {
    voice.stop();
  }
  if (!enabled && liveVoice && typeof liveVoice.disconnect === 'function') {
    liveVoice.disconnect();
    liveVoice = null;
  }
}

function setVoiceStatus(text) {
  if (!voiceStatusEl) return;
  const t = String(text || '').trim();
  voiceStatusEl.textContent = t;
  voiceStatusEl.hidden = !t;
}

function setVoiceBtnState(listening) {
  if (!voiceToggleBtn) return;
  const on = Boolean(listening);
  voiceToggleBtn.classList.toggle('is-listening', on);
  const label = on ? 'Stop voice input' : 'Voice input';
  voiceToggleBtn.title = label;
  voiceToggleBtn.setAttribute('aria-label', label);
}

function setVoiceChatState(text) {
  if (!voiceChatStateEl) return;
  voiceChatStateEl.textContent = String(text || '').trim();
}

function decodeBase64ToBytes(b64) {
  const bin = atob(String(b64 || ''));
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function downsampleFloat32ToInt16(input, inRate, outRate) {
  if (!input || !input.length) return new Int16Array(0);

  if (!inRate || !outRate || inRate === outRate) {
    const out = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      out[i] = s < 0 ? (s * 0x8000) : (s * 0x7fff);
    }
    return out;
  }

  const ratio = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Int16Array(outLen);
  let inputIndex = 0;
  for (let outIndex = 0; outIndex < outLen; outIndex++) {
    const next = Math.floor((outIndex + 1) * ratio);
    let sum = 0;
    let count = 0;
    for (; inputIndex < next && inputIndex < input.length; inputIndex++) {
      sum += input[inputIndex];
      count++;
    }
    const avg = count ? (sum / count) : 0;
    const s = Math.max(-1, Math.min(1, avg));
    out[outIndex] = s < 0 ? (s * 0x8000) : (s * 0x7fff);
  }
  return out;
}

function initVoiceChat() {
  if (!voiceChatToggleBtn || !voiceChatPanelEl || !pushToTalkBtn) return;

  const WS_URL = ((location.protocol === 'https:') ? 'wss://' : 'ws://') + location.host + '/api/live/ws';

  let ws = null;
  let connecting = false;
  let talking = false;
  let liveReady = false;
  let lastWsError = '';
  let inputTx = '';

  let micStream = null;
  let micCtx = null;
  let micSource = null;
  let micProc = null;
  let micWorklet = null;
  let micSilentGain = null;

  let playCtx = null;
  let playHead = 0;
  let playingSources = [];

  let botBubble = null;
  let botText = '';

  function setPanelOpen(open) {
    voiceChatPanelEl.hidden = !open;
    voiceChatToggleBtn.classList.toggle('is-listening', Boolean(open));
    const label = open ? 'Close voice chat' : 'Voice chat';
    voiceChatToggleBtn.title = label;
    voiceChatToggleBtn.setAttribute('aria-label', label);
  }

  function resetBotTurn() {
    botBubble = null;
    botText = '';
  }

  function stopPlayback() {
    try {
      for (const s of playingSources) {
        try { s.stop(0); } catch { /* ignore */ }
      }
    } catch { /* ignore */ }
    playingSources = [];
    if (playCtx) playHead = playCtx.currentTime;
  }

  async function ensurePlayContext() {
    if (!playCtx) {
      playCtx = new (window.AudioContext || window.webkitAudioContext)();
      playHead = playCtx.currentTime;
    }
    if (playCtx.state === 'suspended') {
      try { await playCtx.resume(); } catch { /* ignore */ }
    }
  }

  async function connect() {
    if (ws || connecting) return;
    connecting = true;
    liveReady = false;
    lastWsError = '';
    setVoiceChatState('Connecting…');
    pushToTalkBtn.disabled = true;

    ws = new WebSocket(WS_URL);
    ws.binaryType = 'arraybuffer';

    ws.addEventListener('open', () => {
      connecting = false;
      setVoiceChatState('Connected. Starting session…');
      if (voiceChatDisconnectBtn) voiceChatDisconnectBtn.hidden = false;
      pushToTalkBtn.textContent = 'Hold to talk (release to get reply)';
      // Start the Live session.
      try {
        ws.send(JSON.stringify({
          type: 'start',
        }));
      } catch { /* ignore */ }
    });

    ws.addEventListener('message', async (evt) => {
      let msg = null;
      try { msg = JSON.parse(evt.data); } catch { return; }
      if (!msg || !msg.type) return;

      if (msg.type === 'ready') {
        liveReady = true;
        setVoiceChatState('Connected. Hold to talk.');
        pushToTalkBtn.disabled = false;
        return;
      }

      if (msg.type === 'error') {
        lastWsError = String(msg.detail || 'Voice chat error');
        setVoiceChatState(lastWsError);
        return;
      }

      if (msg.type === 'interrupted') {
        stopPlayback();
        resetBotTurn();
        return;
      }

      if (msg.type === 'input_transcript') {
        if (msg.text) inputTx = String(msg.text);
        if (msg.finished && inputTx) {
          addMessage('me', 'You (voice)', inputTx);
          inputTx = '';
        }
        return;
      }

      if (msg.type === 'output_transcript') {
        const t = (msg.text != null) ? String(msg.text) : '';
        if (!t) return;
        if (!botBubble) botBubble = addMessage('bot', BOT_NAME, '');
        botText = t;
        setBubbleText(botBubble, botText, { markdown: true });
        return;
      }

      if (msg.type === 'model_text') {
        const t = (msg.text != null) ? String(msg.text) : '';
        if (!t) return;
        if (!botBubble) botBubble = addMessage('bot', BOT_NAME, '');
        botText += t;
        setBubbleText(botBubble, botText, { markdown: true });
        return;
      }

      if (msg.type === 'audio') {
        const b = decodeBase64ToBytes(msg.data);
        if (!b || !b.length) return;

        const mime = String(msg.mime || 'audio/pcm;rate=24000').toLowerCase();
        let rate = 24000;
        const m = mime.match(/rate=(\d+)/);
        if (m) rate = parseInt(m[1], 10) || 24000;

        await ensurePlayContext();

        const pcm16 = new Int16Array(b.buffer, b.byteOffset, Math.floor(b.byteLength / 2));
        const floats = new Float32Array(pcm16.length);
        for (let i = 0; i < pcm16.length; i++) floats[i] = pcm16[i] / 0x8000;

        const audioBuffer = playCtx.createBuffer(1, floats.length, rate);
        audioBuffer.copyToChannel(floats, 0);

        const src = playCtx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(playCtx.destination);

        const now = playCtx.currentTime;
        if (!playHead || playHead < now) playHead = now;
        src.start(playHead);
        playHead += audioBuffer.duration;
        playingSources.push(src);
        src.addEventListener('ended', () => {
          playingSources = playingSources.filter((x) => x !== src);
        });
        return;
      }

      if (msg.type === 'turn_complete') {
        resetBotTurn();
      }
    });

    ws.addEventListener('close', (evt) => {
      ws = null;
      connecting = false;
      talking = false;
      liveReady = false;
      pushToTalkBtn.disabled = true;
      pushToTalkBtn.classList.remove('is-talking');

      const code = (evt && typeof evt.code === 'number') ? evt.code : null;
      const reason = (evt && evt.reason) ? String(evt.reason).trim() : '';
      let status = lastWsError ? `Disconnected: ${lastWsError}` : 'Disconnected.';
      if (!lastWsError && code) {
        status = `Disconnected (${code})`;
        if (reason) status += `: ${reason}`;
      }

      // If Gemini Live closes with a policy/permissions error and we didn't receive a JSON error
      // message, show a friendlier hint (the raw close reason can be very confusing).
      if (!lastWsError && code === 1008) {
        const r = (reason || '').toLowerCase();
        if (r.includes('publisher model') || r.includes('policy violation')) {
          status = 'Voice chat blocked by Gemini Live (model/permissions). Check Cloud Run logs + GEMINI_LIVE_MODEL.';
        }
      }
      setVoiceChatState(status);

      if (voiceChatDisconnectBtn) voiceChatDisconnectBtn.hidden = true;
      stopPlayback();
      resetBotTurn();
      cleanupMic();
    });

    ws.addEventListener('error', () => {
      setVoiceChatState('Voice chat connection error.');
    });
  }

  function disconnect() {
    try {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'close' }));
    } catch { /* ignore */ }
    try { if (ws) ws.close(); } catch { /* ignore */ }
  }

  async function setupMicIfNeeded() {
    if (micStream && micCtx && micProc && micSource) return;
    if (micStream && micCtx && micWorklet && micSource) return;
    micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    micCtx = new (window.AudioContext || window.webkitAudioContext)();
    micSource = micCtx.createMediaStreamSource(micStream);

    // Prefer AudioWorklet when available (avoids ScriptProcessor deprecation warnings).
    if (micCtx.audioWorklet && typeof window.AudioWorkletNode !== 'undefined') {
      try {
        await micCtx.audioWorklet.addModule('/static/ptt_worklet.js');
        micSilentGain = micCtx.createGain();
        micSilentGain.gain.value = 0;

        micWorklet = new AudioWorkletNode(micCtx, 'ptt-capture', {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          channelCountMode: 'explicit',
          channelInterpretation: 'speakers',
        });

        micWorklet.port.onmessage = (evt) => {
          if (!talking) return;
          if (!liveReady) return;
          if (!ws || ws.readyState !== WebSocket.OPEN) return;

          const buf = evt && evt.data ? evt.data : null;
          if (!buf) return;
          let f32 = null;
          try {
            if (buf instanceof Float32Array) f32 = buf;
            else if (buf.buffer) f32 = new Float32Array(buf.buffer);
            else if (buf instanceof ArrayBuffer) f32 = new Float32Array(buf);
          } catch {
            f32 = null;
          }
          if (!f32 || !f32.length) return;

          const pcm = downsampleFloat32ToInt16(f32, micCtx.sampleRate, 16000);
          if (!pcm || !pcm.length) return;
          try { ws.send(pcm.buffer); } catch { /* ignore */ }
        };

        micSource.connect(micWorklet);
        micWorklet.connect(micSilentGain);
        micSilentGain.connect(micCtx.destination);
        return;
      } catch {
        // Fall back to ScriptProcessor below.
        try { if (micWorklet) micWorklet.disconnect(); } catch { /* ignore */ }
        try { if (micSilentGain) micSilentGain.disconnect(); } catch { /* ignore */ }
        micWorklet = null;
        micSilentGain = null;
      }
    }

    // ScriptProcessorNode is deprecated but broadly supported and sufficient here.
    micProc = micCtx.createScriptProcessor(2048, 1, 1);
    micProc.onaudioprocess = (e) => {
      if (!talking) return;
      if (!liveReady) return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      const ch0 = e.inputBuffer.getChannelData(0);
      const pcm = downsampleFloat32ToInt16(ch0, micCtx.sampleRate, 16000);
      if (!pcm || !pcm.length) return;
      try {
        ws.send(pcm.buffer);
      } catch {
        // ignore
      }
    };

    micSource.connect(micProc);
    // Avoid feedback; ScriptProcessor needs to be connected to destination to run in some browsers.
    micProc.connect(micCtx.destination);
  }

  function cleanupMic() {
    try { if (micProc) micProc.disconnect(); } catch { /* ignore */ }
    try { if (micWorklet) micWorklet.disconnect(); } catch { /* ignore */ }
    try { if (micSilentGain) micSilentGain.disconnect(); } catch { /* ignore */ }
    try { if (micSource) micSource.disconnect(); } catch { /* ignore */ }
    try {
      if (micStream) micStream.getTracks().forEach((t) => t.stop());
    } catch { /* ignore */ }
    try { if (micCtx) micCtx.close(); } catch { /* ignore */ }
    micProc = null;
    micWorklet = null;
    micSilentGain = null;
    micSource = null;
    micStream = null;
    micCtx = null;
  }

  function startTalking() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    if (talking) return;
    talking = true;
    pushToTalkBtn.classList.add('is-talking');
    pushToTalkBtn.textContent = 'Release to send';
  }

  function stopTalking() {
    if (!talking) return;
    talking = false;
    pushToTalkBtn.classList.remove('is-talking');
    pushToTalkBtn.textContent = 'Hold to talk';
    try {
      if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'audio_stream_end' }));
    } catch { /* ignore */ }
  }

  voiceChatToggleBtn.addEventListener('click', async () => {
    const open = voiceChatPanelEl.hidden;
    setPanelOpen(open);
    if (open) {
      await connect();
      liveVoice = { disconnect };
    } else {
      disconnect();
      liveVoice = null;
    }
  });

  if (voiceChatDisconnectBtn) {
    voiceChatDisconnectBtn.addEventListener('click', () => disconnect());
  }

  // Push-to-talk: use Pointer Events + pointer capture so holding is reliable
  // and we don't need global mouseup/touchend listeners (which can fire in
  // surprising ways and make PTT "flash").
  let activePointerId = null;

  pushToTalkBtn.addEventListener('pointerdown', async (e) => {
    if (e.button != null && e.button !== 0) return; // left click only
    if (activePointerId != null) return;
    if (input && input.disabled) return;

    activePointerId = e.pointerId;
    try { pushToTalkBtn.setPointerCapture(activePointerId); } catch { /* ignore */ }

    // Only prevent default while interacting with PTT.
    e.preventDefault();

    // Start UI immediately; audio will stream as soon as ws+mic are ready.
    startTalking();
    setVoiceChatState('Listening…');

    try {
      // Ensure we're connected and the mic processor is running.
      await connect();
      await setupMicIfNeeded();
    } catch {
      setVoiceChatState('Microphone permission denied.');
      stopTalking();
      activePointerId = null;
      try { pushToTalkBtn.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    }
  });

  pushToTalkBtn.addEventListener('pointerup', (e) => {
    if (activePointerId == null) return;
    if (e.pointerId !== activePointerId) return;
    e.preventDefault();
    stopTalking();
    try { pushToTalkBtn.releasePointerCapture(activePointerId); } catch { /* ignore */ }
    activePointerId = null;
    if (ws && ws.readyState === WebSocket.OPEN) setVoiceChatState('Connected. Hold to talk.');
  });

  pushToTalkBtn.addEventListener('pointercancel', (e) => {
    if (activePointerId == null) return;
    if (e.pointerId !== activePointerId) return;
    stopTalking();
    try { pushToTalkBtn.releasePointerCapture(activePointerId); } catch { /* ignore */ }
    activePointerId = null;
  });

  // Default UI state.
  setPanelOpen(false);
  setVoiceChatState('Voice chat is off.');
  pushToTalkBtn.textContent = 'Hold to talk (release to get reply)';
}

function initVoiceInput() {
  if (!voiceToggleBtn || !input) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    // Fallback: record short audio and send to server for transcription.
    // This requires MediaRecorder support + the backend /api/speech/transcribe endpoint.
    const hasRecorder = typeof window.MediaRecorder !== 'undefined' && navigator && navigator.mediaDevices && navigator.mediaDevices.getUserMedia;
    if (!hasRecorder) {
      voiceToggleBtn.hidden = true;
      setVoiceStatus('');
      return;
    }

    let recording = false;
    let mediaRecorder = null;
    let stream = null;
    let chunks = [];
    let baseText = '';

    async function startRecording() {
      if (recording) return;
      if (input.disabled) return;
      baseText = (input.value || '').trim();
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        setVoiceStatus('Microphone permission denied.');
        return;
      }

      chunks = [];
      const options = {};
      try {
        // Prefer opus in webm when supported.
        if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm;codecs=opus')) {
          options.mimeType = 'audio/webm;codecs=opus';
        } else if (MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('audio/webm')) {
          options.mimeType = 'audio/webm';
        }
      } catch {
        // ignore
      }

      try {
        mediaRecorder = new MediaRecorder(stream, options);
      } catch {
        // Some browsers don't like options; retry without.
        mediaRecorder = new MediaRecorder(stream);
      }

      mediaRecorder.addEventListener('dataavailable', (e) => {
        if (e && e.data && e.data.size > 0) chunks.push(e.data);
      });

      mediaRecorder.addEventListener('stop', async () => {
        recording = false;
        setVoiceBtnState(false);
        setVoiceStatus('Transcribing…');

        try {
          const mime = (mediaRecorder && mediaRecorder.mimeType) ? mediaRecorder.mimeType : 'audio/webm';
          const blob = new Blob(chunks, { type: mime });
          const form = new FormData();
          let ext = 'webm';
          const ml = String(mime || '').toLowerCase();
          if (ml.includes('ogg')) ext = 'ogg';
          else if (ml.includes('wav')) ext = 'wav';
          else if (ml.includes('mpeg') || ml.includes('mp3')) ext = 'mp3';
          else if (ml.includes('mp4')) ext = 'mp4';
          form.append('file', blob, `voice.${ext}`);

          const r = await fetch('/api/speech/transcribe', { method: 'POST', body: form });
          if (!r.ok) {
            const err = await r.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${r.status}`);
          }
          const data = await r.json();
          const t = (data && data.text) ? String(data.text).trim() : '';
          if (t) {
            const parts = [];
            if (baseText) parts.push(baseText);
            parts.push(t);
            input.value = parts.join(baseText ? ' ' : '');
          }
          setVoiceStatus('');
        } catch (err) {
          setVoiceStatus(err.message || String(err));
        } finally {
          try {
            if (stream) {
              stream.getTracks().forEach((tr) => tr.stop());
            }
          } catch {
            // ignore
          }
          stream = null;
          mediaRecorder = null;
          chunks = [];
        }
      });

      try {
        mediaRecorder.start();
        recording = true;
        setVoiceBtnState(true);
        setVoiceStatus('Recording… click again to stop.');
      } catch {
        recording = false;
        setVoiceBtnState(false);
        setVoiceStatus('Could not start recording.');
      }
    }

    function stopRecording() {
      if (!recording) {
        setVoiceBtnState(false);
        setVoiceStatus('');
        return;
      }
      try {
        mediaRecorder && mediaRecorder.stop();
      } catch {
        // ignore
      }
    }

    voiceToggleBtn.addEventListener('click', () => {
      if (input.disabled) return;
      if (recording) stopRecording();
      else startRecording();
    });

    document.addEventListener('visibilitychange', () => {
      if (document.hidden) stopRecording();
    });

    voice = { stop: stopRecording };
    setVoiceBtnState(false);
    return;
  }

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.maxAlternatives = 1;
  recognition.lang = (navigator && navigator.language) ? navigator.language : 'en-US';

  let listening = false;
  let baseText = '';

  function start() {
    if (listening) return;
    if (input.disabled) return;

    baseText = (input.value || '').trim();
    try {
      recognition.start();
      listening = true;
      setVoiceBtnState(true);
      setVoiceStatus('Listening…');
      input.focus();
    } catch {
      // Some browsers throw if start() is called too quickly.
      listening = false;
      setVoiceBtnState(false);
    }
  }

  function stop() {
    if (!listening) {
      setVoiceBtnState(false);
      setVoiceStatus('');
      return;
    }
    try {
      recognition.stop();
    } catch {
      // ignore
    }
    listening = false;
    setVoiceBtnState(false);
    setVoiceStatus('');
  }

  recognition.addEventListener('result', (e) => {
    let interim = '';
    let finalText = '';

    for (let i = e.resultIndex; i < e.results.length; i++) {
      const res = e.results[i];
      const t = (res && res[0] && res[0].transcript) ? String(res[0].transcript) : '';
      if (!t) continue;
      if (res.isFinal) finalText += t;
      else interim += t;
    }

    const parts = [];
    if (baseText) parts.push(baseText);
    const combined = (finalText + interim).trim();
    if (combined) parts.push(combined);

    input.value = parts.join(baseText ? ' ' : '');
  });

  recognition.addEventListener('start', () => {
    listening = true;
    setVoiceBtnState(true);
    setVoiceStatus('Listening…');
  });

  recognition.addEventListener('end', () => {
    listening = false;
    setVoiceBtnState(false);
    setVoiceStatus('');
  });

  recognition.addEventListener('error', (e) => {
    const code = e && e.error ? String(e.error) : '';
    if (code === 'not-allowed' || code === 'service-not-allowed') {
      setVoiceStatus('Microphone permission denied.');
    } else if (code === 'no-speech') {
      setVoiceStatus('No speech detected.');
      setTimeout(() => setVoiceStatus(''), 1200);
    } else {
      setVoiceStatus('Voice input error.');
    }
    listening = false;
    setVoiceBtnState(false);
  });

  voiceToggleBtn.addEventListener('click', () => {
    if (input.disabled) return;
    if (listening) stop();
    else start();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stop();
  });

  voice = { stop };
  setVoiceBtnState(false);
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
  setPrimaryActionBtnState();
  setComposerEnabled(false);

  addMessage('me', 'You', item.user_message);
  addBotAnswer(item.bot_answer || '(no answer)', item.sources || [], item.id, item.videos || []);
}

function exitHistoryView() {
  viewingHistoryId = null;
  if (viewBannerEl) viewBannerEl.hidden = true;
  setPrimaryActionBtnState();
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

function setPrimaryActionBtnState() {
  if (!resumeChatBtn) return;
  // This button doubles as:
  // - "Back to chat" when viewing a history entry
  // - "New chat" during the live session
  resumeChatBtn.hidden = false;
  resumeChatBtn.textContent = viewingHistoryId ? 'Back to chat' : 'New chat';
}

function startNewChat() {
  // If we were viewing history, return to live first.
  if (viewingHistoryId) {
    exitHistoryView();
  }

  // Clear local live transcript and force a new session id so history entries
  // don't get appended to the previous conversation.
  try {
    localStorage.removeItem(LIVE_CHAT_KEY);
    localStorage.removeItem(SESSION_ID_KEY);
  } catch {
    // ignore
  }

  liveChatSnapshot = null;
  messagesEl.innerHTML = '';
  addMessage('bot', BOT_NAME, 'Ask me anything about OSRS.');
  setPrimaryActionBtnState();
  setComposerEnabled(true);
  input.focus();
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

setPrimaryActionBtnState();

if (refreshHistoryBtn) {
  refreshHistoryBtn.addEventListener('click', () => {
    loadHistory();
  });
}

if (resumeChatBtn) {
  resumeChatBtn.addEventListener('click', () => {
    if (viewingHistoryId) exitHistoryView();
    else startNewChat();
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

    // Reddit posts often block preview fetching; just open them normally in a new tab.
    if (isRedditUrl(url)) return;

    if (e.ctrlKey || e.metaKey || e.shiftKey || e.altKey) return;
    e.preventDefault();

    const label = (a.textContent || '').replace(/^\[\d+\]\s*/g, '').trim();
    if (youtubeVideoId(url)) {
      openYouTubePreview(url, label);
    } else {
      openWikiPreview(url);
    }
  });
}

// Load initial history once the page is ready.
document.addEventListener('DOMContentLoaded', () => {
  initMobileViewportSizing();
  // Sidebar toggle state
  const open = localStorage.getItem(HISTORY_SIDEBAR_KEY);
  // Hide public chat log by default unless user explicitly opened it.
  // Stored values: '1' = open, '0' = collapsed.
  const shouldCollapse = (open !== '1');
  if (appLayoutEl && shouldCollapse) {
    appLayoutEl.classList.add('sidebar-collapsed');
    if (open == null) localStorage.setItem(HISTORY_SIDEBAR_KEY, '0');
  }
  setFullscreenBtnState();
  initVoiceInput();
  initVoiceChat();
  loadHistory();
});

function initMobileViewportSizing() {
  // On mobile browsers/PWAs, the on-screen keyboard shrinks the *visual* viewport,
  // but CSS 100vh/100dvh can still behave like the layout viewport.
  // We track VisualViewport.height and expose it as --vvh so the composer stays visible.
  const root = document.documentElement;
  if (!root) return;
  if (!window.visualViewport) return;

  const vv = window.visualViewport;
  const update = () => {
    const h = Math.max(1, Math.floor(vv.height));
    root.style.setProperty('--vvh', `${h}px`);
  };

  vv.addEventListener('resize', update);
  vv.addEventListener('scroll', update);
  window.addEventListener('orientationchange', () => setTimeout(update, 200));
  update();

  // Best-effort: when focusing the input, nudge it into view.
  const msgInput = document.getElementById('message');
  if (msgInput) {
    msgInput.addEventListener('focus', () => {
      setTimeout(() => {
        try {
          msgInput.scrollIntoView({ block: 'center', behavior: 'smooth' });
        } catch {
          // ignore
        }
      }, 250);
    });
  }
}

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
