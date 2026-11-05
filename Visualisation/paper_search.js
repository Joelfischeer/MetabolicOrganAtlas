// Paper Search tab of the connection sidebar.
// The data (every retrieved paper once + the paper indices of each organ pair) is embedded gzip-compressed by
// export_network_to_cytoscape_dashboard() (see paper_search_payload() in run_metabolic_lit_search.py) and is
// unpacked on first use. This file is inserted into the dashboard as it is; the placeholder string of the first
// constant below is replaced by the data. Papers are arrays: [pmid, title, year, doi, label index (-1 = none), keywords].
const PAPER_SEARCH_GZ_B64 = "__PAPER_SEARCH_GZ_B64__";
const PAPER_SEARCH_AVAILABLE = PAPER_SEARCH_GZ_B64.length > 0;
const PS_PAGE = 10;
const PS_STOP = new Set(['the', 'of', 'and', 'in', 'a', 'an', 'to', 'for', 'on', 'with', 'by', 'is', 'are', 'as', 'at',
                         'from', 'or', 'its', 'that', 'this']);
window.__ps = { edge: null, mode: null, query: '', page: 0, cacheKey: null, list: [] };
let _paperSearchPromise = null;

async function getPaperSearchData() {
  if (_paperSearchPromise) return _paperSearchPromise;
  _paperSearchPromise = (async () => {
    if (!PAPER_SEARCH_AVAILABLE || typeof DecompressionStream === 'undefined') return null;
    try {
      const binary = atob(PAPER_SEARCH_GZ_B64);
      const bytes = new Uint8Array(binary.length);
      for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      const d = JSON.parse(await new Response(stream).text());
      d._tok = new Array(d.papers.length);
      return d;
    } catch (err) {
      console.warn('Paper search data failed to load:', err);
      return null;
    }
  })();
  return _paperSearchPromise;
}

function _psTokens(s) { return (String(s || '').toLowerCase().match(/[\p{L}\p{N}]+/gu) || []); }

function _psPaperTokens(d, i) {
  let t = d._tok[i];
  if (!t) {
    const p = d.papers[i];
    t = d._tok[i] = { title: _psTokens(p[1]), kw: _psTokens(p[5]) };
    t.titleSet = new Set(t.title);
    t.kwSet = new Set(t.kw);
    t.titleJoined = t.title.join(' ');
  }
  return t;
}

// 2 = the word occurs, 1 = a word starts with it (3+ letters), 0 = no match
function _psMatch(set, arr, q) {
  if (set.has(q)) return 2;
  if (q.length >= 3) for (const w of arr) if (w.startsWith(q)) return 1;
  return 0;
}

// Newest first (no query / ties): year, then PMID
function _psNewest(d, a, b) {
  return (Number(d.papers[b][2]) || 0) - (Number(d.papers[a][2]) || 0) ||
         (Number(d.papers[b][0]) || 0) - (Number(d.papers[a][0]) || 0);
}

// Best fit first. Every query word is looked up in the title (weight 3, word prefix 2) and in the author keywords
// (1.5 / 0.8), scaled by how rare the word is among the papers of this connection (idf); papers that match more of
// the words rank higher, and the words in the same order in the title get a bonus. Papers without any match are
// left out.
function paperSearchRank(d, ids, query) {
  const qs = _psTokens(query);
  let toks = [...new Set(qs.filter(t => !PS_STOP.has(t)))];
  if (!toks.length) toks = [...new Set(qs)];
  if (!toks.length) return ids.slice().sort((a, b) => _psNewest(d, a, b));
  const N = ids.length;
  const df = toks.map(() => 0);
  const hits = ids.map(i => {
    const t = _psPaperTokens(d, i);
    return toks.map((q, j) => {
      const ht = _psMatch(t.titleSet, t.title, q), hk = _psMatch(t.kwSet, t.kw, q);
      if (ht || hk) df[j]++;
      return [ht, hk];
    });
  });
  const idf = df.map(x => Math.log(1 + N / (1 + x)));
  const phrase = qs.join(' ');
  const scored = [];
  ids.forEach((i, r) => {
    let s = 0, matched = 0;
    hits[r].forEach(([ht, hk], j) => {
      if (!ht && !hk) return;
      matched++;
      s += idf[j] * ((ht === 2 ? 3 : ht === 1 ? 2 : 0) + (hk === 2 ? 1.5 : hk === 1 ? 0.8 : 0));
    });
    if (!matched) return;
    s *= matched / toks.length;
    if (toks.length > 1 && _psPaperTokens(d, i).titleJoined.includes(phrase)) s += 3 * toks.length;
    scored.push([s, i]);
  });
  scored.sort((a, b) => b[0] - a[0] || _psNewest(d, a[1], b[1]));
  return scored.map(x => x[1]);
}

function paperSearchShell() {
  return '<div id="ps-root">' +
    '<input id="ps-input" type="search" autocomplete="off" placeholder="Search title and keywords…" ' +
    'oninput="psOnInput(this.value)" style="width:100%;box-sizing:border-box;background:#0f172a;' +
    'border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:12px;padding:7px 9px;margin-bottom:8px">' +
    '<div id="ps-status" style="font-size:11px;color:#64748b;margin-bottom:8px"></div>' +
    '<div id="ps-results"></div>' +
    '<div id="ps-nav" style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-top:10px"></div>' +
    '</div>';
}

function initPaperSearch() {
  const root = document.getElementById('ps-root');
  if (!root || typeof currentEdgeData === 'undefined' || !currentEdgeData) return;
  const st = window.__ps;
  if (st.edge !== currentEdgeData.id) { st.query = ''; st.page = 0; st.cacheKey = null; }
  st.edge = currentEdgeData.id;
  st.mode = window.__currentDisplayMode || 'all';
  document.getElementById('ps-input').value = st.query;
  document.getElementById('ps-status').textContent = 'Loading papers…';
  getPaperSearchData().then(d => {
    if (!d) {
      document.getElementById('ps-status').textContent =
        'The paper list could not be loaded (this browser lacks gzip decompression support).';
      return;
    }
    psRender(d);
  });
}

function psOnInput(v) {
  clearTimeout(window.__psTimer);
  window.__psTimer = setTimeout(() => {
    const st = window.__ps;
    st.query = v;
    st.page = 0;
    getPaperSearchData().then(d => { if (d) psRender(d); });
  }, 120);
}

function psGo(delta) {
  window.__ps.page = Math.max(0, window.__ps.page + delta);
  getPaperSearchData().then(d => { if (d) psRender(d); });
}

function psRender(d) {
  const root = document.getElementById('ps-root');
  if (!root || typeof currentEdgeData === 'undefined' || !currentEdgeData) return;
  const st = window.__ps;
  const mode = window.__currentDisplayMode || 'all';
  const key = [currentEdgeData.source, currentEdgeData.target].sort().join('|');
  const cacheKey = key + '#' + mode + '#' + st.query;
  if (st.cacheKey !== cacheKey) {
    const li = d.slugs.indexOf(mode);
    const base = d.pairs[key] || [];
    const ids = base.filter(i => mode === 'all' || d.papers[i][4] === li);
    st.total = ids.length;
    st.list = paperSearchRank(d, ids, st.query);
    st.cacheKey = cacheKey;
  }
  const list = st.list, n = list.length;
  const pages = Math.max(1, Math.ceil(n / PS_PAGE));
  st.page = Math.min(st.page, pages - 1);
  const from = st.page * PS_PAGE;
  const modeLabel = mode === 'all' ? 'All papers' : ((window.__condLabels || {})[mode] || mode);
  const q = st.query.trim();
  const status = document.getElementById('ps-status');
  status.innerHTML = !st.total
    ? 'No papers retrieved for this connection in <b style="color:#cbd5e1">' + escHtml(modeLabel) + '</b>.'
    : (q
      ? '<b style="color:#e2e8f0">' + n.toLocaleString() + '</b> of ' + st.total.toLocaleString() +
        ' papers match &ldquo;' + escHtml(q) + '&rdquo; in <b style="color:#cbd5e1">' + escHtml(modeLabel) + '</b>, best fit first'
      : '<b style="color:#e2e8f0">' + st.total.toLocaleString() + '</b> papers in <b style="color:#cbd5e1">' +
        escHtml(modeLabel) + '</b>, newest first');
  document.getElementById('ps-results').innerHTML = (n === 0 && st.total)
    ? '<div style="font-size:12px;color:#475569">No paper matches this search.</div>'
    : list.slice(from, from + PS_PAGE).map((idx, k) => {
        const p = d.papers[idx];
        const href = p[3] ? 'https://doi.org/' + p[3] : 'https://pubmed.ncbi.nlm.nih.gov/' + p[0] + '/';
        return '<div style="display:flex;gap:6px;margin-bottom:6px;font-size:11px;color:#94a3b8">' +
          '<span style="color:#818cf8;font-weight:700;flex-shrink:0">' + (from + k + 1) + '.</span>' +
          '<span><a href="' + escHtml(href) + '" target="_blank" rel="noopener" ' +
          'style="color:#cbd5e1;text-decoration:underline dotted">' + escHtml(p[1]) + '</a>' +
          '<span style="color:#475569"> · ' + escHtml(p[2]) + ' · PMID ' + escHtml(p[0]) + '</span></span></div>';
      }).join('');
  const btn = (label, delta, enabled) =>
    '<button ' + (enabled ? 'onclick="psGo(' + delta + ')"' : 'disabled') +
    ' style="background:#1e293b;border:1px solid #334155;border-radius:6px;color:' + (enabled ? '#cbd5e1' : '#475569') +
    ';font-size:11px;padding:5px 10px;cursor:' + (enabled ? 'pointer' : 'default') + '">' + label + '</button>';
  document.getElementById('ps-nav').innerHTML = n > PS_PAGE
    ? btn('← Previous 10', -1, st.page > 0) +
      '<span style="font-size:11px;color:#64748b">' + (from + 1) + '–' + Math.min(from + PS_PAGE, n) + ' of ' + n.toLocaleString() + '</span>' +
      btn('Next 10 →', 1, st.page < pages - 1)
    : '';
}
