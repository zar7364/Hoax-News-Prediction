/* AntiHoaxID - ML + Explainable AI + Google Custom Search */

if (typeof pdfjsLib !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc =
    'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';
}

const MAX_SIZE      = 10 * 1024 * 1024;
const ALLOW_EXT     = ['.pdf', '.doc', '.docx'];
const ALLOW_MIME    = [
  'application/pdf',
  'application/msword',
  'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
];
const SERPER_KEY = '1aea5942fcd7e598aff85b1f59a5bbed997837f6';
const SERPER_URL = 'https://google.serper.dev/search';
const W_ML       = 0.50;
const W_WEB      = 0.50;
const WEB_PENALTY = 83; // skor hoax default jika 0 hasil ditemukan

let currentTab   = 'upload';
let selectedFile = null;
let MODEL        = null;

/* ══════════════════════════════════════════════════════
   MODEL LOAD
   ══════════════════════════════════════════════════════ */

async function loadModel() {
  try {
    const data = await (await fetch('model.json')).json();
    MODEL = {
      vocab:       data.vocabulary,
      idf:         data.idf,
      sublinearTf: data.sublinear_tf,
      ngramRange:  data.ngram_range,
      classes:     data.classes,
      coef:        data.coef[0],
      intercept:   data.intercept[0],
      nFeatures:   data.idf.length,
    };
  } catch (e) { console.error('Gagal memuat model:', e); }
}

/* ══════════════════════════════════════════════════════
   TEXT PROCESSING
   ══════════════════════════════════════════════════════ */

function tokenize(text) {
  return text.toLowerCase().match(/\b\w\w+\b/g) || [];
}

function getNgrams(tokens, mn, mx) {
  const out = [];
  for (let n = mn; n <= mx; n++)
    for (let i = 0; i <= tokens.length - n; i++)
      out.push(tokens.slice(i, i + n).join(' '));
  return out;
}

function buildTfidf(text) {
  const tokens = tokenize(text);
  const ngrams = getNgrams(tokens, MODEL.ngramRange[0], MODEL.ngramRange[1]);

  const counts = {};
  for (const ng of ngrams)
    if (ng in MODEL.vocab) counts[ng] = (counts[ng] || 0) + 1;

  // Unnormalized TF-IDF
  const unnorm = {};
  let normSq = 0;
  for (const [term, cnt] of Object.entries(counts)) {
    const idx = MODEL.vocab[term];
    const tf  = MODEL.sublinearTf ? Math.log(1 + cnt) : cnt;
    const val = tf * MODEL.idf[idx];
    unnorm[term] = { val, idx };
    normSq += val * val;
  }

  // L2 normalise
  const norm = Math.sqrt(normSq);
  const vec  = new Float64Array(MODEL.nFeatures);
  for (const [, { val, idx }] of Object.entries(unnorm))
    vec[idx] = norm > 0 ? val / norm : val;

  return { vec, norm, unnorm };
}

/* ══════════════════════════════════════════════════════
   ML PREDICT
   ══════════════════════════════════════════════════════ */

function mlPredict(text) {
  const { vec } = buildTfidf(text);
  let dec = MODEL.intercept;
  for (let i = 0; i < vec.length; i++) dec += vec[i] * MODEL.coef[i];
  const pValid = 1 / (1 + Math.exp(-dec));
  const pHoax  = 1 - pValid;
  return {
    proba_hoax:  +(pHoax  * 100).toFixed(1),
    proba_valid: +(pValid * 100).toFixed(1),
  };
}

/* ══════════════════════════════════════════════════════
   EXPLAINABLE AI
   ══════════════════════════════════════════════════════ */

function explainPrediction(text) {
  const { vec, unnorm } = buildTfidf(text);

  // contribution[i] = normalized_tfidf[i] * coef[i]
  // positive coef → pushes toward VALID (classes[1])
  // negative coef → pushes toward HOAX (classes[0])
  const contributions = [];
  for (const [term, { idx }] of Object.entries(unnorm)) {
    const contrib = vec[idx] * MODEL.coef[idx];
    contributions.push({ term, contrib, abs: Math.abs(contrib) });
  }

  contributions.sort((a, b) => b.abs - a.abs);
  const maxAbs = contributions[0]?.abs || 1;

  const hoaxWords = contributions
    .filter(c => c.contrib < 0)
    .slice(0, 6)
    .map(c => ({ term: c.term, pct: Math.round(c.abs / maxAbs * 100), score: c.contrib }));

  const validWords = contributions
    .filter(c => c.contrib > 0)
    .slice(0, 6)
    .map(c => ({ term: c.term, pct: Math.round(c.abs / maxAbs * 100), score: c.contrib }));

  // Summary sentence
  const topHoax  = hoaxWords.slice(0, 3).map(w => `"${w.term}"`).join(', ');
  const topValid = validWords.slice(0, 3).map(w => `"${w.term}"`).join(', ');

  let summary = '';
  if (hoaxWords.length && validWords.length) {
    summary = `Kata seperti ${topHoax} mendorong skor hoax, sedangkan ${topValid || 'tidak ada'} mengindikasikan keaslian.`;
  } else if (hoaxWords.length) {
    summary = `Kata seperti ${topHoax} secara dominan mendorong prediksi hoax.`;
  } else if (validWords.length) {
    summary = `Kata seperti ${topValid} mengindikasikan konten ini lebih condong ke fakta.`;
  }

  return { hoaxWords, validWords, summary };
}

/* ══════════════════════════════════════════════════════
   GOOGLE FACT CHECK API
   ══════════════════════════════════════════════════════ */

async function fetchGoogleSearch(text) {
  try {
    const words = text.replace(/[^a-zA-Z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 3);
    const query = words.slice(0, 8).join(' ');
    const res = await fetch(SERPER_URL, {
      method:  'POST',
      headers: { 'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json' },
      body:    JSON.stringify({ q: query, gl: 'id', hl: 'id', num: 10 }),
    });
    if (!res.ok) {
      if (res.status === 429) showQuotaToast();
      return { _apiError: true, status: res.status };
    }
    return await res.json();
  } catch (e) {
    return { _apiError: true, status: 0 };
  }
}

async function fetchTurnbackHoax(text) {
  try {
    const words = text.replace(/[^a-zA-Z0-9\s]/g, ' ').split(/\s+/).filter(w => w.length > 3);
    const query = 'site:turnbackhoax.id ' + words.slice(0, 6).join(' ');
    const res = await fetch(SERPER_URL, {
      method:  'POST',
      headers: { 'X-API-KEY': SERPER_KEY, 'Content-Type': 'application/json' },
      body:    JSON.stringify({ q: query, gl: 'id', hl: 'id', num: 10 }),
    });
    if (!res.ok) return { _apiError: true, status: res.status };
    return await res.json();
  } catch (e) {
    return { _apiError: true, status: 0 };
  }
}

const HOAX_DOMAINS  = ['turnbackhoax', 'cekfakta', 'mafindo', 'hoaxbuster', 'stophoax', 'kominfo.go.id/content/detail'];
const VALID_DOMAINS = ['kompas', 'detik', 'tempo', 'antaranews', 'cnnindonesia', 'liputan6', 'tribunnews', 'republika', 'sindonews', 'okezone', 'mediaindonesia', 'beritasatu'];
const HOAX_KEYWORDS = ['hoax', 'tidak benar', 'palsu', 'disinformasi', 'klarifikasi', 'keliru', 'menyesatkan', 'dibantah', 'cek fakta', 'fakta atau hoax', 'bukan fakta'];

function computeWebScore(googleData, tbhData) {
  const apiOk = !googleData?._apiError;
  if (!apiOk)
    return { results: [], tbhResults: [], webScore: null, total: 0, apiOk: false, penalty: false, errorStatus: googleData?.status || 0 };

  let googleResults = [];
  let hoaxSignal = 0;

  if (googleData?.organic?.length) {
    for (const item of googleData.organic) {
      const title   = (item.title       || '').toLowerCase();
      const snippet = (item.snippet     || '').toLowerCase();
      const domain  = (item.displayLink || '').toLowerCase();
      let signal = 'neutral';
      if (HOAX_DOMAINS.some(d => domain.includes(d))) {
        hoaxSignal += 2; signal = 'hoax';
      } else if (HOAX_KEYWORDS.some(k => title.includes(k) || snippet.includes(k))) {
        hoaxSignal += 1; signal = 'hoax';
      }
      googleResults.push({ title: item.title || '', link: item.link || '#', domain: item.displayLink || '', snippet: item.snippet || '', signal });
    }
  }

  let tbhResults = [];
  let tbhSignal = 0;

  if (!tbhData?._apiError && tbhData?.organic?.length) {
    for (const item of tbhData.organic) {
      tbhResults.push({ title: item.title || '', link: item.link || '#', domain: item.displayLink || 'turnbackhoax.id', snippet: item.snippet || '', signal: 'hoax' });
      tbhSignal += 2;
    }
  }

  const totalResults = googleResults.length + tbhResults.length;
  const totalSignal  = hoaxSignal + tbhSignal;

  if (totalResults === 0)
    return { results: [], tbhResults: [], webScore: WEB_PENALTY, total: 0, apiOk: true, penalty: true };

  const webScore = totalSignal > 0
    ? Math.min(97, 50 + (totalSignal / totalResults) * 80)
    : 50;

  return { results: googleResults, tbhResults, webScore, total: totalSignal, apiOk: true, penalty: false };
}

function interpolate(mlHoax, webScore) {
  if (webScore === null) return mlHoax;
  return W_ML * mlHoax + W_WEB * webScore;
}

/* ══════════════════════════════════════════════════════
   FILE EXTRACTION
   ══════════════════════════════════════════════════════ */

async function extractPdf(file) {
  const pdf  = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
  let text   = '';
  for (let i = 1; i <= pdf.numPages; i++) {
    const c = await (await pdf.getPage(i)).getTextContent();
    text += c.items.map(s => s.str).join(' ') + '\n';
  }
  return text.trim();
}

async function extractDocx(file) {
  return (await mammoth.extractRawText({ arrayBuffer: await file.arrayBuffer() })).value.trim();
}

async function extractText(file) {
  const ext = file.name.split('.').pop().toLowerCase();
  if (ext === 'pdf')                   return extractPdf(file);
  if (ext === 'docx' || ext === 'doc') return extractDocx(file);
  throw new Error('Format tidak didukung.');
}

/* ══════════════════════════════════════════════════════
   UI HELPERS
   ══════════════════════════════════════════════════════ */

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(`tab-${tab}`).classList.add('active');
  clearError(); updateBtn();
}

function handleDragOver(e)  { e.preventDefault(); document.getElementById('dropZone').classList.add('dragging'); }
function handleDragLeave(e) { e.preventDefault(); document.getElementById('dropZone').classList.remove('dragging'); }
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('dragging');
  if (e.dataTransfer.files.length) processFile(e.dataTransfer.files[0]);
}
function handleFileSelect(e) { if (e.target.files[0]) { processFile(e.target.files[0]); e.target.value = ''; } }

function processFile(file) {
  clearError();
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!ALLOW_MIME.includes(file.type) && !ALLOW_EXT.includes(ext))
    return showError('Format tidak didukung. Gunakan PDF, DOC, atau DOCX.');
  if (file.size > MAX_SIZE)
    return showError(`Ukuran (${(file.size/1048576).toFixed(1)} MB) melebihi batas 10 MB.`);
  selectedFile = file; showPreview(file); updateBtn();
}

function showPreview(file) {
  const ext  = file.name.split('.').pop().toLowerCase();
  const icon = document.getElementById('fileIcon');
  icon.textContent = ext === 'pdf' ? 'PDF' : 'DOC';
  icon.className   = 'fp-icon ' + (ext === 'pdf' ? 'pdf' : 'doc');
  document.getElementById('fileName').textContent = file.name;
  document.getElementById('fileSize').textContent = fmtSize(file.size);
  document.getElementById('filePreview').classList.remove('hidden');
}

function removeFile() {
  selectedFile = null;
  document.getElementById('filePreview').classList.add('hidden');
  clearError(); updateBtn();
}

function fmtSize(b) {
  if (b < 1024)    return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(2) + ' MB';
}

function handleTextInput() {
  document.getElementById('charCount').textContent =
    document.getElementById('textInput').value.length.toLocaleString('id-ID');
  clearError(); updateBtn();
}

function clearText() {
  document.getElementById('textInput').value = '';
  document.getElementById('charCount').textContent = '0';
  clearError(); updateBtn();
}

function updateBtn() {
  const ready = currentTab === 'upload'
    ? selectedFile !== null
    : document.getElementById('textInput').value.trim().length >= 20;
  document.getElementById('submitBtn').disabled = !ready;
}

function showError(msg) {
  document.getElementById('errorText').textContent = msg;
  document.getElementById('errorMsg').classList.remove('hidden');
}

function clearError() { document.getElementById('errorMsg').classList.add('hidden'); }

function setLoading(on, msg = '') {
  document.getElementById('submitLabel').classList.toggle('hidden', on);
  const el = document.getElementById('submitLoading');
  el.classList.toggle('hidden', !on);
  if (on) el.innerHTML = `<span class="spin"></span> ${msg}`;
  document.getElementById('submitBtn').disabled = on;
}

/* ══════════════════════════════════════════════════════
   ANALYZE
   ══════════════════════════════════════════════════════ */

async function analyzeContent() {
  if (!MODEL) return showError('Model belum dimuat. Coba refresh halaman.');
  clearError();
  setLoading(true, 'Menganalisis teks...');

  try {
    const text = currentTab === 'text'
      ? document.getElementById('textInput').value.trim()
      : await extractText(selectedFile);

    if (!text || text.length < 20)
      throw new Error('Teks terlalu pendek atau tidak dapat diekstrak dari file.');

    const mlResult    = mlPredict(text);
    const explanation = explainPrediction(text);

    setLoading(true, 'Mencari di Google & TurnbackHoax.id...');
    const [searchData, tbhData] = await Promise.all([
      fetchGoogleSearch(text),
      fetchTurnbackHoax(text),
    ]);
    const { results, tbhResults, webScore, total, apiOk, penalty } = computeWebScore(searchData, tbhData);

    const finalHoax  = interpolate(mlResult.proba_hoax, webScore);
    const finalValid = 100 - finalHoax;
    const verdict    = finalHoax > 50 ? 'hoax' : 'valid';
    const confidence = +(verdict === 'hoax' ? finalHoax : finalValid).toFixed(1);

    setLoading(false);
    showResult({
      verdict, confidence,
      proba_hoax:  +finalHoax.toFixed(1),
      proba_valid: +finalValid.toFixed(1),
      mlHoax:      mlResult.proba_hoax,
      webScore:    webScore !== null ? +webScore.toFixed(1) : null,
      results, tbhResults, webTotal: total,
      webApiOk: apiOk, webPenalty: penalty, webErrorStatus: searchData?._apiError ? searchData.status : 0,
      explanation,
    });

  } catch (err) {
    setLoading(false);
    showError(err.message || 'Terjadi kesalahan saat analisis.');
  }
}

/* ══════════════════════════════════════════════════════
   RENDER RESULT
   ══════════════════════════════════════════════════════ */

const ratingClass = (r) => {
  const low = r.toLowerCase();
  if (HOAX_WORDS.some(k  => low.includes(k)))  return 'rating-hoax';
  if (VALID_WORDS.some(k => low.includes(k)))  return 'rating-valid';
  return 'rating-neutral';
};

function handleModalClick(e) {
  if (e.target === document.getElementById('resultModal')) analyzeAgain();
}

function showResult(r) {
  document.getElementById('resultModal').classList.remove('hidden');

  // ── Verdict ──────────────────────────────────────────
  const vBox = document.getElementById('verdictBox');
  vBox.className = `verdict-box ${r.verdict}`;
  document.getElementById('verdictIcon').innerHTML = r.verdict === 'hoax'
    ? '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#dc2626" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/></svg>'
    : '<svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#16a34a" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="10"/><polyline points="9 12 11 14 15 10"/></svg>';
  document.getElementById('verdictText').textContent = r.verdict === 'hoax' ? 'Hoax' : 'Fakta';
  const badge = document.getElementById('confidenceBadge');
  badge.className   = `v-badge ${r.verdict}`;
  badge.textContent = `${r.confidence}% Yakin`;

  // ── Confidence bar ────────────────────────────────────
  document.getElementById('confidencePercent').textContent = r.confidence + '%';
  const fill = document.getElementById('confidenceFill');
  fill.className = `conf-fill ${r.verdict}`;
  setTimeout(() => { fill.style.width = r.confidence + '%'; }, 80);

  // ── Score Breakdown ───────────────────────────────────
  const bd = document.getElementById('scoreBreakdown');
  if (r.webApiOk) {
    bd.classList.remove('hidden');
    document.getElementById('mlScore').textContent  = r.mlHoax + '%';
    document.getElementById('fcScore').textContent  = r.webPenalty
      ? `${r.webScore}% (penalti)`
      : r.webScore + '%';
    document.getElementById('wmlScore').textContent = r.proba_hoax + '%';
  } else {
    bd.classList.add('hidden');
  }

  // ── Explainable AI ────────────────────────────────────
  const xai = document.getElementById('xaiSection');
  const { hoaxWords, validWords, summary } = r.explanation;

  if (hoaxWords.length || validWords.length) {
    xai.classList.remove('hidden');
    document.getElementById('xaiSummary').textContent = summary;

    document.getElementById('hoaxWords').innerHTML = hoaxWords.length
      ? hoaxWords.map(w => `
          <div class="xai-word">
            <span class="xai-term">${w.term}</span>
            <div class="xai-bar-wrap"><div class="xai-bar xai-bar-hoax" style="width:${w.pct}%"></div></div>
            <span class="xai-pct">${w.pct}%</span>
          </div>`).join('')
      : '<span class="xai-none">Tidak ada</span>';

    document.getElementById('validWords').innerHTML = validWords.length
      ? validWords.map(w => `
          <div class="xai-word">
            <span class="xai-term">${w.term}</span>
            <div class="xai-bar-wrap"><div class="xai-bar xai-bar-valid" style="width:${w.pct}%"></div></div>
            <span class="xai-pct">${w.pct}%</span>
          </div>`).join('')
      : '<span class="xai-none">Tidak ada</span>';
  } else {
    xai.classList.add('hidden');
  }

  // ── Detail sentences ──────────────────────────────────
  const isHoax = r.verdict === 'hoax';
  const totalWebResults = r.results.length + (r.tbhResults?.length || 0);
  const webSentence = !r.webApiOk
    ? (r.webErrorStatus === 403
        ? 'Akses ditolak (403): API key perlu diizinkan untuk Custom Search API di Google Cloud Console.'
        : `Pencarian web gagal (error ${r.webErrorStatus || 'jaringan'}). Prediksi hanya dari model ML.`)
    : r.webPenalty
      ? `Tidak ditemukan hasil di Google maupun TurnbackHoax.id. Skor penalti <strong>${WEB_PENALTY}%</strong> diterapkan karena klaim tak terverifikasi lebih berisiko.`
      : r.webTotal > 0
        ? `Ditemukan <strong>${r.webTotal} sinyal debunking</strong> dari ${totalWebResults} hasil (Google + TurnbackHoax.id), skor web: ${r.webScore}%.`
        : `Ditemukan ${totalWebResults} hasil web, tidak ada sinyal debunking (skor netral 50%).`;
  document.getElementById('detailsGrid').innerHTML = [
    { c: isHoax ? 'red' : 'green', t: `Model mengklasifikasikan konten ini sebagai <strong>${isHoax ? 'hoax' : 'fakta'}</strong> dengan keyakinan ${r.confidence}%.` },
    { c: r.webPenalty ? 'red' : 'yellow', t: webSentence },
    { c: 'yellow', t: 'Tetap verifikasi secara mandiri ke sumber berita resmi sebelum menyebarkan.' },
  ].map(d => `<div class="det-item"><span class="det-dot ${d.c}"></span><span class="det-text">${d.t}</span></div>`).join('');

  // ── Google Search Results (top 10) ───────────────────────
  const srcSection = document.getElementById('fcSources');
  if (r.results.length) {
    srcSection.classList.remove('hidden');
    document.getElementById('fcCount').textContent = `${r.results.length} hasil`;
    document.getElementById('fcList').innerHTML = r.results.map((s, i) => `
      <a href="${s.link}" target="_blank" rel="noopener" class="fc-item">
        <div class="fc-item-top">
          <span class="fc-num">${i + 1}</span>
          <span class="fc-publisher">${s.domain}</span>
          <span class="fc-badge ${s.signal === 'hoax' ? 'rating-hoax' : s.signal === 'valid' ? 'rating-valid' : 'rating-neutral'}">${s.signal === 'hoax' ? 'Hoax Signal' : s.signal === 'valid' ? 'Valid' : 'Netral'}</span>
        </div>
        <p class="fc-title">${s.title}</p>
        ${s.snippet ? `<p class="fc-claim">${s.snippet.substring(0, 130)}${s.snippet.length > 130 ? '...' : ''}</p>` : ''}
      </a>`).join('');
  } else {
    srcSection.classList.add('hidden');
  }

  // ── TurnbackHoax.id Results (top 10) ──────────────────────
  const tbhSection = document.getElementById('tbhSources');
  if (r.tbhResults?.length) {
    tbhSection.classList.remove('hidden');
    document.getElementById('tbhCount').textContent = `${r.tbhResults.length} hasil`;
    document.getElementById('tbhList').innerHTML = r.tbhResults.map((s, i) => `
      <a href="${s.link}" target="_blank" rel="noopener" class="fc-item">
        <div class="fc-item-top">
          <span class="fc-num">${i + 1}</span>
          <span class="fc-publisher">${s.domain}</span>
          <span class="fc-badge rating-hoax">Hoax Signal</span>
        </div>
        <p class="fc-title">${s.title}</p>
        ${s.snippet ? `<p class="fc-claim">${s.snippet.substring(0, 130)}${s.snippet.length > 130 ? '...' : ''}</p>` : ''}
      </a>`).join('');
  } else {
    tbhSection.classList.add('hidden');
  }
}

/* ══════════════════════════════════════════════════════
   RESET
   ══════════════════════════════════════════════════════ */

function showQuotaToast() {
  const t = document.getElementById('quotaToast');
  t.classList.remove('hidden');
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add('hidden'), 8000);
}

function analyzeAgain() {
  document.getElementById('resultModal').classList.add('hidden');
  document.getElementById('confidenceFill').style.width = '0';
  removeFile(); clearText(); updateBtn();
}

document.addEventListener('DOMContentLoaded', () => { updateBtn(); loadModel(); });
