/* vocab-size · 前端：一个文件，两套视图（做题 / 结果），无依赖 */

const $ = sel => document.querySelector(sel);
const app = $('#app');
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let S = null;          // 服务器给的会话状态
let pending = null;    // 已作答、等待按空格的下一题
let answeredCount = 0;
let totalCount = 0;
let locked = false;

async function api(path, body) {
  const r = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : undefined);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

boot().catch(e => { app.innerHTML = `<div class="booting">出错了：${esc(e.message)}<br>可以关掉这个页面，回到对话里重试。</div>`; });

async function boot() {
  S = await api('/api/state');
  answeredCount = S.item ? S.answered - 1 : S.answered;   // answered 含当前这道未答的
  totalCount = S.total;
  if (S.done) { const r = await api('/api/result'); renderResult(r.report); }
  else renderQuestion(S.item);
}

// ================= 做题视图 =================
const TAGNAME = { zk: '中考', gk: '高考', cet4: '四级', cet6: '六级', ky: '考研', toefl: '托福', ielts: '雅思', gre: 'GRE' };

function renderQuestion(item) {
  if (!item) { app.innerHTML = `<div class="booting">题目发完了。</div>`; return; }
  locked = false;
  pending = null;
  const pct = Math.round((answeredCount / totalCount) * 100);
  const tags = (item.tag || '').split(/\s+/).filter(Boolean);

  app.innerHTML = `
    <div class="topline">
      <span class="mode-chip">${S.mode === 'track' ? '窄带追踪' : '全量测试'}</span>
      <span class="counter">第 <b>${item.index}</b> / ${item.total} 题</span>
    </div>
    <div class="rail"><span style="width:${pct}%"></span></div>

    <div class="word-block">
      <h1 class="word">${esc(item.word)}</h1>
      ${item.phonetic ? `<div class="phonetic">/${esc(item.phonetic)}/</div>` : ''}
      <div class="tags">${tags.map(t => `<span class="tag">${esc(TAGNAME[t] || t)}</span>`).join('')}</div>
    </div>

    <div class="options" id="opts">
      ${item.options.map((lines, i) => `
        <button class="opt" data-i="${i}">
          <span class="keycap">${i + 1}</span>
          <span class="gloss">${lines.map(esc).map(l => `<div>${l}</div>`).join('')}</span>
        </button>`).join('')}
      <button class="unknown" data-i="${item.unknownChoice}">
        <span class="keycap">${item.unknownChoice + 1}</span>
        <span class="gloss"><div>不认识这个词</div></span>
      </button>
    </div>
    <div id="after"></div>`;

  app.querySelectorAll('[data-i]').forEach(b => b.addEventListener('click', () => answer(Number(b.dataset.i))));
}

async function answer(choice) {
  if (locked) return;
  locked = true;
  const t0 = performance.now();
  let r;
  try { r = await api('/api/answer', { choice, ms: Math.round(t0) }); }
  catch (e) { locked = false; return; }

  answeredCount = r.answered;
  const btns = [...app.querySelectorAll('[data-i]')];
  btns.forEach(b => {
    const i = Number(b.dataset.i);
    b.disabled = true;
    if (i === r.answerIndex) b.classList.add('is-correct');
    else if (i === choice) b.classList.add('is-wrong');
    else b.classList.add('is-dim');
  });
  btns[r.answerIndex]?.scrollIntoView({ block: 'nearest' });

  const unknown = choice === r.unknownChoice;
  const verdict = r.correct ? '答对了' : (unknown ? '不认识 —— 这个选择本身就是有效信息' : '答错了');

  $('#after').innerHTML = `
    <div class="reveal">
      <div class="reveal-verdict ${r.correct ? 'good' : 'bad'}">${verdict}</div>
      <div><span class="reveal-word">${esc(r.answerWord)}</span>${r.answerPhonetic ? `<span class="reveal-phon">/${esc(r.answerPhonetic)}/</span>` : ''}</div>
      <div class="reveal-gloss">${r.answerGloss.map(esc).join('<br>')}</div>
    </div>
    <div class="next-row">
      <span class="hint">${r.done ? '这是最后一题' : '按 <kbd>空格</kbd> 继续'}</span>
      <button class="next" id="nextBtn">${r.done ? '看结果' : '下一题'}</button>
    </div>`;

  $('#nextBtn').focus();
  $('#nextBtn').addEventListener('click', advance);

  if (r.done) { pending = { done: true, report: r.report }; }
  else { pending = { done: false, item: r.item }; }
}

function advance() {
  if (!pending) return;
  const p = pending; pending = null;
  if (p.done) renderResult(p.report);
  else renderQuestion(p.item);
  window.scrollTo({ top: 0, behavior: 'instant' });
}

document.addEventListener('keydown', e => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (pending) {
    if (e.code === 'Space' || e.code === 'Enter') { e.preventDefault(); advance(); }
    return;
  }
  if (locked || !S || S.done) return;
  const n = Number(e.key);
  if (n >= 1 && n <= 5) { e.preventDefault(); answer(n - 1); }
});

// ================= 结果视图 =================
function renderResult(rep) {
  S.done = true;
  if (!rep || !rep.headline) {
    app.innerHTML = `<div class="booting">这次答题数太少，算不出结果。已按 incomplete 归档。</div>`;
    return;
  }
  const h = rep.headline;
  document.title = `词汇量约 ${h.vocab}`;

  const parts = [];
  parts.push(`
    <div class="result-head">
      <div class="result-eyebrow">${rep.mode === 'track' ? '窄带追踪' : '全量测试'} · ${rep.date || new Date().toISOString().slice(0, 10)}</div>
      <div class="big-number">${h.vocab.toLocaleString('en-US')}</div>
      <div class="big-unit">词 · 量程 ${rep.scale ? rep.scale.max.toLocaleString('en-US') : '20,000'}（COCA 频率前 2 万）</div>
      <div class="ci-row">
        <span>${h.ci95[0].toLocaleString('en-US')}</span>
        ${ciBar(h.ci95, h.vocab)}
        <span>${h.ci95[1].toLocaleString('en-US')}</span>
      </div>
      <div class="big-unit" style="margin-top:10px">95% 置信区间 · ±${h.seVocab.toLocaleString('en-US')} 词</div>
    </div>`);

  if (rep.scale?.saturated) parts.push(`<div class="warn">这个词表只收了频率前 ${rep.scale.max.toLocaleString('en-US')} 个词。母语成人一般在 2 万–3.5 万，<b>已经超出量程</b>，所以越靠近顶部读数越被压扁、越不准。你现在处在量程上端，这个数字应当看作"至少这么多"。</div>`);
  if (h.atBound) parts.push(`<div class="warn">这次几乎全对或全错，估计撞到了词表的边界，<b>这个数字不可信</b>。题目难度和你的水平差得太远，换个模式重测。</div>`);
  if (rep.thinSample) parts.push(`<div class="warn">总量只用首考词算，这次首考词只有 ${h.nFirstExposure} 个（其余 ${h.nRepeated} 个是复现词）。样本偏薄，区间比显示的更宽。</div>`);
  if (h.shapeUncertain) parts.push(`<div class="warn">你的作答曲线太扁，陡峭度 a 这次钉不死：总量是好几条可能曲线的加权平均，区间也相应加宽。曲线扁不是测失败 —— 它说明你的词汇分布不均匀：常见词里有不熟的，偏门词里有熟的。</div>`);
  parts.push(`<div class="warn" style="background:var(--accent-soft);border-color:var(--accent-line);color:#4E4267">陡峭度 a = ${h.a}${h.aFixed ? '（沿用上次 level 的测量值）' : h.shapeUncertain ? '（多条曲线的加权平均）' : '（本次拟合）'}。a 越大表示"会的就会、不会的就不会"，a 小说明半生不熟的词多。你的 a 会影响所有精度数字，第一次测完之后它就固定下来了。</div>`);

  if (rep.band) {
    const b = rep.band;
    parts.push(`<section class="card">
      <h2 class="sec">窄带追踪 <span>第 ${b.lo}–${b.hi} 名 · 共 ${b.width} 词</span></h2>
      <div class="metrics">
        <div class="metric"><div class="v">${b.hits}<span style="font-size:16px;color:var(--ink-faint)">/${b.items}</span></div><div class="k">本带答对</div><div class="d">命中率 ${b.rate}%</div></div>
        <div class="metric"><div class="v">${b.knownNow}</div><div class="k">推算已掌握</div><div class="d">占本带 ${Math.round(b.knownNow / b.width * 100)}%</div></div>
        <div class="metric ${b.deltaKnown > 0 ? 'good' : b.deltaKnown < 0 ? 'bad' : ''}"><div class="v">${b.deltaKnown == null ? '—' : (b.deltaKnown > 0 ? '+' : '') + b.deltaKnown}</div><div class="k">较上次同带</div><div class="d">${b.deltaRate == null ? '无可比记录' : `命中率 ${b.deltaRate > 0 ? '+' : ''}${b.deltaRate} 个百分点`}</div></div>
      </div>
      ${b.comparable ? '' : `<div class="trend-note" style="margin-top:16px">上次测的不是这一段，所以没有可比的增量。窄带要连续测同一段才能看出趋势 —— 下次 track 会继续用这一段，直到你把它啃完。</div>`}
    </section>`);
  } else if (h.deltaVocab != null) {
    const up = h.deltaVocab >= 0;
    const noise = Math.round(h.seVocab * 1.4);
    parts.push(`<section class="card">
      <h2 class="sec">与上次相比</h2>
      <div class="deltaline ${up ? 'delta-up' : 'delta-down'}"><b>${up ? '+' : ''}${h.deltaVocab.toLocaleString('en-US')}</b> 词 &nbsp;·&nbsp; 上次 ${esc(h.deltaVs || '')} 测的是 ${rep.trend.at(-2)?.vocab?.toLocaleString('en-US') || '?'} 词</div>
      <div class="trend-note">这个差值的测量噪声约 ±${noise.toLocaleString('en-US')} 词。小于这个量级的起伏看不出来是真的涨了还是随机抖动 —— 要盯小步进展请用 track 模式，它对窄带内的变化灵敏约 3 倍。</div>
    </section>`);
  }

  if (rep.mastery) {
    const peakLo = Math.min(...rep.mastery.filter(b => b.frac >= 5 && b.frac <= 95).map(b => b.lo), Infinity);
    const peakHi = Math.max(...rep.mastery.filter(b => b.frac >= 5 && b.frac <= 95).map(b => b.hi), -Infinity);
    parts.push(`<section class="card">
      <h2 class="sec">掌握曲线 <span>每 1000 词一段 · 紫色是过渡区</span></h2>
      <div class="bands">${rep.mastery.map(b => `
        <div class="band-row ${b.frac >= 5 && b.frac <= 95 ? 'peak' : b.frac > 95 ? 'spent' : ''}">
          <span class="band-label">${b.lo.toLocaleString('en-US')}–${b.hi.toLocaleString('en-US')}</span>
          <span class="band-track"><span class="band-fill" style="width:${b.frac}%"></span></span>
          <span class="band-pct">${b.known}</span>
        </div>`).join('')}</div>
      <div class="trend-note" style="margin-top:18px">右列是该段推算认识的词数。过渡区在第 ${peakLo === Infinity ? '—' : peakLo.toLocaleString('en-US')}–${peakHi === -Infinity ? '—' : peakHi.toLocaleString('en-US')} 名之间 —— 这段是 i+1 阅读最该盯的区间，两头（全会 / 全不会）投入产出都很低。</div>
    </section>`);
  }

  if (rep.tags) {
    const rows = Object.entries(rep.tags).filter(([, t]) => t.total >= 100).sort((x, y) => y[1].frac - x[1].frac);
    parts.push(`<section class="card">
      <h2 class="sec">考试词覆盖率 <span>该考试在本词表内的大纲词，推算你认识多少</span></h2>
      <div class="tagrows">${rows.map(([k, t]) => `
        <div class="tagrow">
          <span class="tagname">${esc(t.label)}</span>
          <span class="tagtrack"><span class="tagfill" style="width:${t.frac}%"></span></span>
          <span class="tagval">${t.frac}% · ${t.known}/${t.total}</span>
        </div>`).join('')}</div>
    </section>`);
  }

  if (rep.memory.repeated > 0) {
    const m = rep.memory;
    parts.push(`<section class="card">
      <h2 class="sec">长期记忆 <span>${m.repeated} 个复现词 · 都是 90 天以前见过的</span></h2>
      <div class="metrics">
        <div class="metric good"><div class="v">${m.retention}%</div><div class="k">保持率</div><div class="d">上次对、这次还对</div></div>
        <div class="metric bad"><div class="v">${m.forgetting}%</div><div class="k">遗忘率</div><div class="d">上次对、这次错了</div></div>
        <div class="metric"><div class="v">${m.recovery}%</div><div class="k">挽回率</div><div class="d">上次错、这次对了</div></div>
      </div>
      <div class="trend-note" style="margin-top:16px">这三个数<b>不进总量</b>。总量只用首考词算，否则"记住了上次的答案"会把词汇量越测越高。复现词单独放在这里看记忆牢不牢。</div>
    </section>`);
  }

  const trend = rep.trend || [];
  parts.push(`<section class="card">
    <h2 class="sec">历史趋势 <span>${trend.length} 次记录</span></h2>
    ${trend.length >= 2 ? trendSvg(trend) : `<div class="trend-note">只有一次记录，画不出趋势。level 建议每半年测一次；想盯小步进展用 track，每季度一次。</div>`}
    ${trend.some(t => t.mode === 'track') ? `<div class="trend-note" style="margin-top:10px">空心点 = track 窄带测出的总量推算；实心点 = level 全量测。阴影是 95% 置信区间。</div>` : ''}
  </section>`);

  if (rep.wrong?.length) {
    parts.push(`<details class="wrong">
      <summary>答错的词 · ${rep.wrong.length} 个</summary>
      <div class="wronglist">${rep.wrong.map(w => `
        <div class="wrongitem">
          <div class="w">${esc(w.word)}${w.phonetic ? `<i>/${esc(w.phonetic)}/</i>` : ''}${w.tag ? `<em>${esc(w.tag.split(/\s+/).map(t => TAGNAME[t] || t).join(' '))}</em>` : ''}</div>
          <div class="g">${(w.glossLines || []).map(esc).join('<br>')}</div>
          <div class="r">第 ${w.rank.toLocaleString('en-US')} 名</div>
        </div>`).join('')}</div>
    </details>`);
  }

  parts.push(`<div class="finish-row"><div class="finish-note">
    结果已写入本地历史。<br>
    ${rep.mode === 'level' ? '下一次 level 建议 6–12 个月之后；这期间想看看有没有长进，用 track。' : '下一次 track 建议 3 个月之后 —— 90 天冷却期一过，这次的词才能重新出现在题里。'}
  </div></div>`);

  app.innerHTML = parts.join('');
}

function ciBar(ci, v) {
  const lo = 1, hi = 20000;
  const p = x => Math.max(0, Math.min(100, ((x - lo) / (hi - lo)) * 100));
  return `<span class="ci-bar"><i style="left:${p(ci[0])}%;width:${p(ci[1]) - p(ci[0])}%"></i><b style="left:${p(v)}%"></b></span>`;
}

function trendSvg(trend) {
  const W = 600, H = 150, PL = 46, PR = 12, PT = 14, PB = 26;
  const vs = trend.flatMap(t => [t.vocab, ...(t.ci95 || [])]);
  let lo = Math.min(...vs), hi = Math.max(...vs);
  const pad = Math.max(300, (hi - lo) * 0.25);
  lo = Math.max(0, lo - pad); hi += pad;
  const X = i => PL + (trend.length === 1 ? (W - PL - PR) / 2 : (i / (trend.length - 1)) * (W - PL - PR));
  const Y = v => PT + (1 - (v - lo) / (hi - lo || 1)) * (H - PT - PB);

  const area = trend.length > 1
    ? `<polygon class="band" points="${trend.map((t, i) => `${X(i)},${Y(t.ci95?.[1] ?? t.vocab)}`).join(' ')} ${trend.map((t, i) => `${X(trend.length - 1 - i)},${Y(trend[trend.length - 1 - i].ci95?.[0] ?? trend[trend.length - 1 - i].vocab)}`).join(' ')}"></polygon>` : '';
  const line = trend.length > 1
    ? `<polyline class="line" points="${trend.map((t, i) => `${X(i)},${Y(t.vocab)}`).join(' ')}"></polyline>` : '';
  const dots = trend.map((t, i) => `<circle class="dot${t.mode === 'track' ? ' open' : ''}" cx="${X(i)}" cy="${Y(t.vocab)}" r="${i === trend.length - 1 ? 4 : 3}"></circle>`).join('');
  const labels = trend.map((t, i) => `<text x="${X(i)}" y="${H - 6}" text-anchor="middle">${esc(t.date.slice(5))}</text>`).join('');
  const grid = [0, 0.5, 1].map(f => {
    const v = lo + f * (hi - lo);
    return `<line class="grid" x1="${PL}" y1="${Y(v)}" x2="${W - PR}" y2="${Y(v)}"></line><text x="${PL - 8}" y="${Y(v) + 3}" text-anchor="end">${Math.round(v).toLocaleString('en-US')}</text>`;
  }).join('');
  return `<svg class="trend-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">${grid}${area}${line}${dots}${labels}</svg>`;
}
