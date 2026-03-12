/* ===================================================================
   Pipeline Dashboard — App (main logic)
   Theme management, data loading, render pipeline
   =================================================================== */

/* ===== Theme Management ===== */
const themeToggle = document.getElementById('themeToggle');
const themeIcon   = document.getElementById('themeIcon');
const themeLabel  = document.getElementById('themeLabel');

function getTheme() {
  return localStorage.getItem('dashboard-theme') || 'light';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  themeIcon.textContent  = theme === 'dark' ? '🌙' : '☀️';
  themeLabel.textContent = theme === 'dark' ? 'Escuro' : 'Claro';
  localStorage.setItem('dashboard-theme', theme);

  if (window._chartsRendered && window._cachedF4) {
    rerenderCharts(window._cachedF4);
  }
}

applyTheme(getTheme());

themeToggle.addEventListener('click', () => {
  const current = getTheme();
  applyTheme(current === 'dark' ? 'light' : 'dark');
});

/* ===== Sticky Header Detection ===== */
const stickyObserver = new IntersectionObserver(
  entries => entries.forEach(e => {
    e.target.nextElementSibling?.classList.toggle('phase__header--stuck', !e.isIntersecting);
  }),
  { threshold: 1, rootMargin: '-1px 0px 0px 0px' }
);

// Will be called after render
function observeStickyHeaders() {
  document.querySelectorAll('.phase').forEach(phase => {
    const sentinel = document.createElement('div');
    sentinel.className = 'sticky-sentinel';
    sentinel.style.cssText = 'height:1px;width:100%;position:absolute;top:0;left:0;pointer-events:none;';
    phase.style.position = 'relative';
    phase.insertBefore(sentinel, phase.firstChild);
    stickyObserver.observe(sentinel);
  });
}

/* ===== Helpers ===== */
function fmt(n) {
  if (n == null) return '—';
  return n.toLocaleString('pt-BR');
}

function el(tag, cls) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  return e;
}

function metricCard(label, value, sub, colorClass) {
  const card = el('div', 'card');
  card.innerHTML = `
    <div class="card__label">${label}</div>
    <div class="card__value card__value--${colorClass || 'accent'}">${value}</div>
    ${sub ? `<div class="card__sub">${sub}</div>` : ''}
  `;
  return card;
}

function formatDuration(seconds) {
  if (seconds == null) return null;
  if (seconds < 60) return `${seconds}s`;
  const min = Math.floor(seconds / 60);
  const sec = seconds % 60;
  return sec > 0 ? `${min}min ${sec}s` : `${min}min`;
}

function formatArtifacts(artefatos) {
  if (!artefatos) return [];
  return Object.entries(artefatos).map(([name, size]) => ({
    name,
    size: size != null ? `${size} MB` : '—',
  }));
}

/* ===== Phase Meta Builder (clean, no emojis) ===== */
function buildPhaseMeta(data) {
  const meta = el('div', 'phase__meta');
  let html = '';

  if (data.script) {
    html += `<span class="phase__meta-item">Script: <code>${data.script}</code></span>`;
  }

  const dur = formatDuration(data.duracao_segundos);
  if (dur) {
    html += `<span class="phase__meta-item">Duração: <code>${dur}</code></span>`;
  }

  meta.innerHTML = html;

  // Artifacts as structured list
  if (data.artefatos && Array.isArray(data.artefatos) && data.artefatos.length > 0) {
    const artDiv = el('div', 'phase__artifacts');
    data.artefatos.forEach(a => {
      const isInput = a.tipo === 'entrada';
      const row = el('div', `artifact ${isInput ? 'artifact--input' : 'artifact--output'}`);
      const sizeStr = a.tamanho_mb != null ? `${a.tamanho_mb} MB` : '—';
      const ext = a.nome.split('.').pop();
      const svgIcon = (path) => `<svg class="artifact__icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${path}</svg>`;
      const icons = {
        sql:    svgIcon('<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6a8 3 0 0 0 16 0V6"/><path d="M4 12v6a8 3 0 0 0 16 0v-6"/>'),
        sqlite: svgIcon('<ellipse cx="12" cy="6" rx="8" ry="3"/><path d="M4 6v6a8 3 0 0 0 16 0V6"/><path d="M4 12v6a8 3 0 0 0 16 0v-6"/>'),
        json:   svgIcon('<path d="M7 4a2 2 0 0 0-2 2v3a2 3 0 0 1-2 3 2 3 0 0 1 2 3v3a2 2 0 0 0 2 2"/><path d="M17 4a2 2 0 0 1 2 2v3a2 3 0 0 0 2 3 2 3 0 0 0-2 3v3a2 2 0 0 1-2 2"/>'),
        jsonl:  svgIcon('<path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z"/><line x1="9" y1="13" x2="15" y2="13"/><line x1="9" y1="17" x2="13" y2="17"/>'),
      };
      const icon = icons[ext] || svgIcon('<path d="M14 3v4a1 1 0 0 0 1 1h4"/><path d="M17 21H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7l5 5v11a2 2 0 0 1-2 2z"/>');
      row.innerHTML = `
        <span class="artifact__badge">${isInput ? 'Entrada' : 'Saída'}</span>
        ${icon}
        <code class="artifact__name">${a.nome}</code>
        <span class="artifact__size">${sizeStr}</span>
        ${a.conteudo ? `<span class="artifact__desc">${a.conteudo}</span>` : ''}
      `;
      artDiv.appendChild(row);
    });
    meta.appendChild(artDiv);
  }

  return meta;
}

/* ===== Phase Section Builder ===== */
function buildPhase(num, data, contentFn) {
  const isDone  = data.status === 'concluida';
  const section = el('section', `phase${isDone ? '' : ' phase--pending'}`);
  const header  = el('div', 'phase__header');

  const numBadge = el('div', `phase__number ${isDone ? 'phase__number--done' : 'phase__number--pending'}`);
  numBadge.textContent = num;

  const title = el('span', 'phase__title');
  title.textContent = `Fase ${num}: ${data.nome}`;

  const status = el('span', `phase__status ${isDone ? 'phase__status--done' : 'phase__status--pending'}`);
  status.textContent = isDone ? '✓ Concluída' : '⏳ Pendente';

  header.append(numBadge, title, status);
  section.appendChild(header);

  // Body container — all content below the header
  const body = el('div', 'phase__body');

  if (data.descricao) {
    const desc = el('p', 'phase__desc');
    desc.textContent = data.descricao;
    body.appendChild(desc);
  }

  // Phase metadata (script, time, artifacts)
  if (isDone) {
    body.appendChild(buildPhaseMeta(data));
  }

  // Contextual narrative
  if (data.narrativa) {
    const narr = el('p', 'phase__narrativa');
    narr.textContent = data.narrativa;
    body.appendChild(narr);
  }

  if (contentFn) {
    const content = contentFn();
    if (content) body.appendChild(content);
  }

  section.appendChild(body);
  return section;
}

/* ===== Data Loading ===== */
(async function() {
  const main    = document.getElementById('main');
  const loading = document.getElementById('loading');

  try {
    const resp = await fetch('data/estatisticas_corpus.json');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
    const data = await resp.json();
    loading.remove();
    render(data);
    document.getElementById('footer').style.display = '';
  } catch (err) {
    loading.innerHTML = `
      <div class="error-msg">
        <p style="font-size:2rem;margin-bottom:1rem">⚠️</p>
        <p><strong>Não foi possível carregar os dados.</strong></p>
        <p style="margin-top:0.5rem;font-size:0.85rem;color:var(--text-muted)">
          Execute <code>python3 pipeline/04_estatisticas.py</code> para gerar o JSON.<br>
          Erro: ${err.message}
        </p>
      </div>`;
  }
})();

/* ===== Main Render ===== */
function render(data) {
  const meta = data.meta || {};
  document.getElementById('titulo').textContent       = meta.titulo_pesquisa || 'Dashboard';
  document.getElementById('autor').textContent        = meta.autor || '';
  document.getElementById('meta-gerado').textContent  = `Gerado em ${meta.gerado_em ? meta.gerado_em.replace('T', ' às ') : '—'}`;

  // Pipeline total time
  const totalTime = formatDuration(meta.pipeline_total_segundos);
  if (totalTime) {
    const timeEl = document.getElementById('meta-pipeline-time');
    timeEl.textContent = `Pipeline executado em ${totalTime}`;
    timeEl.style.display = '';
  }

  const fases = data.fases || {};
  const main  = document.getElementById('main');

  // Phase 1
  const f1 = fases.fase1_ingestao;
  if (f1) main.appendChild(buildPhase(1, f1, () => {
    const cards = el('div', 'cards');
    cards.append(
      metricCard('Registros no Dump', fmt(f1.registros_dump), 'pares voto/ementa no PostgreSQL', 'accent'),
      metricCard('Exportados', fmt(f1.registros_exportados), `−${fmt(f1.perda)} sem fundamentação/ementa`, 'green'),
      metricCard('Taxa de Retenção', `${f1.taxa_retencao}%`, 'alta completude da base', 'green'),
      metricCard('Fonte', f1.fonte || '—', '', 'purple'),
    );
    return cards;
  }));

  // Phase 2
  const f2 = fases.fase2_higienizacao;
  if (f2) main.appendChild(buildPhase(2, f2, () => {
    const cards = el('div', 'cards');
    cards.append(
      metricCard('Entrada', fmt(f2.registros_entrada), 'registros brutos', 'accent'),
      metricCard('Saída', fmt(f2.registros_saida), `−${fmt(f2.perda)} removidos por filtro`, 'green'),
      metricCard('Taxa de Retenção', `${f2.taxa_retencao}%`, 'ruído pontual', 'green'),
    );
    if (f2.regras_aplicadas) {
      const rCard = el('div', 'card card--wide');
      rCard.innerHTML = `<div class="card__label">Regras de Limpeza Aplicadas</div>`;
      const ul = el('ul', 'rules');
      f2.regras_aplicadas.forEach(r => {
        const li = el('li');
        li.textContent = r;
        ul.appendChild(li);
      });
      rCard.appendChild(ul);
      cards.appendChild(rCard);
    }
    return cards;
  }));

  // Phase 3
  const f3 = fases.fase3_anonimizacao;
  if (f3) main.appendChild(buildPhase(3, f3, () => {
    const cards = el('div', 'cards');
    cards.append(
      metricCard('Treino', fmt(f3.treino), `${f3.split_ratio} · seed=${f3.random_seed}`, 'accent'),
      metricCard('Teste', fmt(f3.teste), 'holdout para avaliação', 'purple'),
    );
    if (f3.categorias_pii) {
      const piiCard = el('div', 'card card--wide');
      piiCard.innerHTML = `<div class="card__label">Categorias PII Substituídas</div>`;
      const tags = el('div', 'tags');
      f3.categorias_pii.forEach(c => {
        const t = el('span', 'tag');
        t.textContent = `[${c}]`;
        tags.appendChild(t);
      });
      piiCard.appendChild(tags);
      cards.appendChild(piiCard);
    }
    return cards;
  }));

  // Phase 4
  const f4 = fases.fase4_estatisticas;
  if (f4) main.appendChild(buildPhase(4, f4, () => {
    const cards = el('div', 'cards');

    cards.append(
      metricCard('Retenção Global', `${f4.funil.taxa_retencao_global}%`, `${fmt(f4.funil.dump_postgresql)} → ${fmt(f4.funil.dataset_final_fase3)}`, 'green'),
      metricCard('Compressão Média', `${f4.razao_compressao.media}:1`, 'forte compressão', 'accent'),
      metricCard('Palavras (Fund.)', fmt(f4.fundamentacao.total_palavras), `média: ${fmt(f4.fundamentacao.media)} palavras`, 'purple'),
      metricCard('Palavras (Ementa)', fmt(f4.ementa.total_palavras), `média: ${fmt(f4.ementa.media)} palavras`, 'orange'),
    );

    // Vocabulary cards
    if (f4.vocabulario) {
      const v = f4.vocabulario;
      cards.append(
        metricCard('Vocabulário Total', fmt(v.total_unico), 'palavras únicas no corpus', 'accent'),
        metricCard('Vocab. Fundamentação', fmt(v.fundamentacao), '', 'purple'),
        metricCard('Vocab. Ementa', fmt(v.ementa), '', 'orange'),
        metricCard('Sobreposição', fmt(v.sobreposicao), `${(v.sobreposicao / v.total_unico * 100).toFixed(1)}% do vocabulário total`, 'green'),
      );
    }

    // Helper: build a hint paragraph from dicas
    const dicas = f4.dicas || {};
    function hintBlock(key) {
      if (!dicas[key]) return '';
      return `<p class="card__hint">${dicas[key]}</p>`;
    }

    // Funnel chart
    const funnelCard = el('div', 'card card--wide');
    funnelCard.innerHTML = `<div class="card__label">Funil de Attrition</div>${hintBlock('funil')}<div class="chart-container"><canvas id="chart-funnel"></canvas></div>`;
    cards.appendChild(funnelCard);

    // Distribution comparison table
    const distCard = el('div', 'card card--wide');
    distCard.innerHTML = `
      <div class="card__label">Distribuição de Comprimento <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— em palavras (split por espaço)</span></div>
      ${hintBlock('distribuicao')}
      <table class="comp-table">
        <thead><tr><th>Métrica</th><th style="text-align:right">Fundamentação</th><th style="text-align:right">Ementa</th></tr></thead>
        <tbody>
          <tr><td>Média</td><td class="num">${fmt(f4.fundamentacao.media)}</td><td class="num">${fmt(f4.ementa.media)}</td></tr>
          <tr><td>Mediana</td><td class="num">${fmt(f4.fundamentacao.mediana)}</td><td class="num">${fmt(f4.ementa.mediana)}</td></tr>
          <tr><td>Desvio Padrão</td><td class="num">${fmt(f4.fundamentacao.desvio_padrao)}</td><td class="num">${fmt(f4.ementa.desvio_padrao)}</td></tr>
          <tr><td>Mín.</td><td class="num">${fmt(f4.fundamentacao.min)}</td><td class="num">${fmt(f4.ementa.min)}</td></tr>
          <tr><td>P5</td><td class="num">${fmt(f4.fundamentacao.p5)}</td><td class="num">${fmt(f4.ementa.p5)}</td></tr>
          <tr><td>P25</td><td class="num">${fmt(f4.fundamentacao.p25)}</td><td class="num">${fmt(f4.ementa.p25)}</td></tr>
          <tr><td>P75</td><td class="num">${fmt(f4.fundamentacao.p75)}</td><td class="num">${fmt(f4.ementa.p75)}</td></tr>
          <tr><td>P95</td><td class="num">${fmt(f4.fundamentacao.p95)}</td><td class="num">${fmt(f4.ementa.p95)}</td></tr>
          <tr><td>Máx.</td><td class="num">${fmt(f4.fundamentacao.max)}</td><td class="num">${fmt(f4.ementa.max)}</td></tr>
          <tr><td><strong>Total</strong></td><td class="num"><strong>${fmt(f4.fundamentacao.total_palavras)}</strong></td><td class="num"><strong>${fmt(f4.ementa.total_palavras)}</strong></td></tr>
        </tbody>
      </table>`;
    cards.appendChild(distCard);

    // Novel n-grams gauges
    if (f4.novel_ngrams) {
      const ngCard = el('div', 'card card--wide');
      ngCard.innerHTML = `
        <div class="card__label">Novel N-grams — Grau de Abstratividade <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">(See et al., 2017)</span></div>
        ${hintBlock('novel_ngrams')}
        <div class="gauge-grid">
          <div class="gauge-item"><canvas id="gauge-uni"></canvas><div class="gauge-item__value">${f4.novel_ngrams.unigrams?.media || 0}%</div><div class="gauge-item__label">Unigrams</div></div>
          <div class="gauge-item"><canvas id="gauge-bi"></canvas><div class="gauge-item__value">${f4.novel_ngrams.bigrams?.media || 0}%</div><div class="gauge-item__label">Bigrams</div></div>
          <div class="gauge-item"><canvas id="gauge-tri"></canvas><div class="gauge-item__value">${f4.novel_ngrams.trigrams?.media || 0}%</div><div class="gauge-item__label">Trigrams</div></div>
        </div>`;
      cards.appendChild(ngCard);
    }

    // Histogram - Ementa (shorter texts first)
    if (f4.histograma_ementa) {
      const histECard = el('div', 'card card--wide');
      histECard.innerHTML = `
        <div class="card__label">Histograma — Comprimento das Ementas <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">(em palavras)</span></div>
        <p class="card__hint">As ementas concentram-se em poucas dezenas de palavras. Compare com o histograma das fundamentações abaixo para visualizar a compressão extrema.</p>
        <div class="chart-container"><canvas id="chart-histogram-ementa"></canvas></div>`;
      cards.appendChild(histECard);
    }

    // Histogram - Fundamentação (longer texts)
    if (f4.histograma_fundamentacao) {
      const histCard = el('div', 'card card--wide');
      histCard.innerHTML = `
        <div class="card__label">Histograma — Comprimento das Fundamentações <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">(em palavras)</span></div>
        ${hintBlock('histograma')}
        <div class="chart-container"><canvas id="chart-histogram"></canvas></div>`;
      cards.appendChild(histCard);
    }

    // Temporal distribution
    if (f4.periodo_temporal && f4.periodo_temporal.distribuicao_por_ano) {
      const tempCard = el('div', 'card card--wide');
      tempCard.innerHTML = `
        <div class="card__label">Processos por Ano <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— ${f4.periodo_temporal.data_mais_antiga} a ${f4.periodo_temporal.data_mais_recente}</span></div>
        ${hintBlock('temporal')}
        <div class="chart-container"><canvas id="chart-temporal"></canvas></div>`;
      cards.appendChild(tempCard);
    }

    return cards;
  }));

  // Phases 5-7 (pending)
  const pendingPhases = [
    { num: 5, key: 'fase5_finetuning', nome: 'Fine-Tuning', desc: 'Fine-Tuning Supervisionado do Gemini 3.1 Pro via API Google AI Studio.' },
    { num: 6, key: 'fase6_baseline', nome: 'Baseline Zero-Shot', desc: 'Inferência com modelo base (sem fine-tuning) para comparação.' },
    { num: 7, key: 'fase7_avaliacao', nome: 'Avaliação Final', desc: 'ROUGE + BERTScore + NLI (auditoria factual) + Bootstrap significance.' },
  ];
  pendingPhases.forEach(p => {
    const phaseData = fases[p.key];
    if (phaseData === null || phaseData === undefined) {
      main.appendChild(buildPhase(p.num, { nome: p.nome, descricao: p.desc, status: 'pendente' }, null));
    } else {
      main.appendChild(buildPhase(p.num, phaseData, null));
    }
  });

  // Cache f4 and render charts
  window._cachedF4 = f4;
  window._chartsRendered = true;

  requestAnimationFrame(() => {
    renderFunnelChart(f4);
    renderGauges(f4);
    renderTemporalChart(f4);
    renderHistogramChart(f4);
    renderHistogramEmentaChart(f4);
    observeStickyHeaders();
  });
}
