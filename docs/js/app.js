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
    rerenderCharts(window._cachedF4, window._cachedF3);
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

const STATUS_SPECS = {
  implementacao: {
    concluida: {
      short: 'Concluída',
      long: 'Implementação concluída',
      pillClass: 'status-pill--implementation-done',
      valueClass: 'accent',
      label: 'Implementação',
    },
    parcial: {
      short: 'Parcial',
      long: 'Implementação parcial',
      pillClass: 'status-pill--implementation-partial',
      valueClass: 'orange',
      label: 'Implementação',
    },
    pendente: {
      short: 'Pendente',
      long: 'Implementação pendente',
      pillClass: 'status-pill--neutral',
      valueClass: 'muted',
      label: 'Implementação',
    },
  },
  exploratorio: {
    validada: {
      short: 'Validada',
      long: 'Smoke exploratório validado',
      pillClass: 'status-pill--exploratory-done',
      valueClass: 'orange',
      label: 'Exploratório',
    },
    parcial: {
      short: 'Parcial',
      long: 'Smoke exploratório parcial',
      pillClass: 'status-pill--exploratory-partial',
      valueClass: 'orange',
      label: 'Exploratório',
    },
    pendente: {
      short: 'Pendente',
      long: 'Smoke exploratório pendente',
      pillClass: 'status-pill--neutral',
      valueClass: 'muted',
      label: 'Exploratório',
    },
  },
  oficial: {
    concluida: {
      short: 'Concluída',
      long: 'Execução oficial concluída',
      pillClass: 'status-pill--official-done',
      valueClass: 'green',
      label: 'Oficial',
    },
    em_andamento: {
      short: 'Em andamento',
      long: 'Execução oficial em andamento',
      pillClass: 'status-pill--official-progress',
      valueClass: 'accent',
      label: 'Oficial',
    },
    pendente: {
      short: 'Pendente',
      long: 'Execução oficial pendente',
      pillClass: 'status-pill--official-pending',
      valueClass: 'muted',
      label: 'Oficial',
    },
  },
};

function getStatusSpec(scope, status) {
  return STATUS_SPECS[scope]?.[status] || {
    short: status || '—',
    long: status || '—',
    pillClass: 'status-pill--neutral',
    valueClass: 'muted',
    label: scope,
  };
}

function renderStatusPill(scope, status) {
  const spec = getStatusSpec(scope, status);
  return `<span class="status-pill ${spec.pillClass}">${spec.label}: ${spec.short}</span>`;
}

function experimentalStatusCard(scope, status, detail) {
  const spec = getStatusSpec(scope, status);
  const card = el('div', 'card');
  card.innerHTML = `
    <div class="card__label">${spec.label}</div>
    <div class="card__value card__value--${spec.valueClass}">${spec.short}</div>
    ${detail ? `<div class="card__sub">${detail}</div>` : `<div class="card__sub">${spec.long}</div>`}
  `;
  return card;
}

function buildExperimentalOverview(status57) {
  const section = el('section', 'experimental-overview');
  const summary = status57.sumario || {};
  const meta = status57.meta || {};

  section.innerHTML = `
    <div class="experimental-banner">
      <div class="experimental-banner__eyebrow">Fases 5–7 · Prontidão Experimental</div>
      <h2 class="experimental-banner__title">${meta.titulo || 'Prontidão Experimental das Fases 5–7'}</h2>
      <p class="experimental-banner__lead">${meta.aviso_principal || ''}</p>
      <p class="experimental-banner__note">${summary.mensagem || ''}</p>
      <div class="experimental-banner__chips">
        ${renderStatusPill('implementacao', summary.implementacao_status)}
        ${renderStatusPill('exploratorio', summary.validacao_exploratoria_status)}
        ${renderStatusPill('oficial', summary.execucao_oficial_status)}
      </div>
    </div>
  `;

  return section;
}

function buildExperimentalComponentCard(item) {
  const card = el('div', 'card experimental-component');
  const secondary = item.modelo || item.ambiente;
  card.innerHTML = `
    <div class="experimental-component__header">
      <div>
        <div class="card__label">${item.rotulo}</div>
        ${secondary ? `<div class="experimental-component__meta">${secondary}${item.ambiente && item.modelo ? ' · ' + item.ambiente : item.ambiente ? '' : ''}</div>` : ''}
      </div>
    </div>
    <div class="experimental-component__chips">
      ${renderStatusPill('implementacao', item.implementacao_status)}
      ${renderStatusPill('exploratorio', item.validacao_exploratoria_status)}
      ${renderStatusPill('oficial', item.execucao_oficial_status)}
    </div>
    ${item.dataset_exploratorio ? `<p class="experimental-component__text"><strong>Dataset exploratório:</strong> ${item.dataset_exploratorio}</p>` : ''}
    ${item.evidencia_exploratoria ? `<p class="experimental-component__text"><strong>Evidência:</strong> ${item.evidencia_exploratoria}</p>` : ''}
    ${item.artefato_exploratorio ? `<p class="experimental-component__path"><code>${item.artefato_exploratorio}</code></p>` : ''}
    ${item.observacao ? `<p class="experimental-component__note">${item.observacao}</p>` : ''}
  `;
  return card;
}

function buildExperimentalPhase(phase) {
  const section = el('section', 'phase phase--experimental');
  const header = el('div', 'phase__header');
  const numBadge = el('div', 'phase__number phase__number--done');
  numBadge.textContent = phase.numero;

  const title = el('span', 'phase__title');
  title.textContent = `Fase ${phase.numero}: ${phase.nome}`;

  const statusGroup = el('div', 'phase__status-group');
  statusGroup.innerHTML = `
    ${renderStatusPill('implementacao', phase.implementacao_status)}
    ${renderStatusPill('exploratorio', phase.validacao_exploratoria_status)}
    ${renderStatusPill('oficial', phase.execucao_oficial_status)}
  `;

  header.append(numBadge, title, statusGroup);
  section.appendChild(header);

  const body = el('div', 'phase__body');

  if (phase.descricao) {
    const desc = el('p', 'phase__desc');
    desc.textContent = phase.descricao;
    body.appendChild(desc);
  }

  const warning = el('div', 'experimental-callout');
  warning.innerHTML = `
    <p>${phase.aviso_leitor || 'Os artefatos abaixo documentam prontidão técnica e smoke tests exploratórios.'}</p>
    ${phase.bloqueio_principal ? `<p><strong>Bloqueio atual:</strong> ${phase.bloqueio_principal}</p>` : ''}
  `;
  body.appendChild(warning);

  if (phase.componentes && phase.componentes.length > 0) {
    const componentsGrid = el('div', 'cards');
    phase.componentes.forEach(item => componentsGrid.appendChild(buildExperimentalComponentCard(item)));
    body.appendChild(componentsGrid);
  }

  section.appendChild(body);
  return section;
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
    const [respStats, respStatus57] = await Promise.all([
      fetch('data/estatisticas_corpus.json'),
      fetch('data/fases_5_7_status.json').catch(() => null),
    ]);
    if (!respStats.ok) throw new Error(`HTTP ${respStats.status}: ${respStats.statusText}`);
    const data = await respStats.json();
    let status57 = null;
    if (respStatus57 && respStatus57.ok) {
      status57 = await respStatus57.json();
    }
    loading.remove();
    render(data, status57);
    document.getElementById('footer').style.display = '';
  } catch (err) {
    loading.innerHTML = `
      <div class="error-msg">
        <p style="font-size:2rem;margin-bottom:1rem">⚠️</p>
        <p><strong>Não foi possível carregar os dados.</strong></p>
        <p style="margin-top:0.5rem;font-size:0.85rem;color:var(--text-muted)">
          Execute <code>python3 -m pipeline.fase1_4.fase04_estatisticas</code> para gerar o JSON.<br>
          Erro: ${err.message}
        </p>
      </div>`;
  }
})();

/* ===== Main Render ===== */
function render(data, status57) {
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
      metricCard('Treino', fmt(f3.treino), `${f3.split_ratio} · divisão cronológica por data`, 'accent'),
      metricCard('Teste', fmt(f3.teste), 'holdout para avaliação', 'purple'),
    );
    if (f3.categorias_pii) {
      const piiCard = el('div', 'card card--wide');
      piiCard.innerHTML = `<div class="card__label">Categorias de Dados Pessoais Substituídas</div>`;
      const tags = el('div', 'tags');
      f3.categorias_pii.forEach(c => {
        const t = el('span', 'tag');
        t.textContent = `[${c}]`;
        tags.appendChild(t);
      });
      piiCard.appendChild(tags);
      cards.appendChild(piiCard);
    }

    // PII token counts chart
    if (f3.pii_contagem) {
      const piiChartCard = el('div', 'card card--wide');
      piiChartCard.innerHTML = `
        <div class="card__label">Tokens de Dados Pessoais Substituídos <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— conformidade LGPD</span></div>
        <p class="card__hint">Total de ${f3.pii_contagem.total.toLocaleString('pt-BR')} dados pessoais substituídos por tokens neutros. Nomes são identificados em duas passagens: primeiro os precedidos de pronome de tratamento, depois suas re-menções parciais no restante do texto.</p>
        <div class="chart-container chart-container--tall"><canvas id="chart-pii"></canvas></div>`;
      cards.appendChild(piiChartCard);
    }

    // System prompt card
    if (f3.system_prompt) {
      const spCard = el('div', 'card card--wide');
      spCard.innerHTML = `
        <div class="card__label">System Prompt Canônico <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— pipeline/prompts/system_prompt.txt</span></div>
        <p class="card__hint">Instrução embutida em cada registro do JSONL (turno <code>user</code>). O mesmo prompt é usado no fine-tuning (Fase 5) e no baseline zero-shot (Fase 6), garantindo consistência experimental.</p>
        <blockquote class="system-prompt-quote">${f3.system_prompt}</blockquote>`;
      cards.appendChild(spCard);
    }

    return cards;
  }));

  // Phase 4
  const f4 = fases.fase4_estatisticas;
  if (f4) main.appendChild(buildPhase(4, f4, () => {
    const cards = el('div', 'cards');

    // Helper: build a hint paragraph from dicas
    const dicas = f4.dicas || {};
    function hintBlock(key) {
      if (!dicas[key]) return '';
      return `<p class="card__hint">${dicas[key]}</p>`;
    }

    // ── 1. PANORAMA — KPIs de destaque ──
    cards.append(
      metricCard('Retenção Global', `${f4.funil.taxa_retencao_global}%`, `${fmt(f4.funil.dump_postgresql)} → ${fmt(f4.funil.dataset_final_fase3)}`, 'green'),
      metricCard('Compressão Média', `${f4.razao_compressao.media}:1`, 'forte compressão', 'accent'),
      metricCard('Palavras (Fund.)', fmt(f4.fundamentacao.total_palavras), `média: ${fmt(f4.fundamentacao.media)} palavras`, 'purple'),
      metricCard('Palavras (Ementa)', fmt(f4.ementa.total_palavras), `média: ${fmt(f4.ementa.media)} palavras`, 'orange'),
    );

    // ── 2. ORIGEM — Como chegamos aqui ──
    const funnelCard = el('div', 'card card--wide');
    funnelCard.innerHTML = `<div class="card__label">Funil de Attrition</div>${hintBlock('funil')}<div class="chart-container"><canvas id="chart-funnel"></canvas></div>`;
    cards.appendChild(funnelCard);

    if (f4.periodo_temporal && f4.periodo_temporal.distribuicao_por_ano) {
      const tempCard = el('div', 'card card--wide');
      tempCard.innerHTML = `
        <div class="card__label">Processos por Ano <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— ${f4.periodo_temporal.data_mais_antiga} a ${f4.periodo_temporal.data_mais_recente}</span></div>
        ${hintBlock('temporal')}
        <div class="chart-container"><canvas id="chart-temporal"></canvas></div>`;
      cards.appendChild(tempCard);
    }

    // ── 3. ESTRUTURA — Distribuições de comprimento ──
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
          <tr><td>Assimetria</td><td class="num">${fmt(f4.fundamentacao.assimetria)}</td><td class="num">${fmt(f4.ementa.assimetria)}</td></tr>
          <tr><td>Curtose (Fisher)</td><td class="num">${fmt(f4.fundamentacao.curtose)}</td><td class="num">${fmt(f4.ementa.curtose)}</td></tr>
          <tr><td>CV%</td><td class="num">${f4.fundamentacao.coeficiente_variacao_pct != null ? f4.fundamentacao.coeficiente_variacao_pct + '%' : '—'}</td><td class="num">${f4.ementa.coeficiente_variacao_pct != null ? f4.ementa.coeficiente_variacao_pct + '%' : '—'}</td></tr>
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

    // Box-plot: visualização dos mesmos percentis
    const boxCard = el('div', 'card card--wide');
    boxCard.innerHTML = `
      <div class="card__label">Box-Plot Comparativo <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— Fundamentação vs. Ementa (em palavras)</span></div>
      <p class="card__hint">Cada painel usa sua própria escala. A barra central (colorida) mostra onde estão 50% dos textos; as áreas mais claras cobrem 90% do total. P25 corresponde ao primeiro quartil (Q1), P75 ao terceiro quartil (Q3), e a faixa entre eles é o intervalo interquartílico (IQR).</p>
      <div class="boxplot-grid">
        <div class="boxplot-panel"><div class="boxplot-panel__title">Fundamentação</div><div class="chart-container" style="height:260px"><canvas id="chart-boxplot-fund"></canvas></div></div>
        <div class="boxplot-panel"><div class="boxplot-panel__title">Ementa</div><div class="chart-container" style="height:260px"><canvas id="chart-boxplot-ementa"></canvas></div></div>
      </div>`;
    cards.appendChild(boxCard);

    if (f4.histograma_ementa) {
      const histECard = el('div', 'card card--wide');
      histECard.innerHTML = `
        <div class="card__label">Histograma — Comprimento das Ementas <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">(em palavras)</span></div>
        <p class="card__hint">As ementas concentram-se em poucas dezenas de palavras. Compare com o histograma das fundamentações abaixo para visualizar a compressão extrema.</p>
        <div class="chart-container"><canvas id="chart-histogram-ementa"></canvas></div>`;
      cards.appendChild(histECard);
    }

    if (f4.histograma_fundamentacao) {
      const histCard = el('div', 'card card--wide');
      histCard.innerHTML = `
        <div class="card__label">Histograma — Comprimento das Fundamentações <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">(em palavras)</span></div>
        ${hintBlock('histograma')}
        <div class="chart-container"><canvas id="chart-histogram"></canvas></div>`;
      cards.appendChild(histCard);
    }

    // ── 4. COMPRESSÃO — Relação fundamentação × ementa ──
    if (f4.scatter_compressao) {
      const scatterCard = el('div', 'card card--wide');
      scatterCard.innerHTML = `
        <div class="card__label">Dispersão dos Dados (Scatter Plot) <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— amostra de ${(Array.isArray(f4.scatter_compressao) ? f4.scatter_compressao.length : f4.scatter_compressao.n_amostra).toLocaleString('pt-BR')} pares</span></div>
        <p class="card__hint">Cada ponto representa um par fundamentação–ementa. O eixo X mostra o comprimento da fundamentação e o Y o da ementa. A concentração no canto inferior esquerdo com dispersão horizontal confirma a compressão extrema da tarefa.</p>
        <div class="chart-container chart-container--tall"><canvas id="chart-scatter"></canvas></div>`;
      cards.appendChild(scatterCard);
    }

    // ── 5. ABSTRATIVIDADE — Novel n-grams ──
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

    // ── 6. COMPOSIÇÃO TEMÁTICA — Áreas do direito ──
    if (f4.distribuicao_materias) {
      const matCard = el('div', 'card card--wide');
      matCard.innerHTML = `
        <div class="card__label">Composição Temática do Corpus <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— área do direito por ementa</span></div>
        <p class="card__hint">Distribuição das ementas por área do direito, extraída do prefixo de cada ementa. O corpus é predominantemente previdenciário e assistencial, mas inclui matérias diversas (administrativo, processual, FGTS, tributário, civil), refletindo a competência ampla dos Juizados Especiais Federais.</p>
        <div class="chart-container chart-container--tall"><canvas id="chart-materias"></canvas></div>`;
      cards.appendChild(matCard);
    }

    // ── 7. VOCABULÁRIO — Composição e domínio ──
    if (f4.vocabulario) {
      const v = f4.vocabulario;
      cards.append(
        metricCard('Vocabulário Total', fmt(v.total_unico), 'palavras únicas no corpus', 'accent'),
        metricCard('Vocab. Fundamentação', fmt(v.fundamentacao), '', 'purple'),
        metricCard('Vocab. Ementa', fmt(v.ementa), '', 'orange'),
        metricCard('Sobreposição', fmt(v.sobreposicao), `${(v.sobreposicao / v.total_unico * 100).toFixed(1)}% do vocabulário total`, 'green'),
      );
    }

    if (f4.wordcloud) {
      const wcCard = el('div', 'card card--wide');
      wcCard.innerHTML = `
        <div class="card__label">Nuvem de Palavras — Ementas <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— top-100 termos mais frequentes</span></div>
        <p class="card__hint">Termos mais frequentes nas ementas do corpus, com stop words e tokens de anonimização removidos. O tamanho é proporcional à frequência. Reflete o vocabulário jurídico que o modelo precisa dominar.</p>
        <div class="wordcloud-container"><canvas id="chart-wordcloud"></canvas></div>`;
      cards.appendChild(wcCard);
    }

    if (f4.vocabulario) {
      const vocabCard = el('div', 'card card--wide');
      vocabCard.innerHTML = `
        <div class="card__label">Sobreposição de Vocabulário <span style="color:var(--text-muted);font-weight:400;text-transform:none;letter-spacing:0">— Fundamentação vs. Ementa</span></div>
        <p class="card__hint">Embora ${(f4.vocabulario.sobreposicao / f4.vocabulario.ementa * 100).toFixed(1)}% do vocabulário das ementas apareça nas fundamentações, isso não implica viabilidade extrativa: os novel trigrams de 86,6% mostram que as ementas <strong>recombinam</strong> esse vocabulário compartilhado em sequências inteiramente novas. O modelo não precisa inventar palavras — precisa aprender a reorganizá-las. Isso é precisamente o que a sumarização abstrativa faz.</p>
        <div class="chart-container"><canvas id="chart-vocab"></canvas></div>`;
      cards.appendChild(vocabCard);
    }

    return cards;
  }));

  if (status57) {
    main.appendChild(buildExperimentalOverview(status57));
    ['fase5', 'fase6', 'fase7'].forEach(key => {
      const phase = status57.fases?.[key];
      if (phase) {
        main.appendChild(buildExperimentalPhase(phase));
      }
    });
  } else {
    const pendingPhases = [
      { num: 5, nome: 'Fine-Tuning', desc: 'Fase implementada no código, mas ainda sem snapshot de prontidão experimental disponível no dashboard.' },
      { num: 6, nome: 'Inferência e Ementas Geradas', desc: 'Runners implementados, mas sem snapshot de prontidão experimental disponível no dashboard.' },
      { num: 7, nome: 'Avaliação Final', desc: 'Infraestrutura implementada, mas sem snapshot de prontidão experimental disponível no dashboard.' },
    ];
    pendingPhases.forEach(p => {
      main.appendChild(buildPhase(p.num, { nome: p.nome, descricao: p.desc, status: 'pendente' }, null));
    });
  }

  // Cache f4/f3 and render charts
  window._cachedF4 = f4;
  window._cachedF3 = f3;
  window._chartsRendered = true;

  requestAnimationFrame(() => {
    rerenderCharts(f4, f3);
    observeStickyHeaders();
  });
}
