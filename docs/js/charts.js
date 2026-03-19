/* ===================================================================
   Pipeline Dashboard — Charts (Chart.js)
   Funnel, Gauges, Temporal, Histogramas
   Requer: chart.js + chartjs-plugin-datalabels
   =================================================================== */

/**
 * Lê cores do CSS custom properties (theme-aware)
 */
function getChartColors() {
  const style = getComputedStyle(document.documentElement);
  return {
    accent:   style.getPropertyValue('--accent').trim(),
    accent2:  style.getPropertyValue('--accent-2').trim(),
    accent3:  style.getPropertyValue('--accent-3').trim(),
    accent4:  style.getPropertyValue('--accent-4').trim(),
    grid:     style.getPropertyValue('--chart-grid').trim(),
    text:     style.getPropertyValue('--chart-text').trim(),
    gaugeBg:  style.getPropertyValue('--gauge-bg').trim(),
    label:    style.getPropertyValue('--datalabel-color').trim(),
  };
}

/**
 * Destroi todos os Chart.js instances (para troca de tema)
 */
function destroyAllCharts() {
  Object.values(Chart.instances).forEach(i => i.destroy());
}

/**
 * Re-renderiza todos os gráficos com cores do tema atual
 */
function rerenderCharts(f4, f3) {
  destroyAllCharts();
  renderFunnelChart(f4);
  renderGauges(f4);
  renderTemporalChart(f4);
  renderHistogramChart(f4);
  renderHistogramEmentaChart(f4);
  renderBoxPlot(f4);
  renderScatterCompression(f4);
  renderPiiChart(f3);
  renderWordCloud(f4);
  renderVocabOverlap(f4);
}

/* ===== Funil de Attrition ===== */
function renderFunnelChart(f4) {
  const canvas = document.getElementById('chart-funnel');
  if (!canvas || !f4?.funil) return;
  const c = getChartColors();

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Dump PostgreSQL', 'Pós-filtro nulos', 'Pós-limpeza', 'Dataset Final', 'Treino', 'Teste'],
      datasets: [{
        data: [
          f4.funil.dump_postgresql,
          f4.funil.apos_filtro_nulos_fase1,
          f4.funil.apos_limpeza_fase2,
          f4.funil.dataset_final_fase3,
          f4.funil.treino,
          f4.funil.teste,
        ],
        backgroundColor: [
          c.accent + '55', c.accent + '77', c.accent + '99',
          c.accent3 + 'cc', c.accent2 + 'cc', c.accent4 + 'cc',
        ],
        borderColor: [c.accent, c.accent, c.accent, c.accent3, c.accent2, c.accent4],
        borderWidth: 2,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw.toLocaleString('pt-BR')} registros` } },
        datalabels: {
          anchor: 'end',
          align(ctx) {
            const max = Math.max(...ctx.dataset.data);
            return ctx.dataset.data[ctx.dataIndex] < max * 0.2 ? 'right' : 'left';
          },
          color: c.label,
          font: { size: 12, weight: '700', family: 'Inter' },
          formatter: v => v.toLocaleString('pt-BR'),
          padding: { right: 8, left: 4 },
        },
      },
      scales: {
        x: {
          grid: { color: c.grid },
          ticks: { color: c.text, font: { size: 12, weight: '500' }, callback: v => v.toLocaleString('pt-BR') },
        },
        y: {
          grid: { display: false },
          ticks: { color: c.text, font: { size: 12, weight: '600' } },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}

/* ===== Novel N-Grams (Gauges) ===== */
function renderGauges(f4) {
  if (!f4?.novel_ngrams) return;
  const c = getChartColors();

  const gauges = [
    { id: 'gauge-uni', value: f4.novel_ngrams.unigrams?.media || 0, color: c.accent },
    { id: 'gauge-bi',  value: f4.novel_ngrams.bigrams?.media || 0,  color: c.accent2 },
    { id: 'gauge-tri', value: f4.novel_ngrams.trigrams?.media || 0, color: c.accent3 },
  ];

  gauges.forEach(g => {
    const canvas = document.getElementById(g.id);
    if (!canvas) return;
    new Chart(canvas, {
      type: 'doughnut',
      data: {
        datasets: [{
          data: [g.value, 100 - g.value],
          backgroundColor: [g.color, c.gaugeBg],
          borderWidth: 0,
        }],
      },
      options: {
        cutout: '75%',
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { enabled: false },
          datalabels: { display: false },
        },
      },
    });
  });
}

/* ===== Distribuição Temporal ===== */
function renderTemporalChart(f4) {
  const canvas = document.getElementById('chart-temporal');
  if (!canvas || !f4?.periodo_temporal?.distribuicao_por_ano) return;
  const c = getChartColors();

  const dist = f4.periodo_temporal.distribuicao_por_ano;
  const labels = Object.keys(dist);
  const values = Object.values(dist);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Processos cadastrados',
        data: values,
        backgroundColor: c.accent2 + 'aa',
        borderColor: c.accent2,
        borderWidth: 2,
        borderRadius: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw.toLocaleString('pt-BR')} processos cadastrados` } },
        datalabels: {
          anchor: 'end',
          align: 'end',
          color: c.label,
          font: { size: 13, weight: '700', family: 'Inter' },
          formatter: v => v.toLocaleString('pt-BR'),
        },
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: c.text, font: { size: 14, weight: '700' } },
        },
        y: {
          grid: { color: c.grid },
          ticks: { color: c.text, font: { size: 12, weight: '500' }, callback: v => v.toLocaleString('pt-BR') },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}

/* ===== Histograma de Comprimento ===== */
function renderHistogramChart(f4) {
  const canvas = document.getElementById('chart-histogram');
  if (!canvas || !f4?.histograma_fundamentacao) return;
  const c = getChartColors();

  const labels = Object.keys(f4.histograma_fundamentacao);
  const values = Object.values(f4.histograma_fundamentacao);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Fundamentações',
        data: values,
        backgroundColor: c.accent + 'aa',
        borderColor: c.accent,
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw.toLocaleString('pt-BR')} fundamentações` } },
        datalabels: {
          anchor: 'end',
          align: 'end',
          color: c.text,
          font: { size: 10, weight: '600', family: 'Inter' },
          formatter: v => v.toLocaleString('pt-BR'),
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Faixa de palavras', color: c.text, font: { size: 11, weight: '600' } },
          grid: { display: false },
          ticks: { color: c.text, font: { size: 10 }, maxRotation: 45 },
        },
        y: {
          title: { display: true, text: 'Nº de fundamentações', color: c.text, font: { size: 11, weight: '600' } },
          grid: { color: c.grid },
          ticks: { color: c.text, callback: v => v.toLocaleString('pt-BR') },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}

/* ===== Histograma de Comprimento — Ementas ===== */
function renderHistogramEmentaChart(f4) {
  const canvas = document.getElementById('chart-histogram-ementa');
  if (!canvas || !f4?.histograma_ementa) return;
  const c = getChartColors();

  const labels = Object.keys(f4.histograma_ementa);
  const values = Object.values(f4.histograma_ementa);

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: 'Ementas',
        data: values,
        backgroundColor: c.accent4 + 'aa',
        borderColor: c.accent4,
        borderWidth: 1,
        borderRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw.toLocaleString('pt-BR')} ementas` } },
        datalabels: {
          anchor: 'end',
          align: 'end',
          color: c.text,
          font: { size: 10, weight: '600', family: 'Inter' },
          formatter: v => v.toLocaleString('pt-BR'),
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Faixa de palavras', color: c.text, font: { size: 11, weight: '600' } },
          grid: { display: false },
          ticks: { color: c.text, font: { size: 10 }, maxRotation: 45 },
        },
        y: {
          title: { display: true, text: 'Nº de ementas', color: c.text, font: { size: 11, weight: '600' } },
          grid: { color: c.grid },
          ticks: { color: c.text, callback: v => v.toLocaleString('pt-BR') },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}

/* ===== Box-Plot Comparativo (Fundamentação vs. Ementa) ===== */
function renderBoxPlot(f4) {
  const canvas = document.getElementById('chart-boxplot');
  if (!canvas || !f4?.fundamentacao || !f4?.ementa) return;
  const c = getChartColors();

  // Simula box-plot com barras flutuantes (IQR = P25-P75)
  // e barras finas para whiskers (P5-P25 e P75-P95)
  const fund = f4.fundamentacao;
  const eme = f4.ementa;

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Fundamentação', 'Ementa'],
      datasets: [
        {
          label: 'P5 — P25',
          data: [[fund.p5, fund.p25], [eme.p5, eme.p25]],
          backgroundColor: c.accent + '30',
          borderColor: c.accent + '60',
          borderWidth: 1,
          borderRadius: 2,
          barPercentage: 0.6,
        },
        {
          label: 'P25 — P75 (IQR)',
          data: [[fund.p25, fund.p75], [eme.p25, eme.p75]],
          backgroundColor: [c.accent + 'aa', c.accent4 + 'aa'],
          borderColor: [c.accent, c.accent4],
          borderWidth: 2,
          borderRadius: 6,
          barPercentage: 0.6,
        },
        {
          label: 'P75 — P95',
          data: [[fund.p75, fund.p95], [eme.p75, eme.p95]],
          backgroundColor: c.accent + '30',
          borderColor: c.accent + '60',
          borderWidth: 1,
          borderRadius: 2,
          barPercentage: 0.6,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'bottom', labels: { color: c.text, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const range = ctx.raw;
              return `${ctx.dataset.label}: ${Math.round(range[0])} — ${Math.round(range[1])} palavras`;
            }
          }
        },
        datalabels: {
          display: ctx => ctx.datasetIndex === 1,
          anchor: 'center',
          align: 'center',
          color: '#fff',
          font: { size: 12, weight: '700' },
          formatter: (v, ctx) => {
            const med = ctx.dataIndex === 0 ? fund.mediana : eme.mediana;
            return `med: ${Math.round(med)}`;
          },
        },
      },
      scales: {
        x: {
          grid: { color: c.grid },
          ticks: { color: c.text, callback: v => v.toLocaleString('pt-BR') },
          title: { display: true, text: 'Palavras', color: c.text, font: { size: 11, weight: '600' } },
        },
        y: {
          stacked: false,
          grid: { display: false },
          ticks: { color: c.text, font: { size: 13, weight: '700' } },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}

/* ===== Scatter de Compressão ===== */
function renderScatterCompression(f4) {
  const canvas = document.getElementById('chart-scatter');
  if (!canvas || !f4?.scatter_compressao) return;
  const c = getChartColors();

  new Chart(canvas, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Pares (fund. × ementa)',
        data: f4.scatter_compressao,
        backgroundColor: c.accent + '44',
        borderColor: c.accent + '88',
        borderWidth: 1,
        pointRadius: 2.5,
        pointHoverRadius: 5,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `Fundamentação: ${ctx.raw.x} palavras · Ementa: ${ctx.raw.y} palavras · Razão: ${(ctx.raw.x / ctx.raw.y).toFixed(1)}:1`,
          }
        },
        datalabels: { display: false },
      },
      scales: {
        x: {
          grid: { color: c.grid },
          ticks: { color: c.text, callback: v => v.toLocaleString('pt-BR') },
          title: { display: true, text: 'Palavras na fundamentação', color: c.text, font: { size: 11, weight: '600' } },
        },
        y: {
          grid: { color: c.grid },
          ticks: { color: c.text },
          title: { display: true, text: 'Palavras na ementa', color: c.text, font: { size: 11, weight: '600' } },
        },
      },
    },
  });
}

/* ===== Top PII Tokens (Barras Horizontais) ===== */
function renderPiiChart(f3) {
  const canvas = document.getElementById('chart-pii');
  if (!canvas || !f3?.pii_contagem) return;
  const c = getChartColors();

  const pii = f3.pii_contagem;
  // Filtrar 'total' e 'descartados_pos_anon', ordenar por contagem
  const entries = Object.entries(pii)
    .filter(([k]) => k !== 'total' && k !== 'descartados_pos_anon')
    .sort((a, b) => b[1] - a[1]);

  const labels = entries.map(([k]) => `[${k}]`);
  const values = entries.map(([, v]) => v);

  const colors = [c.accent, c.accent2, c.accent3, c.accent4,
                  c.accent + 'cc', c.accent2 + 'cc', c.accent3 + 'cc',
                  c.accent4 + 'cc', c.accent + '99'];

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors.slice(0, values.length),
        borderColor: colors.slice(0, values.length).map(co => co.replace(/[0-9a-f]{2}$/i, 'ff')),
        borderWidth: 1,
        borderRadius: 6,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `${ctx.raw.toLocaleString('pt-BR')} substituições` } },
        datalabels: {
          anchor: 'end',
          align(ctx) {
            const max = Math.max(...ctx.dataset.data);
            return ctx.dataset.data[ctx.dataIndex] < max * 0.15 ? 'right' : 'left';
          },
          color: c.label,
          font: { size: 12, weight: '700', family: 'Inter' },
          formatter: v => v.toLocaleString('pt-BR'),
          padding: { right: 8, left: 4 },
        },
      },
      scales: {
        x: {
          grid: { color: c.grid },
          ticks: { color: c.text, callback: v => v.toLocaleString('pt-BR') },
        },
        y: {
          grid: { display: false },
          ticks: { color: c.text, font: { size: 12, weight: '600', family: 'monospace' } },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}

/* ===== Word Cloud ===== */
function renderWordCloud(f4) {
  const canvas = document.getElementById('chart-wordcloud');
  if (!canvas || !f4?.wordcloud || !window.WordCloud) return;
  const c = getChartColors();

  const colors = [c.accent, c.accent2, c.accent3, c.accent4];
  const maxWeight = f4.wordcloud[0]?.weight || 1;

  // wordcloud2.js expects [[word, size], ...]
  const list = f4.wordcloud.map(w => [w.text, w.weight]);

  // Clear previous render
  const ctx2d = canvas.getContext('2d');
  ctx2d.clearRect(0, 0, canvas.width, canvas.height);

  // Set canvas size based on container
  const container = canvas.parentElement;
  canvas.width = container.offsetWidth || 600;
  canvas.height = container.offsetHeight || 350;

  WordCloud(canvas, {
    list,
    gridSize: 8,
    weightFactor: size => Math.max(12, (size / maxWeight) * 72),
    fontFamily: 'Inter, sans-serif',
    fontWeight: '700',
    color: () => colors[Math.floor(Math.random() * colors.length)],
    rotateRatio: 0.3,
    rotationSteps: 2,
    backgroundColor: 'transparent',
    drawOutOfBound: false,
    shrinkToFit: true,
  });
}

/* ===== Sobreposição de Vocabulário ===== */
function renderVocabOverlap(f4) {
  const canvas = document.getElementById('chart-vocab');
  if (!canvas || !f4?.vocabulario) return;
  const c = getChartColors();

  const v = f4.vocabulario;
  const fundExclusive = v.fundamentacao - v.sobreposicao;
  const ementaExclusive = v.ementa - v.sobreposicao;

  new Chart(canvas, {
    type: 'bar',
    data: {
      labels: ['Fundamentação', 'Ementa'],
      datasets: [
        {
          label: 'Exclusivo',
          data: [fundExclusive, ementaExclusive],
          backgroundColor: [c.accent + '88', c.accent4 + '88'],
          borderColor: [c.accent, c.accent4],
          borderWidth: 2,
          borderRadius: 6,
        },
        {
          label: 'Compartilhado',
          data: [v.sobreposicao, v.sobreposicao],
          backgroundColor: [c.accent3 + '88', c.accent3 + '88'],
          borderColor: [c.accent3, c.accent3],
          borderWidth: 2,
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, position: 'bottom', labels: { color: c.text, font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.raw.toLocaleString('pt-BR')} palavras únicas`,
          }
        },
        datalabels: {
          anchor: 'center',
          align: 'center',
          color: '#fff',
          font: { size: 11, weight: '700' },
          formatter: v => v.toLocaleString('pt-BR'),
          display: ctx => ctx.raw > 1500,
        },
      },
      scales: {
        x: {
          stacked: true,
          grid: { color: c.grid },
          ticks: { color: c.text, callback: v => (v / 1000).toFixed(0) + 'k' },
          title: { display: true, text: 'Palavras únicas', color: c.text, font: { size: 11, weight: '600' } },
        },
        y: {
          stacked: true,
          grid: { display: false },
          ticks: { color: c.text, font: { size: 13, weight: '700' } },
        },
      },
    },
    plugins: [ChartDataLabels],
  });
}
