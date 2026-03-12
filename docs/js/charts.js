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
function rerenderCharts(f4) {
  destroyAllCharts();
  renderFunnelChart(f4);
  renderGauges(f4);
  renderTemporalChart(f4);
  renderHistogramChart(f4);
  renderHistogramEmentaChart(f4);
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
