const summaryEl = document.getElementById('summary');
const instanceTableBody = document.getElementById('instanceTableBody');
const instanceChartsEl = document.getElementById('instanceCharts');
const configTextEl = document.getElementById('configText');
const lastUpdateEl = document.getElementById('lastUpdate');

let throughputChart = null;
let latencyChart = null;
let tokenChart = null;

const history = {
  timestamps: [],
  qps: [],
  latency: [],
  prefillTps: [],
  decodeTps: [],
  maxPoints: 60
};

function initCharts() {
  const throughputEl = document.getElementById('throughputChart');
  const latencyEl = document.getElementById('latencyChart');
  const tokenEl = document.getElementById('tokenChart');

  if (!throughputEl || !latencyEl || !tokenEl) return;

  throughputChart = echarts.init(throughputEl);
  throughputChart.setOption({
    tooltip: { trigger: 'axis' },
    legend: { data: ['QPS'], textStyle: { color: '#e2e8f0' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '40px', containLabel: true },
    xAxis: { type: 'category', data: [], axisLabel: { color: '#94a3b8', fontSize: 11 } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
    series: [
      { name: 'QPS', type: 'line', smooth: true, data: [], itemStyle: { color: '#3b82f6' }, areaStyle: { opacity: 0.2 } }
    ]
  });

  latencyChart = echarts.init(latencyEl);
  latencyChart.setOption({
    tooltip: { trigger: 'axis', formatter: '{b}<br/>{a}: {c} ms' },
    legend: { data: ['Latency'], textStyle: { color: '#e2e8f0' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '40px', containLabel: true },
    xAxis: { type: 'category', data: [], axisLabel: { color: '#94a3b8', fontSize: 11 } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8', formatter: '{value} ms' } },
    series: [
      { name: 'Latency', type: 'line', smooth: true, data: [], itemStyle: { color: '#f59e0b' }, areaStyle: { opacity: 0.2 } }
    ]
  });

  tokenChart = echarts.init(tokenEl);
  tokenChart.setOption({
    tooltip: { trigger: 'axis', formatter: '{b}<br/>{a0}: {c0} tok/s<br/>{a1}: {c1} tok/s' },
    legend: { data: ['Prefill', 'Decode'], textStyle: { color: '#e2e8f0' }, top: 0 },
    grid: { left: '3%', right: '4%', bottom: '3%', top: '40px', containLabel: true },
    xAxis: { type: 'category', data: [], axisLabel: { color: '#94a3b8', fontSize: 11 } },
    yAxis: { type: 'value', axisLabel: { color: '#94a3b8' } },
    series: [
      { name: 'Prefill', type: 'bar', data: [], itemStyle: { color: '#8b5cf6' } },
      { name: 'Decode', type: 'bar', data: [], itemStyle: { color: '#10b981' } }
    ]
  });
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function renderSummary(monitor, healthyCount, totalCount) {
  if (!summaryEl) return;
  const successRate = (monitor.success_rate ?? 0) * 100;

  summaryEl.innerHTML = `
    <div class="card"><div class="card-title">Instances</div><div class="card-value ${healthyCount > 0 ? 'success' : 'error'}">${healthyCount}/${totalCount}</div></div>
    <div class="card"><div class="card-title">Total Requests</div><div class="card-value">${formatNumber(monitor.total_requests ?? 0)}</div></div>
    <div class="card"><div class="card-title">QPS</div><div class="card-value">${(monitor.recent_qps ?? 0).toFixed(2)}</div></div>
    <div class="card"><div class="card-title">Avg Latency</div><div class="card-value">${(monitor.recent_avg_latency_ms ?? 0).toFixed(1)} ms</div></div>
    <div class="card"><div class="card-title">Success</div><div class="card-value ${successRate >= 95 ? 'success' : successRate >= 80 ? 'warning' : 'error'}">${successRate.toFixed(1)}%</div></div>
    <div class="card"><div class="card-title">Errors</div><div class="card-value ${monitor.total_errors > 0 ? 'error' : ''}">${formatNumber(monitor.total_errors ?? 0)}</div></div>
  `;
}

function renderInstanceTable(instances) {
  if (!instanceTableBody) return;

  instanceTableBody.innerHTML = instances.map(inst => {
    const km = inst.key_metrics || {};
    const status = inst.healthy ? 'healthy' : 'unhealthy';
    const prefillTok = km.prompt_tokens_total || 0;
    const decodeTok = km.generation_tokens_total || 0;
    const gpuUtil = (km.gpu_utilization || 0) * 100;
    const kvUtil = (km.kv_cache_usage || 0) * 100;

    return `
      <tr class="${inst.healthy ? '' : 'unhealthy-row'}">
        <td><strong>${inst.config.id}</strong></td>
        <td><span class="badge ${status}">${inst.healthy ? '✓' : '✗'}</span></td>
        <td class="model-cell" title="${inst.config.model}">${truncate(inst.config.model, 20)}</td>
        <td><code>${inst.config.host}:${inst.config.port}</code></td>
        <td>${inst.pid || '-'}</td>
        <td>${inst.inflight_requests}</td>
        <td>${km.waiting_requests || 0}</td>
        <td>${km.running_requests || 0}</td>
        <td>
          <div class="progress-mini">
            <div class="progress-fill-mini" style="width:${gpuUtil}%;background:${gpuUtil > 80 ? '#ef4444' : gpuUtil > 50 ? '#f59e0b' : '#10b981'}"></div>
          </div>
          <span>${gpuUtil.toFixed(0)}%</span>
        </td>
        <td>
          <div class="progress-mini">
            <div class="progress-fill-mini kv" style="width:${kvUtil}%"></div>
          </div>
          <span>${kvUtil.toFixed(0)}%</span>
        </td>
        <td>${formatNumber(prefillTok)}</td>
        <td>${formatNumber(decodeTok)}</td>
        <td>
          <button class="btn-xs" onclick="toggleInstance('${inst.config.id}', true)">▶</button>
          <button class="btn-xs btn-secondary" onclick="toggleInstance('${inst.config.id}', false)">⏹</button>
        </td>
      </tr>
    `;
  }).join('');
}

function updateCharts(monitor, instances) {
  if (!throughputChart || !latencyChart || !tokenChart) return;

  const now = new Date();
  const timeStr = now.toLocaleTimeString();

  // Calculate token throughput from instances
  let totalPrefillRate = 0;
  let totalDecodeRate = 0;

  instances.forEach(inst => {
    if (!inst.healthy) return;
    const km = inst.key_metrics || {};
    // These are cumulative, we'll compute rate on frontend
    totalPrefillRate = km.prompt_tokens_total || 0;
    totalDecodeRate = km.generation_tokens_total || 0;
  });

  history.timestamps.push(timeStr);
  history.qps.push(monitor.recent_qps || 0);
  history.latency.push(monitor.recent_avg_latency_ms || 0);
  history.prefillTps.push(totalPrefillRate);
  history.decodeTps.push(totalDecodeRate);

  if (history.timestamps.length > history.maxPoints) {
    history.timestamps.shift();
    history.qps.shift();
    history.latency.shift();
    history.prefillTps.shift();
    history.decodeTps.shift();
  }

  throughputChart.setOption({ xAxis: { data: history.timestamps }, series: [{ data: history.qps }] });
  latencyChart.setOption({ xAxis: { data: history.timestamps }, series: [{ data: history.latency }] });

  // Token chart shows cumulative values scaled
  const lastPrefill = history.prefillTps[history.prefillTps.length - 1] || 0;
  const lastDecode = history.decodeTps[history.decodeTps.length - 1] || 0;
  tokenChart.setOption({
    xAxis: { data: ['Tokens'] },
    series: [
      { data: [lastPrefill / 1000] },
      { data: [lastDecode / 1000] }
    ]
  });
}

function renderInstanceCharts(instances) {
  if (!instanceChartsEl) return;

  instanceChartsEl.innerHTML = instances.filter(i => i.healthy).map(inst => {
    const km = inst.key_metrics || {};
    const gpuUtil = (km.gpu_utilization || 0) * 100;
    const kvUtil = (km.kv_cache_usage || 0) * 100;
    const prefillTok = km.prompt_tokens_total || 0;
    const decodeTok = km.generation_tokens_total || 0;

    return `
      <div class="instance-metrics-card">
        <h3>📊 ${inst.config.id}</h3>
        <div class="metrics-grid">
          <div class="metric-item">
            <label>GPU Util</label>
            <div class="progress-bar"><div class="progress-fill" style="width:${gpuUtil}%"></div></div>
            <span>${gpuUtil.toFixed(1)}%</span>
          </div>
          <div class="metric-item">
            <label>KV Cache</label>
            <div class="progress-bar"><div class="progress-fill kv" style="width:${kvUtil}%"></div></div>
            <span>${kvUtil.toFixed(1)}%</span>
          </div>
          <div class="metric-item">
            <label>Prefill Tokens</label>
            <span class="metric-value">${formatNumber(prefillTok)}</span>
          </div>
          <div class="metric-item">
            <label>Decode Tokens</label>
            <span class="metric-value">${formatNumber(decodeTok)}</span>
          </div>
          <div class="metric-item">
            <label>Running</label>
            <span class="metric-value">${km.running_requests || 0}</span>
          </div>
          <div class="metric-item">
            <label>Waiting</label>
            <span class="metric-value">${km.waiting_requests || 0}</span>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

async function toggleInstance(id, start) {
  const url = start ? `/admin/instances/${id}/start` : `/admin/instances/${id}/stop`;
  await fetchJson(url, { method: 'POST' });
  refresh();
}
window.toggleInstance = toggleInstance;

async function refresh() {
  try {
    const [state, cfg] = await Promise.all([fetchJson('/admin/state'), fetchJson('/admin/config')]);
    const healthy = state.instances.filter(i => i.healthy).length;
    renderSummary(state.monitor, healthy, state.instances.length);
    renderInstanceTable(state.instances);
    updateCharts(state.monitor, state.instances);
    renderInstanceCharts(state.instances);
    if (configTextEl) configTextEl.value = cfg.text;
    if (lastUpdateEl) lastUpdateEl.textContent = new Date().toLocaleTimeString();
  } catch (e) {
    console.error(e);
  }
}

function formatNumber(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
  return n.toString();
}

function truncate(s, len) { return s && s.length > len ? s.slice(0, len) + '…' : s; }

document.getElementById('refreshBtn')?.addEventListener('click', refresh);
document.getElementById('reloadBtn')?.addEventListener('click', async () => { await fetchJson('/admin/reload', { method: 'POST' }); refresh(); });
document.getElementById('saveConfigBtn')?.addEventListener('click', async () => {
  await fetchJson('/admin/config', { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: configTextEl.value }) });
  refresh();
});

window.addEventListener('resize', () => { throughputChart?.resize(); latencyChart?.resize(); tokenChart?.resize(); });

initCharts();
refresh();
setInterval(refresh, 3000);
