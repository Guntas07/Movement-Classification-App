const currentLabel = document.querySelector('[data-current-label]');
const currentState = document.querySelector('[data-current-state]');
const statusEl = document.querySelector('[data-status]');
const toggleButton = document.querySelector('[data-toggle]');
const clearButton = document.querySelector('[data-clear]');
const historyCount = document.querySelector('[data-history-count]');
const walkingCount = document.querySelector('[data-walking-count]');
const jumpingCount = document.querySelector('[data-jumping-count]');
const listEl = document.querySelector('[data-recent-list]');
const chartCanvas = document.getElementById('predChart');

let timer = null;
let values = [];
let chart;

function labelFor(value) {
  return value === 1 ? 'Jumping' : 'Walking';
}

function setStatus(message) {
  if (statusEl) statusEl.textContent = message;
}

function renderStats() {
  const walking = values.filter(value => value === 0).length;
  const jumping = values.filter(value => value === 1).length;
  historyCount.textContent = values.length.toLocaleString();
  walkingCount.textContent = walking.toLocaleString();
  jumpingCount.textContent = jumping.toLocaleString();
}

function renderRecent() {
  listEl.innerHTML = '';
  values.slice(-12).reverse().forEach((value, index) => {
    const chip = document.createElement('span');
    chip.className = `chip ${value === 1 ? 'jump' : 'walk'}`;
    chip.textContent = `${values.length - index}: ${labelFor(value)}`;
    listEl.appendChild(chip);
  });
}

function renderChart() {
  if (!chartCanvas || !window.Chart) return;
  const ctx = chartCanvas.getContext('2d');
  const labels = values.map((_, index) => index + 1);

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Realtime movement',
        data: values,
        stepped: true,
        borderColor: '#a78bfa',
        backgroundColor: 'rgba(167, 139, 250, 0.18)',
        fill: true,
        borderWidth: 3,
        pointRadius: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: -0.1,
          max: 1.1,
          grid: { color: 'rgba(148, 163, 184, 0.16)' },
          ticks: {
            color: '#94a3b8',
            stepSize: 1,
            callback: value => Number(value) === 1 ? 'Jumping' : Number(value) === 0 ? 'Walking' : '',
          },
        },
        x: {
          grid: { display: false },
          ticks: { color: '#94a3b8' },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: item => labelFor(item.raw) } },
      },
    },
  });
}

function renderPrediction(value) {
  currentLabel.textContent = labelFor(value);
  currentState.textContent = value === 1 ? 'High-impact motion detected' : 'Steady gait detected';
  values.push(value);
  renderStats();
  renderRecent();
  renderChart();
}

async function pollRealtime() {
  setStatus('Reading the PhyPhox stream…');

  try {
    const response = await fetch('/api/predict/realtime');
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || 'Realtime endpoint failed.');

    renderPrediction(Number(payload.prediction));
    setStatus(`Updated ${new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit', second: '2-digit' })}`);
  } catch (error) {
    setStatus(`${error.message} Check your PhyPhox stream address.`);
  }
}

function start() {
  values = [];
  renderStats();
  renderRecent();
  renderChart();
  currentLabel.textContent = 'Listening';
  currentState.textContent = 'Polling every 5 seconds';
  toggleButton.textContent = 'Pause stream';
  pollRealtime();
  timer = setInterval(pollRealtime, 5000);
}

function stop() {
  clearInterval(timer);
  timer = null;
  toggleButton.textContent = 'Start stream';
  setStatus('Paused');
}

if (toggleButton) {
  toggleButton.addEventListener('click', () => {
    if (timer) stop();
    else start();
  });
}

if (clearButton) {
  clearButton.addEventListener('click', () => {
    values = [];
    currentLabel.textContent = 'Ready';
    currentState.textContent = 'Start the stream to classify live motion';
    renderStats();
    renderRecent();
    renderChart();
    setStatus('History cleared');
  });
}

renderStats();
renderChart();
