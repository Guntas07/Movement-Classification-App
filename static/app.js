const form = document.querySelector('[data-upload-form]');
const fileInput = document.querySelector('[data-file-input]');
const dropzone = document.querySelector('[data-dropzone]');
const fileMeta = document.querySelector('[data-file-meta]');
const statusEl = document.querySelector('[data-status]');
const summaryEl = document.querySelector('[data-summary]');
const totalEl = document.querySelector('[data-total-windows]');
const walkingEl = document.querySelector('[data-walking-count]');
const jumpingEl = document.querySelector('[data-jumping-count]');
const listEl = document.querySelector('[data-prediction-list]');
const chartCanvas = document.getElementById('predChart');

let chart;

function setStatus(message, tone = 'neutral') {
  if (!statusEl) return;
  statusEl.textContent = message;
  statusEl.dataset.tone = tone;
}

function updateFileMeta(file) {
  if (!fileMeta) return;
  if (!file) {
    fileMeta.textContent = 'Drop a CSV here or browse your files.';
    return;
  }
  const sizeKb = Math.max(1, Math.round(file.size / 1024));
  fileMeta.textContent = `${file.name} · ${sizeKb.toLocaleString()} KB`;
}

function buildGradient(ctx) {
  const gradient = ctx.createLinearGradient(0, 0, 0, 280);
  gradient.addColorStop(0, 'rgba(56, 189, 248, 0.35)');
  gradient.addColorStop(1, 'rgba(56, 189, 248, 0.02)');
  return gradient;
}

function renderChart(values) {
  if (!chartCanvas || !window.Chart) return;
  const context = chartCanvas.getContext('2d');
  const labels = values.map((_, index) => `Window ${index + 1}`);

  if (chart) chart.destroy();
  chart = new Chart(context, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: 'Movement class',
        data: values,
        stepped: true,
        borderColor: '#38bdf8',
        backgroundColor: buildGradient(context),
        borderWidth: 3,
        fill: true,
        pointRadius: 4,
        pointHoverRadius: 7,
        tension: 0.25,
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
          ticks: { color: '#94a3b8', maxRotation: 0, autoSkip: true },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: item => item.raw === 1 ? 'Jumping' : 'Walking',
          },
        },
      },
    },
  });
}

function renderPredictionChips(values) {
  if (!listEl) return;
  listEl.innerHTML = '';
  values.slice(0, 36).forEach((value, index) => {
    const chip = document.createElement('span');
    chip.className = `chip ${value === 1 ? 'jump' : 'walk'}`;
    chip.textContent = `${index + 1}: ${value === 1 ? 'Jump' : 'Walk'}`;
    listEl.appendChild(chip);
  });

  if (values.length > 36) {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = `+${values.length - 36} more`;
    listEl.appendChild(chip);
  }
}

function showSummary(values) {
  const walking = values.filter(value => value === 0).length;
  const jumping = values.filter(value => value === 1).length;

  totalEl.textContent = values.length.toLocaleString();
  walkingEl.textContent = walking.toLocaleString();
  jumpingEl.textContent = jumping.toLocaleString();
  summaryEl.classList.remove('hidden');
  renderChart(values);
  renderPredictionChips(values);
}

function attachDropEvents() {
  if (!dropzone || !fileInput) return;

  ['dragenter', 'dragover'].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
      event.preventDefault();
      dropzone.classList.add('is-dragging');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropzone.addEventListener(eventName, event => {
      event.preventDefault();
      dropzone.classList.remove('is-dragging');
    });
  });

  dropzone.addEventListener('drop', event => {
    const [file] = event.dataTransfer.files;
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    updateFileMeta(file);
  });

  fileInput.addEventListener('change', () => updateFileMeta(fileInput.files[0]));
}

async function submitCsv(event) {
  event.preventDefault();
  const file = fileInput.files[0];
  if (!file) {
    setStatus('Choose a CSV before running the classifier.', 'warn');
    return;
  }

  const data = new FormData();
  data.append('file', file);

  summaryEl.classList.add('hidden');
  setStatus('Uploading, windowing, and classifying motion data…');
  form.querySelector('button[type="submit"]').disabled = true;

  try {
    const response = await fetch('/api/predict/csv', { method: 'POST', body: data });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || 'Classification failed.');

    const predictions = payload.predictions || [];
    showSummary(predictions);
    setStatus(`Done — classified ${predictions.length.toLocaleString()} motion windows.`, 'success');
  } catch (error) {
    setStatus(error.message, 'error');
  } finally {
    form.querySelector('button[type="submit"]').disabled = false;
  }
}

if (form) {
  attachDropEvents();
  updateFileMeta();
  form.addEventListener('submit', submitCsv);
}
