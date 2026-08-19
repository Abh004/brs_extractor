/* ── UI State ── */
let currentJobId = null;
let ws = null;

/* ── Pipeline Control ── */
async function startPipeline() {
  const runBtn = document.getElementById('run-btn');
  runBtn.disabled = true;
  runBtn.innerHTML = '<span class="btn-icon">&#9699;</span> Starting...';

  const formData = new FormData();
  const videoFile = document.getElementById('video-file').files[0];

  if (!videoFile) {
    showError('Upload a video file.');
    resetButton();
    return;
  }

  formData.append('video_file', videoFile);

  try {
    const resp = await fetch('/api/run', { method: 'POST', body: formData });
    if (!resp.ok) {
      const err = await resp.json();
      showError(err.detail || 'Failed to start pipeline');
      resetButton();
      return;
    }
    const data = await resp.json();
    currentJobId = data.job_id;

    document.getElementById('input-section').classList.add('hidden');
    document.getElementById('progress-section').classList.remove('hidden');
    document.getElementById('cancel-btn').classList.remove('hidden');

    setupStageIndicators();
    connectWebSocket(currentJobId);
  } catch (e) {
    showError('Network error: ' + e.message);
    resetButton();
  }
}

async function cancelPipeline() {
  if (!currentJobId) return;
  const btn = document.getElementById('cancel-btn');
  btn.disabled = true;
  btn.textContent = 'Cancelling...';
  try {
    await fetch('/api/cancel/' + currentJobId, { method: 'POST' });
  } catch (e) {
    console.error('Cancel failed:', e);
  }
}

/* ── WebSocket ── */
function connectWebSocket(jobId) {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url = `${protocol}//${window.location.host}/ws/${jobId}`;
  ws = new WebSocket(url);

  ws.onopen = () => {
    console.log('WebSocket connected');
  };

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);

    if (msg.type === 'update') {
      handleUpdate(msg);
    } else if (msg.type === 'job_done') {
      handleJobDone(msg);
    } else if (msg.type === 'error') {
      showError(msg.message);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected');
    if (currentJobId) {
      pollStatus(currentJobId);
    }
  };

  ws.onerror = (e) => {
    console.error('WebSocket error:', e);
  };
}

/* ── Status Polling (fallback when WS disconnects) ── */
async function pollStatus(jobId) {
  const maxAttempts = 600;
  for (let i = 0; i < maxAttempts; i++) {
    try {
      const resp = await fetch('/api/status/' + jobId);
      if (!resp.ok) break;
      const status = await resp.json();
      handlePollUpdate(status);
      if (['completed', 'failed', 'cancelled'].includes(status.status)) {
        handleJobDone(status);
        return;
      }
    } catch (e) {
      console.error('Poll failed:', e);
    }
    await new Promise(r => setTimeout(r, 1000));
  }
}

/* ── Stage updates ── */
const STAGE_NAMES = [
  null,
  'AVI Index',
  'Model Init',
  'Proposal',
  'Judges',
  'Post-process',
  'Config Gen',
  'DOCX',
];

function setupStageIndicators() {
  const container = document.getElementById('stage-indicators');
  container.innerHTML = '';
  for (let i = 1; i <= 7; i++) {
    const pill = document.createElement('span');
    pill.className = 'stage-pill';
    pill.id = 'stage-' + i;
    pill.textContent = STAGE_NAMES[i];
    container.appendChild(pill);
  }
}

function updateStagePills(stage, status) {
  for (let i = 1; i <= 7; i++) {
    const pill = document.getElementById('stage-' + i);
    pill.classList.remove('active', 'done', 'error', 'skipped');
    if (i < stage) pill.classList.add('done');
    else if (i === stage) {
      if (status === 'done') pill.classList.add('done');
      else if (status === 'error') pill.classList.add('error');
      else if (status === 'skipped') pill.classList.add('skipped');
      else pill.classList.add('active');
    }
  }
}

function handleUpdate(msg) {
  const progress = msg.progress || 0;
  document.getElementById('progress-bar').style.width = (progress * 100) + '%';

  if (msg.stage > 0 && msg.stage <= 7) {
    updateStagePills(msg.stage, msg.status);
  }
}

function handlePollUpdate(status) {
  document.getElementById('progress-bar').style.width = (status.progress || 0) + '%';
}

/* ── Job completion ── */
async function handleJobDone(msg) {
  document.getElementById('cancel-btn').classList.add('hidden');

  if ((msg.status === 'failed' || msg.status === 'error') && msg.error) {
    showError(msg.error);
    return;
  }

  if (msg.status === 'cancelled') {
    showError('Pipeline was cancelled.');
    return;
  }

  document.getElementById('progress-section').classList.add('hidden');
  document.getElementById('results-section').classList.remove('hidden');

  const jobId = currentJobId;
  try {
    const resp = await fetch('/api/results/' + jobId);
    if (!resp.ok) throw new Error('Failed to fetch results');
    const data = await resp.json();
    displayResults(data.results);
  } catch (e) {
    console.error('Failed to load results:', e);
  }
  setupDownloads();
}

function displayResults(results) {
  const reqs = results.requirements || [];
  const reqList = document.getElementById('requirements-list');
  reqList.innerHTML = '<h3>Extracted Requirements</h3>';

  for (const block of reqs) {
    const div = document.createElement('div');
    div.className = 'requirement-block';

    const header = document.createElement('div');
    header.className = 'block-header';
    header.innerHTML = `
      <span>Block ${block.minute_block + 1} (${block.time_range})</span>
      <span>${(block.bullets || []).length} reqs</span>
    `;

    const body = document.createElement('div');
    body.className = 'block-bullets';

    if (block.error) {
      const errDiv = document.createElement('div');
      errDiv.className = 'block-error';
      errDiv.textContent = 'Error: ' + block.error;
      body.appendChild(errDiv);
    } else if (!block.bullets || block.bullets.length === 0) {
      const emptyDiv = document.createElement('div');
      emptyDiv.className = 'bullet-item';
      emptyDiv.style.color = '#6e7681';
      emptyDiv.textContent = 'No requirements extracted';
      body.appendChild(emptyDiv);
    } else {
      for (const bullet of block.bullets) {
        const text = typeof bullet === 'string' ? bullet : bullet.text;
        const priority = typeof bullet === 'string' ? null : bullet.priority;
        const item = document.createElement('div');
        item.className = 'bullet-item';

        if (priority) {
          const badge = document.createElement('span');
          badge.className = 'priority-badge priority-' + priority;
          badge.textContent = priority;
          item.appendChild(badge);
        }

        const textSpan = document.createElement('span');
        textSpan.className = 'bullet-text';
        textSpan.innerHTML = highlightFlagged(text);
        item.appendChild(textSpan);
        body.appendChild(item);
      }
    }

    const wasOpen = sessionStorage.getItem('block-' + block.minute_block) !== 'closed';
    if (!wasOpen) body.style.display = 'none';

    header.addEventListener('click', () => {
      const isOpen = body.style.display !== 'none';
      body.style.display = isOpen ? 'none' : 'block';
      sessionStorage.setItem('block-' + block.minute_block, isOpen ? 'closed' : 'open');
    });

    div.appendChild(header);
    div.appendChild(body);
    reqList.appendChild(div);
  }
}

function highlightFlagged(text) {
  return text.replace(/\[\[(.*?)\]\]/g, '<span class="flagged-word">$1</span>');
}

function setupDownloads() {
  const el = document.getElementById('dl-docx');
  el.href = '/api/download/BRS.docx';
  el.textContent = 'BRS.docx';
}

/* ── Error handling ── */
function showError(message) {
  document.getElementById('input-section').classList.add('hidden');
  document.getElementById('progress-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('error-section').classList.remove('hidden');
  document.getElementById('error-message').textContent = message;
}

function resetUI() {
  currentJobId = null;
  if (ws) { ws.close(); ws = null; }
  document.getElementById('error-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('progress-section').classList.add('hidden');
  document.getElementById('input-section').classList.remove('hidden');
  document.getElementById('run-btn').disabled = false;
  resetButton();
  window.scrollTo(0, 0);
}

function resetButton() {
  const btn = document.getElementById('run-btn');
  btn.disabled = false;
  btn.innerHTML = '<span class="btn-icon">&#9654;</span> Start Pipeline';
}
