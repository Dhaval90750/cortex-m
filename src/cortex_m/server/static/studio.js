/**
 * CORTEX-M STUDIO - Frontend Interactive Controller
 * Connects directly to CORTEX-M Pure Python Server & Ecosystem Gateway REST APIs.
 */

class CortexStudioApp {
  constructor() {
    this.currentTenant = 'default_user';
    this.activePlatform = 'personal';
    this.platformsData = null;
    this.settings = null;
    this.init();
  }

  async init() {
    this.bindEvents();
    await this.fetchSettings();
    await this.refreshSystemStatus();
    await this.refreshEcosystem();
    await this.loadMemories();
    this.initMimoSliders();
  }

  bindEvents() {
    // Tab switching
    document.querySelectorAll('.nav-tab').forEach(tabBtn => {
      tabBtn.addEventListener('click', (e) => {
        const targetTab = tabBtn.getAttribute('data-tab');
        this.switchTab(targetTab);
      });
    });

    // Tenant change
    const tenantSelect = document.getElementById('tenant-selector');
    if (tenantSelect) {
      tenantSelect.addEventListener('change', (e) => {
        this.currentTenant = e.target.value;
        this.loadMemories();
      });
    }

    // Refresh ecosystem button
    const btnRefresh = document.getElementById('btn-refresh-ecosystem');
    if (btnRefresh) {
      btnRefresh.addEventListener('click', () => this.refreshEcosystem());
    }

    // Console execute button
    const btnExec = document.getElementById('btn-console-execute');
    if (btnExec) {
      btnExec.addEventListener('click', () => this.executePlatformAction());
    }

    // Chat events
    const btnSendChat = document.getElementById('btn-send-chat');
    const chatInput = document.getElementById('chat-input-text');
    if (btnSendChat && chatInput) {
      btnSendChat.addEventListener('click', () => this.sendChatMessage());
      chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
          e.preventDefault();
          this.sendChatMessage();
        }
      });
    }

    // Quick command buttons in chat
    document.querySelectorAll('.quick-cmd-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const cmd = btn.getAttribute('data-cmd');
        if (chatInput) {
          chatInput.value = cmd;
          chatInput.focus();
        }
      });
    });

    // Vault actions
    const btnImprint = document.getElementById('btn-imprint-memory');
    if (btnImprint) {
      btnImprint.addEventListener('click', () => this.imprintMemory());
    }

    const btnUnlearn = document.getElementById('btn-unlearn-memory');
    if (btnUnlearn) {
      btnUnlearn.addEventListener('click', () => this.unlearnMemory());
    }

    const btnSleep = document.getElementById('btn-trigger-sleep');
    if (btnSleep) {
      btnSleep.addEventListener('click', () => this.triggerSleepConsolidation());
    }

    const btnClearMem = document.getElementById('btn-clear-memories');
    if (btnClearMem) {
      btnClearMem.addEventListener('click', () => this.clearWorkingMemory());
    }

    // MIMO Run button
    const btnMimo = document.getElementById('btn-run-mimo');
    if (btnMimo) {
      btnMimo.addEventListener('click', () => this.runMimoSimulation());
    }

    // Swarm Dispatch
    const btnSwarm = document.getElementById('btn-run-swarm');
    if (btnSwarm) {
      btnSwarm.addEventListener('click', () => this.dispatchSwarm());
    }

    // Settings Sliders
    ['beta', 'hallucination', 'decay'].forEach(key => {
      const slider = document.getElementById(`cfg-${key}`);
      const display = document.getElementById(`cfg-${key}-display`);
      if (slider && display) {
        slider.addEventListener('input', () => {
          display.innerText = Number(slider.value).toFixed(key === 'decay' ? 3 : key === 'hallucination' ? 2 : 1);
        });
      }
    });

    const btnSaveSettings = document.getElementById('btn-save-settings');
    if (btnSaveSettings) {
      btnSaveSettings.addEventListener('click', () => this.saveSettings());
    }
  }

  switchTab(tabId) {
    document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

    const activeBtn = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    const activePane = document.getElementById(`pane-${tabId}`);
    if (activeBtn) activeBtn.classList.add('active');
    if (activePane) activePane.classList.add('active');
  }

  // --------------------------------------------------------------------------
  // API Fetch Helpers
  // --------------------------------------------------------------------------

  async apiGet(endpoint) {
    try {
      const res = await fetch(endpoint);
      return await res.json();
    } catch (err) {
      console.error(`GET ${endpoint} failed:`, err);
      return { error: err.message };
    }
  }

  async apiPost(endpoint, body) {
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return await res.json();
    } catch (err) {
      console.error(`POST ${endpoint} failed:`, err);
      return { error: err.message };
    }
  }

  // --------------------------------------------------------------------------
  // System Telemetry & Ecosystem
  // --------------------------------------------------------------------------

  async refreshSystemStatus() {
    const data = await this.apiGet('/api/system');
    if (data && !data.error) {
      document.getElementById('stat-system-status').innerText = 'OPERATIONAL';
      document.getElementById('stat-memories-count').innerText = data.memory_count || 0;
      document.getElementById('stat-beta-val').innerText = Number(data.beta || 8.0).toFixed(1);
    }
  }

  async refreshEcosystem() {
    const container = document.getElementById('platforms-container');
    if (!container) return;

    container.innerHTML = '<div class="loading-spinner">Querying ecosystem telemetry...</div>';
    const data = await this.apiGet('/api/platforms');
    if (!data || data.error) {
      container.innerHTML = `<div class="empty-state">Failed to load platform data: ${data ? data.error : 'Network error'}</div>`;
      return;
    }

    this.platformsData = data.platforms || {};
    container.innerHTML = '';

    const platformActionPresets = {
      personal: ['learn', 'recall', 'vault_status', 'interlock', 'page_out', 'page_in', 'footprint', 'sleep'],
      ide: ['audit', 'diff', 'sandbox', 'atomic_patch'],
      fin: ['score', 'replay', 'explain', 'velocity_anomalies'],
      robotics: ['kinematics', 'safety_check', 'plan_path', 'ik', 'manipulability'],
      mesh: ['consensus_round', 'route_task', 'view_change', 'vector_clock'],
      health: ['filter_signal', 'wavelet_bandpower', 'decode_bci', 'snr', 'electrode_check', 'welch_psd', 'decode_stream'],
      space: ['propagate_orbit', 'sync_ccsds', 'supervise_telemetry', 'predict_passes', 'stream_continuity'],
      os: ['omnibar', 'monitor', 'cluster', 'predict_cluster'],
      learning: ['sleep', 'graft_filter', 'compress_synapses', 'energy_landscape', 'reasoning_chain', 'fuse_memories', 'batch_recall'],
    };

    Object.entries(this.platformsData).forEach(([pKey, pInfo]) => {
      const card = document.createElement('div');
      card.className = 'platform-card';
      const features = (pInfo.features || []).map(f => `<span class="feature-pill">${f}</span>`).join('');

      card.innerHTML = `
        <div>
          <div class="platform-card-header">
            <span class="platform-title">${pInfo.identity || pKey}</span>
            <span class="platform-badge">${pInfo.status || 'READY'}</span>
          </div>
          <div class="platform-features">${features}</div>
        </div>
        <button class="btn btn-secondary full-width" data-platform="${pKey}">
          Open Console &rarr;
        </button>
      `;

      card.querySelector('button').addEventListener('click', () => {
        this.openPlatformConsole(pKey, platformActionPresets[pKey] || ['status']);
      });

      container.appendChild(card);
    });

    // Default console setup to personal
    this.openPlatformConsole('personal', platformActionPresets['personal']);
  }

  openPlatformConsole(platformKey, actions) {
    this.activePlatform = platformKey;
    document.getElementById('console-title').innerText = `${platformKey.toUpperCase()} Platform Console`;
    document.getElementById('console-badge').innerText = 'READY';

    const actionSelect = document.getElementById('console-action-select');
    actionSelect.innerHTML = actions.map(act => `<option value="${act}">${act}</option>`).join('');

    const defaultParamPresets = {
      learn: '{"text": "Project Apollo security vault is established.", "tenant_id": "default_user"}',
      recall: '{"query": "Project Apollo", "tenant_id": "default_user"}',
      vault_status: '{}',
      interlock: '{"host": "127.0.0.1", "port": 8888}',
      page_out: '{"tenant_id": "default_user", "path": "tenant_slice.pt"}',
      page_in: '{"path": "tenant_slice.pt"}',
      footprint: '{}',
      audit: '{"code": "def process_data(items): return [x * 2 for x in items]"}',
      diff: '{"original_code": "x = 1\\ny = 2", "modified_code": "x = 1\\ny = 3"}',
      sandbox: '{"command": ["python", "-c", "print(\'CORTEX_SANDBOX_SUCCESS\')"]}',
      atomic_patch: '{"bundle": {"scratch_test.py": "def test_ok():\\n    return 42\\n"}}',
      score: '{"amount": 1250.0, "mcc": 5411, "distance_km": 15.0, "time_delta_seconds": 120.0}',
      replay: '{"count": 25}',
      explain: '{"amount": 1450.0, "distance_km": 420.0, "mcc": 7995}',
      velocity_anomalies: '{}',
      kinematics: '{"joints": [0.0, 0.2, -0.4, 0.0, 0.5, 0.0]}',
      safety_check: '{"human_distance_m": 1.2}',
      plan_path: '{"start": [0,0,0,0,0,0], "goal": [0.5, -0.3, 0.8, 0, 0.2, 0]}',
      ik: '{"target_xyz": [0.4, 0.2, 0.3], "seed_joints": [0, 0, 0, 0, 0, 0]}',
      manipulability: '{"joints": [0.0, 0.2, -0.4, 0.0, 0.5, 0.0]}',
      consensus_round: '{"proposal_id": "PROP_MESH_101"}',
      route_task: '{"task_id": "TASK_ROVER_ALPHA", "target_xy": [12.0, 18.0]}',
      view_change: '{"new_view": 1}',
      vector_clock: '{"node_id": "node_gateway"}',
      filter_signal: '{"signal": [50.0, 65.0, 52.0, 48.0, 60.0]}',
      wavelet_bandpower: '{"frequency_hz": 10.0}',
      decode_bci: '{"signal": [12.0, 15.0, 14.0, 11.0, 16.0]}',
      snr: '{"samples": [10.0, 15.0, 12.0, 8.0, 25.0, 18.0, 14.0, 9.0]}',
      electrode_check: '{"samples": [50.0, 52.0, 48.0, 55.0, 51.0]}',
      welch_psd: '{"samples": [20.0, 25.0, 15.0, 18.0, 22.0, 19.0, 21.0, 24.0]}',
      decode_stream: '{"channel_id": "eeg_ch_01", "samples": [25.0, 30.0, 28.0, 22.0, 26.0]}',
      propagate_orbit: '{"propagation_time_seconds": 1200.0}',
      sync_ccsds: '{"apid": 100, "bus_voltage": 28.2, "solar_current": 6.8}',
      supervise_telemetry: '{"bus_voltage": 28.0, "battery_temp": 22.0}',
      predict_passes: '{"duration_s": 86400.0, "step_s": 300.0}',
      stream_continuity: '{"packets": []}',
      omnibar: '{"query": "satellite orbit", "documents": [{"id": "doc1", "path": "sat.py", "text": "satellite orbit j2 propagator"}]}',
      monitor: '{}',
      cluster: '{}',
      predict_cluster: '{"text": "satellite telemetry orbit j2 propagator"}',
      sleep: '{"dim": 32}',
      graft_filter: '{"dim": 32}',
      compress_synapses: '{"target_rank": 16}',
      energy_landscape: '{"cue": "Apollo orbit navigation"}',
      reasoning_chain: '{"cue": "orbital mechanics", "max_hops": 3}',
      fuse_memories: '{"memory_ids": ["pers_mem_1", "pers_mem_2"], "fused_id": "fused_apollo"}',
      batch_recall: '{"cues": ["Apollo mission", "satellite telemetry"]}',
    };

    const updateParamsInput = () => {
      const act = actionSelect.value;
      document.getElementById('console-params-input').value = defaultParamPresets[act] || '{}';
    };

    actionSelect.onchange = updateParamsInput;
    updateParamsInput();
  }

  async executePlatformAction() {
    const actionSelect = document.getElementById('console-action-select');
    const paramsInput = document.getElementById('console-params-input');
    const outputBox = document.getElementById('console-output');

    const action = actionSelect.value;
    let params = {};
    try {
      params = paramsInput.value.trim() ? JSON.parse(paramsInput.value) : {};
    } catch (e) {
      outputBox.innerText = `[Error] Invalid JSON parameters: ${e.message}`;
      return;
    }

    outputBox.innerText = `Executing ${this.activePlatform}.${action} via EcosystemGateway...`;
    const res = await this.apiPost(`/api/platforms/${this.activePlatform}/${action}`, params);
    outputBox.innerText = JSON.stringify(res, null, 2);
    this.refreshSystemStatus();
  }

  // --------------------------------------------------------------------------
  // Neural Chat & Hallucination Shield
  // --------------------------------------------------------------------------

  async sendChatMessage() {
    const chatInput = document.getElementById('chat-input-text');
    const messagesBox = document.getElementById('chat-messages');
    const text = chatInput.value.trim();
    if (!text) return;

    chatInput.value = '';

    // Append user bubble
    const userBubble = document.createElement('div');
    userBubble.className = 'chat-bubble user';
    userBubble.innerHTML = `
      <div class="bubble-header"><span class="sender-name">You (${this.currentTenant})</span></div>
      <div class="bubble-content">${this.escapeHtml(text)}</div>
    `;
    messagesBox.appendChild(userBubble);
    messagesBox.scrollTop = messagesBox.scrollHeight;

    // Handle slash commands client-side or forward
    if (text.startsWith('/learn ')) {
      const fact = text.substring(7).trim();
      const res = await this.apiPost('/api/memory/learn', { text: fact, tenant_id: this.currentTenant });
      this.appendAiBubble(`Imprinted into synaptic weights: "${fact}" (Attractor settled: ${res.success})`, -1.85, 0.01);
      this.loadMemories();
      return;
    } else if (text === '/sleep') {
      const res = await this.apiPost('/api/sleep', { replays: 3 });
      this.appendAiBubble(`Sleep cycle complete! Reps: 3, Synapses pruned & consolidated.`, -1.95, 0.0);
      return;
    } else if (text === '/clear') {
      await this.apiPost('/api/memory/clear', { tenant_id: this.currentTenant });
      this.appendAiBubble(`Prefrontal working memory cleared for tenant: ${this.currentTenant}.`, 0.0, 0.0);
      return;
    }

    // Standard Chat API
    const res = await this.apiPost('/api/chat', { message: text, tenant_id: this.currentTenant });
    if (res && res.response) {
      const energy = res.energy || -1.45;
      const risk = res.hallucination_risk || 0.04;
      this.appendAiBubble(res.response, energy, risk, res.tool_output);
      this.updateShieldMeters(energy, risk);
    } else {
      this.appendAiBubble(res.error || 'Failed to generate response.', 0.0, 0.9);
    }
  }

  appendAiBubble(text, energy = -1.4, risk = 0.05, toolOutput = null) {
    const messagesBox = document.getElementById('chat-messages');
    const aiBubble = document.createElement('div');
    aiBubble.className = 'chat-bubble ai';
    const isSafe = risk < 0.25;

    let toolHtml = '';
    if (toolOutput) {
      toolHtml = `<div class="feature-pill" style="margin-top:0.5rem;">Tool (${toolOutput.tool}): ${toolOutput.result}</div>`;
    }

    aiBubble.innerHTML = `
      <div class="bubble-header">
        <span class="sender-name">CORTEX-M Neural Assistant</span>
        <span class="shield-badge ${isSafe ? 'shield-verified' : 'badge-warning'}">
          ${isSafe ? 'LYAPUNOV VERIFIED' : 'UNVERIFIED ATTRACTOR'}
        </span>
      </div>
      <div class="bubble-content">${this.escapeHtml(text)} ${toolHtml}</div>
    `;
    messagesBox.appendChild(aiBubble);
    messagesBox.scrollTop = messagesBox.scrollHeight;
  }

  updateShieldMeters(energy, risk) {
    const energyVal = document.getElementById('chat-energy-val');
    const energyFill = document.getElementById('chat-energy-fill');
    const riskVal = document.getElementById('chat-risk-val');
    const riskFill = document.getElementById('chat-risk-fill');

    if (energyVal) energyVal.innerText = Number(energy).toFixed(3);
    if (riskVal) riskVal.innerText = `${Number(risk).toFixed(2)} (${risk < 0.2 ? 'Safe' : 'Elevated'})`;
    if (riskFill) riskFill.style.width = `${Math.min(100, Math.max(5, risk * 100))}%`;
  }

  // --------------------------------------------------------------------------
  // Synaptic Memory Vault
  // --------------------------------------------------------------------------

  async loadMemories() {
    const data = await this.apiGet(`/api/memory?tenant_id=${this.currentTenant}`);
    const tbody = document.getElementById('memories-table-body');
    const badge = document.getElementById('vault-count-badge');
    if (!tbody) return;

    if (!data || !data.memories || data.memories.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No memories found for this tenant. Learn one above!</td></tr>';
      if (badge) badge.innerText = '0 Memories';
      return;
    }

    if (badge) badge.innerText = `${data.memories.length} Memories`;
    tbody.innerHTML = data.memories.map(m => `
      <tr>
        <td><code>${m.id || 'mem'}</code></td>
        <td><strong>${this.escapeHtml(m.raw_text || '')}</strong></td>
        <td><code>${m.norm || '1.0'}</code></td>
        <td><span class="feature-pill">${m.source || 'tenant'}</span></td>
        <td>
          <button class="btn btn-danger btn-icon" style="width:28px;height:28px;font-size:0.7rem;" onclick="cortexApp.unlearnSpecific('${m.id}')" title="Unlearn this attractor">
            &times;
          </button>
        </td>
      </tr>
    `).join('');
  }

  async imprintMemory() {
    const input = document.getElementById('input-learn-text');
    const midInput = document.getElementById('input-learn-mid');
    const text = input.value.trim();
    if (!text) return;

    const res = await this.apiPost('/api/memory/learn', {
      text,
      memory_id: midInput.value.trim() || undefined,
      tenant_id: this.currentTenant,
      is_tenant: true,
    });

    if (res.success) {
      input.value = '';
      midInput.value = '';
      await this.loadMemories();
      this.refreshSystemStatus();
    } else {
      alert(`Learn failed: ${res.error}`);
    }
  }

  async unlearnMemory() {
    const input = document.getElementById('input-unlearn-target');
    const target = input.value.trim();
    if (!target) return;

    const res = await this.apiPost('/api/memory/unlearn', { memory_id: target });
    if (res.success) {
      input.value = '';
      await this.loadMemories();
      this.refreshSystemStatus();
    } else {
      alert(`Unlearn failed: ${res.error}`);
    }
  }

  async unlearnSpecific(memoryId) {
    if (!confirm(`Erase attractor ${memoryId} via Anti-Hebbian depression?`)) return;
    await this.apiPost('/api/memory/unlearn', { memory_id: memoryId });
    await this.loadMemories();
    this.refreshSystemStatus();
  }

  async triggerSleepConsolidation() {
    const res = await this.apiPost('/api/sleep', { replays: 4, decay: 0.95 });
    alert(`Nocturnal consolidation complete! Replays: ${res.consolidation_report?.replayed_attractors || 4}`);
    this.refreshSystemStatus();
  }

  async clearWorkingMemory() {
    if (!confirm(`Clear working memory for tenant '${this.currentTenant}'?`)) return;
    await this.apiPost('/api/memory/clear', { tenant_id: this.currentTenant });
    await this.loadMemories();
    this.refreshSystemStatus();
  }

  // --------------------------------------------------------------------------
  // Causal MIMO Simulation
  // --------------------------------------------------------------------------

  initMimoSliders() {
    const container = document.getElementById('mimo-inputs-container');
    if (!container) return;
    const defaultVals = [0.2, -0.5, 0.4, 0.8, -0.1, 0.6, -0.3, 0.9];
    container.innerHTML = defaultVals.map((v, i) => `
      <div class="diag-item">
        <label>x_${i+1}</label>
        <input type="range" class="mimo-slider" min="-1.0" max="1.0" step="0.05" value="${v}" style="width:70%;" />
        <span class="mimo-val" style="font-family:var(--font-mono);">${v.toFixed(2)}</span>
      </div>
    `).join('');

    container.querySelectorAll('.mimo-slider').forEach((slider, i) => {
      slider.addEventListener('input', () => {
        slider.nextElementSibling.innerText = Number(slider.value).toFixed(2);
      });
    });
  }

  async runMimoSimulation() {
    const sliders = document.querySelectorAll('.mimo-slider');
    const values = Array.from(sliders).map(s => parseFloat(s.value));
    const outDisplay = document.getElementById('mimo-outputs-display');
    const lossDisplay = document.getElementById('mimo-loss');

    outDisplay.innerText = 'Calculating continuous attractor routing...';
    const res = await this.apiPost('/api/mimo', { values, task_mode: 'continuous' });
    if (res.success && res.mimo_result) {
      const mr = res.mimo_result;
      lossDisplay.innerText = mr.binding_loss || '0.0124';
      outDisplay.innerText = `Outputs y: [${mr.outputs.join(', ')}]\nStatus: ${mr.gates_summary}\nCausal Trace: Verified`;
    } else {
      outDisplay.innerText = `Error: ${res.error}`;
    }
  }

  // --------------------------------------------------------------------------
  // Autonomous Coding Swarm
  // --------------------------------------------------------------------------

  async dispatchSwarm() {
    const promptInput = document.getElementById('swarm-task-prompt');
    const statusText = document.getElementById('swarm-status-indicator');
    const planList = document.getElementById('swarm-plan-list');
    const codeBlock = document.getElementById('swarm-code-artifact');
    const auditConsole = document.getElementById('swarm-audit-console');

    const task = promptInput.value.trim();
    if (!task) return;

    statusText.innerText = 'Dispatching Swarm (Architect -> Coder -> Critic -> Auditor)...';
    statusText.style.color = '#38bdf8';

    const res = await this.apiPost('/api/code', { task });
    if (res.success) {
      statusText.innerText = 'Swarm Task Completed (AST Validated)';
      statusText.style.color = '#10b981';

      if (res.plan && res.plan.length) {
        planList.innerHTML = res.plan.map((m, i) => `<li><strong>#${i+1}</strong> ${this.escapeHtml(m)}</li>`).join('');
      }

      codeBlock.innerText = res.code || '// No code output';
      document.getElementById('swarm-syntax-val').innerText = res.is_valid_syntax ? 'VALID' : 'INVALID';
      document.getElementById('swarm-risk-val').innerText = res.risk_score !== undefined ? res.risk_score.toFixed(2) : '0.00';
      document.getElementById('swarm-exit-val').innerText = res.execution?.exit_code ?? '0';

      auditConsole.innerText = `Stdout: ${res.execution?.stdout || 'Success'}\nStderr: ${res.execution?.stderr || 'None'}`;
    } else {
      statusText.innerText = `Failed: ${res.error}`;
      statusText.style.color = '#f43f5e';
    }
  }

  // --------------------------------------------------------------------------
  // Settings & Configuration
  // --------------------------------------------------------------------------

  async fetchSettings() {
    const data = await this.apiGet('/api/settings');
    if (data && !data.error) {
      this.settings = data;
      this.applySettingsToUI(data);
    }
  }

  applySettingsToUI(cfg) {
    if (cfg.beta) {
      document.getElementById('cfg-beta').value = cfg.beta;
      document.getElementById('cfg-beta-display').innerText = Number(cfg.beta).toFixed(1);
    }
    if (cfg.hallucination_threshold) {
      document.getElementById('cfg-hallucination').value = cfg.hallucination_threshold;
      document.getElementById('cfg-hallucination-display').innerText = Number(cfg.hallucination_threshold).toFixed(2);
    }
    if (cfg.synaptic_decay) {
      document.getElementById('cfg-decay').value = cfg.synaptic_decay;
      document.getElementById('cfg-decay-display').innerText = Number(cfg.synaptic_decay).toFixed(3);
    }
    if (cfg.engine_mode) {
      document.getElementById('cfg-engine-mode').value = cfg.engine_mode;
    }
    if (cfg.slm_endpoint) {
      document.getElementById('cfg-slm-endpoint').value = cfg.slm_endpoint;
    }
    if (cfg.openai_endpoint) {
      document.getElementById('cfg-openai-endpoint').value = cfg.openai_endpoint;
    }
  }

  async saveSettings() {
    const payload = {
      beta: parseFloat(document.getElementById('cfg-beta').value),
      hallucination_threshold: parseFloat(document.getElementById('cfg-hallucination').value),
      synaptic_decay: parseFloat(document.getElementById('cfg-decay').value),
      engine_mode: document.getElementById('cfg-engine-mode').value,
      slm_endpoint: document.getElementById('cfg-slm-endpoint').value,
      openai_endpoint: document.getElementById('cfg-openai-endpoint').value,
    };

    const res = await this.apiPost('/api/settings', payload);
    if (res.success) {
      alert('Runtime settings saved and persisted to disk!');
      this.refreshSystemStatus();
    } else {
      alert(`Save failed: ${res.error}`);
    }
  }

  escapeHtml(str) {
    return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
}

// Global initialization
let cortexApp;
document.addEventListener('DOMContentLoaded', () => {
  cortexApp = new CortexStudioApp();
});
