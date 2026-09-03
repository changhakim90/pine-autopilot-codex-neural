// ==UserScript==
// @name         Pine Autopilot — Joe Learning Loop
// @namespace    https://pineandco.online/
// @version      12.0.0
// @description  Local-only neural-network Joe player for Pine & Co Cocktail Defense.
// @match        https://pineandco.online/*
// @homepageURL  https://github.com/changhakim90/pine-autopilot-codex-neural
// @updateURL    https://raw.githubusercontent.com/changhakim90/pine-autopilot-codex-neural/main/pine-autopilot.user.js
// @downloadURL  https://raw.githubusercontent.com/changhakim90/pine-autopilot-codex-neural/main/pine-autopilot.user.js
// @grant        none
// @run-at       document-idle
// ==/UserScript==

/*
 * Pine Autopilot
 *
 * It observes only the visible HUD and upgrade-card text, presses the same
 * inputs a player can use, and saves its own learning data to localStorage.
 * It deliberately never enters a name, clicks SAVE, or submits a rank.
 */
(function pineAutopilot() {
  'use strict';

  if (window.__pineAutopilotLoaded) return;
  window.__pineAutopilotLoaded = true;

  const VERSION = '12.0.0';
  // v12 retains v11's durable transfer path and captures manual key presses
  // immediately, including short taps that an interval sampler can miss.
  const STORE = 'pine-autopilot:joe:neural:v12';
  const LEGACY_STORE = 'pine-autopilot:joe:neural:v11';
  const CHANNEL = 'pine-autopilot:neural:v12';
  // At 100× game speed, a 55 ms wall-clock poll skips several game seconds.
  // Ten ms keeps the 1.2-second policy horizon meaningful while the page is
  // active, without asking the game for more animation frames.
  const TICK_MS = 10;
  const DECISION_GAME_SECONDS = 1.2;
  const ULTIMATE_GAME_SECONDS = 5;
  const FLOW_GAP_MS = 110;
  const VISION_HISTORY = 4;
  const VISION_FEATURES = 27 * VISION_HISTORY;
  const OBJECT_FEATURES = 10;
  const STATE_INPUTS = 20 + OBJECT_FEATURES + VISION_FEATURES;
  const MOVE_INPUTS = STATE_INPUTS;
  const CARD_INPUTS = STATE_INPUTS + 14;
  const HIDDEN_UNITS = 64;
  const LEARNING_RATE = 0.0018;
  const DISCOUNT = 0.985;
  const N_STEP = 4;
  const REPLAY_LIMIT = 24000;
  const REPLAY_BATCH = 8;
  const MIN_REPLAY_BATCH = 3;
  const MAX_REPLAY_BATCH = 12;
  const PRIORITY_ALPHA = 0.60;
  const PRIORITY_BETA_START = 0.40;
  const ELITE_LIMIT = 1800;
  const DEMONSTRATION_LIMIT = 500;
  const TARGET_SYNC = 1500;
  const CARD_CREDIT_SECONDS = 18;
  // A policy action that spans a larger observed game-time jump is ambiguous:
  // it may contain several unseen collisions. Keep it out of replay.
  const MAX_QUALITY_TIME_JUMP = 1.75;
  // The visible timer is integer-granularity, so a single two-second jump is
  // not enough to invalidate a full greedy evaluation. Estimate decisions at
  // this cadence and accept runs with <=10% dropped decision outcomes.
  const EVALUATION_DECISION_ESTIMATE_SECONDS = 2;
  const MAX_EVALUATION_DROPPED_RATE = 0.10;
  const EVALUATION_INTERVAL_RUNS = 6;
  const TOURNAMENT_WINDOW = 5;
  const PROMOTION_MARGIN = 1.03;
  const CHALLENGER_REJECTION_RATIO = 0.85;
  const ROLLBACK_COOLDOWN_MS = 5 * 60 * 1000;
  const ELITE_IMITATION_MARGIN = 0.28;
  const ELITE_IMITATION_WEIGHT = 0.32;
  const DEMONSTRATION_REPLAY_RATE = 0.34;
  const MASTERY_STATS_LIMIT = 1800;
  const MASTERY_MIN_VISITS = 3;
  const MASTERY_MAX_BONUS = 0.82;
  const MANUAL_VIDEO_FPS = 12;
  const MANUAL_VIDEO_BITS_PER_SECOND = 1800000;
  const TRANSFER_OUTBOX_PREFIX = 'pine-autopilot:joe:transfer:v12:';
  const MANUAL_OUTBOX_LIMIT = 180;
  const ELITE_OUTBOX_LIMIT = 4;
  const TRANSFER_ID_LIMIT = 2500;
  const TRANSFER_DRAIN_MS = 350;
  const TRANSFER_DRAIN_BATCH = 18;
  const RISK_MIN_SAMPLES = 8;
  const RISK_STATS_LIMIT = 1800;
  const IDLE_CONSOLIDATION_MAX_STEPS = 28;
  const IDLE_CONSOLIDATION_BUDGET_MS = 6;
  const AUTO_ARM = true;
  const ACTIONS = [
    { id: 'orbit-cw', label: 'Orbit ↻', steps: [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]] },
    { id: 'orbit-ccw', label: 'Orbit ↺', steps: [[1, 0], [1, -1], [0, -1], [-1, -1], [-1, 0], [-1, 1], [0, 1], [1, 1]] },
    { id: 'horizontal', label: 'Side sweep', steps: [[1, 0], [1, 0], [1, 1], [-1, 0], [-1, 0], [-1, -1]] },
    { id: 'vertical', label: 'Vertical sweep', steps: [[0, 1], [1, 1], [0, 1], [0, -1], [-1, -1], [0, -1]] },
    { id: 'weave', label: 'Diagonal weave', steps: [[1, 1], [-1, 1], [-1, -1], [1, -1]] },
    { id: 'perimeter', label: 'Perimeter', steps: [[1, 0], [1, 0], [0, 1], [-1, 0], [-1, 0], [0, -1]] },
    { id: 'north', label: 'North', steps: [[0, -1]] },
    { id: 'north-east', label: 'North-east', steps: [[1, -1]] },
    { id: 'east', label: 'East', steps: [[1, 0]] },
    { id: 'south-east', label: 'South-east', steps: [[1, 1]] },
    { id: 'south', label: 'South', steps: [[0, 1]] },
    { id: 'south-west', label: 'South-west', steps: [[-1, 1]] },
    { id: 'west', label: 'West', steps: [[-1, 0]] },
    { id: 'north-west', label: 'North-west', steps: [[-1, -1]] },
    { id: 'dash-east', label: 'Dash east', steps: [[1, 0]], dash: true },
    { id: 'dash-south', label: 'Dash south', steps: [[0, 1]], dash: true },
    { id: 'dash-west', label: 'Dash west', steps: [[-1, 0]], dash: true },
    { id: 'dash-north', label: 'Dash north', steps: [[0, -1]], dash: true },
    { id: 'ultimate', label: 'Ultimate', steps: [[0, 0]], ultimate: true },
  ];

  const model = loadModel();
  // Earlier demonstrations become the protected policy on first load. The
  // compact policy survives after the raw 500-sample demo window rotates.
  if (!Object.keys(model.masteryPolicy || {}).length && model.demonstrations.length) rebuildMasteryFromDemonstrations();
  let adaptiveReplayBatch = REPLAY_BATCH;
  const priorityTree = new Float64Array(REPLAY_LIMIT * 2);
  let replayPeakPriority = 1;
  const replayPhaseCounts = { early: 0, mid: 0, late: 0, hell: 0 };
  rebuildReplayTree();
  const learnerId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
  const peerChannel = typeof BroadcastChannel === 'function' ? new BroadcastChannel(CHANNEL) : null;
  const peerLastSeen = {};
  let coordinatorId = learnerId;

  function refreshCoordinator() {
    const now = Date.now();
    Object.keys(peerLastSeen).forEach((id) => {
      if (now - peerLastSeen[id] > 6500) delete peerLastSeen[id];
    });
    coordinatorId = [learnerId].concat(Object.keys(peerLastSeen)).sort()[0];
  }

  function observePeer(id) {
    if (!id || id === learnerId) return;
    peerLastSeen[id] = Date.now();
    refreshCoordinator();
  }

  function isCoordinator() {
    refreshCoordinator();
    return !peerChannel || coordinatorId === learnerId;
  }

  function learnerRole() {
    return isCoordinator() ? 'central learner' : 'experience worker';
  }

  function hashText(text) {
    let hash = 2166136261;
    for (let index = 0; index < text.length; index += 1) {
      hash ^= text.charCodeAt(index);
      hash = Math.imul(hash, 16777619);
    }
    return hash >>> 0;
  }

  const workerSeed = hashText(learnerId);
  const workerExplorationScale = 0.82 + ((workerSeed % 37) / 100);
  const workerDecisionSeconds = DECISION_GAME_SECONDS * (0.80 + (((workerSeed >>> 8) % 41) / 100));
  const workerStyle = workerSeed % 4;
  const workerProfile = ['orbit', 'sweep', 'ability', 'cautious'][workerStyle];
  const run = {
    enabled: false,
    timer: null,
    startedAt: 0,
    gameStartedAt: 0,
    lastFlowAt: 0,
    lastDecisionGameTime: 0,
    lastObservedGameTime: null,
    lastGameTimeJump: 0,
    largestGameTimeJump: 0,
    timingWarnings: 0,
    runTimingWarningsAtStart: 0,
    lowQualityDropped: 0,
    lowQualityDroppedAtStart: 0,
    maxPhase: 'early',
    intent: 'survive',
    lastMotionStepGameTime: 0,
    lastUltimateGameTime: 0,
    lastDashGameTime: -Infinity,
    pattern: null,
    patternIndex: 0,
    decision: null,
    nStep: [],
    episodeTransitions: [],
    evaluation: false,
    evaluationPolicy: 'candidate',
    pressed: new Set(),
    lastCard: null,
    lastCardAt: 0,
    lastCardGameTime: 0,
    forceRestart: false,
    startupAttempted: false,
    retryAt: 0,
    hellMode: false,
    status: 'Ready. Arm Autopilot to begin the Joe loop.',
  };
  const manualDemo = {
    active: false,
    timer: null,
    pressed: new Set(),
    decision: null,
    video: null,
    chunks: [],
    stream: null,
    startedAt: 0,
    keyDownAt: {},
    burstActionIndex: -1,
    burstUntil: 0,
    sequence: 0,
  };

  const panel = mountPanel();
  let lastPanelRender = 0;
  if (!model.champion && cleanEvaluationResults().length >= 6) {
    bootstrapChampionFromHistory();
  }
  renderPanel();
  if (AUTO_ARM) setTimeout(arm, 900);

  function createHead(seed) {
    const movementNet = seed ? copyNetwork(seed.movementNet) : createNetwork(MOVE_INPUTS, HIDDEN_UNITS, ACTIONS.length);
    const cardNet = seed ? copyNetwork(seed.cardNet) : createNetwork(CARD_INPUTS, HIDDEN_UNITS, 1);
    return {
      movementNet,
      movementTarget: copyNetwork(movementNet),
      cardNet,
      cardTarget: copyNetwork(cardNet),
      uniqueExperiences: 0,
    };
  }

  function blankModel() {
    const earlyHead = createHead();
    const midHead = createHead(earlyHead);
    const lateHead = createHead(midHead);
    return {
      version: 12,
      completedRuns: 0,
      bestSeconds: 0,
      totalSeconds: 0,
      movementNet: earlyHead.movementNet,
      movementTarget: earlyHead.movementTarget,
      cardNet: earlyHead.cardNet,
      cardTarget: earlyHead.cardTarget,
      midHead,
      lateHead,
      hellHead: createHead(lateHead),
      localExperiences: 0,
      peerExperiences: 0,
      uniqueExperiences: 0,
      settings: { safetyRestartAt60: true, panelMinimized: false, immortalMode: true },
      replay: [],
      replayCursor: 0,
      elite: [],
      eliteCount: 0,
      eliteScores: [],
      demonstrations: [],
      demonstrationCount: 0,
      masteryPolicy: {},
      masteryUpdates: 0,
      masteryActionUses: 0,
      manualTransferIds: [],
      eliteTransferIds: [],
      manualTransfersAccepted: 0,
      eliteTransfersAccepted: 0,
      evaluationResults: [],
      champion: null,
      lastRollbackAt: 0,
      tournamentGeneration: 0,
      migratedFromV8: false,
      riskStats: {},
      riskUpdates: 0,
      eliteImitationSteps: 0,
      consolidationSteps: 0,
      consolidationPasses: 0,
      events: [],
    };
  }

  function validHead(head) {
    return head && validNetwork(head.movementNet, MOVE_INPUTS, ACTIONS.length) && validNetwork(head.cardNet, CARD_INPUTS, 1);
  }

  function normalizedHead(saved, fallback) {
    if (!validHead(saved)) return fallback;
    return {
      ...fallback,
      ...saved,
      movementNet: saved.movementNet,
      movementTarget: validNetwork(saved.movementTarget, MOVE_INPUTS, ACTIONS.length) ? saved.movementTarget : copyNetwork(saved.movementNet),
      cardNet: saved.cardNet,
      cardTarget: validNetwork(saved.cardTarget, CARD_INPUTS, 1) ? saved.cardTarget : copyNetwork(saved.cardNet),
      uniqueExperiences: Number(saved.uniqueExperiences) || 0,
    };
  }

  function loadModel() {
    try {
      const v12Stored = localStorage.getItem(STORE);
      const saved = unpackStoredModel(JSON.parse(v12Stored || localStorage.getItem(LEGACY_STORE) || 'null'));
      const fresh = blankModel();
      if (!saved || typeof saved !== 'object') return fresh;
      return {
        ...fresh,
        ...saved,
        version: 12,
        migratedFromV8: !!saved.migratedFromV8,
        movementNet: validNetwork(saved.movementNet, MOVE_INPUTS, ACTIONS.length) ? saved.movementNet : fresh.movementNet,
        cardNet: validNetwork(saved.cardNet, CARD_INPUTS, 1) ? saved.cardNet : fresh.cardNet,
        movementTarget: validNetwork(saved.movementTarget, MOVE_INPUTS, ACTIONS.length) ? saved.movementTarget : copyNetwork(validNetwork(saved.movementNet, MOVE_INPUTS, ACTIONS.length) ? saved.movementNet : fresh.movementNet),
        cardTarget: validNetwork(saved.cardTarget, CARD_INPUTS, 1) ? saved.cardTarget : copyNetwork(validNetwork(saved.cardNet, CARD_INPUTS, 1) ? saved.cardNet : fresh.cardNet),
        midHead: normalizedHead(saved.midHead, fresh.midHead),
        lateHead: normalizedHead(saved.lateHead, fresh.lateHead),
        hellHead: normalizedHead(saved.hellHead, fresh.hellHead),
        localExperiences: Number(saved.localExperiences) || 0,
        peerExperiences: Number(saved.peerExperiences) || 0,
        uniqueExperiences: Number(saved.uniqueExperiences) || 0,
        settings: { ...fresh.settings, ...(saved.settings || {}) },
        replay: Array.isArray(saved.replay) ? saved.replay.slice(-REPLAY_LIMIT) : [],
        replayCursor: Number(saved.replayCursor) || 0,
        elite: Array.isArray(saved.elite) ? saved.elite.slice(-ELITE_LIMIT) : [],
        eliteCount: Number(saved.eliteCount) || (Array.isArray(saved.elite) ? saved.elite.length : 0),
        eliteScores: Array.isArray(saved.eliteScores) ? saved.eliteScores.slice(-12) : [],
        demonstrations: Array.isArray(saved.demonstrations) ? saved.demonstrations.slice(-DEMONSTRATION_LIMIT) : [],
        demonstrationCount: Number(saved.demonstrationCount) || (Array.isArray(saved.demonstrations) ? saved.demonstrations.length : 0),
        masteryPolicy: normalizeMasteryPolicy(saved.masteryPolicy),
        masteryUpdates: Number(saved.masteryUpdates) || 0,
        masteryActionUses: Number(saved.masteryActionUses) || 0,
        manualTransferIds: Array.isArray(saved.manualTransferIds) ? saved.manualTransferIds.slice(-TRANSFER_ID_LIMIT).filter((id) => typeof id === 'string') : [],
        eliteTransferIds: Array.isArray(saved.eliteTransferIds) ? saved.eliteTransferIds.slice(-TRANSFER_ID_LIMIT).filter((id) => typeof id === 'string') : [],
        manualTransfersAccepted: Number(saved.manualTransfersAccepted) || 0,
        eliteTransfersAccepted: Number(saved.eliteTransfersAccepted) || 0,
        evaluationResults: Array.isArray(saved.evaluationResults) ? saved.evaluationResults.slice(-60).map(normalizeEvaluationRecord) : [],
        champion: saved.champion && validHead(saved.champion.earlyHead) && validHead(saved.champion.midHead) && validHead(saved.champion.lateHead) && validHead(saved.champion.hellHead) ? saved.champion : null,
        riskStats: normalizeRiskStats(saved.riskStats),
        riskUpdates: Number(saved.riskUpdates) || 0,
        eliteImitationSteps: Number(saved.eliteImitationSteps) || 0,
        consolidationSteps: Number(saved.consolidationSteps) || 0,
        consolidationPasses: Number(saved.consolidationPasses) || 0,
        events: Array.isArray(saved.events) ? saved.events.slice(-60) : [],
      };
    } catch (_) {
      return blankModel();
    }
  }

  function saveModel(force = false) {
    // One writer prevents an experience-worker tab from overwriting the
    // central learner's newer weights in same-origin localStorage.
    if (!force && peerChannel && !isCoordinator()) return;
    try {
      // Standard replay stays RAM-only. Network tensors are packed into
      // Float32/base64 so the four curriculum heads and champion fit within
      // browser localStorage without silently exceeding its quota.
      localStorage.setItem(STORE, JSON.stringify(packStoredModel()));
    } catch (_) { /* private mode can block persistence */ }
  }

  function randomWeight(fanIn) {
    return (Math.random() * 2 - 1) * Math.sqrt(2 / fanIn);
  }

  function createNetwork(inputs, hidden, outputs) {
    return {
      inputs,
      hidden,
      outputs,
      w1: Array.from({ length: inputs * hidden }, () => randomWeight(inputs)),
      b1: Array(hidden).fill(0),
      wValue: Array.from({ length: hidden }, () => randomWeight(hidden)),
      bValue: 0,
      wAdvantage: Array.from({ length: hidden * outputs }, () => randomWeight(hidden)),
      bAdvantage: Array(outputs).fill(0),
      samples: 0,
    };
  }

  function copyNetwork(network) {
    return {
      inputs: network.inputs,
      hidden: network.hidden,
      outputs: network.outputs,
      w1: network.w1.slice(),
      b1: network.b1.slice(),
      wValue: network.wValue.slice(),
      bValue: network.bValue,
      wAdvantage: network.wAdvantage.slice(),
      bAdvantage: network.bAdvantage.slice(),
      samples: network.samples || 0,
    };
  }

  function validNetwork(network, inputs, outputs) {
    return network && network.inputs === inputs && network.hidden === HIDDEN_UNITS && network.outputs === outputs
      && Array.isArray(network.w1) && network.w1.length === inputs * HIDDEN_UNITS
      && Array.isArray(network.b1) && network.b1.length === HIDDEN_UNITS
      && Array.isArray(network.wValue) && network.wValue.length === HIDDEN_UNITS
      && Array.isArray(network.wAdvantage) && network.wAdvantage.length === HIDDEN_UNITS * outputs
      && Array.isArray(network.bAdvantage) && network.bAdvantage.length === outputs
      && Number.isFinite(network.bValue);
  }

  function packFloat32(values) {
    if (!Array.isArray(values) || typeof btoa !== 'function') return values;
    const bytes = new Uint8Array(new Float32Array(values).buffer);
    let binary = '';
    for (let index = 0; index < bytes.length; index += 0x4000) binary += String.fromCharCode.apply(null, bytes.subarray(index, index + 0x4000));
    return { float32: btoa(binary) };
  }

  function unpackFloat32(value) {
    if (Array.isArray(value)) return value;
    if (!value || typeof value.float32 !== 'string' || typeof atob !== 'function') return [];
    const binary = atob(value.float32);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) bytes[index] = binary.charCodeAt(index);
    return Array.from(new Float32Array(bytes.buffer));
  }

  function packNetwork(network) {
    if (!network) return network;
    return {
      ...network,
      w1: packFloat32(network.w1),
      b1: packFloat32(network.b1),
      wValue: packFloat32(network.wValue),
      wAdvantage: packFloat32(network.wAdvantage),
      bAdvantage: packFloat32(network.bAdvantage),
    };
  }

  function unpackNetwork(network) {
    if (!network) return network;
    return {
      ...network,
      w1: unpackFloat32(network.w1),
      b1: unpackFloat32(network.b1),
      wValue: unpackFloat32(network.wValue),
      wAdvantage: unpackFloat32(network.wAdvantage),
      bAdvantage: unpackFloat32(network.bAdvantage),
    };
  }

  function packHead(head) {
    if (!head) return head;
    return {
      ...head,
      movementNet: packNetwork(head.movementNet),
      movementTarget: packNetwork(head.movementTarget),
      cardNet: packNetwork(head.cardNet),
      cardTarget: packNetwork(head.cardTarget),
    };
  }

  function unpackHead(head) {
    if (!head) return head;
    return {
      ...head,
      movementNet: unpackNetwork(head.movementNet),
      movementTarget: unpackNetwork(head.movementTarget),
      cardNet: unpackNetwork(head.cardNet),
      cardTarget: unpackNetwork(head.cardTarget),
    };
  }

  function packExperience(experience) {
    return { ...experience, input: packFloat32(experience.input), nextState: packFloat32(experience.nextState) };
  }

  function unpackExperience(experience) {
    return { ...experience, input: unpackFloat32(experience.input), nextState: unpackFloat32(experience.nextState) };
  }

  function packStoredModel() {
    const champion = model.champion ? {
      ...model.champion,
      earlyHead: packHead(model.champion.earlyHead),
      midHead: packHead(model.champion.midHead),
      lateHead: packHead(model.champion.lateHead),
      hellHead: packHead(model.champion.hellHead),
    } : null;
    return {
      ...model,
      movementNet: packNetwork(model.movementNet),
      movementTarget: packNetwork(model.movementTarget),
      cardNet: packNetwork(model.cardNet),
      cardTarget: packNetwork(model.cardTarget),
      midHead: packHead(model.midHead),
      lateHead: packHead(model.lateHead),
      hellHead: packHead(model.hellHead),
      champion,
      replay: [],
      elite: model.elite.slice(-400).map(packExperience),
      eliteCount: Math.min(model.elite.length, 400),
      demonstrations: model.demonstrations.slice(-DEMONSTRATION_LIMIT).map(packExperience),
      demonstrationCount: Math.min(model.demonstrations.length, DEMONSTRATION_LIMIT),
    };
  }

  function unpackStoredModel(saved) {
    if (!saved || typeof saved !== 'object') return saved;
    const champion = saved.champion ? {
      ...saved.champion,
      earlyHead: unpackHead(saved.champion.earlyHead),
      midHead: unpackHead(saved.champion.midHead),
      lateHead: unpackHead(saved.champion.lateHead),
      hellHead: unpackHead(saved.champion.hellHead),
    } : null;
    return {
      ...saved,
      movementNet: unpackNetwork(saved.movementNet),
      movementTarget: unpackNetwork(saved.movementTarget),
      cardNet: unpackNetwork(saved.cardNet),
      cardTarget: unpackNetwork(saved.cardTarget),
      midHead: unpackHead(saved.midHead),
      lateHead: unpackHead(saved.lateHead),
      hellHead: unpackHead(saved.hellHead),
      champion,
      elite: Array.isArray(saved.elite) ? saved.elite.map(unpackExperience) : [],
      demonstrations: Array.isArray(saved.demonstrations) ? saved.demonstrations.map(unpackExperience) : [],
    };
  }

  function event(message) {
    model.events.push({ at: Date.now(), message });
    model.events = model.events.slice(-60);
  }

  function transferOutboxKey(kind) {
    return `${TRANSFER_OUTBOX_PREFIX}${kind}:${learnerId}`;
  }

  function readTransferOutbox(key) {
    try {
      const entries = JSON.parse(localStorage.getItem(key) || '[]');
      return Array.isArray(entries) ? entries : [];
    } catch (_) {
      return [];
    }
  }

  function writeTransferOutbox(key, entries) {
    try {
      localStorage.setItem(key, JSON.stringify(entries));
      return true;
    } catch (_) {
      return false;
    }
  }

  function queueTransfer(kind, payload, limit) {
    const key = transferOutboxKey(kind);
    const entries = readTransferOutbox(key);
    entries.push(payload);
    writeTransferOutbox(key, entries.slice(-limit));
  }

  function rememberTransferId(kind, id) {
    if (!id || typeof id !== 'string') return true;
    const field = kind === 'manual' ? 'manualTransferIds' : 'eliteTransferIds';
    if (model[field].includes(id)) return false;
    model[field].push(id);
    if (model[field].length > TRANSFER_ID_LIMIT) model[field].splice(0, model[field].length - TRANSFER_ID_LIMIT);
    return true;
  }

  function acceptManualTransfer(experience) {
    if (!experience || !Array.isArray(experience.input)) return false;
    if (!rememberTransferId('manual', experience.transferId)) return true;
    rememberDemonstration(experience);
    trainExperience(experience, true);
    model.manualTransfersAccepted += 1;
    return true;
  }

  function acceptEliteTransfer(episode) {
    if (!episode || !Array.isArray(episode.transitions)) return false;
    if (!rememberTransferId('elite', episode.transferId)) return true;
    rememberEliteEpisode(episode);
    model.eliteTransfersAccepted += 1;
    return true;
  }

  let lastTransferDrainAt = 0;

  function drainDurableTransfers(force = false) {
    if (!isCoordinator() || (!force && Date.now() - lastTransferDrainAt < TRANSFER_DRAIN_MS)) return 0;
    lastTransferDrainAt = Date.now();
    let accepted = 0;
    const keys = [];
    try {
      for (let index = 0; index < localStorage.length; index += 1) {
        const key = localStorage.key(index);
        if (key && key.startsWith(TRANSFER_OUTBOX_PREFIX)) keys.push(key);
      }
    } catch (_) { return 0; }
    keys.forEach((key) => {
      if (accepted >= TRANSFER_DRAIN_BATCH) return;
      const kind = key.slice(TRANSFER_OUTBOX_PREFIX.length).split(':')[0];
      const entries = readTransferOutbox(key);
      entries.slice(0, TRANSFER_DRAIN_BATCH - accepted).forEach((entry) => {
        const received = kind === 'manual' ? acceptManualTransfer(entry) : kind === 'elite' ? acceptEliteTransfer(entry) : false;
        if (received) accepted += 1;
      });
      // Re-read before retaining unseen entries so a worker append that landed
      // while this tab trained is kept for the next drain.
      const latest = readTransferOutbox(key);
      const idField = kind === 'manual' ? 'manualTransferIds' : 'eliteTransferIds';
      const pending = latest.filter((entry) => !entry || !entry.transferId || !model[idField].includes(entry.transferId));
      writeTransferOutbox(key, pending);
    });
    if (accepted) {
      event(`Durably accepted ${accepted} worker learning transfer${accepted === 1 ? '' : 's'}.`);
      saveModel(true);
      shareSnapshot(null, true);
    }
    return accepted;
  }

  function normalizeRiskStats(value) {
    if (!value || typeof value !== 'object') return {};
    const entries = Object.entries(value)
      .map(([key, stat]) => [key, {
        visits: Math.max(0, Math.min(100000, Number(stat && stat.visits) || 0)),
        deaths: Math.max(0, Math.min(100000, Number(stat && stat.deaths) || 0)),
        at: Math.max(0, Number(stat && stat.at) || 0),
      }])
      .filter(([key, stat]) => key.length < 80 && stat.visits > 0)
      .sort((left, right) => right[1].at - left[1].at)
      .slice(0, RISK_STATS_LIMIT);
    return Object.fromEntries(entries);
  }

  function phaseFor(observationOrExperience) {
    if (!observationOrExperience) return 'early';
    if (observationOrExperience.phase) return observationOrExperience.phase;
    if (observationOrExperience.hell) return 'hell';
    if ((observationOrExperience.time || 0) < 300) return 'early';
    if ((observationOrExperience.time || 0) < 900) return 'mid';
    return 'late';
  }

  function phaseRank(phase) {
    return ({ early: 0, mid: 1, late: 2, hell: 3 })[phase] ?? 0;
  }

  function furthestPhase(current, candidate) {
    return phaseRank(candidate) > phaseRank(current) ? candidate : current;
  }

  function headFor(observationOrExperience) {
    const phase = phaseFor(observationOrExperience);
    if (phase === 'hell') return model.hellHead;
    if (phase === 'mid') return model.midHead;
    if (phase === 'late') return model.lateHead;
    return model;
  }

  function championHeadFor(observationOrExperience) {
    if (!model.champion) return headFor(observationOrExperience);
    const phase = phaseFor(observationOrExperience);
    if (phase === 'hell') return model.champion.hellHead;
    if (phase === 'mid') return model.champion.midHead;
    if (phase === 'late') return model.champion.lateHead;
    return model.champion.earlyHead;
  }

  function actingHeadFor(observationOrExperience) {
    return run.evaluation && run.evaluationPolicy === 'champion' ? championHeadFor(observationOrExperience) : headFor(observationOrExperience);
  }

  function headSampleCount(head) {
    return (head.movementNet.samples || 0) + (head.cardNet.samples || 0);
  }

  function experienceCount() {
    return headSampleCount(model) + headSampleCount(model.midHead) + headSampleCount(model.lateHead) + headSampleCount(model.hellHead);
  }

  function cloneForPeer(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function shareExperience(experience) {
    if (!peerChannel) return;
    peerChannel.postMessage({ type: 'experience', from: learnerId, experience });
  }

  function headSnapshot(head) {
    return {
      movementNet: head.movementNet,
      movementTarget: head.movementTarget,
      cardNet: head.cardNet,
      cardTarget: head.cardTarget,
      uniqueExperiences: head.uniqueExperiences || 0,
    };
  }

  function applyHeadSnapshot(target, incoming) {
    if (!validHead(incoming)) return false;
    target.movementNet = incoming.movementNet;
    target.cardNet = incoming.cardNet;
    target.movementTarget = validNetwork(incoming.movementTarget, MOVE_INPUTS, ACTIONS.length) ? incoming.movementTarget : copyNetwork(incoming.movementNet);
    target.cardTarget = validNetwork(incoming.cardTarget, CARD_INPUTS, 1) ? incoming.cardTarget : copyNetwork(incoming.cardNet);
    target.uniqueExperiences = Math.max(target.uniqueExperiences || 0, Number(incoming.uniqueExperiences) || 0);
    return true;
  }

  function shareSnapshot(target, force = false) {
    if (!peerChannel) return;
    const champion = model.champion ? {
      ...model.champion,
      earlyHead: packHead(model.champion.earlyHead),
      midHead: packHead(model.champion.midHead),
      lateHead: packHead(model.champion.lateHead),
      hellHead: packHead(model.champion.hellHead),
    } : null;
    peerChannel.postMessage({
      type: 'snapshot',
      from: learnerId,
      target,
      force,
      uniqueExperiences: model.uniqueExperiences,
      model: cloneForPeer({
        earlyHead: packHead(headSnapshot(model)),
        midHead: packHead(headSnapshot(model.midHead)),
        lateHead: packHead(headSnapshot(model.lateHead)),
        hellHead: packHead(headSnapshot(model.hellHead)),
        localExperiences: model.localExperiences,
        peerExperiences: model.peerExperiences,
        uniqueExperiences: model.uniqueExperiences,
        eliteCount: model.elite.length,
        eliteScores: model.eliteScores,
        demonstrationCount: model.demonstrations.length,
        masteryPolicy: model.masteryPolicy,
        masteryUpdates: model.masteryUpdates,
        masteryActionUses: model.masteryActionUses,
        manualTransfersAccepted: model.manualTransfersAccepted,
        eliteTransfersAccepted: model.eliteTransfersAccepted,
        riskStats: model.riskStats,
        riskUpdates: model.riskUpdates,
        eliteImitationSteps: model.eliteImitationSteps,
        consolidationSteps: model.consolidationSteps,
        consolidationPasses: model.consolidationPasses,
        evaluationResults: model.evaluationResults,
        completedRuns: model.completedRuns,
        bestSeconds: model.bestSeconds,
        totalSeconds: model.totalSeconds,
        champion,
        tournamentGeneration: model.tournamentGeneration,
        lastRollbackAt: model.lastRollbackAt,
      }),
    });
  }

  let lastSnapshotExperienceCount = model.uniqueExperiences;

  function maybeShareSnapshot() {
    // Raw experiences remain the main shared learning signal. A periodic
    // canonical snapshot lets a newly opened or lagging tab catch up without
    // waiting for an entire replay buffer to refill.
    if (!isCoordinator() || model.uniqueExperiences - lastSnapshotExperienceCount < 500) return;
    lastSnapshotExperienceCount = model.uniqueExperiences;
    shareSnapshot(null);
  }

  if (peerChannel) {
    peerChannel.onmessage = ({ data }) => {
      if (!data || data.from === learnerId) return;
      observePeer(data.from);
      if (data.type === 'hello' || data.type === 'presence') {
        if (data.type === 'hello' && isCoordinator() && model.uniqueExperiences > (data.uniqueExperiences || 0)) shareSnapshot(data.from);
        return;
      }
      if (data.type === 'snapshot' && (!data.target || data.target === learnerId)) {
        const incomingUnique = Number(data.uniqueExperiences) || (data.model && Number(data.model.uniqueExperiences)) || 0;
        const minimumLead = data.target === learnerId ? 1 : 500;
        const incomingGeneration = Number(data.model && data.model.tournamentGeneration) || 0;
        const tournamentAdvance = data.force && incomingGeneration > (Number(model.tournamentGeneration) || 0);
        if (data.model && (incomingUnique >= model.uniqueExperiences + minimumLead || tournamentAdvance)) {
          const incomingModel = {
            ...data.model,
            earlyHead: unpackHead(data.model.earlyHead),
            midHead: unpackHead(data.model.midHead),
            lateHead: unpackHead(data.model.lateHead),
            hellHead: unpackHead(data.model.hellHead),
            champion: data.model.champion ? {
              ...data.model.champion,
              earlyHead: unpackHead(data.model.champion.earlyHead),
              midHead: unpackHead(data.model.champion.midHead),
              lateHead: unpackHead(data.model.champion.lateHead),
              hellHead: unpackHead(data.model.champion.hellHead),
            } : null,
          };
          const earlyApplied = applyHeadSnapshot(model, incomingModel.earlyHead);
          const midApplied = applyHeadSnapshot(model.midHead, incomingModel.midHead);
          const lateApplied = applyHeadSnapshot(model.lateHead, incomingModel.lateHead);
          const hellApplied = applyHeadSnapshot(model.hellHead, incomingModel.hellHead);
          if (earlyApplied || midApplied || lateApplied || hellApplied) {
          model.localExperiences = Math.max(model.localExperiences, Number(incomingModel.localExperiences) || 0);
          model.peerExperiences = Math.max(model.peerExperiences, Number(incomingModel.peerExperiences) || 0);
          model.uniqueExperiences = Math.max(model.uniqueExperiences, incomingUnique);
          model.eliteCount = Math.max(model.eliteCount || 0, Number(incomingModel.eliteCount) || 0);
          model.eliteScores = Array.isArray(incomingModel.eliteScores) ? incomingModel.eliteScores.slice(-12) : model.eliteScores;
          model.demonstrationCount = Math.max(model.demonstrationCount || 0, Number(incomingModel.demonstrationCount) || 0);
          if (incomingModel.masteryPolicy) model.masteryPolicy = normalizeMasteryPolicy(incomingModel.masteryPolicy);
          model.masteryUpdates = Math.max(model.masteryUpdates || 0, Number(incomingModel.masteryUpdates) || 0);
          model.masteryActionUses = Math.max(model.masteryActionUses || 0, Number(incomingModel.masteryActionUses) || 0);
          model.manualTransfersAccepted = Math.max(model.manualTransfersAccepted || 0, Number(incomingModel.manualTransfersAccepted) || 0);
          model.eliteTransfersAccepted = Math.max(model.eliteTransfersAccepted || 0, Number(incomingModel.eliteTransfersAccepted) || 0);
          if (incomingModel.riskStats) model.riskStats = normalizeRiskStats(incomingModel.riskStats);
          model.riskUpdates = Math.max(model.riskUpdates || 0, Number(incomingModel.riskUpdates) || 0);
          model.eliteImitationSteps = Math.max(model.eliteImitationSteps || 0, Number(incomingModel.eliteImitationSteps) || 0);
          model.consolidationSteps = Math.max(model.consolidationSteps || 0, Number(incomingModel.consolidationSteps) || 0);
          model.consolidationPasses = Math.max(model.consolidationPasses || 0, Number(incomingModel.consolidationPasses) || 0);
          model.evaluationResults = Array.isArray(incomingModel.evaluationResults) ? incomingModel.evaluationResults.slice(-40) : model.evaluationResults;
          model.completedRuns = Math.max(model.completedRuns, Number(incomingModel.completedRuns) || 0);
          model.bestSeconds = Math.max(model.bestSeconds, Number(incomingModel.bestSeconds) || 0);
          model.totalSeconds = Math.max(model.totalSeconds, Number(incomingModel.totalSeconds) || 0);
          if (incomingModel.champion && validHead(incomingModel.champion.earlyHead) && validHead(incomingModel.champion.midHead) && validHead(incomingModel.champion.lateHead) && validHead(incomingModel.champion.hellHead)) {
            model.champion = incomingModel.champion;
          }
          model.tournamentGeneration = Math.max(Number(model.tournamentGeneration) || 0, Number(incomingModel.tournamentGeneration) || 0);
          model.lastRollbackAt = Math.max(Number(model.lastRollbackAt) || 0, Number(incomingModel.lastRollbackAt) || 0);
          lastSnapshotExperienceCount = model.uniqueExperiences;
          event(`Synced neural model at ${model.uniqueExperiences} shared experiences.`);
          saveModel();
          }
        }
        return;
      }
      if (data.type === 'experience' && data.experience) {
        if (isCoordinator()) trainExperience(data.experience, true);
        else model.peerExperiences += 1;
        return;
      }
      if (data.type === 'demonstration' && data.experience && isCoordinator()) {
        acceptManualTransfer(data.experience);
        return;
      }
      if (data.type === 'eliteEpisode' && data.episode && isCoordinator()) {
        acceptEliteTransfer(data.episode);
        return;
      }
      if (data.type === 'restoreChampionRequest' && isCoordinator()) {
        if (restoreChampion('Champion restore requested by a worker tab')) event('Shared champion restore completed.');
        return;
      }
      if (data.type === 'runSummary' && data.result && isCoordinator()) {
        const result = data.result;
        model.completedRuns += 1;
        model.totalSeconds += Number(result.seconds) || 0;
        model.bestSeconds = Math.max(model.bestSeconds, Number(result.seconds) || 0);
        if (result.evaluation) rememberEvaluation(result);
        else {
          event(`Peer run ${model.completedRuns}: ${formatTime(result.seconds)}`);
          scheduleConsolidation('peer run');
        }
        saveModel();
        return;
      }
      if (data.type === 'reset') {
        Object.assign(model, blankModel());
        rebuildReplayTree();
        lastSnapshotExperienceCount = 0;
        event('Shared neural model reset.');
        saveModel(true);
      }
    };
    peerChannel.postMessage({ type: 'hello', from: learnerId, uniqueExperiences: model.uniqueExperiences });
    setInterval(() => {
      refreshCoordinator();
      peerChannel.postMessage({ type: 'presence', from: learnerId, uniqueExperiences: model.uniqueExperiences });
    }, 2000);
  }

  function visible(element) {
    if (!element || !element.isConnected) return false;
    for (let node = element; node && node !== document.documentElement; node = node.parentElement) {
      if (node.classList && node.classList.contains('hidden')) return false;
      const style = getComputedStyle(node);
      if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || 1) === 0) return false;
    }
    const rect = element.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  }

  function click(element, label) {
    if (!element || Date.now() - run.lastFlowAt < FLOW_GAP_MS) return false;
    element.click();
    run.lastFlowAt = Date.now();
    if (label) event(label);
    return true;
  }

  function findVisible(selector, predicate = () => true) {
    return [...document.querySelectorAll(selector)].find((element) => visible(element) && predicate(element));
  }

  function parseClock(text) {
    const fields = String(text || '').trim().split(':').map(Number);
    if (fields.length === 2 && fields.every(Number.isFinite)) return fields[0] * 60 + fields[1];
    if (fields.length === 3 && fields.every(Number.isFinite)) return fields[0] * 3600 + fields[1] * 60 + fields[2];
    return 0;
  }

  let lastVisionAt = 0;
  let latestVision = Array(27).fill(0);
  let visionHistory = Array.from({ length: VISION_HISTORY }, () => Array(27).fill(0));
  // These are lightweight colour-cluster statistics, not claims about the
  // game's internal entities. They give the network object-like position and
  // density cues while remaining robust to asset changes and cheap enough for
  // parallel tabs.
  let latestObjectFeatures = [0.5, 0.5, 0, 0.5, 0.5, 0, 0.5, 0.5, 0, 0.5];
  let previousHostileMass = 0;
  const visionCanvas = document.createElement('canvas');
  visionCanvas.width = 48;
  visionCanvas.height = 48;
  const visionContext = visionCanvas.getContext('2d', { willReadFrequently: true });

  function visualState() {
    // A 3×3 spatial map for hostile warm pixels, green hazards, and yellow
    // loot, plus their colour-cluster centroids. This is intentionally not a
    // costly CNN: live game data, not an approximate simulator, remains the
    // source of truth.
    if (Date.now() - lastVisionAt < 25) return visionHistory.reduce((all, frame) => all.concat(frame), []);
    lastVisionAt = Date.now();
    try {
      const canvas = document.querySelector('canvas#game');
      const ctx = canvas && canvas.getContext ? canvas.getContext('2d', { willReadFrequently: true }) : null;
      if (!ctx) return visionHistory.reduce((all, frame) => all.concat(frame), []);
      // Reading a full-size canvas repeatedly is costly in several 100× tabs.
      // Downsample first, then read only a 48×48 image for spatial features.
      if (!visionContext) return visionHistory.reduce((all, frame) => all.concat(frame), []);
      visionContext.clearRect(0, 0, 48, 48);
      visionContext.drawImage(canvas, 0, canvas.height * 0.08, canvas.width, canvas.height * 0.92, 0, 0, 48, 48);
      const pixels = visionContext.getImageData(0, 0, 48, 48).data;
      const bins = Array.from({ length: 9 }, () => ({ hostile: 0, hazard: 0, loot: 0, samples: 0 }));
      const clusters = {
        hostile: { mass: 0, x: 0, y: 0 },
        hazard: { mass: 0, x: 0, y: 0 },
        loot: { mass: 0, x: 0, y: 0 },
      };
      for (let y = 3; y < 48; y += 3) {
        for (let x = 3; x < 45; x += 3) {
          const offset = (y * 48 + x) * 4;
          const r = pixels[offset]; const g = pixels[offset + 1]; const b = pixels[offset + 2];
          const col = Math.min(2, Math.floor(x / 48 * 3));
          const row = Math.min(2, Math.floor(y / 48 * 3));
          const bin = bins[row * 3 + col];
          bin.samples += 1;
          const add = (cluster) => {
            cluster.mass += 1;
            cluster.x += x / 48;
            cluster.y += y / 48;
          };
          if (r > 105 && r > g * 1.35 && r > b * 1.35) { bin.hostile += 1; add(clusters.hostile); }
          if (g > 90 && g > r * 1.3 && g > b * 1.15) { bin.hazard += 1; add(clusters.hazard); }
          if (r > 135 && g > 110 && b < 95) { bin.loot += 1; add(clusters.loot); }
        }
      }
      latestVision = [];
      bins.forEach((bin) => { latestVision.push(bin.hostile / bin.samples, bin.hazard / bin.samples, bin.loot / bin.samples); });
      const featuresFor = (cluster) => {
        if (!cluster.mass) return [0.5, 0.5, 0];
        return [cluster.x / cluster.mass, cluster.y / cluster.mass, clip(cluster.mass / 40, 0, 1)];
      };
      const hostile = featuresFor(clusters.hostile);
      latestObjectFeatures = hostile.concat(featuresFor(clusters.hazard), featuresFor(clusters.loot), [
        clip((hostile[2] - previousHostileMass) * 4 + 0.5, 0, 1),
      ]);
      previousHostileMass = hostile[2];
      visionHistory.shift();
      visionHistory.push(latestVision);
    } catch (_) {
      // If a future game asset taints the canvas, the learner gracefully falls
      // back to HUD-only features rather than stopping the autoplay loop.
    }
    return visionHistory.reduce((all, frame) => all.concat(frame), []);
  }

  function observe() {
    const hpNode = document.querySelector('#hpText');
    const hpLabel = hpNode && hpNode.textContent ? hpNode.textContent.trim() : '';
    const hpMatch = hpLabel.match(/(\d+)\s*\/\s*(\d+)/);
    const hp = hpMatch ? Number(hpMatch[1]) : null;
    const maxHp = hpMatch ? Math.max(1, Number(hpMatch[2])) : null;
    const hudNode = document.querySelector('#hud');
    const hudText = hudNode ? hudNode.innerText || '' : '';
    const downMatch = hudText.match(/DOWN\s*([\d,]+)/i) || [];
    const downs = Number(downMatch[1] ? downMatch[1].replace(/,/g, '') : 0) || 0;
    const level = Number((hudText.match(/LV\s*(\d+)/i) || [])[1]) || 0;
    const timeNode = document.querySelector('#timeText');
    const time = parseClock(timeNode ? timeNode.textContent : '');
    const levelScreen = document.querySelector('#levelScreen');
    const crafting = visible(document.querySelector('#craftScreen')) || visible(document.querySelector('#craftChoiceScreen'));
    const notices = visible(document.querySelector('#noticeScreen'));
    const intro = visible(document.querySelector('#introScreen'));
    const over = visible(document.querySelector('#overScreen'));
    const selecting = visible(document.querySelector('#selectScreen'));
    const finale = visible(document.querySelector('#finaleMsg'));
    const hell = run.hellMode || (time > 1201 && !finale);
    const dash = document.querySelector('#dashBtn');
    const ultimate = document.querySelector('#ultBtn');
    return {
      hp,
      maxHp,
      health: hp !== null && maxHp ? Math.max(0, Math.min(1, hp / maxHp)) : null,
      time,
      downs,
      level,
      choosing: visible(levelScreen),
      crafting,
      notices,
      intro,
      over,
      selecting,
      finale,
      hell,
      dashReady: !!(dash && visible(dash) && dash.classList.contains('ready')),
      ultimateReady: !!(ultimate && visible(ultimate) && ultimate.classList.contains('ready')),
      vision: visualState(),
      objects: latestObjectFeatures.slice(),
      playing: visible(document.querySelector('#hud')) && !over && !intro && !crafting && !notices && !finale && !visible(levelScreen),
    };
  }

  function sendKey(key, down) {
    window.dispatchEvent(new KeyboardEvent(down ? 'keydown' : 'keyup', {
      key,
      code: key === ' ' ? 'Space' : `Key${key.toUpperCase()}`,
      bubbles: true,
      cancelable: true,
    }));
  }

  function releaseMovement() {
    for (const key of run.pressed) sendKey(key, false);
    run.pressed.clear();
  }

  function setMovement(vector) {
    const wanted = new Set();
    const [x, y] = vector;
    if (x < 0) wanted.add('a');
    if (x > 0) wanted.add('d');
    if (y < 0) wanted.add('w');
    if (y > 0) wanted.add('s');
    for (const key of run.pressed) if (!wanted.has(key)) sendKey(key, false);
    for (const key of wanted) if (!run.pressed.has(key)) sendKey(key, true);
    run.pressed = wanted;
  }

  function dash(vector) {
    const [x, y] = vector;
    const key = x < 0 ? 'a' : x > 0 ? 'd' : y < 0 ? 'w' : 's';
    sendKey(key, false);
    setTimeout(() => {
      sendKey(key, true);
      setTimeout(() => sendKey(key, false), 28);
      setTimeout(() => { if (run.pressed.has(key)) sendKey(key, true); }, 88);
    }, 30);
  }

  function manualKey(event) {
    if (event.code === 'Space') return ' ';
    const key = String(event.key || '').toLowerCase();
    return /^[wasd]$/.test(key) ? key : '';
  }

  function manualDashAction(key) {
    const id = key === 'd' ? 'dash-east' : key === 's' ? 'dash-south' : key === 'a' ? 'dash-west' : 'dash-north';
    return ACTIONS.findIndex((action) => action.id === id);
  }

  function captureManualBurst(actionIndex) {
    if (actionIndex < 0) return;
    manualDemo.burstActionIndex = actionIndex;
    manualDemo.burstUntil = Date.now() + 320;
    captureManualDemo();
  }

  function manualActionIndex() {
    if (manualDemo.burstActionIndex >= 0 && Date.now() < manualDemo.burstUntil) return manualDemo.burstActionIndex;
    if (manualDemo.pressed.has(' ')) return ACTIONS.findIndex((action) => action.ultimate);
    const x = (manualDemo.pressed.has('d') ? 1 : 0) - (manualDemo.pressed.has('a') ? 1 : 0);
    const y = (manualDemo.pressed.has('s') ? 1 : 0) - (manualDemo.pressed.has('w') ? 1 : 0);
    if (!x && !y) return -1;
    return ACTIONS.findIndex((action) => !action.dash && !action.ultimate && action.steps.length === 1
      && action.steps[0][0] === x && action.steps[0][1] === y);
  }

  function rememberDemonstration(experience) {
    if (!experience || !['movement', 'card'].includes(experience.kind) || !Array.isArray(experience.input)) return;
    const saved = { ...experience, isDemonstration: true, priority: Number(experience.priority) || 1 };
    model.demonstrations.push(saved);
    if (model.demonstrations.length > DEMONSTRATION_LIMIT) model.demonstrations.splice(0, model.demonstrations.length - DEMONSTRATION_LIMIT);
    model.demonstrationCount = model.demonstrations.length;
    if (saved.kind === 'movement') recordMasteryAction(saved);
  }

  function recordManualDemonstration(experience) {
    const transfer = {
      ...experience,
      transferId: experience.transferId || `${learnerId}:manual:${Date.now()}:${++manualDemo.sequence}`,
    };
    // Broadcast is fast when every tab is healthy; the per-worker outbox is
    // the durable path when a tab reloads or a broadcast is missed.
    queueTransfer('manual', transfer, MANUAL_OUTBOX_LIMIT);
    if (isCoordinator()) {
      drainDurableTransfers(true);
      return;
    }
    if (peerChannel) peerChannel.postMessage({ type: 'demonstration', from: learnerId, experience: transfer });
  }

  function settleManualDemoDecision(observation, terminal = false) {
    const decision = manualDemo.decision;
    if (!decision) return;
    const boundary = !terminal && decision.phase !== phaseFor(observation);
    const experience = {
      kind: 'movement',
      input: decision.input,
      actionIndex: decision.actionIndex,
      nextState: stateFeatures(observation),
      reward: movementReward(decision, observation, terminal),
      done: terminal || boundary,
      terminal,
      boundary,
      steps: 1,
      phase: decision.phase,
      isDemonstration: true,
    };
    recordManualDemonstration(experience);
    manualDemo.decision = null;
  }

  function captureManualDemo() {
    if (!manualDemo.active) return;
    const observation = observe();
    if (observation.over) {
      settleManualDemoDecision(observation, true);
      return;
    }
    if (!observation.playing) return;
    const actionIndex = manualActionIndex();
    const phase = phaseFor(observation);
    if (!manualDemo.decision && actionIndex >= 0) {
      manualDemo.decision = {
        input: stateFeatures(observation), actionIndex, time: observation.time,
        health: observation.health, downs: observation.downs, hell: observation.hell, phase,
      };
      return;
    }
    if (!manualDemo.decision) return;
    const changedAction = actionIndex >= 0 && actionIndex !== manualDemo.decision.actionIndex;
    if (changedAction || phase !== manualDemo.decision.phase || observation.time - manualDemo.decision.time >= DECISION_GAME_SECONDS) {
      settleManualDemoDecision(observation, false);
      if (actionIndex >= 0) {
        manualDemo.decision = {
          input: stateFeatures(observation), actionIndex, time: observation.time,
          health: observation.health, downs: observation.downs, hell: observation.hell, phase,
        };
      }
    }
  }

  function stopVideoRecording() {
    if (manualDemo.video && manualDemo.video.state !== 'inactive') manualDemo.video.stop();
  }

  function startVideoRecording() {
    const canvas = document.querySelector('canvas#game');
    if (!canvas || typeof canvas.captureStream !== 'function' || typeof MediaRecorder !== 'function') return false;
    try {
      manualDemo.stream = canvas.captureStream(MANUAL_VIDEO_FPS);
      const mimeType = MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported('video/webm;codecs=vp9')
        ? 'video/webm;codecs=vp9' : 'video/webm';
      manualDemo.chunks = [];
      try {
        manualDemo.video = new MediaRecorder(manualDemo.stream, { mimeType, videoBitsPerSecond: MANUAL_VIDEO_BITS_PER_SECOND });
      } catch (_) {
        manualDemo.video = new MediaRecorder(manualDemo.stream, { mimeType });
      }
      manualDemo.video.ondataavailable = (event) => { if (event.data && event.data.size) manualDemo.chunks.push(event.data); };
      manualDemo.video.onstop = () => {
        const chunks = manualDemo.chunks;
        const stream = manualDemo.stream;
        manualDemo.chunks = [];
        manualDemo.video = null;
        manualDemo.stream = null;
        if (stream) stream.getTracks().forEach((track) => track.stop());
        if (!chunks.length) return;
        const anchor = document.createElement('a');
        anchor.href = URL.createObjectURL(new Blob(chunks, { type: 'video/webm' }));
        anchor.download = `pine-autopilot-manual-demo-${Date.now()}.webm`;
        anchor.click();
        setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
        event('Saved local manual-demo video. Attach it to this chat when ready.');
      };
      manualDemo.video.start(1000);
      return true;
    } catch (_) {
      if (manualDemo.stream) manualDemo.stream.getTracks().forEach((track) => track.stop());
      manualDemo.stream = null;
      manualDemo.video = null;
      return false;
    }
  }

  function startManualDemo() {
    if (manualDemo.active) return;
    stop('Manual demonstration: autoplay paused; play the game yourself.');
    manualDemo.active = true;
    manualDemo.startedAt = Date.now();
    manualDemo.pressed.clear();
    manualDemo.decision = null;
    manualDemo.keyDownAt = {};
    manualDemo.burstActionIndex = -1;
    manualDemo.burstUntil = 0;
    const videoStarted = startVideoRecording();
    manualDemo.timer = setInterval(captureManualDemo, 120);
    run.status = videoStarted
      ? 'Recording manual demo and actions. Minimize the panel and play.'
      : 'Recording manual actions. Video capture is unavailable in this tab.';
    event(videoStarted ? 'Manual video/action demonstration started.' : 'Manual action demonstration started; video capture unavailable.');
    renderPanel(true);
  }

  function stopManualDemo(resume = true) {
    if (!manualDemo.active) return;
    const observation = observe();
    settleManualDemoDecision(observation, !!observation.over);
    if (manualDemo.timer) clearInterval(manualDemo.timer);
    manualDemo.timer = null;
    manualDemo.active = false;
    manualDemo.pressed.clear();
    manualDemo.burstActionIndex = -1;
    manualDemo.burstUntil = 0;
    stopVideoRecording();
    if (isCoordinator()) {
      saveModel(true);
      shareSnapshot(null, true);
    }
    event(`Manual demonstration ended with ${model.demonstrationCount} retained action samples.`);
    run.status = resume ? 'Manual demo saved locally; autoplay resumes shortly.' : 'Manual demo saved locally.';
    renderPanel(true);
    if (resume) setTimeout(arm, 750);
  }

  addEventListener('keydown', (event) => {
    if (!manualDemo.active) return;
    const key = manualKey(event);
    if (!key || event.repeat) return;
    const now = Date.now();
    manualDemo.pressed.add(key);
    if (key === ' ') captureManualBurst(ACTIONS.findIndex((action) => action.ultimate));
    else if (now - (manualDemo.keyDownAt[key] || 0) < 280) captureManualBurst(manualDashAction(key));
    else captureManualDemo();
    manualDemo.keyDownAt[key] = now;
  }, true);
  addEventListener('keyup', (event) => {
    if (!manualDemo.active) return;
    const key = manualKey(event);
    if (key) {
      manualDemo.pressed.delete(key);
      captureManualDemo();
    }
  }, true);

  addEventListener('pointerdown', (event) => {
    if (!manualDemo.active || !event.target || !event.target.closest) return;
    const selected = event.target.closest('#upRow .up-card');
    if (!selected || !visible(selected)) return;
    const observation = observe();
    const cards = [...document.querySelectorAll('#upRow .up-card')].filter(visible);
    if (!cards.length) return;
    const slate = cardSlateFeatures(cards);
    cards.forEach((card) => {
      const chosen = card === selected;
      recordManualDemonstration({
        kind: 'card',
        input: cardInput(observation, card.innerText || '', slate),
        reward: chosen ? 5 : -1.25,
        done: true,
        steps: 1,
        phase: phaseFor(observation),
        isDemonstration: true,
        manualCardChoice: chosen,
      });
    });
  }, true);

  function clip(value, low, high) { return Math.max(low, Math.min(high, value)); }

  function visualSummary(vision) {
    const frameSize = 27;
    const current = vision.slice(-frameSize);
    const previous = vision.slice(-frameSize * 2, -frameSize);
    const density = (index, channel) => current[index * 3 + channel] || 0;
    const previousDensity = (index, channel) => previous[index * 3 + channel] || 0;
    const hostile = Array.from({ length: 9 }, (_, index) => density(index, 0));
    const hazard = Array.from({ length: 9 }, (_, index) => density(index, 1));
    const hostileDelta = hostile.map((_, index) => density(index, 0) - previousDensity(index, 0));
    const north = hostile[0] + hostile[1] + hostile[2];
    const south = hostile[6] + hostile[7] + hostile[8];
    const west = hostile[0] + hostile[3] + hostile[6];
    const east = hostile[2] + hostile[5] + hostile[8];
    return [
      clip(hostile[4] * 12, 0, 1),
      clip((north - south) * 7 + 0.5, 0, 1),
      clip((east - west) * 7 + 0.5, 0, 1),
      clip(hazard.reduce((sum, item) => sum + item, 0) * 2, 0, 1),
      clip(hostileDelta[4] * 14 + 0.5, 0, 1),
      clip(hostileDelta.reduce((sum, item) => sum + item, 0) * 5 + 0.5, 0, 1),
    ];
  }

  function stateFeatures(observation) {
    const health = observation.health === null ? 0.5 : observation.health;
    const time = clip(observation.time / 1200, 0, 1);
    const downsPerMinute = observation.time ? clip((observation.downs / observation.time) * 60 / 80, 0, 1) : 0;
    const level = clip(observation.level / 60, 0, 1);
    const vision = observation.vision || Array(VISION_FEATURES).fill(0);
    const objects = Array.isArray(observation.objects) && observation.objects.length === OBJECT_FEATURES
      ? observation.objects : latestObjectFeatures;
    const phase = phaseFor(observation);
    const phaseFlags = ['early', 'mid', 'late', 'hell'].map((name) => phase === name ? 1 : 0);
    const summary = visualSummary(vision);
    return [
      health, time, downsPerMinute, level,
      health < 0.55 ? 1 : 0, health < 0.28 ? 1 : 0, observation.hell ? 1 : 0, 1,
      ...phaseFlags,
      observation.dashReady ? 1 : 0,
      observation.ultimateReady ? 1 : 0,
      ...summary,
      ...objects,
    ].concat(vision);
  }

  function movementInput(observation, patternIndex) {
    return stateFeatures(observation);
  }

  function riskKey(input, phase, actionIndex) {
    const health = clip(Number(input && input[0]) || 0.5, 0, 1);
    const centreThreat = clip(Number(input && input[14]) || 0, 0, 1);
    const hazard = clip(Number(input && input[17]) || 0, 0, 1);
    const healthBin = Math.min(4, Math.floor(health * 5));
    const pressureBin = Math.min(3, Math.floor((centreThreat * 0.65 + hazard * 0.35) * 4));
    return `${phase || 'early'}:${healthBin}:${pressureBin}:${actionIndex}`;
  }

  function updateRiskEstimate(experience) {
    if (!experience || experience.kind !== 'movement' || !Array.isArray(experience.input)) return;
    const key = riskKey(experience.input, phaseFor(experience), experience.actionIndex);
    const previous = model.riskStats[key] || { visits: 0, deaths: 0, at: 0 };
    previous.visits = Math.min(100000, previous.visits + 1);
    // Phase-boundary transitions stop DQN bootstrapping but are not deaths.
    previous.deaths = Math.min(previous.visits, previous.deaths + (experience.terminal ? 1 : 0));
    previous.at = Date.now();
    model.riskStats[key] = previous;
    model.riskUpdates += 1;
    if (Object.keys(model.riskStats).length > RISK_STATS_LIMIT * 1.08) model.riskStats = normalizeRiskStats(model.riskStats);
  }

  function riskPenaltyFor(input, phase, actionIndex) {
    const stat = model.riskStats[riskKey(input, phase, actionIndex)];
    if (!stat || stat.visits < RISK_MIN_SAMPLES) return 0;
    // Beta(1, 3) prior and an uncertainty term make this a conservative
    // probability of near-term death, not an overconfident action blacklist.
    const mean = (stat.deaths + 1) / (stat.visits + 4);
    const upper = clip(mean + Math.sqrt(Math.max(0, mean * (1 - mean) / (stat.visits + 4))) * 1.35, 0, 0.9);
    const health = clip(Number(input[0]) || 0.5, 0, 1);
    const urgency = 0.35 + (1 - health) * 0.9 + (phase === 'hell' ? 0.18 : phase === 'late' ? 0.10 : 0);
    return upper * urgency * 0.48;
  }

  function masteryKey(input, phase, includeCooldowns = true) {
    const value = (index) => clip(Number(input && input[index]) || 0.5, 0, 1);
    const health = Math.min(4, Math.floor(value(0) * 5));
    const centreThreat = Math.min(3, Math.floor(value(14) * 4));
    const hazard = Math.min(3, Math.floor(value(17) * 4));
    const vertical = value(15) < 0.42 ? 'n' : value(15) > 0.58 ? 's' : 'c';
    const horizontal = value(16) < 0.42 ? 'w' : value(16) > 0.58 ? 'e' : 'c';
    const cooldowns = includeCooldowns ? `:${value(12) > 0.5 ? 1 : 0}${value(13) > 0.5 ? 1 : 0}` : '';
    return `${phase || 'early'}:${health}:${centreThreat}:${hazard}:${vertical}${horizontal}${cooldowns}`;
  }

  function normalizeMasteryPolicy(value) {
    if (!value || typeof value !== 'object') return {};
    return Object.entries(value)
      .map(([key, stat]) => [key, {
        visits: Math.max(0, Math.min(100000, Number(stat && stat.visits) || 0)),
        actions: Array.from({ length: ACTIONS.length }, (_, index) => Math.max(0, Math.min(100000, Number(stat && stat.actions && stat.actions[index]) || 0))),
        at: Math.max(0, Number(stat && stat.at) || 0),
      }])
      .filter(([key, stat]) => key.length < 60 && stat.visits > 0)
      .sort((left, right) => right[1].at - left[1].at)
      .slice(0, MASTERY_STATS_LIMIT)
      .reduce((all, [key, stat]) => ({ ...all, [key]: stat }), {});
  }

  function updateMasteryKey(key, actionIndex) {
    if (!key || actionIndex < 0 || actionIndex >= ACTIONS.length) return;
    const stat = model.masteryPolicy[key] || { visits: 0, actions: Array(ACTIONS.length).fill(0), at: 0 };
    stat.visits = Math.min(100000, stat.visits + 1);
    stat.actions[actionIndex] = Math.min(stat.visits, (stat.actions[actionIndex] || 0) + 1);
    stat.at = Date.now();
    model.masteryPolicy[key] = stat;
  }

  function recordMasteryAction(experience) {
    if (!experience || experience.kind !== 'movement' || !Array.isArray(experience.input) || !Number.isInteger(experience.actionIndex)) return;
    const phase = phaseFor(experience);
    updateMasteryKey(masteryKey(experience.input, phase, true), experience.actionIndex);
    updateMasteryKey(masteryKey(experience.input, phase, false), experience.actionIndex);
    model.masteryUpdates += 1;
    if (Object.keys(model.masteryPolicy).length > MASTERY_STATS_LIMIT * 1.08) model.masteryPolicy = normalizeMasteryPolicy(model.masteryPolicy);
  }

  function rebuildMasteryFromDemonstrations() {
    model.masteryPolicy = {};
    model.masteryUpdates = 0;
    model.demonstrations.forEach((experience) => {
      if (experience && experience.kind === 'movement') recordMasteryAction(experience);
    });
  }

  function masteryActionPrior(input, phase) {
    if (!model.settings.immortalMode) return { bonuses: Array(ACTIONS.length).fill(0), confidence: 0 };
    const exact = model.masteryPolicy[masteryKey(input, phase, true)];
    const broad = model.masteryPolicy[masteryKey(input, phase, false)];
    const stat = exact && exact.visits >= MASTERY_MIN_VISITS ? exact : broad;
    if (!stat || stat.visits < MASTERY_MIN_VISITS) return { bonuses: Array(ACTIONS.length).fill(0), confidence: 0 };
    const counts = stat.actions || [];
    const winner = Math.max(...counts, 0);
    const purity = winner / Math.max(1, stat.visits);
    const support = clip(Math.log2(stat.visits + 1) / 4, 0, 1);
    const confidence = support * purity;
    if (confidence < 0.28) return { bonuses: Array(ACTIONS.length).fill(0), confidence: 0 };
    const peak = 0.12 + MASTERY_MAX_BONUS * confidence;
    return {
      confidence,
      bonuses: counts.map((count) => count ? peak * (count / winner) : 0),
    };
  }

  function cardSlateFeatures(cards) {
    const labels = cards.map((card) => String(card.innerText || '').toUpperCase());
    const fraction = (pattern) => labels.length ? labels.filter((label) => pattern.test(label)).length / labels.length : 0;
    return [
      fraction(/EVOLVE|SUPER|RAINBOW/),
      fraction(/ATTACK|POWER|RAPID|DAMAGE|FIRE/),
      fraction(/MAX HP|HEALTH|DEFEN[CS]E|REGEN/),
      fraction(/MOVE|SPEED|DASH|MINT/),
      fraction(/GOLD|CASH|MONEY|LUCK|SUGAR/),
      clip(labels.length / 5, 0, 1),
    ];
  }

  function cardInput(observation, text, slate) {
    const label = String(text).toUpperCase();
    const isNew = /NEW\s*[·.]?\s*LV\s*1/.test(label) ? 1 : 0;
    return stateFeatures(observation).concat([
      /EVOLVE|SUPER|RAINBOW/.test(label) ? 1 : 0,
      /WHISKY SOUR|MOJITO|NEGRONI|MARTINI|MANHATTAN|OLD FASHIONED|ATTACK|POWER|RAPID|DAMAGE|FIRE/.test(label) ? 1 : 0,
      /MAX HP|HEALTH|DEFEN[CS]E|REGEN/.test(label) ? 1 : 0,
      /MOVE|SPEED|DASH|MINT/.test(label) ? 1 : 0,
      isNew,
      /GOLD|CASH|MONEY|LUCK|SUGAR/.test(label) ? 1 : 0,
      /BASE/.test(label) ? 1 : 0,
      1,
    ]).concat(slate || Array(6).fill(0));
  }

  function networkForward(network, input) {
    const pre = Array(network.hidden).fill(0);
    const hidden = Array(network.hidden).fill(0);
    for (let h = 0; h < network.hidden; h += 1) {
      let sum = network.b1[h];
      for (let i = 0; i < network.inputs; i += 1) sum += network.w1[h * network.inputs + i] * input[i];
      pre[h] = sum;
      hidden[h] = Math.max(0, sum);
    }
    let value = network.bValue;
    for (let h = 0; h < network.hidden; h += 1) value += network.wValue[h] * hidden[h];
    const advantage = network.bAdvantage.slice();
    for (let out = 0; out < network.outputs; out += 1) {
      for (let h = 0; h < network.hidden; h += 1) advantage[out] += network.wAdvantage[out * network.hidden + h] * hidden[h];
    }
    const meanAdvantage = advantage.reduce((sum, item) => sum + item, 0) / network.outputs;
    const output = advantage.map((item) => value + item - meanAdvantage);
    return { pre, hidden, output, advantage };
  }

  function trainNetwork(network, input, target, outputIndex, importanceWeight) {
    const { pre, hidden, output } = networkForward(network, input);
    const error = clip(output[outputIndex] - target, -3, 3); // Huber-style gradient clip keeps online learning stable.
    const oldWValue = network.wValue.slice();
    const oldWAdvantage = network.wAdvantage.slice();
    const rate = LEARNING_RATE * (importanceWeight || 1);
    for (let h = 0; h < network.hidden; h += 1) network.wValue[h] -= rate * error * hidden[h];
    network.bValue -= rate * error;
    for (let out = 0; out < network.outputs; out += 1) {
      const advantageGradient = error * (out === outputIndex ? 1 - 1 / network.outputs : -1 / network.outputs);
      const offset = out * network.hidden;
      for (let h = 0; h < network.hidden; h += 1) network.wAdvantage[offset + h] -= rate * advantageGradient * hidden[h];
      network.bAdvantage[out] -= rate * advantageGradient;
    }
    for (let h = 0; h < network.hidden; h += 1) {
      let outputGradient = error * oldWValue[h];
      for (let out = 0; out < network.outputs; out += 1) {
        const advantageGradient = error * (out === outputIndex ? 1 - 1 / network.outputs : -1 / network.outputs);
        outputGradient += advantageGradient * oldWAdvantage[out * network.hidden + h];
      }
      const gradient = pre[h] > 0 ? outputGradient : 0;
      network.b1[h] -= rate * gradient;
      for (let i = 0; i < network.inputs; i += 1) network.w1[h * network.inputs + i] -= rate * gradient * input[i];
    }
    network.samples += 1;
    return Math.abs(error);
  }

  function argMax(values) {
    let index = 0;
    for (let candidate = 1; candidate < values.length; candidate += 1) if (values[candidate] > values[index]) index = candidate;
    return index;
  }

  function targetFor(experience, head) {
    if (experience.kind !== 'movement' || experience.done || !Array.isArray(experience.nextState)) return experience.reward;
    // Double DQN: the online head chooses the next action; the frozen target
    // head evaluates it. This cuts the optimistic-value bias of vanilla DQN.
    const bestAction = argMax(networkForward(head.movementNet, experience.nextState).output);
    const future = networkForward(head.movementTarget, experience.nextState).output[bestAction];
    return experience.reward + Math.pow(DISCOUNT, experience.steps || 1) * (Number.isFinite(future) ? future : 0);
  }

  function trainOne(experience) {
    if (!experience) return null;
    const head = headFor(experience);
    const network = experience.kind === 'card' ? head.cardNet : head.movementNet;
    if (!Array.isArray(experience.input) || experience.input.length !== network.inputs || !Number.isFinite(experience.reward)) return null;
    const outputIndex = experience.kind === 'card' ? 0 : experience.actionIndex;
    if (!Number.isInteger(outputIndex) || outputIndex < 0 || outputIndex >= network.outputs) return null;
    const samplesBefore = network.samples || 0;
    const priority = trainNetwork(network, experience.input, targetFor(experience, head), outputIndex, experience.importanceWeight);
    if (experience.kind === 'movement' && (experience.isEliteSample || experience.isDemonstration)) {
      // Advantage-weighted imitation: retain the chosen action from a proven
      // survivor without discarding the Bellman target from real game data.
      const output = networkForward(network, experience.input).output;
      const imitationTarget = Math.max(...output) + ELITE_IMITATION_MARGIN;
      trainNetwork(network, experience.input, imitationTarget, outputIndex,
        (experience.isDemonstration ? 1.25 : 1) * ELITE_IMITATION_WEIGHT);
      model.eliteImitationSteps += 1;
    }
    experience.priority = priority + 0.01;
    refreshReplayPriority(experience);
    if (Math.floor(network.samples / TARGET_SYNC) > Math.floor(samplesBefore / TARGET_SYNC)) {
      if (experience.kind === 'card') head.cardTarget = copyNetwork(head.cardNet);
      else head.movementTarget = copyNetwork(head.movementNet);
    }
    return priority;
  }

  function treeSet(index, value) {
    let node = index + REPLAY_LIMIT;
    priorityTree[node] = value;
    while (node > 1) {
      node = Math.floor(node / 2);
      priorityTree[node] = priorityTree[node * 2] + priorityTree[node * 2 + 1];
    }
  }

  function treeFind(value) {
    let node = 1;
    while (node < REPLAY_LIMIT) {
      const left = node * 2;
      if (value <= priorityTree[left]) node = left;
      else {
        value -= priorityTree[left];
        node = left + 1;
      }
    }
    return node - REPLAY_LIMIT;
  }

  function priorityWeight(priority) {
    return Math.pow(Math.max(0.001, Number(priority) || 1), PRIORITY_ALPHA);
  }

  function rebuildReplayTree() {
    priorityTree.fill(0);
    replayPeakPriority = 1;
    Object.keys(replayPhaseCounts).forEach((phase) => { replayPhaseCounts[phase] = 0; });
    model.replay.forEach((experience, index) => {
      experience.replayIndex = index;
      experience.priority = Number(experience.priority) || 1;
      replayPeakPriority = Math.max(replayPeakPriority, experience.priority);
      replayPhaseCounts[phaseFor(experience)] += 1;
      priorityTree[REPLAY_LIMIT + index] = priorityWeight(experience.priority);
    });
    for (let node = REPLAY_LIMIT - 1; node > 0; node -= 1) priorityTree[node] = priorityTree[node * 2] + priorityTree[node * 2 + 1];
  }

  function refreshReplayPriority(experience) {
    const index = experience && experience.replayIndex;
    if (!Number.isInteger(index) || model.replay[index] !== experience) return;
    replayPeakPriority = Math.max(replayPeakPriority, experience.priority || 1);
    treeSet(index, priorityWeight(experience.priority));
  }

  function remember(experience) {
    const index = model.replay.length < REPLAY_LIMIT ? model.replay.length : model.replayCursor % REPLAY_LIMIT;
    const replaced = model.replay[index];
    if (replaced) replayPhaseCounts[phaseFor(replaced)] = Math.max(0, replayPhaseCounts[phaseFor(replaced)] - 1);
    experience.priority = Number(experience.priority) || replayPeakPriority;
    experience.replayIndex = index;
    model.replay[index] = experience;
    replayPhaseCounts[phaseFor(experience)] += 1;
    model.replayCursor = (index + 1) % REPLAY_LIMIT;
    replayPeakPriority = Math.max(replayPeakPriority, experience.priority);
    treeSet(index, priorityWeight(experience.priority));
  }

  function preferredReplayPhase() {
    const targets = { early: 0.30, mid: 0.30, late: 0.23, hell: 0.17 };
    const total = Object.values(replayPhaseCounts).reduce((sum, count) => sum + count, 0) || 1;
    const available = Object.keys(targets).filter((phase) => replayPhaseCounts[phase] > 0);
    if (!available.length) return 'early';
    const deficits = available.map((phase) => Math.max(0.02, targets[phase] - replayPhaseCounts[phase] / total));
    const sum = deficits.reduce((value, item) => value + item, 0);
    let draw = Math.random() * sum;
    for (let index = 0; index < available.length; index += 1) {
      draw -= deficits[index];
      if (draw <= 0) return available[index];
    }
    return available[available.length - 1];
  }

  function sampleEliteForPhase(phase) {
    if (!model.elite.length) return null;
    for (let attempt = 0; attempt < 18; attempt += 1) {
      const elite = model.elite[Math.floor(Math.random() * model.elite.length)];
      if (phaseFor(elite) === phase) return elite;
    }
    return model.elite[Math.floor(Math.random() * model.elite.length)];
  }

  function samplePrioritizedReplay() {
    const preferredPhase = preferredReplayPhase();
    if (model.demonstrations.length && Math.random() < DEMONSTRATION_REPLAY_RATE) {
      for (let attempt = 0; attempt < 18; attempt += 1) {
        const demo = model.demonstrations[Math.floor(Math.random() * model.demonstrations.length)];
        if (demo && phaseFor(demo) === preferredPhase) return { ...demo, importanceWeight: 1.25, isDemonstration: true };
      }
      const demo = model.demonstrations[Math.floor(Math.random() * model.demonstrations.length)];
      if (demo) return { ...demo, importanceWeight: 1.25, isDemonstration: true };
    }
    if (model.elite.length && Math.random() < 0.45) {
      const elite = sampleEliteForPhase(preferredPhase);
      if (elite) {
        return { ...elite, importanceWeight: 1, isEliteSample: true };
      }
    }
    if (!model.replay.length || !priorityTree[1]) return null;
    const total = priorityTree[1];
    let index = 0;
    let experience = null;
    // Sum-tree draws retain prioritized replay, while rejection sampling gives
    // scarce mid/late/Hell experience a deliberate curriculum share.
    for (let attempt = 0; attempt < 18; attempt += 1) {
      index = treeFind(Math.max(0.000001, Math.random() * total));
      experience = model.replay[index];
      if (experience && phaseFor(experience) === preferredPhase) break;
    }
    if (!experience || phaseFor(experience) !== preferredPhase) {
      for (let attempt = 0; attempt < 48; attempt += 1) {
        const fallback = model.replay[Math.floor(Math.random() * model.replay.length)];
        if (fallback && phaseFor(fallback) === preferredPhase) {
          experience = fallback;
          index = experience.replayIndex;
          break;
        }
      }
    }
    if (!experience) return model.replay[Math.floor(Math.random() * model.replay.length)];
    const probability = priorityTree[REPLAY_LIMIT + index] / total;
    const beta = Math.min(1, PRIORITY_BETA_START + model.uniqueExperiences / 500000);
    experience.importanceWeight = clip(Math.pow(model.replay.length * probability, -beta), 0.20, 1);
    return experience;
  }

  function trainExperience(experience, fromPeer = false) {
    if (!experience || !Array.isArray(experience.input) || !Number.isFinite(experience.reward)) return;
    const startedAt = performance.now();
    updateRiskEstimate(experience);
    remember(experience);
    if (trainOne(experience) === null) return;
    for (let i = 0; i < adaptiveReplayBatch && model.replay.length > 1; i += 1) {
      trainOne(samplePrioritizedReplay());
    }
    const trainingMs = performance.now() - startedAt;
    if (trainingMs > 8) adaptiveReplayBatch = Math.max(MIN_REPLAY_BATCH, adaptiveReplayBatch - 1);
    else if (trainingMs < 3) adaptiveReplayBatch = Math.min(MAX_REPLAY_BATCH, adaptiveReplayBatch + 1);
    const head = headFor(experience);
    head.uniqueExperiences = (head.uniqueExperiences || 0) + 1;
    model.uniqueExperiences += 1;
    if (fromPeer) model.peerExperiences += 1;
    else {
      model.localExperiences += 1;
      shareExperience(experience);
    }
    maybeShareSnapshot();
    if (experienceCount() % 100 < adaptiveReplayBatch + 1) saveModel();
  }

  let consolidationQueued = false;

  function scheduleConsolidation(reason) {
    if (!isCoordinator() || consolidationQueued || model.replay.length < MIN_REPLAY_BATCH) return;
    consolidationQueued = true;
    const consolidate = () => {
      consolidationQueued = false;
      // Never let a training burst contend with an active movement decision.
      if (run.decision || !isCoordinator()) return;
      const startedAt = performance.now();
      let steps = 0;
      while (steps < IDLE_CONSOLIDATION_MAX_STEPS && performance.now() - startedAt < IDLE_CONSOLIDATION_BUDGET_MS) {
        const sample = samplePrioritizedReplay();
        if (!sample || trainOne(sample) === null) break;
        steps += 1;
      }
      if (!steps) return;
      model.consolidationSteps += steps;
      model.consolidationPasses += 1;
      if (model.consolidationPasses % 5 === 0) event(`Idle replay consolidation: ${steps} updates after ${reason}.`);
      saveModel();
    };
    if (typeof requestIdleCallback === 'function') requestIdleCallback(consolidate, { timeout: 180 });
    else setTimeout(consolidate, 90);
  }

  function recordExperience(experience) {
    // Exactly one elected tab trains. The other windows continuously collect
    // independent game traces and stream them to it, avoiding five divergent
    // replay schedules and keeping CPU available for the 100× game windows.
    if (isCoordinator()) {
      trainExperience(experience, false);
      return;
    }
    model.localExperiences += 1;
    shareExperience(experience);
  }

  function explorationRate(head) {
    // Decay only with actual cross-window experience, never replay gradient
    // steps. This preserves useful exploration throughout early training.
    return Math.max(0.025, 0.28 * Math.exp(-(head.uniqueExperiences || 0) / 72000));
  }

  function actingExplorationRate(head) {
    if (run.evaluation) return 0;
    // Each parallel worker has a stable, slightly different exploration rate
    // and decision rhythm, making its game traces less correlated with peers.
    return Math.min(0.42, explorationRate(head) * workerExplorationScale);
  }

  function currentThreatBins(observation) {
    const current = (observation.vision || []).slice(-27);
    return Array.from({ length: 9 }, (_, index) => {
      const hostile = current[index * 3] || 0;
      const hazard = current[index * 3 + 1] || 0;
      return hostile + hazard * 1.25;
    });
  }

  function patternThreat(observation, pattern) {
    const [x, y] = pattern.steps[0] || [0, 0];
    const row = y < 0 ? 0 : y > 0 ? 2 : 1;
    const col = x < 0 ? 0 : x > 0 ? 2 : 1;
    const bins = currentThreatBins(observation);
    const target = bins[row * 3 + col] || 0;
    const centre = bins[4] || 0;
    return target * 1.45 + centre * 0.30;
  }

  function tacticalIntent(observation) {
    const health = observation.health === null ? 0.5 : observation.health;
    const threat = currentThreatBins(observation);
    const pressure = threat.reduce((sum, item) => sum + item, 0);
    if (health < 0.35 || threat[4] > 0.12) return 'escape';
    if (observation.ultimateReady && pressure > 0.35) return 'clear';
    // Object features contain the yellow-cluster position/mass. This only
    // nudges exploration; the value network still makes the final decision.
    if ((observation.objects || [])[8] > 0.16 && health > 0.55) return 'collect';
    return 'survive';
  }

  function intentBonus(intent, pattern, observation) {
    if (intent === 'escape') return (/orbit|perimeter/.test(pattern.id) ? 0.035 : 0) + (pattern.dash && observation.dashReady ? 0.025 : 0);
    if (intent === 'clear') return pattern.ultimate && observation.ultimateReady ? 0.045 : 0;
    if (intent === 'collect' && !pattern.ultimate) return 0.012;
    return 0;
  }

  function safetyPenalty(observation, pattern) {
    const health = observation.health === null ? 0.5 : observation.health;
    if (health >= 0.62) return 0;
    const vulnerability = clip((0.62 - health) / 0.62, 0, 1);
    // This is a soft shield, rather than a hard scripted override: it keeps
    // exploration possible but strongly down-ranks moving into visible danger.
    return vulnerability * patternThreat(observation, pattern) * 0.24;
  }

  function selectPattern(observation) {
    const head = actingHeadFor(observation);
    const intent = tacticalIntent(observation);
    run.intent = intent;
    const input = movementInput(observation);
    const phase = phaseFor(observation);
    const mastery = masteryActionPrior(input, phase);
    // Where the player's demonstrated policy has repeated the same safe
    // choice, follow it instead of injecting random exploration into it.
    if (mastery.confidence < 0.28 && Math.random() < actingExplorationRate(head)) {
      const candidates = ACTIONS.filter((pattern, index) => safetyPenalty(observation, pattern) < 0.04
        && riskPenaltyFor(input, phase, index) < 0.12);
      return (candidates.length ? candidates : ACTIONS)[Math.floor(Math.random() * (candidates.length || ACTIONS.length))];
    }
    const values = networkForward(head.movementNet, input).output;
    const targetValues = networkForward(head.movementTarget, input).output;
    let choice = null;
    ACTIONS.forEach((pattern, index) => {
      const prediction = values[index];
      const defensiveNudge = observation.health !== null && observation.health < 0.45 && /orbit|perimeter/.test(pattern.id) ? 0.03 : 0;
      const styleNudge = run.evaluation ? 0 : workerStyle === 0 && /orbit|perimeter/.test(pattern.id) ? 0.025
        : workerStyle === 1 && /dash|horizontal|vertical|weave/.test(pattern.id) ? 0.025
          : workerStyle === 2 && pattern.ultimate && observation.ultimateReady ? 0.025
            : workerStyle === 3 && safetyPenalty(observation, pattern) < 0.01 ? 0.018 : 0;
      // Online-versus-frozen-target disagreement is a cheap uncertainty proxy.
      // It is limited to a small exploration bonus and disabled for evaluation.
      const uncertaintyBonus = run.evaluation ? 0 : Math.min(0.055, Math.abs(prediction - targetValues[index]) * 0.025);
      const learnedRiskPenalty = riskPenaltyFor(input, phase, index);
      const masteryBonus = mastery.bonuses[index] || 0;
      const score = prediction + defensiveNudge + styleNudge + intentBonus(intent, pattern, observation) + uncertaintyBonus + masteryBonus - safetyPenalty(observation, pattern) - learnedRiskPenalty;
      if (!choice || score > choice.score) choice = { pattern, score, masteryBonus };
    });
    if (choice.masteryBonus > 0.08) model.masteryActionUses += 1;
    return choice.pattern;
  }

  function cardKey(text) {
    return String(text).replace(/NEW\s*[·.]?\s*LV\s*1/gi, '').replace(/LV\s*\d+\s*[→>-]\s*\d+\s*\/\s*\d+/gi, '').replace(/\s+/g, ' ').trim().toUpperCase().slice(0, 100);
  }

  function cardWarmStart(text, observation) {
    const label = String(text).toUpperCase();
    let value = 0;
    if (/EVOLVE|SUPER|RAINBOW/.test(label)) value += 0.8;
    if (/MAX HP|HEALTH|DEFEN[CS]E|REGEN/.test(label) && observation.health !== null && observation.health < 0.55) value += 0.45;
    if (/ATTACK|POWER|RAPID|DAMAGE|FIRE/.test(label)) value += 0.2;
    if (/GOLD|CASH|MONEY|LUCK|SUGAR/.test(label)) value -= 0.08;
    return value;
  }

  function chooseCard(observation) {
    const cards = [...document.querySelectorAll('#upRow .up-card')].filter(visible);
    if (!cards.length || Date.now() - run.lastFlowAt < FLOW_GAP_MS) return false;
    const head = actingHeadFor(observation);
    const exploring = Math.random() < actingExplorationRate(head);
    const slate = cardSlateFeatures(cards);
    let choice = null;
    cards.forEach((element, index) => {
      const text = element.innerText || '';
      const prediction = networkForward(head.cardNet, cardInput(observation, text, slate)).output[0];
      const score = prediction + (head.cardNet.samples < 80 ? cardWarmStart(text, observation) : 0);
      if (!choice || score > choice.score || (exploring && Math.random() < 1 / (index + 1))) choice = { element, text, score };
    });
    if (!choice) return false;
    click(choice.element, `Card: ${cardKey(choice.text).slice(0, 34)}`);
    run.lastCard = {
      input: cardInput(observation, choice.text, slate),
      chosenAtGameTime: observation.time,
      health: observation.health,
      downs: observation.downs,
      hell: observation.hell,
      phase: phaseFor(observation),
      timingWarningsAtStart: run.timingWarnings,
    };
    run.lastCardAt = Date.now();
    run.lastCardGameTime = observation.time;
    return true;
  }

  function movementReward(start, observation, terminal) {
    const elapsed = Math.max(0, observation.time - start.time);
    const hpDelta = start.health !== null && observation.health !== null ? observation.health - start.health : 0;
    const downs = Math.max(0, observation.downs - start.downs);
    const phase = start.phase || phaseFor(start);
    const phaseSurvivalWeight = phase === 'hell' ? 0.36 : phase === 'late' ? 0.30 : phase === 'mid' ? 0.25 : 0.20;
    const deathPenalty = phase === 'hell' ? 15 : phase === 'late' ? 13 : phase === 'mid' ? 12 : 11;
    const milestone = (start.time < 300 && observation.time >= 300 ? 2.5 : 0)
      + (start.time < 900 && observation.time >= 900 ? 4.5 : 0)
      + (start.phase !== 'hell' && observation.hell ? 6 : 0);
    // Survival, phase progression, and avoiding death dominate incidental
    // combat score. This aligns optimization with the user's survival goal.
    return elapsed * phaseSurvivalWeight + downs * 0.012 + hpDelta * 7 + milestone - (terminal ? deathPenalty : 0);
  }

  function emitNstepTransition() {
    const horizon = run.nStep.slice(0, N_STEP);
    if (!horizon.length) return;
    const first = horizon[0];
    const last = horizon[horizon.length - 1];
    let reward = 0;
    for (let index = 0; index < horizon.length; index += 1) reward += Math.pow(DISCOUNT, index) * horizon[index].reward;
    const experience = {
      ...first,
      reward,
      nextState: last.nextState,
      done: last.done,
      terminal: horizon.some((step) => step.terminal),
      boundary: !horizon.some((step) => step.terminal) && horizon.some((step) => step.boundary),
      steps: horizon.length,
    };
    if (!run.evaluation) {
      recordExperience(experience);
      run.episodeTransitions.push(experience);
    }
    run.nStep.shift();
  }

  function queueMovementStep(step) {
    run.nStep.push(step);
    if (step.done) {
      while (run.nStep.length) emitNstepTransition();
    } else if (run.nStep.length >= N_STEP) {
      emitNstepTransition();
    }
  }

  function settleDecision(observation, terminal = false, boundary = false) {
    const decision = run.decision;
    if (!decision || (!terminal && !boundary && observation.time <= decision.time)) return;
    if (run.timingWarnings > decision.timingWarningsAtStart) {
      // Do not bridge a bad sampling gap with n-step returns. Clearing the
      // pending sequence prevents a later clean decision from inheriting it.
      run.nStep = [];
      run.lowQualityDropped += 1;
      run.decision = null;
      return;
    }
    queueMovementStep({
      kind: 'movement',
      input: decision.input,
      actionIndex: decision.actionIndex,
      nextState: stateFeatures(observation),
      reward: movementReward(decision, observation, terminal),
      done: terminal || boundary,
      terminal,
      boundary,
      hell: decision.hell,
      phase: decision.phase,
    });
    run.decision = null;
  }

  function startDecision(observation) {
    run.pattern = selectPattern(observation);
    run.patternIndex = 0;
    run.lastDecisionGameTime = observation.time;
    run.lastMotionStepGameTime = observation.time;
    run.decision = {
      input: stateFeatures(observation),
      actionIndex: ACTIONS.findIndex((action) => action.id === run.pattern.id),
      time: observation.time,
      health: observation.health,
      downs: observation.downs,
      hell: observation.hell,
      phase: phaseFor(observation),
      timingWarningsAtStart: run.timingWarnings,
    };
  }

  function settleCard(observation, force = false, terminal = false) {
    const card = run.lastCard;
    if (!card || (!terminal && observation.time < card.chosenAtGameTime)) return;
    const elapsed = Math.max(0, observation.time - card.chosenAtGameTime);
    if (!force && elapsed < CARD_CREDIT_SECONDS) return;
    if (run.timingWarnings > card.timingWarningsAtStart) {
      run.lowQualityDropped += 1;
      run.lastCard = null;
      return;
    }
    const hpDelta = card.health !== null && observation.health !== null ? observation.health - card.health : 0;
    const downs = Math.max(0, observation.downs - card.downs);
    // Card effects often emerge over several upgrade intervals, so use a
    // longer capped survival window rather than near-instant attribution.
    const reward = Math.min(elapsed, CARD_CREDIT_SECONDS) * 0.025 + downs * 0.08 + hpDelta * 5 - (terminal ? 2.5 : 0);
    if (!run.evaluation) recordExperience({ kind: 'card', input: card.input, reward, done: true, hell: card.hell, phase: card.phase });
    run.lastCard = null;
  }

  function median(values) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((left, right) => left - right);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
  }

  function evaluationQuality(result) {
    const seconds = Math.max(1, Number(result && result.seconds) || 0);
    const timingWarnings = Math.max(0, Number(result && result.timingWarnings) || 0);
    // v6 records do not have a per-run dropped count. Their timing-warning
    // count is the conservative fallback; v7 uses actual dropped transitions.
    const droppedTransitions = Number.isFinite(result && result.droppedTransitions)
      ? Math.max(0, Number(result.droppedTransitions)) : timingWarnings;
    const decisionEstimate = Math.max(1, Math.ceil(seconds / EVALUATION_DECISION_ESTIMATE_SECONDS));
    const droppedRate = droppedTransitions / decisionEstimate;
    return {
      timingWarnings,
      droppedTransitions,
      decisionEstimate,
      droppedRate,
      quality: droppedRate <= MAX_EVALUATION_DROPPED_RATE,
    };
  }

  function normalizeEvaluationRecord(result) {
    const quality = evaluationQuality(result);
    const policy = result && (result.policy === 'champion' || result.policy === 'candidate') ? result.policy : 'legacy';
    return {
      ...result,
      seconds: Number(result && result.seconds) || 0,
      phase: (result && result.phase) || 'early',
      timingWarnings: quality.timingWarnings,
      droppedTransitions: quality.droppedTransitions,
      decisionEstimate: quality.decisionEstimate,
      droppedRate: quality.droppedRate,
      quality: quality.quality,
      policy,
      generation: Number.isInteger(result && result.generation) ? result.generation : -1,
      at: Number(result && result.at) || Date.now(),
    };
  }

  function cleanEvaluationResults() {
    return model.evaluationResults.filter((entry) => entry && Number.isFinite(entry.seconds) && evaluationQuality(entry).quality);
  }

  function tournamentEvaluationResults(policy) {
    return cleanEvaluationResults().filter((entry) => entry.policy === policy && entry.generation === model.tournamentGeneration);
  }

  function candidateEvaluationResults() {
    const current = tournamentEvaluationResults('candidate');
    return current.length ? current : cleanEvaluationResults().filter((entry) => entry.policy !== 'champion');
  }

  function tournamentScorecard() {
    const candidate = tournamentEvaluationResults('candidate').slice(-TOURNAMENT_WINDOW);
    const champion = tournamentEvaluationResults('champion').slice(-TOURNAMENT_WINDOW);
    if (candidate.length < TOURNAMENT_WINDOW || champion.length < TOURNAMENT_WINDOW) return null;
    const candidateSeconds = candidate.map((entry) => entry.seconds).sort((left, right) => left - right);
    const championSeconds = champion.map((entry) => entry.seconds).sort((left, right) => left - right);
    return {
      candidate: median(candidateSeconds),
      champion: median(championSeconds),
      candidateLower: candidateSeconds[Math.floor(candidateSeconds.length * 0.2)],
      championLower: championSeconds[Math.floor(championSeconds.length * 0.2)],
      candidateRuns: candidate.length,
      championRuns: champion.length,
      generation: model.tournamentGeneration,
    };
  }

  function evaluationTrend() {
    const values = candidateEvaluationResults().map((entry) => entry.seconds);
    if (values.length < 6) return null;
    const middle = Math.floor(values.length / 2);
    const early = median(values.slice(0, middle));
    const recent = median(values.slice(middle));
    return { early, recent, percent: early ? ((recent - early) / early) * 100 : 0 };
  }

  function championSnapshot(score) {
    return {
      score,
      at: Date.now(),
      earlyHead: cloneForPeer(headSnapshot(model)),
      midHead: cloneForPeer(headSnapshot(model.midHead)),
      lateHead: cloneForPeer(headSnapshot(model.lateHead)),
      hellHead: cloneForPeer(headSnapshot(model.hellHead)),
    };
  }

  function bootstrapChampionFromHistory() {
    if (model.champion) return false;
    const values = cleanEvaluationResults().map((entry) => entry.seconds);
    if (values.length < 6) return false;
    const score = median(values.slice(-Math.min(12, values.length)));
    model.champion = championSnapshot(score);
    event(`Bootstrapped champion from pre-tournament history: ${formatTime(score)}.`);
    saveModel();
    if (isCoordinator()) shareSnapshot(null, true);
    return true;
  }

  function promoteChampionIfBetter() {
    const scorecard = tournamentScorecard();
    if (!scorecard || scorecard.candidate < scorecard.champion * PROMOTION_MARGIN
      || scorecard.candidateLower < scorecard.championLower * 0.95) return false;
    model.champion = championSnapshot(scorecard.candidate);
    model.tournamentGeneration += 1;
    event(`Challenger promoted: ${formatTime(scorecard.candidate)} vs ${formatTime(scorecard.champion)} median; lower tail ${formatTime(scorecard.candidateLower)}.`);
    saveModel();
    if (isCoordinator()) shareSnapshot(null, true);
    return true;
  }

  function restoreChampion(reason = 'Restored champion') {
    if (!model.champion) return false;
    const champion = model.champion;
    const early = applyHeadSnapshot(model, champion.earlyHead);
    const mid = applyHeadSnapshot(model.midHead, champion.midHead);
    const late = applyHeadSnapshot(model.lateHead, champion.lateHead);
    const hell = applyHeadSnapshot(model.hellHead, champion.hellHead);
    if (!early || !mid || !late || !hell) return false;
    model.replay = [];
    model.replayCursor = 0;
    rebuildReplayTree();
    model.tournamentGeneration += 1;
    event(`${reason} (${formatTime(champion.score)} held-out median).`);
    saveModel(true);
    if (isCoordinator()) shareSnapshot(null, true);
    return true;
  }

  function maybeAutomaticRollback() {
    const scorecard = tournamentScorecard();
    if (!model.champion || !scorecard) return false;
    const now = Date.now();
    if (now - (Number(model.lastRollbackAt) || 0) < ROLLBACK_COOLDOWN_MS) return false;
    if (scorecard.candidate >= scorecard.champion * CHALLENGER_REJECTION_RATIO) return false;
    model.lastRollbackAt = now;
    return restoreChampion(`Tournament rollback after ${formatTime(scorecard.candidate)} vs ${formatTime(scorecard.champion)}`);
  }

  function rememberEvaluation(result) {
    if (!result || !Number.isFinite(result.seconds)) return;
    const evaluation = normalizeEvaluationRecord({ ...result, at: Date.now() });
    model.evaluationResults.push(evaluation);
    model.evaluationResults = model.evaluationResults.slice(-40);
    if (evaluation.quality) event(`${evaluation.policy} evaluation: ${formatTime(result.seconds)} · ${evaluation.phase} (${(evaluation.droppedRate * 100).toFixed(1)}% dropped)`);
    else event(`Evaluation excluded: ${(evaluation.droppedRate * 100).toFixed(1)}% dropped decisions.`);
    if (!model.champion) bootstrapChampionFromHistory();
    else if (!promoteChampionIfBetter()) maybeAutomaticRollback();
    saveModel();
  }

  function rememberEliteEpisode(episode) {
    if (!episode || !Number.isFinite(episode.seconds) || !Array.isArray(episode.transitions) || episode.seconds < 120) return;
    const scores = model.eliteScores.slice().sort((left, right) => right - left);
    const cutoff = scores.length < 6 ? 120 : scores[Math.min(11, scores.length - 1)];
    if (episode.seconds < cutoff) return;
    const candidates = episode.transitions.filter((transition) => transition && transition.kind === 'movement' && Array.isArray(transition.input)).slice(-360);
    if (!candidates.length) return;
    candidates.forEach((transition) => { transition.priority = Number(transition.priority) || 1; });
    model.elite.push(...candidates);
    if (model.elite.length > ELITE_LIMIT) model.elite.splice(0, model.elite.length - ELITE_LIMIT);
    model.eliteCount = model.elite.length;
    model.eliteScores.push(episode.seconds);
    model.eliteScores = model.eliteScores.sort((left, right) => right - left).slice(0, 12);
    event(`Elite replay retained from ${formatTime(episode.seconds)}.`);
    saveModel();
  }

  function submitEliteEpisode(seconds) {
    if (run.evaluation || !run.episodeTransitions.length) return;
    const episode = {
      seconds,
      transitions: run.episodeTransitions.slice(),
      transferId: `${learnerId}:elite:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`,
    };
    queueTransfer('elite', episode, ELITE_OUTBOX_LIMIT);
    if (isCoordinator()) drainDurableTransfers(true);
    else if (peerChannel) peerChannel.postMessage({ type: 'eliteEpisode', from: learnerId, episode });
  }

  function finishRun(observation, terminal = false) {
    settleDecision(observation, terminal, !terminal);
    settleCard(observation, true, terminal);
    const seconds = observation.time || Math.max(0, Math.round((Date.now() - run.gameStartedAt) / 1000));
    if (seconds > 0) {
      model.completedRuns += 1;
      model.totalSeconds += seconds;
      model.bestSeconds = Math.max(model.bestSeconds, seconds);
      if (run.evaluation) {
        const timingWarnings = Math.max(0, run.timingWarnings - run.runTimingWarningsAtStart);
        const droppedTransitions = Math.max(0, run.lowQualityDropped - run.lowQualityDroppedAtStart);
        const result = {
          seconds,
          phase: run.maxPhase,
          timingWarnings,
          droppedTransitions,
          evaluation: true,
          policy: run.evaluationPolicy,
          generation: model.tournamentGeneration,
        };
        if (isCoordinator()) rememberEvaluation(result);
      } else {
        submitEliteEpisode(seconds);
        event(`Run ${model.completedRuns}: ${formatTime(seconds)}`);
        if (isCoordinator()) scheduleConsolidation('local run');
      }
      if (!isCoordinator() && peerChannel) {
        peerChannel.postMessage({ type: 'runSummary', from: learnerId, result: {
          seconds,
          phase: run.maxPhase,
          evaluation: run.evaluation,
          policy: run.evaluationPolicy,
          generation: model.tournamentGeneration,
          timingWarnings: Math.max(0, run.timingWarnings - run.runTimingWarningsAtStart),
          droppedTransitions: Math.max(0, run.lowQualityDropped - run.lowQualityDroppedAtStart),
        } });
      }
      saveModel();
    }
    run.gameStartedAt = 0;
    run.pattern = null;
    run.lastCard = null;
    run.decision = null;
    run.nStep = [];
    run.episodeTransitions = [];
    run.evaluation = false;
    run.evaluationPolicy = 'candidate';
    run.hellMode = false;
    run.maxPhase = 'early';
  }

  function handleFlow(observation) {
    if (observation.finale) {
      releaseMovement();
      // First appears at normal closing time. Start the scripted chase; after
      // that chase ends, select the explicitly labelled After-hours Hell mode.
      const hell = findVisible('#finaleMsg .fin-continue', (button) => /HELL/i.test(button.innerText));
      if (hell) {
        if (click(hell, 'Entered After-hours Hell mode')) {
          settleDecision(observation, false, true);
          settleCard(observation, true);
          run.hellMode = true;
          run.maxPhase = 'hell';
          run.status = 'Hell mode entered; continuing autonomous play.';
        }
        return true;
      }
      const startRunning = findVisible('#finaleMsg .fin-continue', (button) => /START RUNNING/i.test(button.innerText));
      if (click(startRunning, 'Finale: start running')) run.status = 'Finale chase started.';
      return true;
    }

    if (run.forceRestart || (model.settings.safetyRestartAt60 && observation.playing && observation.time >= 60 * 60)) {
      releaseMovement();
      if (!run.forceRestart) {
        finishRun(observation, false);
        run.forceRestart = true;
        run.status = '60-minute safety retry: exiting this run.';
        click(findVisible('#pauseBtn'), '60-minute safety pause');
        return true;
      }
      const exit = findVisible('button.pause-exit');
      if (click(exit, '60-minute safety exit')) {
        run.forceRestart = false;
        run.startupAttempted = false;
        run.status = '60-minute safety retry: starting Joe again.';
      } else if (visible(findVisible('#pauseBtn'))) {
        click(findVisible('#pauseBtn'), '60-minute safety pause');
      }
      return true;
    }

    if (observation.over) {
      releaseMovement();
      if (!run.retryAt) {
        finishRun(observation, true);
        run.retryAt = Date.now() + 1300;
        run.status = 'Run finished; retrying with Joe…';
      }
      if (Date.now() >= run.retryAt) {
        const retry = findVisible('#overScreen button', (button) => /RETRY/i.test(button.innerText));
        if (click(retry, 'Retry')) run.retryAt = 0;
      }
      return true;
    }

    if (observation.choosing) {
      releaseMovement();
      settleCard(observation, true);
      if (chooseCard(observation)) run.status = 'Selected an upgrade card.';
      return true;
    }

    if (visible(document.querySelector('#craftChoiceScreen'))) {
      releaseMovement();
      const choice = findVisible('#craftChoices > *, #craftChoices button, #craftChoices [onclick]');
      if (click(choice, 'Craft choice')) run.status = 'Crafting the available cocktail.';
      return true;
    }

    if (visible(document.querySelector('#craftScreen'))) {
      releaseMovement();
      if (click(findVisible('#craftBtn'), 'Craft confirmed')) run.status = 'Confirmed cocktail craft.';
      return true;
    }

    if (visible(document.querySelector('#noticeScreen'))) {
      releaseMovement();
      if (click(findVisible('#noticeScreen .tip-go, #noticeScreen button'), 'Notice acknowledged')) run.status = 'Acknowledged unlock.';
      return true;
    }

    if (observation.intro) {
      releaseMovement();
      if (click(findVisible('.crawl-skip'), 'Intro skipped')) run.status = 'Skipping intro.';
      return true;
    }

    if (observation.selecting) {
      releaseMovement();
      if (click(findVisible('.char.joe'), 'Joe selected')) {
        run.hellMode = false;
        run.status = 'Joe selected.';
      }
      return true;
    }

    // Boss-tip and other transient dialogs are all safe local game prompts.
    const continueButton = findVisible('button.tip-go', (button) => /CHEERS|GOT IT/i.test(button.innerText));
    if (continueButton) {
      releaseMovement();
      if (click(continueButton, `Continue: ${continueButton.innerText.trim()}`)) run.status = 'Continuing prompt.';
      return true;
    }

    return false;
  }

  function play(observation) {
    if (!run.gameStartedAt) {
      run.gameStartedAt = Date.now() - observation.time * 1000;
      run.evaluation = (model.completedRuns + 1) % EVALUATION_INTERVAL_RUNS === 0;
      const evaluationNumber = Math.floor((model.completedRuns + 1) / EVALUATION_INTERVAL_RUNS);
      run.evaluationPolicy = run.evaluation && model.champion
        ? (evaluationNumber % 2 ? 'candidate' : 'champion') : 'candidate';
      run.episodeTransitions = [];
      run.nStep = [];
      run.runTimingWarningsAtStart = run.timingWarnings;
      run.lowQualityDroppedAtStart = run.lowQualityDropped;
      run.maxPhase = phaseFor(observation);
      event('Gameplay detected.');
    }
    run.maxPhase = furthestPhase(run.maxPhase, phaseFor(observation));
    settleCard(observation);
    const phaseChanged = run.decision && run.decision.phase !== phaseFor(observation);
    if (!run.decision || phaseChanged || observation.time - run.decision.time >= workerDecisionSeconds) {
      settleDecision(observation, false, phaseChanged);
      startDecision(observation);
    } else if (run.pattern.steps.length > 1 && observation.time - run.lastMotionStepGameTime >= 0.24) {
      run.patternIndex = (run.patternIndex + 1) % run.pattern.steps.length;
      run.lastMotionStepGameTime = observation.time;
    }
    const vector = run.pattern.steps[run.patternIndex];
    setMovement(vector);
    if ((run.pattern.dash || (observation.health !== null && observation.health < 0.22)) && observation.time - run.lastDashGameTime >= 0.35) {
      dash(vector);
      run.lastDashGameTime = observation.time;
    }

    if (run.pattern.ultimate && observation.time - run.lastUltimateGameTime >= ULTIMATE_GAME_SECONDS) {
      const ultimate = findVisible('#ultBtn');
      if (ultimate) ultimate.click();
      else {
        sendKey(' ', true);
        setTimeout(() => sendKey(' ', false), 35);
      }
      run.lastUltimateGameTime = observation.time;
    }
    run.status = `${run.evaluation ? `Eval ${run.evaluationPolicy} · ` : ''}${run.pattern.label} · ${observation.health === null ? 'HP ?' : `${Math.round(observation.health * 100)}% HP`} · ${formatTime(observation.time)}`;
  }

  function attemptInitialJoeStart(observation) {
    // The public title routes to a different mini-game, while the Joe card is
    // the Cocktail Defense game's own start control. Trigger it once only when
    // Autopilot is first armed and no active game screen is present.
    if (run.startupAttempted || observation.playing || observation.intro || observation.selecting || observation.over) return;
    const joe = document.querySelector('.char.joe');
    if (joe && Date.now() - run.startedAt > 700 && click(joe, 'Joe start')) {
      run.hellMode = false;
      run.startupAttempted = true;
      run.status = 'Starting Cocktail Defense as Joe.';
    }
  }

  function tick() {
    if (!run.enabled) return;
    drainDurableTransfers();
    const observation = observe();
    if (observation.playing && run.lastObservedGameTime !== null && observation.time >= run.lastObservedGameTime) {
      const jump = observation.time - run.lastObservedGameTime;
      run.lastGameTimeJump = jump;
      run.largestGameTimeJump = Math.max(run.largestGameTimeJump, jump);
      if (jump > MAX_QUALITY_TIME_JUMP) run.timingWarnings += 1;
    }
    run.lastObservedGameTime = observation.time;

    if (handleFlow(observation)) {
      renderPanel();
      return;
    }
    attemptInitialJoeStart(observation);
    if (observation.playing) play(observation);
    else {
      releaseMovement();
      if (!run.startupAttempted) run.status = 'Looking for the Joe game start.';
    }
    renderPanel();
  }

  function arm() {
    if (run.enabled) return;
    run.enabled = true;
    run.startedAt = Date.now();
    run.lastUltimateGameTime = -ULTIMATE_GAME_SECONDS;
    run.lastDecisionGameTime = 0;
    run.lastMotionStepGameTime = 0;
    run.lastDashGameTime = -Infinity;
    run.lastObservedGameTime = null;
    run.lastGameTimeJump = 0;
    run.largestGameTimeJump = 0;
    run.timingWarnings = 0;
    run.runTimingWarningsAtStart = 0;
    run.lowQualityDropped = 0;
    run.lowQualityDroppedAtStart = 0;
    run.maxPhase = 'early';
    run.startupAttempted = false;
    run.retryAt = 0;
    run.status = 'Autopilot armed.';
    run.timer = setInterval(tick, TICK_MS);
    event('Autopilot armed.');
    renderPanel();
  }

  function stop(reason = 'Stopped.') {
    releaseMovement();
    if (run.timer) clearInterval(run.timer);
    run.timer = null;
    run.enabled = false;
    run.status = reason;
    saveModel();
    renderPanel();
  }

  function formatTime(seconds) {
    const whole = Math.max(0, Math.floor(seconds || 0));
    return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
  }

  function downloadTrainingData() {
    const cleanExperience = (experience) => ({
      kind: experience.kind,
      input: experience.input,
      actionIndex: experience.actionIndex,
      nextState: experience.nextState,
      reward: experience.reward,
      done: experience.done,
      steps: experience.steps,
      phase: experience.phase,
      priority: experience.priority,
    });
    const payload = {
      format: 'pine-autopilot-training-v12',
      exportedAt: new Date().toISOString(),
      stateInputs: STATE_INPUTS,
      cardInputs: CARD_INPUTS,
      actions: ACTIONS.map((action) => ({ id: action.id, label: action.label })),
      metrics: {
        completedRuns: model.completedRuns,
        bestSeconds: model.bestSeconds,
        totalSeconds: model.totalSeconds,
        uniqueExperiences: model.uniqueExperiences,
        evaluationResults: model.evaluationResults,
        cleanEvaluationRuns: candidateEvaluationResults().length,
        championEvaluationRuns: tournamentEvaluationResults('champion').length,
        tournamentGeneration: model.tournamentGeneration,
        eliteScores: model.eliteScores,
        championMedian: model.champion ? model.champion.score : null,
        lastRollbackAt: model.lastRollbackAt || null,
        immortalPolicyUpdates: model.masteryUpdates,
        immortalSituations: Object.keys(model.masteryPolicy).length,
        manualTransfersAccepted: model.manualTransfersAccepted,
        eliteTransfersAccepted: model.eliteTransfersAccepted,
      },
      recentReplay: model.replay.slice(-3000).map(cleanExperience),
      eliteReplay: model.elite.slice(-600).map(cleanExperience),
      demonstrationReplay: model.demonstrations.slice(-DEMONSTRATION_LIMIT).map(cleanExperience),
    };
    downloadJson(payload, `pine-autopilot-training-${Date.now()}.json`);
    event('Downloaded local training export.');
  }

  function downloadJson(payload, filename) {
    const anchor = document.createElement('a');
    anchor.href = URL.createObjectURL(new Blob([JSON.stringify(payload)], { type: 'application/json' }));
    anchor.download = filename;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(anchor.href), 1000);
  }

  function downloadMpsCheckpoint() {
    const payload = {
      format: 'pine-autopilot-checkpoint-v12',
      exportedAt: new Date().toISOString(),
      contract: {
        stateInputs: STATE_INPUTS,
        cardInputs: CARD_INPUTS,
        hiddenUnits: HIDDEN_UNITS,
        movementActions: ACTIONS.map((action) => ({ id: action.id, label: action.label })),
        discount: DISCOUNT,
      },
      // This is a complete browser checkpoint. The local MPS trainer modifies
      // the four challenger heads; importing never replaces this champion.
      model: packStoredModel(),
    };
    downloadJson(payload, `pine-autopilot-checkpoint-${Date.now()}.json`);
    event('Downloaded MPS challenger checkpoint.');
  }

  function importMpsCheckpoint(file) {
    if (!file) return;
    if (!isCoordinator()) {
      run.status = 'Import from the central learner tab so it can safely share the challenger.';
      renderPanel(true);
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => {
      run.status = 'Could not read that checkpoint file.';
      renderPanel(true);
    };
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result || ''));
        if (!payload || !['pine-autopilot-checkpoint-v8', 'pine-autopilot-checkpoint-v9', 'pine-autopilot-checkpoint-v10', 'pine-autopilot-checkpoint-v11', 'pine-autopilot-checkpoint-v12'].includes(payload.format) || !payload.model) throw new Error('format');
        const incoming = unpackStoredModel(payload.model);
        if (!validHead(incoming) || !validHead(incoming.midHead) || !validHead(incoming.lateHead) || !validHead(incoming.hellHead)) throw new Error('network shape');
        const early = applyHeadSnapshot(model, headSnapshot(incoming));
        const mid = applyHeadSnapshot(model.midHead, headSnapshot(incoming.midHead));
        const late = applyHeadSnapshot(model.lateHead, headSnapshot(incoming.lateHead));
        const hell = applyHeadSnapshot(model.hellHead, headSnapshot(incoming.hellHead));
        if (!early || !mid || !late || !hell) throw new Error('network shape');
        model.replay = [];
        model.replayCursor = 0;
        rebuildReplayTree();
        model.tournamentGeneration += 1;
        model.lastRollbackAt = 0;
        event('Imported offline MPS challenger; champion remains frozen for tournament testing.');
        saveModel();
        shareSnapshot(null, true);
        run.status = 'Offline challenger imported. Tournament testing has restarted.';
      } catch (_) {
        run.status = 'Invalid MPS checkpoint; the live challenger was unchanged.';
      }
      renderPanel(true);
    };
    reader.readAsText(file);
  }

  function diagnosticsPayload() {
    const actionStats = {};
    model.replay.slice(-3000).concat(model.elite.slice(-600)).forEach((experience) => {
      if (!experience || experience.kind !== 'movement') return;
      const action = ACTIONS[experience.actionIndex];
      const key = `${experience.phase || 'early'} · ${action ? action.label : 'unknown'}`;
      const stat = actionStats[key] || { samples: 0, reward: 0, terminal: 0 };
      stat.samples += 1;
      stat.reward += Number(experience.reward) || 0;
      stat.terminal += experience.done ? 1 : 0;
      actionStats[key] = stat;
    });
    const supportedActions = Object.entries(actionStats)
      .map(([key, stat]) => ({ action: key, samples: stat.samples, averageReward: stat.reward / stat.samples, terminal: stat.terminal }))
      .filter((stat) => stat.samples >= 20)
      .sort((left, right) => right.averageReward - left.averageReward)
      .slice(0, 12);
    const cleanEvaluations = candidateEvaluationResults();
    const championEvaluations = tournamentEvaluationResults('champion');
    const tournament = tournamentScorecard();
    const phaseEvaluations = ['early', 'mid', 'late', 'hell'].map((phase) => {
      const scores = cleanEvaluations.filter((entry) => entry.phase === phase).map((entry) => entry.seconds);
      return { phase, runs: scores.length, medianSeconds: scores.length ? median(scores) : null };
    });
    return {
      format: 'pine-autopilot-diagnostics-v12',
      role: learnerRole(),
      workerProfile,
      uniqueExperiences: model.uniqueExperiences,
      completedRuns: model.completedRuns,
      bestSeconds: model.bestSeconds,
      heldOutMedian: median(cleanEvaluations.map((entry) => entry.seconds)),
      rawHeldOutMedian: median(model.evaluationResults.filter((entry) => entry.policy !== 'champion').map((entry) => entry.seconds)),
      cleanEvaluationRuns: cleanEvaluations.length,
      championEvaluationRuns: championEvaluations.length,
      evaluationResults: model.evaluationResults,
      evaluationTrend: evaluationTrend(),
      evaluationByFurthestPhase: phaseEvaluations,
      tournament,
      tournamentGeneration: model.tournamentGeneration,
      evaluationQuality: {
        acceptedDroppedRateMaximum: MAX_EVALUATION_DROPPED_RATE,
        estimatedDecisionSeconds: EVALUATION_DECISION_ESTIMATE_SECONDS,
      },
      championMedian: model.champion ? model.champion.score : null,
      lastRollbackAt: model.lastRollbackAt || null,
      replaySamplesVisible: model.replay.length,
      eliteSamplesVisible: model.elite.length,
      demonstrationSamplesVisible: model.demonstrations.length,
      immortalPolicyEnabled: !!model.settings.immortalMode,
      immortalSituationsVisible: Object.keys(model.masteryPolicy).length,
      immortalPolicyUpdates: model.masteryUpdates,
      immortalPolicyUses: model.masteryActionUses,
      manualTransfersAccepted: model.manualTransfersAccepted,
      eliteTransfersAccepted: model.eliteTransfersAccepted,
      eliteImitationSteps: model.eliteImitationSteps,
      riskStatesVisible: Object.keys(model.riskStats).length,
      riskUpdates: model.riskUpdates,
      consolidationSteps: model.consolidationSteps,
      consolidationPasses: model.consolidationPasses,
      timingWarnings: run.timingWarnings,
      largestGameTimeJump: run.largestGameTimeJump,
      lastGameTimeJump: run.lastGameTimeJump,
      transitionQualityThresholdSeconds: MAX_QUALITY_TIME_JUMP,
      lowQualityDropped: run.lowQualityDropped,
      currentIntent: run.intent,
      replayByPhase: { ...replayPhaseCounts },
      supportedActions,
    };
  }

  function copyDiagnostics() {
    const text = JSON.stringify(diagnosticsPayload());
    const copied = navigator.clipboard && navigator.clipboard.writeText ? navigator.clipboard.writeText(text) : null;
    if (copied && typeof copied.then === 'function') {
      copied.then(() => { run.status = 'Diagnostics copied; paste them into this chat.'; renderPanel(true); })
        .catch(() => window.prompt('Copy these diagnostics into chat:', text));
    } else {
      window.prompt('Copy these diagnostics into chat:', text);
    }
  }

  function mountPanel() {
    const host = document.createElement('div');
    host.id = 'pine-autopilot-root';
    const shadow = host.attachShadow({ mode: 'open' });
    shadow.innerHTML = `
      <style>
        .panel { position:fixed; z-index:2147483647; left:16px; bottom:16px; width:252px; overflow:hidden; border:1px solid #c89a3c; border-radius:8px; color:#f4e9cf; background:rgba(18,19,24,.95); box-shadow:0 8px 24px #0008; font:12px/1.35 ui-monospace,Menlo,monospace; }
        .panel.minimized main { display:none; } header { display:flex; justify-content:space-between; align-items:center; padding:9px 10px 7px; color:#ffd36d; border-bottom:1px solid #c89a3c55; font-weight:700; } .header-right { display:flex; align-items:center; gap:7px; } header button { flex:0; min-width:24px; padding:0 5px; border:0; background:transparent; color:#ffd36d; font-size:16px; line-height:16px; }
        main { padding:9px 10px 10px; } .status { min-height:32px; color:#e4dccd; } .stats { margin:7px 0; color:#bdb4a4; }
        .controls { display:flex; gap:6px; } button { flex:1; padding:6px; border:1px solid #9e7a32; border-radius:4px; background:#372b19; color:#fff0cd; font:inherit; cursor:pointer; } button:hover { background:#514020; } button.stop { border-color:#965355; background:#482728; }
        details { margin-top:8px; color:#aba292; } summary { cursor:pointer; } .note { margin-top:7px; color:#8f877a; font-size:10px; }
      </style>
      <section class="panel" aria-label="Pine Autopilot"><header><span>🍸 Pine Autopilot</span><span class="header-right"><span>v${VERSION}</span><button id="minimize" type="button" title="Minimize panel">−</button></span></header><main>
        <div class="status" id="status"></div><div class="stats" id="stats"></div>
        <div class="controls"><button id="arm" type="button">Arm loop</button><button id="stop" class="stop" type="button">Stop</button></div>
        <label style="display:block;margin-top:8px;color:#cfc5b3"><input id="safetyRestart" type="checkbox"> Restart safely at 60:00</label>
        <label style="display:block;margin-top:5px;color:#cfc5b3"><input id="immortalMode" type="checkbox"> Protect manual “immortal” policy</label>
        <details><summary>Neural learner</summary><div id="learning"></div><button id="manualDemo" type="button" style="margin-top:7px">Record manual demo</button><button id="diagnostics" type="button" style="margin-top:7px">Copy diagnostics</button><button id="export" type="button" style="margin-top:7px">Download training data</button><button id="checkpoint" type="button" style="margin-top:7px">Download MPS checkpoint</button><button id="import" type="button" style="margin-top:7px">Import MPS challenger</button><input id="checkpointFile" type="file" accept="application/json,.json" hidden><button id="restore" type="button" style="margin-top:7px">Restore champion</button><button id="reset" type="button" style="margin-top:7px">Reset neural model</button></details>
        <div class="note">Auto-starts on page load · Double DQN · risk-aware survival · one central learner. Every sixth run is a blind champion/challenger evaluation. Manual demo records local video/actions; no rank submission.</div>
      </main></section>`;
    document.documentElement.append(host);
    shadow.querySelector('#arm').addEventListener('click', arm);
    shadow.querySelector('#stop').addEventListener('click', () => stop('Stopped by player.'));
    shadow.querySelector('#safetyRestart').addEventListener('change', (event) => {
      model.settings.safetyRestartAt60 = event.target.checked;
      saveModel();
      renderPanel();
    });
    shadow.querySelector('#immortalMode').addEventListener('change', (event) => {
      model.settings.immortalMode = event.target.checked;
      saveModel(true);
      if (isCoordinator()) shareSnapshot(null, true);
      renderPanel();
    });
    shadow.querySelector('#export').addEventListener('click', downloadTrainingData);
    shadow.querySelector('#manualDemo').addEventListener('click', () => {
      if (manualDemo.active) stopManualDemo();
      else startManualDemo();
    });
    shadow.querySelector('#minimize').addEventListener('click', () => {
      model.settings.panelMinimized = !model.settings.panelMinimized;
      saveModel();
      renderPanel(true);
    });
    shadow.querySelector('#checkpoint').addEventListener('click', downloadMpsCheckpoint);
    shadow.querySelector('#import').addEventListener('click', () => shadow.querySelector('#checkpointFile').click());
    shadow.querySelector('#checkpointFile').addEventListener('change', (event) => {
      importMpsCheckpoint(event.target.files && event.target.files[0]);
      event.target.value = '';
    });
    shadow.querySelector('#diagnostics').addEventListener('click', copyDiagnostics);
    shadow.querySelector('#reset').addEventListener('click', () => {
      if (!confirm('Reset the shared neural model for every active Pine tab?')) return;
      Object.assign(model, blankModel());
      rebuildReplayTree();
      lastSnapshotExperienceCount = 0;
      if (peerChannel) peerChannel.postMessage({ type: 'reset', from: learnerId });
      saveModel(true);
      run.status = 'Local model reset.';
      renderPanel();
    });
    shadow.querySelector('#restore').addEventListener('click', () => {
      if (!model.champion) return;
      if (!confirm(`Restore the champion model (${formatTime(model.champion.score)} held-out median)? Recent replay data will be cleared.`)) return;
      if (!isCoordinator() && peerChannel) {
        peerChannel.postMessage({ type: 'restoreChampionRequest', from: learnerId });
        run.status = 'Requested champion restore from the central learner.';
        renderPanel();
        return;
      }
      if (restoreChampion()) run.status = 'Champion restored; fresh replay collection started.';
      renderPanel();
    });
    return shadow;
  }

  function renderPanel(force = false) {
    if (!force && Date.now() - lastPanelRender < 250) return;
    lastPanelRender = Date.now();
    panel.querySelector('#status').textContent = run.status;
    panel.querySelector('#stats').textContent = `Runs ${model.completedRuns} · best ${formatTime(model.bestSeconds)} · total ${formatTime(model.totalSeconds)}`;
    panel.querySelector('#arm').disabled = run.enabled || manualDemo.active;
    panel.querySelector('#stop').disabled = !run.enabled;
    panel.querySelector('#safetyRestart').checked = !!model.settings.safetyRestartAt60;
    panel.querySelector('#immortalMode').checked = !!model.settings.immortalMode;
    panel.querySelector('#restore').disabled = !model.champion;
    panel.querySelector('#import').disabled = !isCoordinator();
    panel.querySelector('#manualDemo').textContent = manualDemo.active ? 'Stop demo & resume bot' : 'Record manual demo';
    const panelRoot = panel.querySelector('.panel');
    panelRoot.classList.toggle('minimized', !!model.settings.panelMinimized);
    panel.querySelector('#minimize').textContent = model.settings.panelMinimized ? '+' : '−';
    const cleanEvaluations = candidateEvaluationResults();
    const championEvaluations = tournamentEvaluationResults('champion');
    const evaluationMedian = median(cleanEvaluations.map((entry) => entry.seconds));
    const tournament = tournamentScorecard();
    const trend = evaluationTrend();
    const trendText = trend ? `\nEval trend: ${formatTime(trend.early)} → ${formatTime(trend.recent)} (${trend.percent >= 0 ? '+' : ''}${trend.percent.toFixed(1)}%)` : '';
    const championText = model.champion ? `\nFrozen champion: ${formatTime(model.champion.score)} · generation ${model.tournamentGeneration}` : '';
    const tournamentText = tournament
      ? `\nTournament: candidate ${formatTime(tournament.candidate)} vs champion ${formatTime(tournament.champion)} · lower ${formatTime(tournament.candidateLower)}/${formatTime(tournament.championLower)} (${tournament.candidateRuns}/${TOURNAMENT_WINDOW} each)`
      : `\nTournament: candidate ${cleanEvaluations.length}/${TOURNAMENT_WINDOW} · champion ${championEvaluations.length}/${TOURNAMENT_WINDOW} clean evals`;
    const migrationText = model.migratedFromV8 ? '\nImported earlier challenger and champion' : '';
    const timingText = run.timingWarnings ? `\nTiming: ${run.largestGameTimeJump.toFixed(1)} s max jump · ${run.timingWarnings} warning(s) · ${run.lowQualityDropped} transition(s) dropped` : '';
    const top = `Role: ${learnerRole()} · ${workerProfile} worker\nDQN gradient steps: ${experienceCount()}\nUnique shared experiences: ${model.uniqueExperiences}\nEarly ε: ${(explorationRate(model) * 100).toFixed(1)}% · Mid: ${(explorationRate(model.midHead) * 100).toFixed(1)}%\nLate ε: ${(explorationRate(model.lateHead) * 100).toFixed(1)}% · Hell: ${(explorationRate(model.hellHead) * 100).toFixed(1)}%\nReplay: ${model.replay.length}/${REPLAY_LIMIT} · elite: ${model.eliteCount || model.elite.length}/${ELITE_LIMIT} · demos: ${model.demonstrationCount || model.demonstrations.length}/${DEMONSTRATION_LIMIT}\nImmortal policy: ${model.settings.immortalMode ? 'protected' : 'off'} · ${Object.keys(model.masteryPolicy).length} learned situations · used ${model.masteryActionUses} times\nDurable transfers: manual ${model.manualTransfersAccepted} · elite ${model.eliteTransfersAccepted}\nReplay phase mix: E${replayPhaseCounts.early} M${replayPhaseCounts.mid} L${replayPhaseCounts.late} H${replayPhaseCounts.hell}\nAdaptive replay: ${adaptiveReplayBatch}/transition · ${Math.round(DEMONSTRATION_REPLAY_RATE * 100)}% demo target · imitate ${model.eliteImitationSteps}\nRisk shield: ${Object.keys(model.riskStats).length} learned states · ${model.riskUpdates} outcomes\nIdle consolidation: ${model.consolidationSteps} updates / ${model.consolidationPasses} passes\nIntent: ${run.intent} · shield: colour-threat aware\nCandidate held-out: ${cleanEvaluations.length ? formatTime(evaluationMedian) : 'collecting…'} (${cleanEvaluations.length} accepted / ${model.evaluationResults.length} total; ≤${(MAX_EVALUATION_DROPPED_RATE * 100).toFixed(0)}% drops)${tournamentText}${trendText}${championText}${migrationText}${timingText}`;
    const learning = panel.querySelector('#learning');
    learning.textContent = top;
    learning.style.whiteSpace = 'pre-line';
  }

  addEventListener('pagehide', () => {
    stopManualDemo(false);
    stop('Page left; model saved.');
  }, { once: true });
})();
