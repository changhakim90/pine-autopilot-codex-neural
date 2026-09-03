#!/usr/bin/env node

/*
 * Local-only report for a file downloaded from Pine Autopilot v5 through v11.
 * Usage: node offline-analysis.mjs pine-autopilot-training-....json
 */

import fs from 'node:fs';

const file = process.argv[2];
if (!file) {
  console.error('Usage: node offline-analysis.mjs <training-export.json>');
  process.exit(1);
}

const report = JSON.parse(fs.readFileSync(file, 'utf8'));
if (!['pine-autopilot-training-v5', 'pine-autopilot-training-v6', 'pine-autopilot-training-v7', 'pine-autopilot-training-v8', 'pine-autopilot-training-v9', 'pine-autopilot-training-v10', 'pine-autopilot-training-v11'].includes(report.format)) {
  throw new Error('Not a Pine Autopilot v5 through v11 training export.');
}

const median = (values) => {
  if (!values.length) return 0;
  const sorted = values.slice().sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
};

const formatTime = (seconds) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
const actionNames = new Map(report.actions.map((action, index) => [index, action.label]));
const transitions = [...(report.recentReplay || []), ...(report.eliteReplay || []), ...(report.demonstrationReplay || [])].filter((item) => item.kind === 'movement');
const groups = new Map();

for (const transition of transitions) {
  const key = `${transition.phase || 'early'} · ${actionNames.get(transition.actionIndex) || 'unknown'}`;
  const group = groups.get(key) || { count: 0, reward: 0, deaths: 0 };
  group.count += 1;
  group.reward += transition.reward || 0;
  group.deaths += transition.done ? 1 : 0;
  groups.set(key, group);
}

console.log(`Exported: ${report.exportedAt}`);
console.log(`Unique experiences: ${report.metrics.uniqueExperiences}`);
console.log(`Best run: ${formatTime(report.metrics.bestSeconds)}`);
console.log(`Champion held-out median: ${report.metrics.championMedian ? formatTime(report.metrics.championMedian) : 'none'}`);

const allEvaluations = report.metrics.evaluationResults || [];
const evaluationQuality = (entry) => {
  const seconds = Math.max(1, Number(entry?.seconds) || 0);
  const fallbackDrops = Math.max(0, Number(entry?.timingWarnings) || 0);
  const drops = Number.isFinite(entry?.droppedTransitions) ? Math.max(0, Number(entry.droppedTransitions)) : fallbackDrops;
  const estimatedDecisions = Math.max(1, Math.ceil(seconds / 2));
  return { drops, rate: drops / estimatedDecisions, clean: drops / estimatedDecisions <= 0.10 };
};
const cleanEvaluations = allEvaluations.filter((entry) => entry && evaluationQuality(entry).clean);
const tournamentGeneration = Number(report.metrics.tournamentGeneration);
const currentCandidate = ['pine-autopilot-training-v8', 'pine-autopilot-training-v9', 'pine-autopilot-training-v10', 'pine-autopilot-training-v11'].includes(report.format)
  ? cleanEvaluations.filter((entry) => entry.policy === 'candidate' && entry.generation === tournamentGeneration)
  : [];
const currentChampion = ['pine-autopilot-training-v8', 'pine-autopilot-training-v9', 'pine-autopilot-training-v10', 'pine-autopilot-training-v11'].includes(report.format)
  ? cleanEvaluations.filter((entry) => entry.policy === 'champion' && entry.generation === tournamentGeneration)
  : [];
const candidateEvaluations = currentCandidate.length
  ? currentCandidate
  : cleanEvaluations.filter((entry) => entry.policy !== 'champion');
const evaluations = candidateEvaluations.map((entry) => entry.seconds);
if (allEvaluations.length !== cleanEvaluations.length) {
  console.log(`Excluded held-out runs with timing gaps: ${allEvaluations.length - cleanEvaluations.length}`);
}
if (evaluations.length >= 2) {
  const middle = Math.floor(evaluations.length / 2);
  const early = median(evaluations.slice(0, middle));
  const recent = median(evaluations.slice(middle));
  const change = early ? ((recent - early) / early) * 100 : 0;
  console.log(`Held-out trend: ${formatTime(early)} -> ${formatTime(recent)} (${change >= 0 ? '+' : ''}${change.toFixed(1)}%)`);
}

if (['pine-autopilot-training-v8', 'pine-autopilot-training-v9', 'pine-autopilot-training-v10', 'pine-autopilot-training-v11'].includes(report.format)) {
  console.log(`Tournament generation: ${Number.isFinite(tournamentGeneration) ? tournamentGeneration : 'unknown'}`);
  console.log(`Current tournament clean runs: candidate ${currentCandidate.length}/5 · champion ${currentChampion.length}/5`);
  if (currentCandidate.length >= 5 && currentChampion.length >= 5) {
    const candidate = median(currentCandidate.slice(-5).map((entry) => entry.seconds));
    const champion = median(currentChampion.slice(-5).map((entry) => entry.seconds));
    const ratio = champion ? (candidate / champion) * 100 : 0;
    console.log(`Tournament medians: challenger ${formatTime(candidate)} vs champion ${formatTime(champion)} (${ratio.toFixed(1)}%)`);
  }
}

if (['pine-autopilot-training-v6', 'pine-autopilot-training-v7', 'pine-autopilot-training-v8', 'pine-autopilot-training-v9', 'pine-autopilot-training-v10', 'pine-autopilot-training-v11'].includes(report.format)) {
  console.log('\nClean held-out results by furthest phase:');
  ['early', 'mid', 'late', 'hell'].forEach((phase) => {
    const values = candidateEvaluations.filter((entry) => (entry.phase || 'early') === phase).map((entry) => entry.seconds);
    if (values.length) console.log(`${phase}: ${values.length} run(s), median ${formatTime(median(values))}`);
  });
}

const phaseReplay = transitions.reduce((counts, transition) => {
  const phase = transition.phase || 'early';
  counts[phase] = (counts[phase] || 0) + 1;
  return counts;
}, {});
console.log(`\nExported replay by phase: early ${phaseReplay.early || 0} · mid ${phaseReplay.mid || 0} · late ${phaseReplay.late || 0} · hell ${phaseReplay.hell || 0}`);

console.log('\nBest sampled action / phase combinations:');
const supportedGroups = [...groups.entries()]
  .map(([key, group]) => ({ key, ...group, average: group.reward / group.count }))
  .filter((group) => group.count >= 20)
  .sort((left, right) => right.average - left.average)
  .slice(0, 12);
if (!supportedGroups.length) console.log('Not enough replay support yet; collect more than 20 samples per action/phase.');
supportedGroups.forEach((group) => console.log(`${group.key}: reward ${group.average.toFixed(3)} across ${group.count} samples; terminal ${group.deaths}`));
