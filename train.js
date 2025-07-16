const {
  WEAPONS,
  createRandomNet,
  neuralDecision,
  GRAVITY,
} = require('./assets/js/game');

const POPULATION = 20;
const GENERATIONS = 50;
const MUTATION_RATE = 0.1;
const MUTATION_STRENGTH = 0.5;

function cloneNet(net) {
  return JSON.parse(JSON.stringify(net));
}

function mutate(net) {
  const mutateArray = (arr) => {
    for (let i = 0; i < arr.length; i++) {
      if (Array.isArray(arr[i])) mutateArray(arr[i]);
      else if (Math.random() < MUTATION_RATE)
        arr[i] += (Math.random() * 2 - 1) * MUTATION_STRENGTH;
    }
  };
  mutateArray(net.hiddenWeights);
  mutateArray(net.hiddenBias);
  mutateArray(net.outputWeights);
  mutateArray(net.outputBias);
}

function simulateShot(net, enemyX) {
  const inputs = [
    enemyX / 800, // dx
    0, // dy
    0, // front slope
    0, // back slope
    Math.abs(enemyX) / 800, // enemy distance
  ];
  const { angle, power, weapon } = neuralDecision(net, inputs);
  const rad = (angle * Math.PI) / 180;
  const w = WEAPONS[weapon];
  const speed = w.speed * power;
  const vx = Math.cos(rad) * speed;
  const vy = Math.sin(rad) * speed;
  const t = (vy * 2) / GRAVITY;
  const landingX = vx * t;
  const dist = Math.abs(landingX - enemyX);
  let damage = 0;
  if (dist <= w.radius) {
    damage = w.damage * (1 - dist / w.radius);
  }
  const closeness = Math.max(0, (w.radius - dist) / w.radius);
  return damage + closeness * 10;
}

function evaluate(net) {
  let fitness = 0;
  for (let i = 0; i < 5; i++) {
    const enemyX = 300 + Math.random() * 200;
    fitness += simulateShot(net, enemyX);
  }
  return fitness / 5;
}

function evolve() {
  let population = Array.from({ length: POPULATION }, createRandomNet);
  let best = population[0];
  for (let g = 0; g < GENERATIONS; g++) {
    const scored = population.map((net) => ({ net, score: evaluate(net) }));
    scored.sort((a, b) => b.score - a.score);
    best = scored[0].net;
    const survivors = scored.slice(0, POPULATION / 2).map((s) => s.net);
    population = survivors.map(cloneNet);
    while (population.length < POPULATION) {
      const parent = cloneNet(survivors[Math.floor(Math.random() * survivors.length)]);
      mutate(parent);
      population.push(parent);
    }
    console.log(`Generation ${g + 1}: best fitness = ${scored[0].score.toFixed(2)}`);
  }
  return best;
}

if (require.main === module) {
  const trained = evolve();
  const fs = require('fs');
  fs.writeFileSync('trained_net.json', JSON.stringify(trained, null, 2));
  console.log('Training complete. Network saved to trained_net.json');
}
