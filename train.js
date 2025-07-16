const fs = require('fs');

const INPUT_SIZE = 13;
const HIDDEN_SIZE = 20;
const OUTPUT_SIZE = 3;
const TRAINING_ITERATIONS = 500;
const MAX_POWER = 20;

class NeuralNetwork {
  constructor(inputSize, hiddenSize, outputSize) {
    this.weights1 = this.randomMatrix(inputSize, hiddenSize);
    this.bias1 = this.randomMatrix(1, hiddenSize);
    this.weights2 = this.randomMatrix(hiddenSize, outputSize);
    this.bias2 = this.randomMatrix(1, outputSize);
  }

  randomMatrix(rows, cols) {
    return Array.from({ length: rows }, () =>
      Array.from({ length: cols }, () => (Math.random() - 0.5) * 2)
    );
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  forward(inputs) {
    // normalize like game.js assumes
    inputs = inputs.map((val, idx) => {
      if (idx < 12) return val / 1000; // canvas width
      if (idx === 12) return val / 0.03;
      return val / 200;
    });

    const hidden = inputs
      .map((_, i) =>
        this.weights1[i].reduce((sum, w, j) => sum + w * inputs[j], 0) +
        this.bias1[0][i]
      )
      .map(this.sigmoid);

    const outputs = this.weights2[0]
      .map((_, j) =>
        hidden.reduce((sum, h, i) => sum + h * this.weights2[i][j], 0) +
        this.bias2[0][j]
      )
      .map(this.sigmoid);

    return outputs;
  }

  mutate(rate = 0.1) {
    const mutate = val => (Math.random() < rate ? val + (Math.random() - 0.5) * 0.5 : val);
    this.weights1 = this.weights1.map(r => r.map(mutate));
    this.bias1 = this.bias1.map(r => r.map(mutate));
    this.weights2 = this.weights2.map(r => r.map(mutate));
    this.bias2 = this.bias2.map(r => r.map(mutate));
  }

  copyFrom(other) {
    this.weights1 = JSON.parse(JSON.stringify(other.weights1));
    this.bias1 = JSON.parse(JSON.stringify(other.bias1));
    this.weights2 = JSON.parse(JSON.stringify(other.weights2));
    this.bias2 = JSON.parse(JSON.stringify(other.bias2));
  }

  toJSON() {
    return {
      weights1: this.weights1,
      bias1: this.bias1,
      weights2: this.weights2,
      bias2: this.bias2
    };
  }
}

class Tank {
  constructor(x, y, id) {
    this.x = x;
    this.y = y;
    this.id = id;
    this.brain = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    this.alive = true;
    this.fitness = 0;
    this.angle = 0;
    this.power = 0;
  }
}

let terrainFreq = 0.02;
let terrainAmp = 100;
let trainingPool = Array.from({ length: 50 }, () => new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE));
let globalBestModel = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
let globalBestFitness = -Infinity;

function simulateTraining() {
  let generationBest = -Infinity;
  for (let sim = 0; sim < TRAINING_ITERATIONS; sim++) {
    const simTanks = [
      new Tank(100, 300, 1),
      new Tank(300, 300, 2),
      new Tank(500, 300, 3),
      new Tank(700, 300, 4)
    ];

    simTanks.forEach(tank => {
      tank.brain.copyFrom(trainingPool[Math.floor(Math.random() * trainingPool.length)]);
    });

    for (let round = 0; round < 5; round++) {
      simTanks.forEach(tank => {
        if (!tank.alive) return;
        const inputs = [
          tank.x,
          tank.y,
          ...simTanks.filter(t => t.id !== tank.id).flatMap(t => [t.x, t.y, t.alive ? 1 : 0]),
          terrainFreq,
          terrainAmp
        ];
        const outputs = tank.brain.forward(inputs);
        tank.angle = (outputs[0] - 0.5) * Math.PI;
        tank.power = outputs[1] * MAX_POWER;
        const target = simTanks.find(t => t.id !== tank.id && t.alive);
        if (target) {
          const distance = Math.hypot(target.x - tank.x, target.y - tank.y);
          tank.fitness += Math.max(0, 1000 - distance);
        }
      });
    }

    simTanks.sort((a, b) => b.fitness - a.fitness);
    const bestTank = simTanks[0];
    if (bestTank.fitness > generationBest) generationBest = bestTank.fitness;
    if (bestTank.fitness > globalBestFitness) {
      globalBestFitness = bestTank.fitness;
      globalBestModel.copyFrom(bestTank.brain);
    }
    trainingPool[Math.floor(Math.random() * trainingPool.length)].copyFrom(bestTank.brain);
  }
  return generationBest;
}

function train(generations) {
  for (let g = 0; g < generations; g++) {
    const genBest = simulateTraining();
    trainingPool.forEach(net => net.mutate());
    console.log(
      `Generation ${g + 1}/${generations} - best: ${genBest.toFixed(2)} global best: ${globalBestFitness.toFixed(2)}`
    );
  }
  fs.writeFileSync(
    'trained_net.json',
    JSON.stringify(globalBestModel.toJSON(), null, 2)
  );
  console.log('Saved weights to trained_net.json');
  return globalBestFitness;
}

if (require.main === module) {
  const gens = parseInt(process.argv[2], 10) || 500;
  train(gens);
}

module.exports = {
  train,
  simulateTraining,
  getGlobalBestFitness: () => globalBestFitness
};
