# TenK

This project contains a small web game. It now includes a Node.js setup with Jest for running tests.

## Development

Install dependencies and run tests:

```bash
npm install
npm test
```

## Training AI Tanks

To train AI tanks using reinforcement learning, run:

```bash
node train.js
```

The script evolves neural networks and saves the best network to `trained_net.json`.
The game will automatically load this file for AI tanks if it exists.

### Wind

Each round features a random wind value that influences projectile paths. The AI
tanks receive the current wind as a sensor input and the training simulation
accounts for it as well.

