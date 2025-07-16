const fs = require('fs');

// delete any existing trained_net.json before test
if (fs.existsSync('trained_net.json')) {
  fs.unlinkSync('trained_net.json');
}

// create minimal canvas for module to initialize
document.body.innerHTML = '<canvas id="gameCanvas" width="1000" height="600"></canvas>';
const canvas = document.getElementById('gameCanvas');
canvas.getContext = jest.fn(() => ({}));

const { trainCLI } = require('../assets/js/game');

test('trainCLI produces positive fitness', () => {
  const fitness = trainCLI(1); // run for 1 generation to keep test quick
  if (fs.existsSync('trained_net.json')) {
    fs.unlinkSync('trained_net.json');
  }
  expect(fitness).toBeGreaterThan(0);
});
