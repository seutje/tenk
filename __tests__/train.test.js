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

test('trainCLI improves fitness over generations', () => {
  const history = trainCLI(10); // run full 10 generations
  if (fs.existsSync('trained_net.json')) {
    fs.unlinkSync('trained_net.json');
  }
  expect(history[history.length - 1]).toBeGreaterThan(history[0]);
});
