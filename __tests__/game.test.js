const { WEAPONS, simulateShot, HIDDEN_WEIGHTS, INPUT_SIZE } = require('../assets/js/game');

test('weapon configuration includes mega blast damage', () => {
  expect(WEAPONS.mega.damage).toBe(50);
});

test('weapon keys', () => {
  expect(Object.keys(WEAPONS)).toEqual([
    'standard',
    'rapid',
    'triple',
    'bouncy',
    'mega',
  ]);
});

test('simulateShot penalizes self hits', () => {
  const net = {
    hiddenWeights: Array.from({ length: HIDDEN_WEIGHTS }, () => Array(INPUT_SIZE).fill(0)),
    hiddenBias: Array(HIDDEN_WEIGHTS).fill(0),
    outputWeights: Array.from({ length: 7 }, () => Array(HIDDEN_WEIGHTS).fill(0)),
    outputBias: [1, -1, 0, 0, 0, 0, 0],
  };
  const score = simulateShot(net, 300);
  expect(score).toBeLessThan(0);
});
