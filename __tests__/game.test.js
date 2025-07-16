const { WEAPONS, simulateShot } = require('../assets/js/game');

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
    hiddenWeights: Array.from({ length: 3 }, () => Array(5).fill(0)),
    hiddenBias: Array(3).fill(0),
    outputWeights: Array.from({ length: 7 }, () => Array(3).fill(0)),
    outputBias: [1, -1, 0, 0, 0, 0, 0],
  };
  const score = simulateShot(net, 300);
  expect(score).toBeLessThan(0);
});
