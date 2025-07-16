const { train, getGlobalBestFitness } = require('../train');

test('training improves fitness', () => {
  train(10);
  expect(getGlobalBestFitness()).toBeGreaterThan(0);
});
