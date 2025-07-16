const { WEAPONS } = require('../assets/js/game');

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
