let generateTerrain;
let getTerrainY;
let initGame;

describe('terrain utility', () => {
  beforeAll(() => {
    document.body.innerHTML = `
      <canvas id="gameCanvas" width="1000" height="600"></canvas>
      <div id="stats"></div>
    `;
    const canvas = document.getElementById('gameCanvas');
    canvas.getContext = jest.fn(() => ({}));
    ({ generateTerrain, getTerrainY, initGame } = require('../assets/js/game'));
  });

  beforeEach(() => {
    generateTerrain();
  });

  test('getTerrainY returns canvas height for invalid x', () => {
    expect(getTerrainY(NaN)).toBe(600);
    expect(getTerrainY(Infinity)).toBe(600);
  });

  test('getTerrainY returns canvas height for out of bounds x', () => {
    expect(getTerrainY(-10)).toBe(600);
    expect(getTerrainY(2000)).toBe(600);
  });

  test('initGame runs without throwing', () => {
    expect(() => initGame()).not.toThrow();
  });
});

  describe('Projectile', () => {
  let Tank, Projectile, TANK_WIDTH, TANK_HEIGHT, gameState;

  beforeAll(() => {
    document.body.innerHTML = `
      <canvas id="gameCanvas" width="1000" height="600"></canvas>
      <div id="stats"></div>
    `;
    const canvas = document.getElementById('gameCanvas');
    canvas.getContext = jest.fn(() => ({}));
    const gameModule = require('../assets/js/game');
    ({ generateTerrain, getTerrainY, initGame, Tank, Projectile, gameState, TANK_WIDTH, TANK_HEIGHT } = gameModule);
  });

  beforeEach(() => {
    generateTerrain();
    gameState.tanks = [];
    gameState.projectiles = [];
    const tankY = 500; // Fixed Y coordinate for the tank
    const tank = new Tank(100, tankY, '#ff0000', 1);
    gameState.tanks.push(tank);
  });

  test('explosion 5px next to a tank harms the tank', () => {
    const tankInGameState = gameState.tanks[0];
    const initialHealth = tankInGameState.health;

    const projectileX = tankInGameState.x + TANK_WIDTH + 60; // 60px to the right of the tank's right edge
    const projectileY = tankInGameState.y; // Same Y coordinate as the tank
    const projectile = new Projectile(projectileX, projectileY, 0, 0, 100, 2); // Explode 5px from tank center
    projectile.explode();

    expect(tankInGameState.health).toBeLessThan(initialHealth);
  });
});
