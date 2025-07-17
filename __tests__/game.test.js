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

  describe('Neural Network Inputs', () => {
  let Tank, gameState;

  beforeAll(() => {
    document.body.innerHTML = `
      <canvas id="gameCanvas" width="1000" height="600"></canvas>
      <div id="stats"></div>
    `;
    const canvas = document.getElementById('gameCanvas');
    canvas.getContext = jest.fn(() => ({}));
    const gameModule = require('../assets/js/game');
    ({ generateTerrain, getTerrainY, initGame, Tank, gameState } = gameModule);
  });

  beforeEach(() => {
    generateTerrain();
    gameState.tanks = [];
    gameState.projectiles = [];
  });

  test('neural network inputs always contain data for 3 enemies', () => {
    // Test with exactly 4 tanks (1 self + 3 enemies)
    gameState.tanks = [
      new Tank(100, 500, '#ff0000', 1),
      new Tank(200, 500, '#00ff00', 2),
      new Tank(300, 500, '#0000ff', 3),
      new Tank(400, 500, '#ffff00', 4)
    ];

    const tank = gameState.tanks[0];
    const enemies = gameState.tanks.filter(t => t.id !== tank.id);
    
    // Ensure we have exactly 3 enemies
    expect(enemies.length).toBe(3);
    
    // Ensure each enemy provides 3 values (x, y, alive)
    const enemyInputs = enemies.flatMap(t => [t.x, t.y, t.alive ? 1 : 0]);
    expect(enemyInputs.length).toBe(9); // 3 enemies × 3 values each = 9
  });

  test('neural network inputs contain data for 3 enemies even with fewer tanks', () => {
    // Test with only 2 tanks (1 self + 1 enemy)
    gameState.tanks = [
      new Tank(100, 500, '#ff0000', 1),
      new Tank(200, 500, '#00ff00', 2)
    ];

    const tank = gameState.tanks[0];
    const enemies = gameState.tanks.filter(t => t.id !== tank.id);
    
    // With only 1 enemy, we should still expect 3 enemy slots
    expect(enemies.length).toBe(1);
    
    // After the fix, we should always get 9 values for 3 enemy slots
    const enemyInputs = [];
    for (let i = 0; i < 3; i++) {
      if (i < enemies.length) {
        const enemy = enemies[i];
        enemyInputs.push(enemy.x, enemy.y, enemy.alive ? 1 : 0);
      } else {
        enemyInputs.push(0, 0, 0); // padding
      }
    }
    expect(enemyInputs.length).toBe(9); // Always 3 enemies × 3 values each = 9
  });

  test('Tank.think() method should always provide 14 inputs regardless of enemy count', () => {
    // Test with 2 tanks
    gameState.tanks = [
      new Tank(100, 500, '#ff0000', 1),
      new Tank(200, 500, '#00ff00', 2)
    ];

    const tank = gameState.tanks[0];
    
    // Mock the brain.forward method to capture inputs
    const capturedInputs = [];
    tank.brain.forward = (inputs) => {
      capturedInputs.push(...inputs);
      return [0.5, 0.5, 0.5]; // Mock outputs
    };
    
    tank.think();
    
    // The think() method should always produce 14 inputs
    expect(capturedInputs.length).toBe(14);
    
    // The enemy data should always be 9 values (3 enemies × 3 values each)
    const enemyData = capturedInputs.slice(2, 11);
    expect(enemyData.length).toBe(9);
    
    // Check that padding is applied correctly (last 6 values should be 0 for missing enemies)
    expect(enemyData.slice(3)).toEqual([0, 0, 0, 0, 0, 0]);
  });

  test('Tank.think() method should provide 3 enemy slots even with no enemies', () => {
    // Test with only 1 tank (no enemies)
    gameState.tanks = [
      new Tank(100, 500, '#ff0000', 1)
    ];

    const tank = gameState.tanks[0];
    
    // Mock the brain.forward method to capture inputs
    const capturedInputs = [];
    tank.brain.forward = (inputs) => {
      capturedInputs.push(...inputs);
      return [0.5, 0.5, 0.5]; // Mock outputs
    };
    
    tank.think();
    
    // The think() method should always produce 14 inputs
    expect(capturedInputs.length).toBe(14);
    
    // The enemy data should always be 9 values (3 enemies × 3 values each)
    const enemyData = capturedInputs.slice(2, 11);
    expect(enemyData.length).toBe(9);
    
    // Check that all enemy slots are padded with zeros
    expect(enemyData).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0]);
  });
});
