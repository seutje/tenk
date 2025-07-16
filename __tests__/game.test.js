let generateTerrain;
let getTerrainY;

describe('terrain utility', () => {
  beforeAll(() => {
    document.body.innerHTML = `
      <canvas id="gameCanvas" width="1000" height="600"></canvas>
      <div id="stats"></div>
    `;
    ({ generateTerrain, getTerrainY } = require('../assets/js/game'));
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
});
