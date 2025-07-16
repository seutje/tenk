const { evolve } = require('./assets/js/game');

if (require.main === module) {
  const trained = evolve();
  const fs = require('fs');
  fs.writeFileSync('trained_net.json', JSON.stringify(trained, null, 2));
  console.log('Training complete. Network saved to trained_net.json');
}
