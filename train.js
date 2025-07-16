const { trainCLI } = require('./assets/js/game');

const gens = parseInt(process.argv[2], 10) || 10;
trainCLI(gens);
