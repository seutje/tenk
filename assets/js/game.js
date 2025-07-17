let canvas;
let ctx;

if (typeof document !== 'undefined') {
    canvas = document.getElementById('gameCanvas');
    ctx = canvas.getContext('2d');
} else {
    canvas = { width: 1000, height: 600 };
    ctx = {};
}

// Game state
let gameSpeed = 1;
let isPaused = false;
let generation = 1;
let frameCount = 0;
let isTraining = false;

// Neural Network parameters
const INPUT_SIZE = 13;
const HIDDEN_SIZE = 20;
const OUTPUT_SIZE = 3;
const LEARNING_RATE = 0.1;
const MUTATION_RATE = 0.1;

// Terrain parameters
let terrainFreq = 0.02;
let terrainAmp = 100;
const TERRAIN_POINTS = 200;
let terrain = [];

// Tank parameters
const TANK_WIDTH = 40;
const TANK_HEIGHT = 25;
const GRAVITY = 0.2;
const MAX_POWER = 20;

// Game objects
const gameState = {
    tanks: [],
    projectiles: [],
    particles: [],
    splashVisuals: []
};

// Neural network storage
let globalBestModel = null;
let globalBestFitness = -Infinity;
let trainingPool = [];
const TRAINING_ITERATIONS = 500; // increased from 10 for faster background training

function loadTrainedNet() {
    if (typeof fetch === 'function') {
        fetch('trained_net.json')
            .then(res => (res.ok ? res.json() : null))
            .then(data => {
                if (data) globalBestModel = NeuralNetwork.fromJSON(data);
            })
            .catch(() => {});
    } else if (typeof require !== 'undefined') {
        try {
            const fs = require('fs');
            if (fs.existsSync('trained_net.json')) {
                const data = JSON.parse(fs.readFileSync('trained_net.json', 'utf8'));
                globalBestModel = NeuralNetwork.fromJSON(data);
            }
        } catch (e) {}
    }
}

loadTrainedNet();

// Initialize terrain
function generateTerrain() {
    terrain = [];
    terrainFreq = 0.015 + Math.random() * 0.02;
    terrainAmp = 80 + Math.random() * 120;

    for (let i = 0; i <= TERRAIN_POINTS; i++) {
        const x = (i / TERRAIN_POINTS) * canvas.width;
        const y = canvas.height - 60 - Math.sin(x * terrainFreq) * terrainAmp;
        terrain.push({x, y});
    }

    // Raise terrain if any points fall below 5px from the bottom of the canvas
    const maxY = Math.max(...terrain.map(p => p.y));
    if (maxY > canvas.height - 5) {
        const shift = maxY - (canvas.height - 5);
        for (const point of terrain) {
            point.y -= shift;
        }
    }
}

// Neural Network class
class NeuralNetwork {
    constructor(inputSize, hiddenSize, outputSize) {
        this.weights1 = this.randomMatrix(inputSize, hiddenSize);
        this.bias1 = this.randomMatrix(1, hiddenSize);
        this.weights2 = this.randomMatrix(hiddenSize, outputSize);
        this.bias2 = this.randomMatrix(1, outputSize);
    }
    
    randomMatrix(rows, cols) {
        return Array(rows).fill().map(() => 
            Array(cols).fill().map(() => (Math.random() - 0.5) * 2)
        );
    }
    
    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }
    
    forward(inputs) {
        // Normalize inputs
        const normalizedInputs = inputs.map((val, idx) => {
            if (idx < 12) return val / canvas.width; // Positions
            if (idx === 12) return val / 0.035; // Frequency, corrected max
            return val / 200; // Amplitude
        });

        // Hidden layer
        const hidden = new Array(HIDDEN_SIZE).fill(0).map((_, i) => {
            const sum = normalizedInputs.reduce((acc, input, j) => acc + input * this.weights1[j][i], 0);
            return this.sigmoid(sum + this.bias1[0][i]);
        });

        // Output layer
        const outputs = new Array(OUTPUT_SIZE).fill(0).map((_, i) => {
            const sum = hidden.reduce((acc, h, j) => acc + h * this.weights2[j][i], 0);
            return this.sigmoid(sum + this.bias2[0][i]);
        });

        return outputs;
    }
    
    mutate(rate = MUTATION_RATE) {
        const mutate = (val) => Math.random() < rate ? val + (Math.random() - 0.5) * 0.5 : val;
        
        this.weights1 = this.weights1.map(row => row.map(mutate));
        this.bias1 = this.bias1.map(row => row.map(mutate));
        this.weights2 = this.weights2.map(row => row.map(mutate));
        this.bias2 = this.bias2.map(row => row.map(mutate));
    }
    
    copyFrom(other) {
        this.weights1 = JSON.parse(JSON.stringify(other.weights1));
        this.bias1 = JSON.parse(JSON.stringify(other.bias1));
        this.weights2 = JSON.parse(JSON.stringify(other.weights2));
        this.bias2 = JSON.parse(JSON.stringify(other.bias2));
    }

    toJSON() {
        return {
            weights1: this.weights1,
            bias1: this.bias1,
            weights2: this.weights2,
            bias2: this.bias2
        };
    }

    static fromJSON(data) {
        const nn = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        nn.weights1 = data.weights1;
        nn.bias1 = data.bias1;
        nn.weights2 = data.weights2;
        nn.bias2 = data.bias2;
        return nn;
    }
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

// Tank class
class Tank {
    constructor(x, y, color, id) {
        this.x = x;
        this.y = y;
        this.color = color;
        this.id = id;
        this.health = 100;
        this.alive = true;
        this.angle = -Math.PI / 4;
        this.power = 10;
        this.weapon = 0; // 0 = shell, 1 = shotgun
        this.brain = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        this.fitness = 0;
        this.lastDamage = 0;
        this.shotsFired = 0;
    }
    
    updateBrain() {
        if (globalBestModel) {
            this.brain.copyFrom(globalBestModel);
            this.brain.mutate();
        }
    }
    
    think() {
        if (!this.alive) return;
        
        const inputs = [
            this.x,
            this.y,
            ...gameState.tanks.filter(t => t.id !== this.id).flatMap(t => [t.x, t.y, t.alive ? 1 : 0]),
            terrainFreq,
            terrainAmp
        ];
        
        const outputs = this.brain.forward(inputs);
        
        this.angle = (outputs[0] * 2 - 1) * Math.PI;
        this.power = outputs[1] * MAX_POWER;
        this.weapon = outputs[2] > 0.5 ? 1 : 0;
    }
    
    fire() {
        if (!this.alive) return;
        
        this.shotsFired++;
        const velocity = {
            x: Math.cos(this.angle) * this.power,
            y: Math.sin(this.angle) * this.power
        };

        const barrelTipX = this.x + TANK_WIDTH / 2 + Math.cos(this.angle) * 31;
        const barrelTipY = this.y - TANK_HEIGHT / 2 + Math.sin(this.angle) * 31;
        
        if (this.weapon === 0) {
            // Standard shell
            gameState.projectiles.push(new Projectile(
                barrelTipX,
                barrelTipY,
                velocity.x,
                velocity.y,
                15,
                this.id
            ));
        } else {
            // Shotgun
            const spread = 0.2;
            for (let i = -1; i <= 1; i++) {
                const spreadAngle = this.angle + i * spread;
                const spreadVelocity = {
                    x: Math.cos(spreadAngle) * this.power,
                    y: Math.sin(spreadAngle) * this.power
                };
                gameState.projectiles.push(new Projectile(
                    barrelTipX,
                    barrelTipY,
                    spreadVelocity.x,
                    spreadVelocity.y,
                    5,
                    this.id
                ));
            }
        }
    }
    
    takeDamage(amount, sourceId) {
        this.health = Math.round(this.health - amount);
        this.lastDamage = amount;
        
        if (sourceId === this.id) {
            this.fitness -= amount * 2; // Penalty for self-damage
        } else {
            this.fitness += amount; // Reward when damaged by others
            const source = gameState.tanks.find(t => t.id === sourceId);
            if (source) {
                source.fitness += amount * 1.5; // Bonus for dealing damage
            }
        }
        
        if (this.health <= 0) {
            this.health = 0;
            this.alive = false;
        }
        
        // Add damage particles
        if (typeof requestAnimationFrame === 'function' && !isTraining) {
            for (let i = 0; i < 5; i++) {
                gameState.particles.push(new Particle(
                    this.x + TANK_WIDTH/2 + (Math.random() - 0.5) * TANK_WIDTH,
                    this.y - TANK_HEIGHT/2,
                    (Math.random() - 0.5) * 5,
                    -Math.random() * 5,
                    '#ff4444'
                ));
            }
        }
    }
    
    draw() {
        if (!this.alive) return;
        
        ctx.save();
        ctx.translate(this.x + TANK_WIDTH/2, this.y - TANK_HEIGHT/2);
        
        // Tank body
        ctx.fillStyle = this.color;
        ctx.fillRect(-TANK_WIDTH/2, -TANK_HEIGHT/2, TANK_WIDTH, TANK_HEIGHT);
        
        // Tank barrel
        ctx.strokeStyle = '#333';
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(Math.cos(this.angle) * 30, Math.sin(this.angle) * 30);
        ctx.stroke();
        
        // Health bar
        ctx.fillStyle = '#333';
        ctx.fillRect(-TANK_WIDTH/2, -TANK_HEIGHT/2 - 15, TANK_WIDTH, 8);
        ctx.fillStyle = this.health > 50 ? '#00ff00' : this.health > 25 ? '#ffff00' : '#ff0000';
        ctx.fillRect(-TANK_WIDTH/2, -TANK_HEIGHT/2 - 15, TANK_WIDTH * (this.health / 100), 8);
        
        ctx.restore();
        
        // Draw tank ID
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Tank ${this.id}`, this.x + TANK_WIDTH/2, this.y + 15);
    }
}

// Projectile class
function getTerrainY(x) {
    if (!Number.isFinite(x)) {
        return canvas.height;
    }

    if (x < 0 || x > canvas.width) return canvas.height;

    const index = Math.floor((x / canvas.width) * TERRAIN_POINTS);
    if (index < 0 || index >= terrain.length) return canvas.height;
    return terrain[index].y;
}

// Projectile class
class Projectile {
    constructor(x, y, vx, vy, damage, ownerId) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.damage = damage;
        this.ownerId = ownerId;
        this.active = true;
    }
    
    update() {
        if (!this.active) return;

        const prevX = this.x;
        const prevY = this.y;

        this.vy += GRAVITY;
        const nextX = this.x + this.vx;
        const nextY = this.y + this.vy;

        const steps = Math.max(1, Math.ceil(Math.max(Math.abs(nextX - prevX), Math.abs(nextY - prevY))));

        for (let i = 1; i <= steps && this.active; i++) {
            const t = i / steps;
            const stepX = prevX + (nextX - prevX) * t;
            const stepY = prevY + (nextY - prevY) * t;

            const terrainY = getTerrainY(stepX);
            if (stepY >= terrainY) {
                this.x = stepX;
                this.y = stepY;
                this.explode();
                this.destroyTerrain(stepX, terrainY);
                this.active = false;
                break;
            }

            for (const tank of gameState.tanks) {
                if (tank.alive &&
                    stepX >= tank.x && stepX <= tank.x + TANK_WIDTH &&
                    stepY >= tank.y - TANK_HEIGHT && stepY <= tank.y) {
                    tank.takeDamage(this.damage, this.ownerId);
                    this.x = stepX;
                    this.y = stepY;
                    this.explode();
                    this.active = false;
                    break;
                }
            }
        }

        if (!this.active) return;

        this.x = nextX;
        this.y = nextY;

        // Check bounds
        if (this.x < 0 || this.x > canvas.width || this.y > canvas.height) {
            this.active = false;
        }
    }
    
    destroyTerrain(x, y) {
        const destroyRadius = 50 + this.damage / 5;
        
        for (let i = 0; i < terrain.length; i++) {
            const distance = Math.sqrt(
                Math.pow(terrain[i].x - x, 2) + 
                Math.pow(terrain[i].y - y, 2)
            );
            
            if (distance < destroyRadius) {
                const newY = terrain[i].y + (destroyRadius - distance);
                terrain[i].y = Math.min(newY, canvas.height - 5);
            }
        }
        
        // Update tank positions based on new terrain
        gameState.tanks.forEach(tank => {
            if (tank.alive) {
                tank.y = getTerrainY(tank.x + TANK_WIDTH/2);
            }
        });
        
        // Add explosion particles
        if (typeof requestAnimationFrame === 'function' && !isTraining) {
            for (let i = 0; i < 10; i++) {
                gameState.particles.push(new Particle(
                    x + (Math.random() - 0.5) * destroyRadius * 2,
                    y + (Math.random() - 0.5) * destroyRadius * 2,
                    (Math.random() - 0.5) * 10,
                    -Math.random() * 10,
                    '#ff6600'
                ));
            }
        }
    }
    
    explode() {
        // Create explosion effect
        const explosion = {
            x: this.x,
            y: this.y,
            radius: 0,
            maxRadius: 70 + this.damage / 5,
            alpha: 1
        };

        // Apply splash damage
        gameState.tanks.forEach(tank => {
            if (tank.alive) {
                const tankCenterX = tank.x + TANK_WIDTH / 2;
                const tankCenterY = tank.y - TANK_HEIGHT / 2;
                const distance = Math.sqrt(Math.pow(tankCenterX - explosion.x, 2) + Math.pow(tankCenterY - explosion.y, 2));
                if (distance < explosion.maxRadius) {
                    const damage = (1 - distance / explosion.maxRadius) * this.damage;
                    tank.takeDamage(damage, this.ownerId);
                }
            }
        });
        
        if (typeof requestAnimationFrame === 'function' && !isTraining) {
            gameState.splashVisuals.push(new SplashDamageVisual(explosion.x, explosion.y, explosion.maxRadius));
        }
    }
    
    draw() {
        const elapsed = Date.now() - this.startTime;
        const progress = elapsed / this.duration;

        if (progress >= 1) {
            return false; // Indicate that this visual is done
        }

        const currentRadius = this.maxRadius * progress;
        const alpha = 1 - progress; // Fade out

        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = 'rgba(255, 69, 0, 0.7)'; // Fire color (OrangeRed with some transparency)
        ctx.beginPath();
        ctx.arc(this.x, this.y, currentRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return true; // Indicate that this visual is still active
    }
    
    draw() {
        if (!this.active) return;
        
        ctx.fillStyle = '#333';
        ctx.beginPath();
        ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
        ctx.fill();
    }
}

// New class for splash damage visualization
class SplashDamageVisual {
    constructor(x, y, maxRadius) {
        this.x = x;
        this.y = y;
        this.maxRadius = maxRadius;
        this.startTime = Date.now();
        this.duration = 1000; // 1 second
    }

    draw() {
        const elapsed = Date.now() - this.startTime;
        const progress = elapsed / this.duration;

        if (progress >= 1) {
            return false; // Indicate that this visual is done
        }

        const currentRadius = this.maxRadius * progress;
        const alpha = 1 - progress; // Fade out

        ctx.save();
        ctx.globalAlpha = alpha;
        ctx.fillStyle = 'rgba(255, 69, 0, 0.7)'; // Fire color (OrangeRed with some transparency)
        ctx.beginPath();
        ctx.arc(this.x, this.y, currentRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
        return true; // Indicate that this visual is still active
    }
}

// Particle class for effects
class Particle {
    constructor(x, y, vx, vy, color) {
        this.x = x;
        this.y = y;
        this.vx = vx;
        this.vy = vy;
        this.color = color;
        this.life = 30;
        this.maxLife = 30;
    }
    
    update() {
        this.x += this.vx;
        this.y += this.vy;
        this.vy += 0.1;
        this.life--;
        
        if (this.life <= 0) {
            const index = gameState.particles.indexOf(this);
            if (index > -1) gameState.particles.splice(index, 1);
        }
    }
    
    draw() {
        ctx.save();
        ctx.globalAlpha = this.life / this.maxLife;
        ctx.fillStyle = this.color;
        ctx.beginPath();
        ctx.arc(this.x, this.y, 2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }
}

// Initialize game
function initGame() {
    generateTerrain();
    gameState.tanks = [];
    gameState.projectiles = [];
    gameState.particles = [];
    gameState.splashVisuals = [];
    
    const colors = ['#ff4444', '#44ff44', '#4444ff', '#ffff44'];
    const minDistance = 100;
    const maxAttempts = 100;

    for (let i = 0; i < 4; i++) {
        let x, y;
        let placed = false;
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            x = Math.random() * (canvas.width - TANK_WIDTH);
            y = getTerrainY(x + TANK_WIDTH / 2);

            let tooClose = false;
            for (const existingTank of gameState.tanks) {
                if (Math.abs(x - existingTank.x) < minDistance) {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose) {
                placed = true;
                break;
            }
        }

        if (!placed) {
            // Fallback to default positioning if random placement fails after maxAttempts
            x = (i + 1) * canvas.width / 5 - TANK_WIDTH / 2;
            y = getTerrainY(x + TANK_WIDTH / 2);
        }
        
        gameState.tanks.push(new Tank(x, y, colors[i], i + 1));
        gameState.tanks[i].updateBrain();
    }
    
    // Initialize training pool
    trainingPool = Array(50).fill().map(() => new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE));
    
    updateStats();
}

// Training simulation
function simulateTraining(isCLI = false) {
    isTraining = !isCLI;

    if (!globalBestModel) {
        globalBestModel = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }

    let generationBestFitness = -Infinity;
    let generationBestBrain = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);

    const fitnessScores = trainingPool.map(brain => {
        let totalFitness = 0;
        const evaluationRounds = 5;

        for (let i = 0; i < evaluationRounds; i++) {
            // Store original game state
            const originalTanks = gameState.tanks;
            const originalProjectiles = gameState.projectiles;
            const originalTerrain = terrain;

            // Setup simulation environment
            generateTerrain();
            const simTanks = [];
            const colors = ['#ff4444', '#44ff44', '#4444ff', '#ffff44'];
            const minDistance = 100; // Define minDistance here
            for (let j = 0; j < 4; j++) {
                let x, y;
                let placed = false;
                for (let attempt = 0; attempt < 100; attempt++) {
                    x = Math.random() * (canvas.width - TANK_WIDTH);
                    y = getTerrainY(x + TANK_WIDTH / 2);

                    let tooClose = false;
                    for (const existingTank of simTanks) {
                        if (Math.abs(x - existingTank.x) < minDistance) {
                            tooClose = true;
                            break;
                        }
                    }

                    if (!tooClose) {
                        placed = true;
                        break;
                    }
                }

                if (!placed) {
                    // Fallback to default positioning if random placement fails after maxAttempts
                    x = (j + 1) * canvas.width / 5 - TANK_WIDTH / 2;
                    y = getTerrainY(x + TANK_WIDTH / 2);
                }
                simTanks.push(new Tank(x, y, colors[j], j + 1));
            }

            const mainTank = simTanks[0];
            mainTank.brain.copyFrom(brain);

            simTanks.slice(1).forEach(otherTank => {
                const randomBrain = trainingPool[Math.floor(Math.random() * trainingPool.length)];
                otherTank.brain.copyFrom(randomBrain);
            });

            gameState.tanks = simTanks;
            gameState.projectiles = [];

            // Run simulation for a fixed number of frames or until the game ends
            const maxFrames = 1800; // 30 seconds
            for (let frame = 0; frame < maxFrames; frame++) {
                if (frame % 60 === 0) {
                    shuffleArray(gameState.tanks);
                    gameState.tanks.forEach(tank => tank.think());
                    gameState.tanks.forEach(tank => tank.fire());
                }

                gameState.projectiles.forEach(p => p.update());
                gameState.projectiles = gameState.projectiles.filter(p => p.active);

                const aliveTanks = gameState.tanks.filter(t => t.alive);
                if (aliveTanks.length <= 1) {
                    // Award bonus for survival
                    aliveTanks.forEach(tank => tank.fitness += 50);
                    break;
                }
            }
            
            totalFitness += mainTank.fitness;

            // Restore original game state
            gameState.tanks = originalTanks;
            gameState.projectiles = originalProjectiles;
            terrain = originalTerrain;
        }
        
        return { fitness: totalFitness / evaluationRounds, brain };
    });

    fitnessScores.sort((a, b) => b.fitness - a.fitness);

    if (fitnessScores.length > 0 && fitnessScores[0].fitness > generationBestFitness) {
        generationBestFitness = fitnessScores[0].fitness;
        generationBestBrain.copyFrom(fitnessScores[0].brain);
    }

    if (generationBestFitness > globalBestFitness) {
        globalBestFitness = generationBestFitness;
        globalBestModel.copyFrom(generationBestBrain);
    }
    
    isTraining = false;
    return { bestFitness: generationBestFitness, bestBrain: generationBestBrain };
}""

function trainCLI(generations = 10) {
    if (!globalBestModel) {
        globalBestModel = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    }

    if (trainingPool.length === 0) {
        trainingPool = Array(50).fill().map(() => new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE));
    }

    for (let g = 0; g < generations; g++) {
        const { bestFitness, bestBrain } = simulateTraining(true);

        if (bestFitness > globalBestFitness) {
            globalBestFitness = bestFitness;
            globalBestModel.copyFrom(bestBrain);
        }

        const newPool = [];
        // Elitism: Keep the best model from the last generation.
        const eliteBrain = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        eliteBrain.copyFrom(globalBestModel);
        newPool.push(eliteBrain);

        // Create the rest of the new generation by mutating the best model.
        while (newPool.length < trainingPool.length) {
            const newNet = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
            newNet.copyFrom(globalBestModel);
            newNet.mutate(MUTATION_RATE);
            newPool.push(newNet);
        }
        trainingPool = newPool;

        console.log(
            `Generation ${g + 1}/${generations} - Best Fitness: ${bestFitness.toFixed(2)}, Global Best: ${globalBestFitness.toFixed(2)}`
        );
    }

    if (typeof require !== 'undefined') {
        const fs = require('fs');
        fs.writeFileSync('trained_net.json', JSON.stringify(globalBestModel.toJSON(), null, 2));
        console.log('Saved weights to trained_net.json');
    }
}

// Game loop
function gameLoop() {
    if (!isPaused) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw terrain
        ctx.fillStyle = '#8B4513';
        ctx.beginPath();
        ctx.moveTo(0, canvas.height);
        terrain.forEach(point => ctx.lineTo(point.x, point.y));
        ctx.lineTo(canvas.width, canvas.height);
        ctx.closePath();
        ctx.fill();
        
        // Update and draw particles
        gameState.particles.forEach(p => {
            p.update();
            p.draw();
        });

        // Update and draw splash visuals
        gameState.splashVisuals = gameState.splashVisuals.filter(visual => visual.draw());
        
        // Tank decision phase every 60 frames
        if (frameCount % 60 === 0) {
            gameState.tanks.forEach(tank => tank.think());
            
            // All tanks fire at once
            gameState.tanks.forEach(tank => tank.fire());
        }
        
        // Update and draw projectiles
        gameState.projectiles.forEach(p => {
            p.update();
            p.draw();
        });
        
        // Remove inactive projectiles
        gameState.projectiles = gameState.projectiles.filter(p => p.active);
        
        // Draw tanks
        gameState.tanks.forEach(tank => tank.draw());
        
        // Update stats
        if (frameCount % 30 === 0) {
            updateStats();
        }
        
        // Training simulation
        if (frameCount % 300 === 0) {
            simulateTraining(false);
            gameState.tanks.forEach(tank => tank.updateBrain());
            generation++;
        }
        
        // Check for game end
        const aliveTanks = gameState.tanks.filter(t => t.alive);
        if (aliveTanks.length <= 1 && frameCount > 60) {
            resetGame();
        }
        
        frameCount++;
    }
    
    requestAnimationFrame(gameLoop);
}

function updateStats() {
    const statsDiv = document.getElementById('stats');
    statsDiv.innerHTML = '';
    
    gameState.tanks.forEach(tank => {
        const div = document.createElement('div');
        div.className = 'tank-stats';
        div.innerHTML = `
            <h3 style="color: ${tank.color}">Tank ${tank.id}</h3>
            <p>Health: ${tank.health}</p>
            <p>Fitness: ${tank.fitness.toFixed(1)}</p>
            <p>Weapon: ${tank.weapon === 0 ? 'Shell' : 'Shotgun'}</p>
            <p>Status: ${tank.alive ? 'Alive' : 'Destroyed'}</p>
        `;
        statsDiv.appendChild(div);
    });
    
    const infoDiv = document.createElement('div');
    infoDiv.className = 'tank-stats';
    infoDiv.innerHTML = `
        <h3>Game Info</h3>
        <p>Generation: ${generation}</p>
        <p>Frame: ${frameCount}</p>
        <p>Terrain Freq: ${terrainFreq.toFixed(3)}</p>
        <p>Terrain Amp: ${terrainAmp.toFixed(0)}</p>
    `;
    statsDiv.appendChild(infoDiv);
}

function togglePause() {
    isPaused = !isPaused;
}

function resetGame() {
    frameCount = 0;
    initGame();
}

function toggleSpeed() {
    gameSpeed = gameSpeed === 1 ? 3 : gameSpeed === 3 ? 0.5 : 1;
    document.getElementById('speedDisplay').textContent = gameSpeed + 'x';
}

// Start game in browser environment
if (
    typeof window !== 'undefined' &&
    !(typeof process !== 'undefined' && process.env && process.env.JEST_WORKER_ID)
) {
    initGame();
    gameLoop();
}

// Export functions for testing in Node environment
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        generateTerrain,
        getTerrainY,
        initGame,
        trainCLI,
        Tank,
        Projectile,
        gameState,
        TANK_WIDTH,
        TANK_HEIGHT
    };
}
