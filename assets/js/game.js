const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Game state
let gameSpeed = 1;
let isPaused = false;
let generation = 1;
let frameCount = 0;

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
let tanks = [];
let projectiles = [];
let particles = [];

// Neural network storage
let globalBestModel = null;
let trainingPool = [];

// Initialize terrain
function generateTerrain() {
    terrain = [];
    terrainFreq = 0.015 + Math.random() * 0.02;
    terrainAmp = 80 + Math.random() * 120;
    
    for (let i = 0; i <= TERRAIN_POINTS; i++) {
        const x = (i / TERRAIN_POINTS) * canvas.width;
        const y = canvas.height - 50 - Math.sin(x * terrainFreq) * terrainAmp;
        terrain.push({x, y});
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
        inputs = inputs.map((val, idx) => {
            if (idx < 12) return val / canvas.width; // Positions
            if (idx === 12) return val / 0.03; // Frequency
            return val / 200; // Amplitude
        });
        
        // Hidden layer
        let hidden = inputs.map((_, i) => 
            this.weights1[i].reduce((sum, w, j) => sum + w * inputs[j], 0) + this.bias1[0][i]
        ).map(this.sigmoid);
        
        // Output layer
        let outputs = this.weights2[0].map((_, j) =>
            hidden.reduce((sum, h, i) => sum + h * this.weights2[i][j], 0) + this.bias2[0][j]
        ).map(this.sigmoid);
        
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
            ...tanks.filter(t => t.id !== this.id).flatMap(t => [t.x, t.y, t.alive ? 1 : 0]),
            terrainFreq,
            terrainAmp
        ];
        
        const outputs = this.brain.forward(inputs);
        
        this.angle = (outputs[0] - 0.5) * Math.PI;
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
        
        if (this.weapon === 0) {
            // Standard shell
            projectiles.push(new Projectile(
                this.x + TANK_WIDTH/2,
                this.y - TANK_HEIGHT,
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
                projectiles.push(new Projectile(
                    this.x + TANK_WIDTH/2,
                    this.y - TANK_HEIGHT,
                    spreadVelocity.x,
                    spreadVelocity.y,
                    5,
                    this.id
                ));
            }
        }
    }
    
    takeDamage(amount, sourceId) {
        this.health -= amount;
        this.lastDamage = amount;
        
        if (sourceId === this.id) {
            this.fitness -= amount * 2; // Penalty for self-damage
        } else {
            this.fitness += amount; // Reward when damaged by others
            const source = tanks.find(t => t.id === sourceId);
            if (source) {
                source.fitness += amount * 1.5; // Bonus for dealing damage
            }
        }
        
        if (this.health <= 0) {
            this.health = 0;
            this.alive = false;
        }
        
        // Add damage particles
        for (let i = 0; i < 5; i++) {
            particles.push(new Particle(
                this.x + TANK_WIDTH/2 + (Math.random() - 0.5) * TANK_WIDTH,
                this.y - TANK_HEIGHT/2,
                (Math.random() - 0.5) * 5,
                -Math.random() * 5,
                '#ff4444'
            ));
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
        
        this.x += this.vx;
        this.y += this.vy;
        this.vy += GRAVITY;
        
        // Check terrain collision
        const terrainY = this.getTerrainY(this.x);
        if (this.y >= terrainY) {
            this.explode();
            this.destroyTerrain(this.x, terrainY);
        }
        
        // Check bounds
        if (this.x < 0 || this.x > canvas.width || this.y > canvas.height) {
            this.active = false;
        }
        
        // Check tank collision
        tanks.forEach(tank => {
            if (tank.alive && this.active &&
                this.x >= tank.x && this.x <= tank.x + TANK_WIDTH &&
                this.y >= tank.y - TANK_HEIGHT && this.y <= tank.y) {
                tank.takeDamage(this.damage, this.ownerId);
                this.explode();
                this.active = false;
            }
        });
    }
    
    getTerrainY(x) {
        const index = Math.floor((x / canvas.width) * TERRAIN_POINTS);
        if (index < 0 || index >= terrain.length) return canvas.height;
        return terrain[index].y;
    }
    
    destroyTerrain(x, y) {
        const destroyRadius = 20 + this.damage / 5;
        
        for (let i = 0; i < terrain.length; i++) {
            const distance = Math.sqrt(
                Math.pow(terrain[i].x - x, 2) + 
                Math.pow(terrain[i].y - y, 2)
            );
            
            if (distance < destroyRadius) {
                terrain[i].y += destroyRadius - distance;
            }
        }
        
        // Update tank positions based on new terrain
        tanks.forEach(tank => {
            if (tank.alive) {
                tank.y = this.getTerrainY(tank.x + TANK_WIDTH/2);
            }
        });
        
        // Add explosion particles
        for (let i = 0; i < 10; i++) {
            particles.push(new Particle(
                x + (Math.random() - 0.5) * destroyRadius * 2,
                y + (Math.random() - 0.5) * destroyRadius * 2,
                (Math.random() - 0.5) * 10,
                -Math.random() * 10,
                '#ff6600'
            ));
        }
    }
    
    explode() {
        // Create explosion effect
        const explosion = {
            x: this.x,
            y: this.y,
            radius: 0,
            maxRadius: 15 + this.damage / 5,
            alpha: 1
        };
        
        const animateExplosion = () => {
            explosion.radius += 2;
            explosion.alpha -= 0.05;
            
            if (explosion.alpha > 0) {
                requestAnimationFrame(animateExplosion);
            }
            
            ctx.save();
            ctx.globalAlpha = explosion.alpha;
            ctx.fillStyle = '#ff6600';
            ctx.beginPath();
            ctx.arc(explosion.x, explosion.y, explosion.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        };
        
        animateExplosion();
    }
    
    draw() {
        if (!this.active) return;
        
        ctx.fillStyle = '#333';
        ctx.beginPath();
        ctx.arc(this.x, this.y, 3, 0, Math.PI * 2);
        ctx.fill();
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
            const index = particles.indexOf(this);
            if (index > -1) particles.splice(index, 1);
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
    tanks = [];
    projectiles = [];
    particles = [];
    
    const colors = ['#ff4444', '#44ff44', '#4444ff', '#ffff44'];
    for (let i = 0; i < 4; i++) {
        const x = (i + 1) * canvas.width / 5 - TANK_WIDTH/2;
        const y = terrain[Math.floor((i + 1) * TERRAIN_POINTS / 5)].y;
        tanks.push(new Tank(x, y, colors[i], i + 1));
        tanks[i].updateBrain();
    }
    
    // Initialize training pool
    trainingPool = Array(50).fill().map(() => new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE));
    
    updateStats();
}

// Training simulation
function simulateTraining() {
    if (!globalBestModel) {
        globalBestModel = new NeuralNetwork(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
        return;
    }
    
    // Run simulation battles for training
    for (let sim = 0; sim < 10; sim++) {
        let simTanks = [
            new Tank(100, 300, '#ff4444', 1),
            new Tank(300, 300, '#44ff44', 2),
            new Tank(500, 300, '#4444ff', 3),
            new Tank(700, 300, '#ffff44', 4)
        ];
        
        simTanks.forEach((tank, i) => {
            tank.brain.copyFrom(trainingPool[Math.floor(Math.random() * trainingPool.length)]);
        });
        
        // Run quick simulation
        for (let round = 0; round < 5; round++) {
            simTanks.forEach(tank => {
                if (tank.alive) {
                    const inputs = [
                        tank.x,
                        tank.y,
                        ...simTanks.filter(t => t.id !== tank.id).flatMap(t => [t.x, t.y, t.alive ? 1 : 0]),
                        terrainFreq,
                        terrainAmp
                    ];
                    
                    const outputs = tank.brain.forward(inputs);
                    tank.angle = (outputs[0] - 0.5) * Math.PI;
                    tank.power = outputs[1] * MAX_POWER;
                    
                    // Simulate shot
                    const target = simTanks.find(t => t.id !== tank.id && t.alive);
                    if (target) {
                        const distance = Math.sqrt(
                            Math.pow(target.x - tank.x, 2) + 
                            Math.pow(target.y - tank.y, 2)
                        );
                        tank.fitness += Math.max(0, 100 - distance);
                    }
                }
            });
        }
        
        // Update training pool with best performers
        simTanks.sort((a, b) => b.fitness - a.fitness);
        trainingPool[Math.floor(Math.random() * trainingPool.length)].copyFrom(simTanks[0].brain);
    }
    
    // Update global best model
    globalBestModel.copyFrom(trainingPool.reduce((best, current) => {
        // Simple fitness evaluation for training pool
        return Math.random() > 0.5 ? current : best;
    }));
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
        particles.forEach(p => {
            p.update();
            p.draw();
        });
        
        // Tank decision phase every 60 frames
        if (frameCount % 60 === 0) {
            tanks.forEach(tank => tank.think());
            
            // All tanks fire at once
            tanks.forEach(tank => tank.fire());
        }
        
        // Update and draw projectiles
        projectiles.forEach(p => {
            p.update();
            p.draw();
        });
        
        // Remove inactive projectiles
        projectiles = projectiles.filter(p => p.active);
        
        // Draw tanks
        tanks.forEach(tank => tank.draw());
        
        // Update stats
        if (frameCount % 30 === 0) {
            updateStats();
        }
        
        // Training simulation
        if (frameCount % 300 === 0) {
            simulateTraining();
            tanks.forEach(tank => tank.updateBrain());
            generation++;
        }
        
        // Check for game end
        const aliveTanks = tanks.filter(t => t.alive);
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
    
    tanks.forEach(tank => {
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

// Start game
initGame();
gameLoop();
