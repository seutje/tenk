// Game constants
const GRAVITY = 0.2;
const TURN_TIME = 30;
const DEMO_TURN_TIME = 1;
const GROUND_HEIGHT = 100;
const TANK_WIDTH = 40;
const TANK_HEIGHT = 20;

const WEAPONS = {
    standard: {
        name: "Standard Shell",
        damage: 25,
        radius: 30,
        speed: 12,
        description: "Standard explosive shell",
    },
    rapid: {
        name: "Rapid Fire",
        damage: 15,
        radius: 20,
        speed: 18,
        shots: 3,
        description: "Three small shells fired in succession",
    },
    triple: {
        name: "Triple Shot",
        damage: 20,
        radius: 25,
        speed: 15,
        spread: 0.3,
        shots: 3,
        description: "Three shells at different angles",
    },
    bouncy: {
        name: "Bouncy Shell",
        damage: 20,
        radius: 25,
        speed: 13.5,
        bounces: 3,
        description: "Bounces off terrain before exploding",
    },
    mega: {
        name: "Mega Blast",
        damage: 50,
        radius: 60,
        speed: 10.5,
        description: "Devastating wide-radius explosion",
    },
};

class NeuralNetwork {
    constructor({ hiddenWeights, hiddenBias, outputWeights, outputBias } = {}) {
        this.hiddenWeights = hiddenWeights || Array.from({ length: 3 }, () => Array(5).fill(0));
        this.hiddenBias = hiddenBias || Array(3).fill(0);
        this.outputWeights = outputWeights || Array.from({ length: 7 }, () => Array(3).fill(0));
        this.outputBias = outputBias || Array(7).fill(0);
    }

    static random() {
        const rand = () => Math.random() * 2 - 1;
        return new NeuralNetwork({
            hiddenWeights: Array.from({ length: 3 }, () => Array.from({ length: 5 }, rand)),
            hiddenBias: Array.from({ length: 3 }, () => rand() / 2),
            outputWeights: Array.from({ length: 7 }, () => Array.from({ length: 3 }, rand)),
            outputBias: Array.from({ length: 7 }, () => rand() / 2),
        });
    }

    static from(obj) {
        return new NeuralNetwork(obj);
    }

    clone() {
        return NeuralNetwork.from(JSON.parse(JSON.stringify(this)));
    }

    mutate(rate = MUTATION_RATE, strength = MUTATION_STRENGTH) {
        const mutateVal = (v) => (Math.random() < rate ? v + (Math.random() * 2 - 1) * strength : v);
        const mutateArr = (a) => a.map((v) => (Array.isArray(v) ? mutateArr(v) : mutateVal(v)));
        this.hiddenWeights = mutateArr(this.hiddenWeights);
        this.hiddenBias = mutateArr(this.hiddenBias);
        this.outputWeights = mutateArr(this.outputWeights);
        this.outputBias = mutateArr(this.outputBias);
    }

    forward(inputs) {
        const hidden = this.hiddenWeights.map((w, i) =>
            Math.tanh(w.reduce((s, wt, j) => s + wt * inputs[j], this.hiddenBias[i]))
        );
        return this.outputWeights.map((w, i) => w.reduce((s, wt, j) => s + wt * hidden[j], this.outputBias[i]));
    }

    decide(inputs) {
        const out = this.forward(inputs);
        const angle = Math.max(0, Math.min(180, (out[0] + 1) * 90));
        const power = Math.max(0.3, Math.min(1, (out[1] + 1) / 2));
        const weaponIndex = out.slice(2).indexOf(Math.max(...out.slice(2)));
        const weapon = Object.keys(WEAPONS)[weaponIndex];
        return { angle, power, weapon };
    }

    toJSON() {
        return {
            hiddenWeights: this.hiddenWeights,
            hiddenBias: this.hiddenBias,
            outputWeights: this.outputWeights,
            outputBias: this.outputBias,
        };
    }
}

let trainedNet = null;
let bestNet = null;
let bestScore = -Infinity;

function loadTrainedNet() {
    if (typeof window === 'undefined') {
        try {
            const obj = require('../trained_net.json');
            trainedNet = NeuralNetwork.from(obj);
            bestNet = trainedNet.clone();
            bestScore = evaluate(trainedNet);
        } catch (e) {
            trainedNet = null;
        }
        return Promise.resolve();
    } else {
        return fetch('trained_net.json')
            .then((r) => r.json())
            .then((j) => {
                trainedNet = NeuralNetwork.from(j);
                bestNet = trainedNet.clone();
                bestScore = evaluate(trainedNet);
            })
            .catch(() => {});
    }
}

// Generate a random neural network for an AI tank
function createRandomNet() {
    return NeuralNetwork.random();
}

function neuralDecision(net, inputs) {
    const nn = net instanceof NeuralNetwork ? net : NeuralNetwork.from(net);
    return nn.decide(inputs);
}

// ==== Training Functions ====
const POPULATION = 20;
const MUTATION_RATE = 0.1;
const MUTATION_STRENGTH = 0.5;
const TRAINING_INTERVAL = 250; // faster background evolution

function cloneNet(net) {
    const nn = net instanceof NeuralNetwork ? net : NeuralNetwork.from(net);
    return nn.clone();
}

function mutate(net) {
    const nn = net instanceof NeuralNetwork ? net : NeuralNetwork.from(net);
    nn.mutate();
}

function simulateShot(net, enemyX) {
    const nn = net instanceof NeuralNetwork ? net : NeuralNetwork.from(net);
    const inputs = [enemyX / 800, 0, 0, 0, Math.abs(enemyX) / 800];
    const { angle, power, weapon } = nn.decide(inputs);
    const rad = (angle * Math.PI) / 180;
    const w = WEAPONS[weapon];
    const speed = w.speed * power;
    const vx = Math.cos(rad) * speed;
    const vy = Math.sin(rad) * speed;
    const t = (vy * 2) / GRAVITY;
    const landingX = vx * t;
    const dist = Math.abs(landingX - enemyX);
    // distance from the firing tank's position at x=0
    const selfDist = Math.abs(landingX);
    let damage = 0;
    if (dist <= w.radius) {
        damage = w.damage * (1 - dist / w.radius);
    }
    let selfDamage = 0;
    if (selfDist <= w.radius) {
        selfDamage = w.damage * (1 - selfDist / w.radius);
    }
    const closeness = Math.max(0, (w.radius - dist) / w.radius);
    return damage + closeness * 10 - selfDamage;
}

function evaluate(net) {
    const nn = net instanceof NeuralNetwork ? net : NeuralNetwork.from(net);
    let fitness = 0;
    for (let i = 0; i < 5; i++) {
        const enemyX = 300 + Math.random() * 200;
        fitness += simulateShot(nn, enemyX);
    }
    return fitness / 5;
}

function evolveStep(population) {
    const scored = population.map((net) => ({ net, score: evaluate(net) }));
    scored.sort((a, b) => b.score - a.score);
    const best = scored[0].net;
    const survivors = scored.slice(0, POPULATION / 2).map((s) => s.net);
    const nextPop = survivors.map(cloneNet);
    while (nextPop.length < POPULATION) {
        const parent = cloneNet(
            survivors[Math.floor(Math.random() * survivors.length)],
        );
        mutate(parent);
        nextPop.push(parent);
    }
    return { best, nextPop, score: scored[0].score };
}

function evolve(populationSize = POPULATION, generations = 50) {
    let population = Array.from({ length: populationSize }, createRandomNet);
    if (bestNet) population[0] = cloneNet(bestNet);
    let genBest = bestNet ? cloneNet(bestNet) : population[0];
    let genScore = bestNet ? bestScore : evaluate(genBest);
    for (let g = 0; g < generations; g++) {
        const result = evolveStep(population);
        population = result.nextPop;
        if (result.score > genScore) {
            genScore = result.score;
            genBest = cloneNet(result.best);
        }
        console.log(
            `Generation ${g + 1}: best fitness = ${result.score.toFixed(2)}, overall best = ${genScore.toFixed(2)}`,
        );
        population[0] = cloneNet(genBest);
    }
    bestNet = cloneNet(genBest);
    bestScore = genScore;
    trainedNet = cloneNet(genBest);
    return genBest;
}

let trainingPopulation = null;
let trainingGen = 0;

function startBackgroundTraining() {
    trainingGen = 0;
    trainingPopulation = [];
    if (trainedNet) {
        bestNet = cloneNet(trainedNet);
        bestScore = evaluate(trainedNet);
        for (let i = 0; i < POPULATION; i++) {
            const net = cloneNet(trainedNet);
            if (i !== 0) mutate(net);
            trainingPopulation.push(net);
        }
    } else {
        trainingPopulation = Array.from({ length: POPULATION }, createRandomNet);
        bestNet = cloneNet(trainingPopulation[0]);
        bestScore = evaluate(bestNet);
    }
    const trainStep = () => {
        const result = evolveStep(trainingPopulation);
        trainingPopulation = result.nextPop;
        if (result.score > bestScore) {
            bestScore = result.score;
            bestNet = cloneNet(result.best);
        }
        trainedNet = cloneNet(bestNet);
        trainingPopulation[0] = cloneNet(bestNet);
        trainingGen++;
        if (tanks) {
            tanks.forEach((t) => {
                if (t.ai) t.aiNet = cloneNet(bestNet);
            });
        }
        console.log(
            `Background generation ${trainingGen} complete, best fitness = ${result.score.toFixed(2)}, overall best = ${bestScore.toFixed(2)}`,
        );
    };
    setInterval(trainStep, TRAINING_INTERVAL);
}

let canvas,
    ctx,
    terrain,
    tanks,
    projectiles,
    explosions,
    timer,
    turnInProgress,
    gameOver,
    aiDecisions,
    hasPlayer = true,
    loopStarted = false,
    eventsBound = false;

function init(player = true) {
    canvas = document.getElementById("gameCanvas");
    ctx = canvas.getContext("2d");
    hasPlayer = player;
    projectiles = [];
    explosions = [];
    gameOver = false;
    resize();
    generateTerrain();
    createTanks(player);
    updateUI();
    if (!eventsBound) {
        bindUI();
        eventsBound = true;
    }
    document.getElementById("controls").style.display = hasPlayer
        ? "block"
        : "none";
    document.getElementById("timerBar").style.display = hasPlayer
        ? "block"
        : "none";
    if (!loopStarted) {
        gameLoop();
        loopStarted = true;
    }
    startTurn();
}

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 260;
}

function generateTerrain() {
    terrain = [];
    const segments = Math.floor(canvas.width / 10);
    const base = canvas.height - GROUND_HEIGHT;
    for (let i = 0; i <= segments; i++) {
        const x = i * 10;
        const y = base - Math.sin(i * 0.5) * 70 - Math.random() * 20;
        terrain.push({ x, y });
    }
}

function createTanks(player = true) {
    tanks = [];
    // Spawn tanks at random horizontal positions across the battlefield
    const positions = Array.from({ length: 4 }, () => Math.random() * 0.8 + 0.1).sort();
    const colors = ["#00ff00", "#00aa00", "#008800", "#006600"];
    for (let i = 0; i < 4; i++) {
        const isPlayer = player && i === 0;
        tanks.push({
            id: i,
            name: isPlayer ? "Player" : `AI-${player ? i : i + 1}`,
            x: canvas.width * positions[i],
            y: 0,
            width: TANK_WIDTH,
            height: TANK_HEIGHT,
            health: 100,
            maxHealth: 100,
            alive: true,
            color: colors[i],
            ai: !isPlayer,
            sensors: { front: 0, back: 0, enemy: 0 },
            aiNet: !isPlayer
                ? trainedNet
                    ? trainedNet.clone()
                    : createRandomNet()
                : null,
        });
    }
    placeTanks();
}

function placeTanks() {
    tanks.forEach((t) => {
        for (let i = 0; i < terrain.length - 1; i++) {
            if (t.x >= terrain[i].x && t.x <= terrain[i + 1].x) {
                const ratio =
                    (t.x - terrain[i].x) / (terrain[i + 1].x - terrain[i].x);
                t.y =
                    terrain[i].y +
                    ratio * (terrain[i + 1].y - terrain[i].y) -
                    t.height;
                break;
            }
        }
    });
}

function bindUI() {
    const wSel = document.getElementById("weaponSelect");
    const aSl = document.getElementById("angleSlider");
    const pSl = document.getElementById("powerSlider");
    const aVal = document.getElementById("angleValue");
    const pVal = document.getElementById("powerValue");
    const wInfo = document.getElementById("weaponInfo");
    aSl.addEventListener("input", () => (aVal.textContent = aSl.value));
    pSl.addEventListener("input", () => (pVal.textContent = pSl.value));
    wSel.addEventListener(
        "change",
        () => (wInfo.textContent = WEAPONS[wSel.value].description),
    );
    document.getElementById("fireButton").addEventListener("click", playerFire);
    window.addEventListener("resize", () => {
        resize();
        generateTerrain();
        placeTanks();
    });
}

function startTurn() {
    if (gameOver) return;
    const turnTime = hasPlayer ? TURN_TIME : DEMO_TURN_TIME;
    timer = turnTime;
    turnInProgress = false;
    aiDecisions = [];
    const interval = setInterval(() => {
        timer -= 0.1;
        document.getElementById("timerFill").style.width =
            (timer / turnTime) * 100 + "%";
        if (timer <= 0) {
            clearInterval(interval);
            hasPlayer && tanks[0].alive ? playerFire() : processAITurn();
        }
    }, 100);
    const startIdx = hasPlayer ? 1 : 0;
    tanks.slice(startIdx).forEach((t, i) => t.alive && makeAIDecision(startIdx + i));
}

function playerFire() {
    if (!hasPlayer || turnInProgress || !tanks[0].alive) return;
    const w = document.getElementById("weaponSelect").value;
    const a = +document.getElementById("angleSlider").value;
    const p = +document.getElementById("powerSlider").value / 100;
    fireProjectile(tanks[0], w, a, p);
    processAITurn();
}

function makeAIDecision(id) {
    const tank = tanks[id];
    if (!tank.alive) return;
    updateSensors(tank);
    const targets = tanks.filter((t) => t.id !== id && t.alive);
    if (!targets.length) return;
    const target = targets.reduce((a, b) =>
        Math.abs(a.x - tank.x) < Math.abs(b.x - tank.x) ? a : b,
    );
    const dx = target.x - tank.x;
    const dy = target.y - tank.y;
    const inputs = [
        dx / canvas.width,
        dy / canvas.height,
        tank.sensors.front,
        tank.sensors.back,
        tank.sensors.enemy,
    ];
    const { angle, power, weapon } = neuralDecision(tank.aiNet, inputs);
    aiDecisions.push({ tank, weapon, angle, power });
}

function processAITurn() {
    if (turnInProgress) return;
    turnInProgress = true;
    aiDecisions.forEach((d) =>
        fireProjectile(d.tank, d.weapon, d.angle, d.power),
    );
    const wait = setInterval(() => {
        if (!projectiles.length && !explosions.length) {
            clearInterval(wait);
            checkGameOver();
            if (!gameOver) setTimeout(startTurn, 1000);
        }
    }, 100);
}

function fireProjectile(tank, type, angle, power) {
    const w = WEAPONS[type];
    const rad = (angle * Math.PI) / 180;
    const speed = w.speed * power;
    const barrel = 30;
    const x = tank.x + Math.cos(rad) * barrel;
    const y = tank.y - Math.sin(rad) * barrel;
    const create = (vx, vy) =>
        projectiles.push({
            x,
            y,
            vx,
            vy,
            weapon: type,
            damage: w.damage,
            radius: w.radius,
            bounces: w.bounces || 0,
            bouncesLeft: w.bounces || 0,
            owner: tank.id,
        });
    if (type === "rapid") {
        for (let i = 0; i < w.shots; i++)
            setTimeout(
                () => create(Math.cos(rad) * speed, -Math.sin(rad) * speed),
                i * 200,
            );
    } else if (type === "triple") {
        for (let i = -1; i <= 1; i++)
            create(
                Math.cos(rad + i * w.spread) * speed,
                -Math.sin(rad + i * w.spread) * speed,
            );
    } else {
        create(Math.cos(rad) * speed, -Math.sin(rad) * speed);
    }
}

function getTerrainHeight(x) {
    if (x < 0) x = 0;
    if (x > canvas.width) x = canvas.width;
    for (let i = 0; i < terrain.length - 1; i++) {
        if (x >= terrain[i].x && x <= terrain[i + 1].x) {
            const r = (x - terrain[i].x) / (terrain[i + 1].x - terrain[i].x);
            return terrain[i].y + r * (terrain[i + 1].y - terrain[i].y);
        }
    }
    return canvas.height - GROUND_HEIGHT;
}

function updateSensors(tank) {
    const base = getTerrainHeight(tank.x);
    const front = getTerrainHeight(tank.x + 20);
    const back = getTerrainHeight(tank.x - 20);
    tank.sensors.front = (front - base) / 50;
    tank.sensors.back = (back - base) / 50;
    const others = tanks.filter((t) => t.id !== tank.id && t.alive);
    if (others.length) {
        const dist = Math.min(
            ...others.map((o) => Math.hypot(o.x - tank.x, o.y - tank.y)),
        );
        tank.sensors.enemy = dist / canvas.width;
    } else {
        tank.sensors.enemy = 1;
    }
}

function updateProjectiles() {
    for (let i = projectiles.length - 1; i >= 0; i--) {
        const p = projectiles[i];
        p.vy += GRAVITY;
        p.x += p.vx;
        p.y += p.vy;
        if (p.y >= getTerrainHeight(p.x)) {
            if (p.bouncesLeft > 0) {
                p.y = getTerrainHeight(p.x);
                p.vy *= -0.7;
                p.vx *= 0.9;
                p.bouncesLeft--;
            } else {
                createExplosion(p.x, p.y, p.radius, p.damage, p.owner);
                projectiles.splice(i, 1);
                continue;
            }
        }
        for (const t of tanks) {
            if (
                t.alive &&
                p.x >= t.x - t.width / 2 &&
                p.x <= t.x + t.width / 2 &&
                p.y >= t.y - t.height &&
                p.y <= t.y
            ) {
                createExplosion(t.x, t.y, p.radius, p.damage, p.owner);
                projectiles.splice(i, 1);
                break;
            }
        }
        if (p.x < 0 || p.x > canvas.width || p.y > canvas.height)
            projectiles.splice(i, 1);
    }
    for (let i = explosions.length - 1; i >= 0; i--) {
        const e = explosions[i];
        e.timer--;
        if (e.timer <= 0) {
            tanks.forEach((t) => {
                if (t.alive) {
                    const d = Math.hypot(t.x - e.x, t.y - e.y);
                    if (d <= e.radius) {
                        t.health -= Math.floor(e.damage * (1 - d / e.radius));
                        t.health = Math.max(0, t.health);
                        if (t.health === 0) {
                            t.alive = false;
                            showMessage(
                                `${tanks[e.owner].name} destroyed ${t.name}!`,
                            );
                        }
                    }
                }
            });
            terrain.forEach((pt) => {
                const d = Math.hypot(pt.x - e.x, pt.y - e.y);
                if (d <= e.radius) pt.y += e.radius - d;
            });
            explosions.splice(i, 1);
            updateUI();
            placeTanks();
        }
    }
}

function createExplosion(x, y, r, d, o) {
    explosions.push({ x, y, radius: r, damage: d, owner: o, timer: 20 });
}

function checkGameOver() {
    const alive = tanks.filter((t) => t.alive);
    if (alive.length <= 1) {
        gameOver = true;
        showMessage(alive.length ? `${alive[0].name} wins!` : "Draw!");
    }
}

function showMessage(msg) {
    const box = document.getElementById("messageBox");
    box.textContent = msg;
    box.style.display = "block";
    setTimeout(() => (box.style.display = "none"), 3000);
}

function startGame() {
    const screen = document.getElementById("startScreen");
    if (screen) screen.style.display = "none";
    init(true);
}

function updateUI() {
    tanks.forEach((t) => {
        const prefix = t.id === 0 ? "player" : `ai${t.id}`;
        const stat = document.getElementById(`${prefix}-stat`);
        const healthSpan = document.getElementById(`${prefix}-health`);
        if (!stat || !healthSpan) return;
        const nameEl = stat.querySelector(".tank-name");
        if (nameEl) nameEl.textContent = t.name;
        healthSpan.textContent = t.health;
        const fill = stat.querySelector(".health-fill");
        if (fill) fill.style.width = `${(t.health / t.maxHealth) * 100}%`;
        stat.classList.toggle("dead", !t.alive);
    });
}

function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    ctx.moveTo(terrain[0].x, terrain[0].y);
    terrain.forEach((p) => ctx.lineTo(p.x, p.y));
    ctx.lineTo(canvas.width, canvas.height);
    ctx.lineTo(0, canvas.height);
    ctx.closePath();
    ctx.fillStyle = "#00ff00";
    ctx.fill();
    tanks.forEach((t) => {
        if (t.alive) {
            ctx.fillStyle = t.color;
            ctx.fillRect(t.x - t.width / 2, t.y - t.height, t.width, t.height);
            ctx.fillStyle = "#000";
            ctx.fillRect(t.x - 2, t.y - t.height - 10, 4, 10);
            ctx.fillStyle = "#000";
            ctx.font = "12px 'Courier New', monospace";
            ctx.textAlign = "center";
            ctx.fillText(t.name, t.x, t.y - t.height + 12);
        }
    });
    ctx.fillStyle = "#00ff00";
    projectiles.forEach((p) => ctx.fillRect(p.x - 2, p.y - 2, 4, 4));
    explosions.forEach((e) => {
        const r = e.radius * (1 - e.timer / 20);
        ctx.beginPath();
        ctx.arc(e.x, e.y, r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(0, 255, 0, ${1 - e.timer / 20})`;
        ctx.fill();
    });
}

function gameLoop() {
    updateProjectiles();
    draw();
    requestAnimationFrame(gameLoop);
}

if (typeof module !== 'undefined') {
    // Export parts of the game logic for Node usage
    module.exports = {
        WEAPONS,
        createRandomNet,
        neuralDecision,
        GRAVITY,
        loadTrainedNet,
        evolve,
        startBackgroundTraining,
        // expose for testing
        evaluate,
        simulateShot,
        NeuralNetwork,
    };
} else {
    window.onload = () => {
        loadTrainedNet().then(() => {
            init(false);
            startBackgroundTraining();
            const btn = document.getElementById("startButton");
            if (btn) btn.addEventListener("click", startGame);
        });
    };
}
