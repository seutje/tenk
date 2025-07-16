// Game constants
const GRAVITY = 0.2;
const TURN_TIME = 30;
const GROUND_HEIGHT = 50;
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

let canvas,
    ctx,
    terrain,
    tanks,
    projectiles,
    explosions,
    timer,
    turnInProgress,
    gameOver,
    aiDecisions;

function init() {
    canvas = document.getElementById("gameCanvas");
    ctx = canvas.getContext("2d");
    projectiles = [];
    explosions = [];
    gameOver = false;
    resize();
    generateTerrain();
    createTanks();
    updateUI();
    bindUI();
    gameLoop();
    startTurn();
}

function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight - 150;
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

function createTanks() {
    tanks = [];
    const positions = [0.15, 0.35, 0.65, 0.85];
    const colors = ["#00ff00", "#00aa00", "#008800", "#006600"];
    for (let i = 0; i < 4; i++) {
        tanks.push({
            id: i,
            name: i ? `AI-${i}` : "Player",
            x: canvas.width * positions[i],
            y: 0,
            width: TANK_WIDTH,
            height: TANK_HEIGHT,
            health: 100,
            maxHealth: 100,
            alive: true,
            color: colors[i],
            ai: !!i,
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
    timer = TURN_TIME;
    turnInProgress = false;
    aiDecisions = [];
    const interval = setInterval(() => {
        timer -= 0.1;
        document.getElementById("timerFill").style.width =
            (timer / TURN_TIME) * 100 + "%";
        if (timer <= 0) {
            clearInterval(interval);
            tanks[0].alive ? playerFire() : processAITurn();
        }
    }, 100);
    tanks.slice(1).forEach((t, i) => t.alive && makeAIDecision(i + 1));
}

function playerFire() {
    if (turnInProgress || !tanks[0].alive) return;
    const w = document.getElementById("weaponSelect").value;
    const a = +document.getElementById("angleSlider").value;
    const p = +document.getElementById("powerSlider").value / 100;
    fireProjectile(tanks[0], w, a, p);
    processAITurn();
}

function makeAIDecision(id) {
    const tank = tanks[id];
    if (!tank.alive) return;
    const targets = tanks.filter((t) => t.id !== id && t.alive);
    if (!targets.length) return;
    const target = targets.reduce((a, b) =>
        Math.abs(a.x - tank.x) < Math.abs(b.x - tank.x) ? a : b,
    );
    const dx = target.x - tank.x;
    const dy = target.y - tank.y;
    const dist = Math.sqrt(dx * dx + dy * dy);
    let angle = (Math.atan2(-dy, dx) * 180) / Math.PI;
    angle = Math.max(0, Math.min(180, angle));
    const power = Math.min(
        1,
        Math.max(0.3, dist / 500 + (Math.random() - 0.5) * 0.2),
    );
    const weapon =
        Object.keys(WEAPONS)[
            Math.floor(Math.random() * Object.keys(WEAPONS).length)
        ];
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

function updateUI() {
    tanks.forEach((t) => {
        const prefix = t.id === 0 ? "player" : `ai${t.id}`;
        const stat = document.getElementById(`${prefix}-stat`);
        const healthSpan = document.getElementById(`${prefix}-health`);
        if (!stat || !healthSpan) return;
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
    // Export for testing in Node environment
    module.exports = { WEAPONS };
} else {
    window.onload = init;
}
