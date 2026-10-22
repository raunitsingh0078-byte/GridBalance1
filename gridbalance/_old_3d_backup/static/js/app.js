/**
 * Grid Balance - Main Application
 * Demand Forecast + Meter Anomaly Detection with 3D Glosomorphic Effects
 */

import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/threejs/r128/three.module.js';
import { OrbitControls } from 'https://cdnjs.cloudflare.com/ajax/libs/threejs/r128/controls/OrbitControls.js';
import { Stats } from 'https://cdnjs.cloudflare.com/ajax/libs/threejs/r128/examples/jsm/libs/Stats.js';

export function createApp() {
    initThreeScene();
    initDataLoader();
    initInteractivity();
    initCursorEffect();
    initResizeHandler();
    
    return true;
}

let threeScene = null;
let camera = null;
let renderer = null;
let controls = null;
let statsPanel = null;
let forecastData = null;
let anomalyData = null;
let isPlaying = true;
let currentRegion = 'PJM_Load';
let animationId = null;

/**
 * Initialize Three.js scene with glosomorphic effects
 */
function initThreeScene() {
    const threeContainer = document.getElementById('three-container');
    if (!threeContainer) return;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(threeContainer.clientWidth, threeContainer.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmToneMapping;
    renderer.toneMappingExposure = 1.2;
    threeContainer.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e17);
    scene.fog = new THREE.Fog(0x0a0e17, 20, 100);

    camera = new THREE.PerspectiveCamera(
        60,
        threeContainer.clientWidth / threeContainer.clientHeight,
        0.1,
        1000
    );
    camera.position.set(0, 10, 25);
    camera.lookAt(0, 0, 0);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI / 2.1;
    controls.minDistance = 5;
    controls.maxDistance = 50;

    // Initialize Three.js elements
    createGlosometricGrid();
    createDemandViz();
    addLighting();

    animate();
}

/**
 * Create glosomorphic grid background
 */
function createGlosometricGrid() {
    const gridCount = 16;
    const gridSize = 8;
    
    for (let i = 0; i < gridCount; i++) {
        for (let j = 0; j < gridCount; j++) {
            const geometry = new THREE.PlaneGeometry(gridSize, gridSize);
            
            // Glosomorphic gradient material
            const color1 = new THREE.Color(0x00d4aa);
            const color2 = new THREE.Color(0x0f172a);
            const t = i / gridCount;
            const interpolatedColor = color1.lerp(color2, t);
            
            const material = new THREE.MeshStandardMaterial({
                color: interpolatedColor,
                transparent: true,
                opacity: 0.02,
                metalness: 0.1,
                roughness: 0.3,
                side: THREE.DoubleSide
            });
            
            const mesh = new THREE.Mesh(geometry, material);
            mesh.position.set(
                (i - gridCount / 2) * gridSize,
                0,
                (j - gridCount / 2) * gridSize
            );
            mesh.rotation.x = -Math.PI / 4;
            mesh.receiveShadow = true;
            scene.add(mesh);
        }
    }
}

/**
 * Create demand forecast visualization
 */
function createDemandViz() {
    const scene = threeScene.scene;
    
    // Generate demand data
    fetch('/api/data/PJM_Load')
        .then(res => res.json())
        .then(data => {
            if (!data || !data.data || data.data.length === 0) {
                // Generate synthetic data if no API data
                createSyntheticDemandViz();
                return;
            }
            
            const values = data.data.map(d => d.value);
            createRealDemandViz(values, data.region);
        })
        .catch(() => createSyntheticDemandViz());
}

/**
 * Create real demand viz from API data
 */
function createRealDemandViz(values, regionName) {
    const points = [];
    const segmentCount = 60;
    
    for (let seg = 0; seg < segmentCount; seg++) {
        const t = (seg / segmentCount) * values.length;
        const index = Math.min(Math.floor(t), values.length - 1);
        const ratio = t - index;
        
        const y1 = values[index] / 50000 * 4;
        const y2 = values[Math.min(index + 1, values.length - 1)] / 50000 * 4;
        const y = y1 + (y2 - y1) * ratio;
        
        const x = (seg - 30) * 1.3;
        const z = y;
        
        points.push(new THREE.Vector3(x, 0, z));
    }
    
    const curve = new THREE.CatmullRomCurve3(points, true, 'catmullrom');
    const geometry = new THREE.TubeGeometry(curve, 200, 0.8, 12, false);
    const material = new THREE.MeshStandardMaterial({
        color: 0x00d4aa,
        transparent: true,
        opacity: 0.9,
        metalness: 0.4,
        roughness: 0.4,
        flatShading: false
    });
    
    const tube = new THREE.Mesh(geometry, material);
    tube.rotation.x = Math.PI / 2;
    tube.geometry.setAttribute('uv2', tube.geometry.attributes 'uv1');
    tube.castShadow = true;
    tube.receiveShadow = true;
    
    // Add glow effect
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0x00d4aa,
        transparent: true,
        opacity: 0.15,
        depthWrite: false
    });
    const glowMesh = new THREE.Mesh(tube.geometry, glowMaterial);
    glowMesh.scale.multiplyScalar(1.02);
    glowMesh.position.z = -0.01;
    scene.add(glowMesh);
    
    scene.add(tube);
    
    // Store for interaction
    threeScene.demandTube = tube;
    threeScene.demandValues = values;
    threeScene.demandRegion = regionName;
    
    // Add data points along the curve
    addDemandPoints(values, points);
}

/**
 * Create synthetic demand viz
 */
function createSyntheticDemandViz() {
    const scene = threeScene.scene;
    
    // Generate synthetic demand pattern
    const baseLoad = 30000;
    const points = [];
    
    for (let i = 0; i < 200; i++) {
        const hour = i % 24;
        let peakFactor = 1;
        
        if (hour >= 7 && hour <= 9) peakFactor = 1.3;
        if (hour >= 17 && hour <= 19) peakFactor = 1.25;
        if (hour >= 0 && hour <= 5) peakFactor = 0.7;
        
        const noise = (Math.random() - 0.5) * 1000;
        const load = baseLoad * peakFactor + noise;
        const normalized = Math.max(load, 10000);
        
        const x = (i - 100) * 0.5;
        const z = normalized / 5000 * 3;
        
        points.push(new THREE.Vector3(x, 0, z));
    }
    
    const curve = new THREE.CatmullRomCurve3(points, true, 'catmullrom');
    const geometry = new THREE.TubeGeometry(curve, 200, 0.8, 12, false);
    const material = new THREE.MeshStandardMaterial({
        color: 0x00d4aa,
        transparent: true,
        opacity: 0.9,
        metalness: 0.4,
        roughness: 0.4
    });
    
    const tube = new THREE.Mesh(geometry, material);
    tube.rotation.x = Math.PI / 2;
    tube.geometry.setAttribute('uv2', tube.geometry.attributes['uv1']);
    tube.castShadow = true;
    tube.receiveShadow = true;
    
    // Glow
    const glowMaterial = new THREE.MeshBasicMaterial({
        color: 0x00d4aa,
        transparent: true,
        opacity: 0.15,
        depthWrite: false
    });
    const glowMesh = new THREE.Mesh(tube.geometry, glowMaterial);
    glowMesh.scale.multiplyScalar(1.02);
    glowMesh.position.z = -0.01;
    scene.add(glowMesh);
    
    scene.add(tube);
    
    threeScene.demandTube = tube;
    threeScene.demandValues = points.map(p => p.z * 5000);
    threeScene.demandRegion = 'Synthetic';
}

/**
 * Add interactive points on the demand curve
 */
function addDemandPoints(values, points) {
    const scene = threeScene.scene;
    
    // Create point lights that react to demand values
    const pointCount = Math.min(20, values.length, points.length);
    
    for (let i = 0; i < pointCount; i++) {
        const idx = Math.floor((i / pointCount) * values.length);
        const value = values[idx] || 30000;
        const intensity = 0.3 + (value / 80000) * 0.7;
        
        const geometry = new THREE.SphereGeometry(0.15, 16, 16);
        const material = new THREE.MeshStandardMaterial({
            color: 0x00d4aa,
            metalness: 0.8,
            roughness: 0.2
        });
        
        const point = new THREE.Mesh(geometry, material);
        point.position.set(
            (i - 10) * 1.5,
            value / 50000 * 2,
            (i / pointCount) * 12 - 6
        );
        point.userData.intensity = intensity;
        point.userData.value = value;
        point.castShadow = true;
        
        // Add emissive glow
        point.emissive = new THREE.Color(0x00d4aa);
        point.emissiveIntensity = intensity * 0.5;
        
        scene.add(point);
        
        // Store for interaction
        point.userData.index = i;
    }
}

/**
 * Add lighting with glow effects
 */
function addLighting() {
    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.2);
    scene.add(ambientLight);
    
    // Main directional light with blue-cyan tint
    const dirLight = new THREE.DirectionalLight(0x00d4aa, 0.7);
    dirLight.position.set(10, 15, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 2048;
    dirLight.shadow.mapSize.height = 2048;
    dirLight.shadow.camera.near = 0.5;
    dirLight.shadow.camera.far = 50;
    dirLight.shadow.radius = 4;
    scene.add(dirLight);
    
    // Top subtle light
    const topLight = new THREE.DirectionalLight(0x63b3e8, 0.2);
    topLight.position.set(0, 20, 0);
    scene.add(topLight);
    
    // Point light for glow effects
    const pointLight = new THREE.PointLight(0x00d4aa, 0.5, 20);
    pointLight.position.set(0, 5, 0);
    pointLight.castShadow = true;
    scene.add(pointLight);
    
    // Helper: create a glowing aura
    const createGlowAura = (position, color = 0x00d4aa) => {
        const auraGeometry = new THREE.SphereGeometry(3, 32, 32);
        const auraMaterial = new THREE.MeshBasicMaterial({
            color: color,
            transparent: true,
            opacity: 0.03,
            depthWrite: false
        });
        const auraMesh = new THREE.Mesh(auraGeometry, auraMaterial);
        auraMesh.position.copy(position);
        scene.add(auraMesh);
        return auraMesh;
    };
    
    threeScene.glowAura = createGlowAura(new THREE.Vector3(0, 3, 0));
}

/* Animation loop */
function animate() {
    animationId = requestAnimationFrame(animate);
    
    if (controls) {
        controls.update();
    }
    
    // Rotate glow aura
    if (threeScene && threeScene.glowAura) {
        threeScene.glowAura.rotation.y += 0.002;
    }
    
    // Animate demand tube if exists
    if (threeScene.demandTube) {
        threeScene.demandTube.rotation.y += 0.001;
    }
    
    renderer.render(scene, camera);
}

/**
 * Initialize data loader with API calls
 */
function initDataLoader() {
    loadRegionData(currentRegion);
    setupRegionSelector();
    setupForecastControls();
}

/**
 * Load region data from API
 */
function loadRegionData(region) {
    fetch(`/api/data/${region}`)
        .then(res => {
            if (!res.ok) throw new Error('Region not found');
            return res.json();
        })
        .then(data => {
            displayRegionInfo(data);
            updateDemandViz(data);
        })
        .catch(err => {
            console.error(err);
            showError('Region data not available, using synthetic data');
        });
}

/**
 * Display region information
 */
function displayRegionInfo(data) {
    const regionEl = document.getElementById('region-value');
    if (regionEl) {
        regionEl.textContent = data.region || 'Unknown';
    }
}

/**
 * Update demand visualization
 */
function updateDemandViz(data) {
    const values = data.data ? data.data.map(d => d.value) : [];
    if (values.length > 0 && threeScene.demandTube) {
        updateDemandCurve(values);
    }
}

/**
 * Update demand curve with new data
 */
function updateDemandCurve(values) {
    if (!threeScene.demandTube) return;
    
    const segmentCount = 60;
    const points = [];
    
    for (let seg = 0; seg < segmentCount; seg++) {
        const t = (seg / segmentCount) * values.length;
        const index = Math.min(Math.floor(t), values.length - 1);
        const ratio = t - index;
        
        const y1 = values[index] / 50000 * 4;
        const y2 = values[Math.min(index + 1, values.length - 1)] / 50000 * 4;
        const y = y1 + (y2 - y1) * ratio;
        
        const x = (seg - 30) * 1.3;
        
        points.push(new THREE.Vector3(x, 0, y));
    }
    
    // Update the tube geometry
    threeScene.demandTube.geometry.dispose();
    const newGeometry = new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(points, true, 'catmullrom'),
        200, 0.8, 12, false
    );
    threeScene.demandTube.geometry = newGeometry;
    threeScene.demandTube.geometry.setAttribute('uv2', threeScene.demandTube.geometry.attributes['uv1']);
    
    // Animate points
    document.querySelectorAll('.demand-point').forEach((point, i) => {
        if (point.userData) {
            point.style.height = `${Math.min(point.userData.value / 50000 * 2, 4)}px`;
        }
    });
}

/**
 * Setup region selector
 */
function setupRegionSelector() {
    const regions = ['PJM_Load', 'AEP_hourly', 'NI_hourly', 'COMED_hourly', 'DOM_hourly'];
    
    const selectorEl = document.getElementById('region-selector');
    if (!selectorEl) return;
    
    selectorEl.innerHTML = '';
    
    regions.forEach(region => {
        const option = document.createElement('option');
        option.value = region;
        option.textContent = region.replace('_', ' ');
        option.dataset region = region;
        
        option.addEventListener('click', () => {
            currentRegion = region;
            loadRegionData(region);
        });
        
        selectorEl.appendChild(option);
    });
    
    // Set default
    const defaultOption = selectorEl.querySelector(`option[value="${currentRegion}"]`);
    if (defaultOption) defaultOption.selected = true;
}

/**
 * Setup forecast controls
 */
function setupForecastControls() {
    const controlsEl = document.getElementById('forecast-controls');
    if (!controlsEl) return;
    
    controlsEl.innerHTML = `
        <div class="glass-panel">
            <h3 style="margin-bottom: 16px; font-size: 14px; color: var(--muted);">Demand Forecast</h3>
            <div style="display: flex; gap: 8px; align-items: center; flex-wrap: wrap;">
                <select id="model-select" style="padding: 8px 12px; background: var(--card-bg); border: 1px solid var(--border); color: var(--fg); border-radius: 8px;">
                    <option value="gradient_boosting">Gradient Boosting</option>
                    <option value="linear">Linear Model</option>
                    <option value="xgboost">XGBoost</option>
                    <option value="lightgbm">LightGBM</option>
                </select>
                <input type="number" id="horizon" min="1" max="168" value="24" style="padding: 8px 12px; background: var(--card-bg); border: 1px solid var(--border); color: var(--fg); border-radius: 8px;">
                <button class="btn" id="run-forecast">Run</button>
            </div>
            <div id="forecast-results" style="margin-top: 16px; display: none;"></div>
        </div>
    `;
    
    document.getElementById('run-forecast').addEventListener('click', () => {
        runForecast();
    });
}

/**
 * Run forecast request
 */
function runForecast() {
    const horizon = parseInt(document.getElementById('horizon').value) || 24;
    const modelType = document.getElementById('model-select').value || 'gradient_boosting';
    
    fetch('/api/forecast', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            region: currentRegion,
            horizon: horizon,
            model_type: modelType
        })
    })
    .then(res => res.json())
    .then(data => {
        displayForecastResults(data);
    })
    .catch(err => {
        console.error('Forecast error:', err);
        showError('Forecast calculation failed');
    });
}

/**
 * Display forecast results
 */
function displayForecastResults(data) {
    const resultsEl = document.getElementById('forecast-results');
    if (!resultsEl) return;
    
    const forecasts = data.forecasts || [];
    const lastValue = data.historical_last || 0;
    
    resultsEl.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 16px;">
            <div class="stat">
                <div class="stat-value">${lastValue.toLocaleString()}</div>
                <div class="stat-label">Last Historical Value</div>
            </div>
            <div class="stat">
                <div class="stat-value" style="color: var(--accent);">+${((forecasts[0] - lastValue) / lastValue * 100).toFixed(1)}%</div>
                <div class="stat-label">Predicted Change</div>
            </div>
        </div>
        <div style="margin-top: 12px;">
            <h4 style="font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Forecast Horizon</h4>
            <div style="display: flex; gap: 4px; flex-wrap: wrap;">
                ${forecasts.slice(0, 24).map((f, i) => `
                    <div style="width: 32px; height: 32px; background: var(--card-bg); border: 1px solid var(--border); border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 11px; color: var(--muted);">
                        H${i + 1}: ${f.toLocaleString()}
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    
    resultsEl.style.display = 'block';
    
    // Update 3D visualization with forecast
    update3DForecast(data);
}

/**
 * Update 3D with forecast data
 */
function update3DForecast(data) {
    if (!threeScene.demandTube) return;
    
    const forecasts = data.forecasts || [];
    const values = [...threeScene.demandValues, ...forecasts.map(f => f * 50000 / 4)];
    
    const segmentCount = 60;
    const totalPoints = values.length;
    const points = [];
    
    for (let seg = 0; seg < segmentCount; seg++) {
        const t = (seg / segmentCount) * totalPoints;
        const index = Math.min(Math.floor(t), totalPoints - 1);
        const ratio = t - index;
        
        const y1 = values[index] / 50000 * 4;
        const y2 = values[Math.min(index + 1, totalPoints - 1)] / 50000 * 4;
        const y = y1 + (y2 - y1) * ratio;
        
        const x = (seg - 30) * 1.3;
        
        points.push(new THREE.Vector3(x, 0, y));
    }
    
    threeScene.demandTube.geometry.dispose();
    const newGeometry = new THREE.TubeGeometry(
        new THREE.CatmullRomCurve3(points, true, 'catmullrom'),
        200, 0.8, 12, false
    );
    threeScene.demandTube.geometry = newGeometry;
    threeScene.demandTube.geometry.setAttribute('uv2', threeScene.demandTube.geometry.attributes['uv1']);
    
    // Change color to indicate forecast
    threeScene.demandTube.material.color.setHex(0x00d4aa);
    threeScene.demandTube.material.opacity = 0.8;
}

/**
 * Initialize interactivity features
 */
function initInteractivity() {
    // Region selector interactivity
    initRegionInteractivity();
    
    // Forecast card interactivity
    initForecastInteractivity();
    
    // Anomaly detection trigger
    initAnomalyInteractivity();
}

/**
 * Region interactivity
 */
function initRegionInteractivity() {
    const regionSelectors = document.querySelectorAll('#region-selector option');
    
    regionSelectors.forEach(opt => {
        opt.addEventListener('mouseenter', () => {
            opt.style.color = var(--accent);
        });
        
        opt.addEventListener('mouseleave', () => {
            opt.style.color = var(--fg);
        });
    });
}

/**
 * Forecast interactivity
 */
function initForecastInteractivity() {
    const forecastCards = document.querySelectorAll('.forecast-card');
    
    forecastCards.forEach(card => {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-4px)';
            card.style.boxShadow = '0 8px 32px var(--glow)';
        });
        
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'translateY(0)';
            card.style.boxShadow = 'none';
        });
    });
}

/**
 * Anomaly detection interactivity
 */
function initAnomalyInteractivity() {
    const anomalyBtn = document.getElementById('detect-anomalies');
    if (!anomalyBtn) return;
    
    anomalyBtn.addEventListener('click', () => {
        detectAnomalies();
    });
    
    // Add hover effect
    anomalyBtn.addEventListener('mouseenter', () => {
        anomalyBtn.style.borderColor = var(--accent);
        anomalyBtn.style.color = var(--accent);
    });
    
    anomalyBtn.addEventListener('mouseleave', () => {
        anomalyBtn.style.borderColor = var(--border);
        anomalyBtn.style.color = var(--fg);
    });
}

/**
 * Detect anomalies in demand data
 */
function detectAnomalies() {
    fetch(`/api/anomalies?region=${currentRegion}&sensitivity=0.05`)
        .then(res => res.json())
        .then(data => {
            displayAnomalyResults(data);
        })
        .catch(err => {
            console.error('Anomaly detection error:', err);
        });
}

/**
 * Display anomaly results
 */
function displayAnomalyResults(data) {
    const anomalyCount = data.anomaly_count || 0;
    const anomalies = data.anomalies || [];
    
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(8px);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 10000;
    `;
    
    const content = document.createElement('div');
    content.style.cssText = `
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 32px;
        width: 90%;
        max-width: 500px;
        max-height: 80%;
        overflow-y: auto;
        color: var(--fg);
    `;
    
    content.innerHTML = `
        <h3 style="margin-bottom: 16px; color: var(--accent);">Anomaly Detection Results</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px;">
            <span>Total Points: ${data.total_points || 0}</span>
            <span style="color: var(--accent); font-weight: bold;">${anomalyCount} anomalies detected</span>
        </div>
        ${anomalies.length > 0 ? `
            <div style="max-height: 300px; overflow-y: auto; margin-top: 16px;">
                ${anomalies.slice(0, 10).map((a, i) => `
                    <div style="padding: 8px 0; border-bottom: 1px solid var(--border);">
                        <div style="font-weight: 500; margin-bottom: 4px;">Anomaly #${i + 1}</div>
                        <div style="font-size: 12px; color: var(--muted);">Index: ${a.index}, Value: ${a.value.toLocaleString()}</div>
                        <div style="font-size: 11px; color: var(--accent);">${a.reason || 'Unknown reason'}</div>
                    </div>
                `).join('')}
            </div>
        ` : '<p style="color: var(--muted);">No anomalies detected</p>'}
        <div style="margin-top: 24px; padding-top: 16px; border-top: 1px solid var(--border);">
            <button class="btn" style="width: 100%; padding: 10px;">Close</button>
        </div>
    `;
    
    modal.appendChild(content);
    document.body.appendChild(modal);
    
    // Close on button click
    content.querySelector('button').addEventListener('click', () => {
        document.body.removeChild(modal);
    });
    
    // Close on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            document.body.removeChild(modal);
        }
    });
}

/* Cursor effect */
function initCursorEffect() {
    const trail = document.createElement('div');
    trail.className = 'cursor-trail circle';
    trail.style.left = `${event.clientX}px`;
    trail.style.top = `${event.clientY}px`;
    document.body.appendChild(trail);
    
    let mouseX = 0,
        mouseY = 0;
    
    document.addEventListener('mousemove', (e) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    });
    
    setInterval(() => {
        const targetX = mouseX;
        const targetY = mouseY;
        
        // Smooth interpolation
        const lerpX = trail.x !== undefined ? 
            (trail.x + (targetX - trail.x) * 0.15) : targetX;
        const lerpY = trail.y !== undefined ? 
            (trail.y + (targetY - trail.y) * 0.15) : targetY;
        
        trail.style.left = `${lerpX}px`;
        trail.style.top = `${lerpY}px`;
    }, 16);
    
    // Change cursor on interactive elements
    const interactiveElements = document.querySelectorAll('button, .option, .option');
    interactiveElements.forEach(el => {
        el.addEventListener('mouseenter', () => {
            trail.classList.remove('circle');
            trail.classList.add('line');
        });
        
        el.addEventListener('mouseleave', () => {
            trail.classList.remove('line');
            trail.classList.add('circle');
        });
    });
}

/**
 * Handle window resize
 */
function initResizeHandler() {
    window.addEventListener('resize', () => {
        if (threeContainer && renderer) {
            const width = threeContainer.clientWidth;
            const height = threeContainer.clientHeight;
            
            renderer.setSize(width, height);
            camera.aspect = width / height;
            camera.updateProjectionMatrix();
        }
    });
}

/**
 * Show error message
 */
function showError(message) {
    const errorEl = document.createElement('div');
    errorEl.style.cssText = `
        background: rgba(255, 0, 0, 0.1);
        border: 1px solid rgba(255, 0, 0, 0.3);
        color: #ff6b6b;
        padding: 12px;
        border-radius: 8px;
        margin: 16px 0;
    `;
    errorEl.textContent = message;
    
    const resultsEl = document.getElementById('forecast-results');
    if (resultsEl && resultsEl.parentNode) {
        resultsEl.parentNode.insertBefore(errorEl, resultsEl);
    }
}

/**
 * Helper: Linear interpolation between two colors
 */
function lerpColor(color1, color2, t) {
    const c1 = new THREE.Color(color1);
    const c2 = new THREE.Color(color2);
    return c1.lerp(c2, t);
}

/**
 * Helper: Convert RGBA to hex
 */
function rgbaToHex(rgba) {
    if (typeof rgba === 'number') {
        const r = Math.round((rgba >> 16) & 0xff) / 255;
        const g = Math.round((rgba >> 8) & 0xff) / 255;
        const b = (rgba & 0xff) / 255;
        return new THREE.Color(r, g, b);
    }
    return new THREE.Color(1, 1, 1);
}