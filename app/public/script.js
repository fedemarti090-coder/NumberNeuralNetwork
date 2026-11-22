const canvas = document.getElementById("digitCanvas");
const predictButton = document.getElementById("predictButton");
const clearButton = document.getElementById("clearButton");
const result = document.getElementById("result");
const modelStatus = document.getElementById("modelStatus");
const probabilityChart = document.getElementById("probabilityChart");
const context = canvas.getContext("2d", { willReadFrequently: true });

context.fillStyle = "black";
context.fillRect(0, 0, canvas.width, canvas.height);

const chartRows = new Map();
for (let digit = 9; digit >= 0; digit--) {
    const row = document.createElement("div");
    row.className = "chart-row";
    row.innerHTML = `
        <span class="chart-label">${digit}</span>
        <div class="bar-track"><div class="bar-fill"></div></div>
        <span class="chart-value">—%</span>
    `;
    probabilityChart.append(row);
    chartRows.set(digit, {
        row,
        fill: row.querySelector(".bar-fill"),
        value: row.querySelector(".chart-value")
    });
}

let drawing = false;
let modelIsReady = false;
let predictionTimer = null;
let predictionInFlight = false;
let predictionPending = false;
let canvasRevision = 0;
let displayedProbabilities = Array(10).fill(0);
let chartAnimationFrame = null;

function schedulePrediction(delay = 180) {
    canvasRevision++;
    window.clearTimeout(predictionTimer);
    if (modelIsReady) {
        predictionTimer = window.setTimeout(() => {
            predictionTimer = null;
            requestPrediction(canvasRevision);
        }, delay);
    }
}

function getCanvasPoint(event) {
    const bounds = canvas.getBoundingClientRect();
    return {
        x: (event.clientX - bounds.left) * (canvas.width / bounds.width),
        y: (event.clientY - bounds.top) * (canvas.height / bounds.height)
    };
}

canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    drawing = true;
    canvas.setPointerCapture(event.pointerId);
    const point = getCanvasPoint(event);
    context.beginPath();
    context.moveTo(point.x, point.y);
    context.lineTo(point.x + 0.01, point.y + 0.01);
    context.strokeStyle = "white";
    context.lineWidth = 1.35;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.stroke();
    schedulePrediction();
});

canvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    const point = getCanvasPoint(event);
    context.strokeStyle = "white";
    context.lineWidth = 1.35;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.lineTo(point.x, point.y);
    context.stroke();
    schedulePrediction();
});

canvas.addEventListener("pointerup", () => {
    drawing = false;
    schedulePrediction(80);
});

canvas.addEventListener("pointercancel", () => {
    drawing = false;
    schedulePrediction(80);
});

clearButton.addEventListener("click", () => {
    canvasRevision++;
    window.clearTimeout(predictionTimer);
    predictionTimer = null;
    predictionPending = false;
    context.fillStyle = "black";
    context.fillRect(0, 0, canvas.width, canvas.height);
    result.textContent = "Disegna una cifra";
    updateChart([]);
});

function readModelInput() {
    const source = context.getImageData(0, 0, canvas.width, canvas.height);
    let minX = canvas.width;
    let minY = canvas.height;
    let maxX = -1;
    let maxY = -1;

    for (let y = 0; y < canvas.height; y++) {
        for (let x = 0; x < canvas.width; x++) {
            const pixel = (y * canvas.width + x) * 4;
            if ((source.data[pixel] + source.data[pixel + 1] + source.data[pixel + 2]) / 3 > 32) {
                minX = Math.min(minX, x);
                minY = Math.min(minY, y);
                maxX = Math.max(maxX, x);
                maxY = Math.max(maxY, y);
            }
        }
    }

    if (maxX < minX || maxY < minY) {
        return null;
    }

    const tinyCanvas = document.createElement("canvas");
    tinyCanvas.width = 28;
    tinyCanvas.height = 28;
    const tinyContext = tinyCanvas.getContext("2d", { willReadFrequently: true });
    tinyContext.fillStyle = "black";
    tinyContext.fillRect(0, 0, 28, 28);

    const digitWidth = maxX - minX + 1;
    const digitHeight = maxY - minY + 1;
    const scale = 20 / Math.max(digitWidth, digitHeight);
    const targetWidth = digitWidth * scale;
    const targetHeight = digitHeight * scale;
    tinyContext.drawImage(
        canvas,
        minX, minY, digitWidth, digitHeight,
        (28 - targetWidth) / 2, (28 - targetHeight) / 2, targetWidth, targetHeight
    );

    const pixels = tinyContext.getImageData(0, 0, 28, 28).data;
    const input = new Float32Array(784);

    for (let i = 0, p = 0; i < pixels.length; i += 4, p++) {
        input[p] = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / (3 * 255);
    }

    return input;
}

async function requestPrediction(revision) {
    if (!modelIsReady) return;
    if (predictionInFlight) {
        predictionPending = true;
        return;
    }

    predictionInFlight = true;
    predictButton.disabled = true;

    try {
        const input = readModelInput();
        if (!input) {
            if (revision === canvasRevision) {
                result.textContent = "Disegna una cifra";
                updateChart([]);
            }
            return;
        }

        result.textContent = "Analisi in corso…";
        const response = await window.electronAPI.predictPixels(input);
        if (revision !== canvasRevision) return;
        if (response.error) {
            result.textContent = response.error;
            return;
        }

        result.textContent = `Il modello riconosce ${response.digit}`;
        updateChart(response.probabilities, response.digit);
    } catch (error) {
        if (revision === canvasRevision) {
            result.textContent = `Errore durante l'inferenza: ${error.message}`;
        }
    } finally {
        predictionInFlight = false;
        predictButton.disabled = !modelIsReady;

        if (predictionPending) {
            predictionPending = false;
            window.clearTimeout(predictionTimer);
            predictionTimer = null;
            requestPrediction(canvasRevision);
        }
    }
}

predictButton.addEventListener("click", () => {
    window.clearTimeout(predictionTimer);
    predictionTimer = null;
    canvasRevision++;
    requestPrediction(canvasRevision);
});

function updateChart(probabilities, predictedDigit = -1) {
    const startValues = displayedProbabilities.slice();
    const targetValues = Array.from({ length: 10 }, (_, digit) => probabilities[digit] || 0);
    const startTime = performance.now();
    const duration = 260;

    if (chartAnimationFrame !== null) {
        cancelAnimationFrame(chartAnimationFrame);
    }

    for (const [digit, elements] of chartRows) {
        elements.row.classList.toggle("is-top", digit === predictedDigit);
    }

    function animateChart(now) {
        const progress = Math.min(1, (now - startTime) / duration);
        const eased = 1 - Math.pow(1 - progress, 3);

        displayedProbabilities = targetValues.map((target, digit) =>
            startValues[digit] + (target - startValues[digit]) * eased
        );

        for (const [digit, elements] of chartRows) {
            const probability = displayedProbabilities[digit];
            elements.fill.style.width = `${Math.max(0, Math.min(100, probability * 100))}%`;
            elements.value.textContent = probabilities.length
                ? `${(probability * 100).toFixed(1)}%`
                : "—%";
        }

        if (progress < 1) {
            chartAnimationFrame = requestAnimationFrame(animateChart);
        } else {
            chartAnimationFrame = null;
        }
    }

    chartAnimationFrame = requestAnimationFrame(animateChart);
}

async function waitForModel() {
    while (!(await window.electronAPI.getModelStatus())) {
        await new Promise((resolve) => setTimeout(resolve, 100));
    }

    modelIsReady = true;
    predictButton.disabled = false;
    modelStatus.textContent = "Modello pronto";
    modelStatus.previousElementSibling.classList.add("ready");
    result.textContent = "Disegna una cifra";
    if (canvasRevision > 0) schedulePrediction(0);
}

waitForModel();