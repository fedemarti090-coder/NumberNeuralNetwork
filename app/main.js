import { app, BrowserWindow, ipcMain } from 'electron';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs/promises';

import Lenet5 from '../wasm/lenet5.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const wasmDirectory = path.join(__dirname, '..', 'wasm');
let win;
let module = null;
let predictProbabilities = null;
let modelReady = false;

function createWindow() {
    win = new BrowserWindow({
        width: 1200,
        height: 800,
        icon: path.join(__dirname, 'public', 'logo.png'),
        autoHideMenuBar: true,
        webPreferences: {
            preload: path.join(__dirname, 'preload.cjs'),
            nodeIntegration: false,
            contextIsolation: true
        }
    });

    win.loadFile('public/index.html');
}

app.whenReady().then(() => {
    createWindow();

    app.on('activate', () => {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') app.quit();
});

async function startModel() {
    const wasmBuffer = await fs.readFile(path.join(wasmDirectory, 'lenet5.wasm'));
    const dataBuffer = await fs.readFile(path.join(wasmDirectory, 'lenet5.data'));

    module = await Lenet5({
        locateFile(packageName) {
            if (packageName.endsWith('lenet5.wasm')) {
                return 'data:application/wasm;base64,' + wasmBuffer.toString('base64');
            }
            if (packageName.endsWith('lenet5.data')) {
                return 'data:application/octet-stream;base64,' + dataBuffer.toString('base64');
            }
            return packageName;
        },
        getPreloadedPackage(packageName, packageSize) {
            if (packageName.endsWith('lenet5.data') && packageSize === dataBuffer.byteLength) {
                return dataBuffer.buffer.slice(
                    dataBuffer.byteOffset,
                    dataBuffer.byteOffset + dataBuffer.byteLength
                );
            }
            return null;
        }
    });

    predictProbabilities = module.cwrap('predict_probs', 'number', ['number', 'number']);
    const loadWeights = module.cwrap('load_weights', 'number', ['number']);
    modelReady = loadWeights(0) === 1;

    if (!modelReady) {
        throw new Error('Impossibile caricare model/lenet5_weights.bin');
    }
}

ipcMain.handle('model-status', () => modelReady);

ipcMain.handle('predict-pixels', (_event, pixels) => {
    if (!modelReady || !(pixels instanceof Float32Array)) {
        return { digit: -1, error: 'Modello non pronto o input non valido' };
    }

    if (pixels.length !== 784) {
        return { digit: -1, error: 'Attesi 784 pixel, ricevuti ' + pixels.length };
    }

    const inputPointer = module._malloc(784 * Float32Array.BYTES_PER_ELEMENT);
    const outputPointer = module._malloc(10 * Float32Array.BYTES_PER_ELEMENT);
    if (!inputPointer || !outputPointer) {
        if (inputPointer) module._free(inputPointer);
        if (outputPointer) module._free(outputPointer);
        return { digit: -1, error: 'Memoria WASM non disponibile' };
    }

    try {
        module.HEAPF32.set(pixels, inputPointer >>> 2);
        if (predictProbabilities(inputPointer, outputPointer) !== 1) {
            return { digit: -1, error: 'Inferenza fallita' };
        }

        const probabilities = Array.from(
            module.HEAPF32.subarray(outputPointer >>> 2, (outputPointer >>> 2) + 10)
        );
        const digit = probabilities.indexOf(Math.max(...probabilities));
        return { digit, probabilities, error: null };
    } finally {
        module._free(inputPointer);
        module._free(outputPointer);
    }
});

app.whenReady().then(async () => {
    try {
        await startModel();
        console.log('LeNet-5 pronto');
    } catch (error) {
        console.error(error);
    }
});