const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    getModelStatus: () => ipcRenderer.invoke('model-status'),
    predictPixels: (pixels) => ipcRenderer.invoke('predict-pixels', new Float32Array(pixels))
});
