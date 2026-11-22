@echo off
setlocal

where em++ >nul 2>nul
if errorlevel 1 (
    echo Emscripten SDK non trovato. Attiva emsdk_env.bat e riprova.
    exit /b 1
)

if not exist wasm mkdir wasm

em++ src/WASM/wasm_bridge.cpp ^
    -I src ^
    -std=c++17 ^
    -O3 ^
    -s WASM=1 ^
    -s MODULARIZE=1 ^
    -s EXPORT_ES6=1 ^
    -s ENVIRONMENT=web ^
    -s ALLOW_MEMORY_GROWTH=1 ^
    -s "EXPORTED_RUNTIME_METHODS=['ccall','cwrap','HEAPF32']" ^
    -s "EXPORTED_FUNCTIONS=['_load_weights','_predict_784','_predict_probs','_malloc','_free']" ^
    --preload-file src/model@/model ^
    -o wasm/lenet5.js

if errorlevel 1 (
    echo Compilazione WASM fallita.
    exit /b 1
)

echo Build completata: wasm/lenet5.js, wasm/lenet5.wasm e wasm/lenet5.data
