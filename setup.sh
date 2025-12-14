#!/bin/bash

# Script de configuration pour Insect-ID-Script
# Détecte l'OS et le matériel pour installer les bonnes dépendances

echo "Détection de l'OS..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "OS détecté : Linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "OS détecté : macOS"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    echo "OS détecté : Windows"
else
    echo "OS non reconnu : $OSTYPE"
    exit 1
fi

echo "Détection du matériel GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU NVIDIA détecté. Installation d'onnxruntime-gpu..."
    pip install onnxruntime-gpu
else
    echo "Pas de GPU NVIDIA détecté ou nvidia-smi non disponible. Installation d'onnxruntime CPU..."
    pip install onnxruntime
fi

echo "Installation des autres dépendances..."
pip install numpy Pillow ijson

echo "Configuration terminée."