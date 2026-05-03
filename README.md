# 🛰️ Satellite Image Land Classification & Environmental Impact Analyzer

This project is a web-based application that classifies satellite images into land cover categories and generates a simple environmental impact report. It uses deep learning and satellite image datasets like **EuroSAT** or **BigEarthNet** to recognize land types such as industrial, residential, forest, etc.

## 🚀 Features

- Upload a satellite image and detect dominant land cover types (multi-label).
- Uses a trained CNN (ResNet18 or EfficientNet) for high-accuracy classification.
- Displays results with clearly labeled land types and colors.

## 🧠 Model Details

- **Architecture**: ResNet18 / EfficientNet-B0
- **Dataset**: EuroSAT (RGB) or BigEarthNet (multi-label)
- **Input Size**: 224x224 pixels
- **Output**: Multi-label land type predictions with analysis

## 🖼️ Land Cover Classes

Example categories used in EuroSAT:
- AnnualCrop
- Forest
- Residential
- Industrial
- SeaLake
- Highway
- River
- PermanentCrop
- Pasture
- HerbaceousVegetation




