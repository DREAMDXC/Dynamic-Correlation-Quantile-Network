# Dynamic-Correlation-Quantile-Network

## 📌 Project Overview

This repository contains the official implementation of the paper:
**"A Data-driven Dynamic Temporal Correlation Modeling Framework for Renewable Energy Scenario Generation"**

The proposed Dynamic Correlation Quantile Network (DCQN) is a novel framework for short-term renewable energy scenario generation that:

- Models time-varying temporal correlations through dynamic covariance matrices
- Performs nonparametric continuous quantile regression using Implicit Quantile Networks (IQN)
- Decouples joint distribution modeling into correlation and marginal mapping stages
- Outperforms state-of-the-art methods in uncertainty quantification and dynamic correlation capture

**Key Innovations**:

- 🌀 **Dynamic Correlation Modeling**: Adaptive covariance matrices capture weather-influenced temporal dependencies

- 📊 **Continuous Inverse Sampling**: Enables flexible scenario generation without predefined quantiles

- 🧠 **Interpretable Representations**: Visualizable correlation structures vs black-box alternatives

- ⚡ **Optimization Framework**: Ensures accuracy through proper scoring rules (quantile loss + MLE)

  

## 🚀 Key Features

- **Dynamic Temporal Correlation**: Models time-varying dependencies influenced by weather systems
- **Nonparametric Quantile Regression**: Continuous modeling of marginal distributions
- **Two-Stage Generation**: Separates correlation and marginal distribution modeling
- **High Performance**: State-of-the-art results on wind and solar datasets
- **Visualizable Covariance**: Provides interpretable correlation heatmaps
- **Modular Architecture**: Easy to extend and adapt to new datasets

## 📦 Installation

### System Requirements

- Python 3.8+
- 

## 🛠 Code Structure

```python
Dynamic-Correlation-Quantile-Network/
├── configs/               # Configuration files
├── data/                  # Data loading and preprocessing
│   ├── dataload.py        # Data loader implementation
│   └── preprocess.py      # Data preprocessing scripts
├── models/                # Model architectures
│   ├── dcn.py             # Dynamic Correlation Network
│   ├── iqn.py             # Implicit Quantile Network
│   └── dcqn.py            # Integrated DCQN model
├── assets/                # Visualizations and diagrams
├── trained_models/        # Pre-trained model weights
├── utils/                 # Utility functions
│   ├── metrics.py         # Evaluation metrics
│   └── visualization.py   # Result visualization
├── DCQN.py                # Main training script
├── generate_scenarios.py  # Scenario generation script
├── requirements.txt       # Python dependencies
└── LICENSE
```

## 🧠 Training DCQN

```
python DCQN.py \
  --epoch 1000 \
  --batch_size 64 \
  --lr 0.001 \
  --target_step 24 \
  --farm_number 1 \
  --scenario_number 500
```

## 📊 Reproducing Paper Results

```
python generate_scenarios.py \
  --checkpoint models/dcqn_wind.pth \
  --dataset data/wind_test.npy \
  --output_dir scenarios/ \
  --num_scenarios 1000
```

## 📜 Citation

If you use DCQN in your research, please cite our paper:

bibtex

```
@article{dong2024data,
  title={A Data-driven Dynamic Temporal Correlation Modeling Framework for Renewable Energy Scenario Generation},
  author={Dong, Xiaochong and Liu, Yilin and Zhang, Xuemin and Mei, Shengwei},
  journal={IEEE Transactions on Sustainable Energy},
  year={2024},
  doi={10.1109/TSTE.2024.XXXXXXX}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://license/) file for details.

## 💬 Contact

For questions and collaborations, please contact:

- Xiaochong Dong: [dream_dxc@163.com](https://mailto:dream_dxc@163.com/)
- Xuemin Zhang: [zhangxuemin@mail.tsinghua.edu.cn](https://mailto:zhangxuemin@mail.tsinghua.edu.cn/)

## 🌐 Acknowledgements

This work was supported by:

- National Key R&D Program of China (2022YFB2403000)
- National Natural Science Foundation of China (U21A20146)
- China Postdoctoral Science Foundation (BX20250414)

We thank the Tsinghua University Power System Operation and Control Laboratory for their computational resources and support.
