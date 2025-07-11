## 👨‍💻 Project Overview

This repository contains the official implementation of the paper:
**Xiaochong Dong, Yilin Liu, Xuemin Zhang, Shengwei Mei. A Data-driven Dynamic Temporal Correlation Modeling Framework for Renewable Energy Scenario Generation. *IEEE Transactions on Sustainable Energy*, Under Review.**
It is a novel framework for short-term renewable energy scenario generation.

## 🏆 **Key Innovations**

- 🧠 **Decoupled Mapping**: using a decoupled transformation process from prior distributions to target joint distributions.

  joint distribution → marginal distributions + correlation structure

- 🌀 **Dynamic Correlation Modeling**: using neural network to capture temporal correlations dynamically through a covariance matrix

- 🚀  **Continuous Inverse Sampling**: using implicit quantile network models the marginal quantile function, enabling scenario generation process (marginal inverse sampling) in a nonparametric, continuous manner.

## 📦 System Requirements

- ```
  python~=3.11.0
  pytorch~=2.1.0
  numpy~=1.24.3
  pandas~=2.0.3
  scipy~=1.11.1
  matplotlib~=3.7.2
  seaborn~=0.12.2
  ```

## 🧠 Training / Testing DCQN

```
python -m WindModel.DCQN.DCQN
python -m PVModel.DCQN.DCQN
```

## 📊 Scenario Generation

🌪 wind power scenario generation：

<img src=".\Fig\wind power scenario generation.png" alt="Fig8" width="400px" />

☀️ solar power scenario generation：

<img src=".\Fig\solar power scenario generation.png" alt="Fig9" width="400px" />

🌍 wind dynamic temporal correlation：

<img src=".\Fig\wind dynamic temporal correlation.png" alt="Fig10" width="400px" />

🌍 solar dynamic temporal correlation：

<img src=".\Fig\solar dynamic temporal correlation.png" alt="Fig10" width="400px" />

## 📜 Citation

If you use DCQN in your research, please cite our paper: (The paper is currently under final review.)

```
@article{dong2024data,
  title={A Data-driven Dynamic Temporal Correlation Modeling Framework for Renewable Energy Scenario Generation},
  author={Dong, Xiaochong and Liu, Yilin and Zhang, Xuemin and Mei, Shengwei},
  journal={IEEE Transactions on Sustainable Energy},
  year={2025},
  doi={XXXXXXX}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Contact

For questions and collaborations, please contact: Xiaochong Dong: dream_dxc@163.com;dream_dxc@mail.tsinghua.edu.cn

## 🌐 Acknowledgements

This work was supported by:

- National Key R&D Program of China (2022YFB2403000)
- National Natural Science Foundation of China (U21A20146)
- Postdoctoral Fellowship Program and China Postdoctoral Science Foundation (BX20250414)
