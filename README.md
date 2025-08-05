# Minimal SVGD in JAX

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/brpuneet898/stein-project/blob/main/gaussian_SVGD.ipynb)

This repo contains a clean, minimal implementation of Stein Variational Gradient Descent (SVGD) using [JAX](https://github.com/google/jax). It includes two common Bayesian inference examples.

---

## Tests Included

### 1. 2-Component Gaussian Mixture

- Demonstrates SVGD approximating a multimodal distribution.
- Plots particle evolution from a single cloud to two modes.

### 2. Bayesian Logistic Regression

- Dataset: UCI Breast Cancer Wisconsin
- SVGD approximates the posterior over logistic weights.
- Compared with Hamiltonian Monte Carlo (via PyMC).

---

## Outputs

- Particle evolution plots for SVGD
- Histogram of posterior predictive probabilities
- Forest plot comparing HMC posterior

---

## Dependencies

Run on [Google Colab](https://colab.research.google.com) or locally with:

```bash
pip install jax jaxlib matplotlib seaborn scikit-learn arviz pymc
```

## clone

```bash
git clone https://github.com/brpuneet898/stein-project
cd stein-project
pip install -r requirements.txt
jupyter notebook notebooks/gaussian_svgd.ipynb
```

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
