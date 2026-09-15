# Temperature Regression Analysis Workflow

**Keywords:** `temperature-regression`, `reproducible-workflow`, `scikit-learn`, `FAIR data`

A from-scratch implementation of multiple linear regression using the
**normal equation**, applied to a Swedish weather-station dataset to model
temperature as a function of latitude, longitude, and elevation.

The solution is computed directly with NumPy matrix operations (no
`scikit-learn`), and cross-checked against NumPy's own least-squares solver to
confirm correctness.

## Background

This was the normal-equation component of a group assignment for
`<course name>`. The full assignment covered several approaches to linear
regression; **this repository contains only the part I implemented** — building
the design matrix, solving the normal equation, validating the result, and
writing per-station predictions to a new CSV.

## The math

We model temperature `y` as a linear combination of three spatial features plus
an intercept:

```
y ≈ β₀ + β₁·lat + β₂·long + β₃·elev
```

Stacking all `n` stations into a design matrix `X` of shape `(n, 4)` (a leading
column of ones supplies the intercept), the least-squares fit minimises the
squared residuals:

```
minimise  ‖ y − Xβ ‖²
```

Setting the gradient with respect to `β` to zero gives the **normal equation**:

```
XᵀX β = Xᵀy        ⟹        β = (XᵀX)⁻¹ Xᵀy
```

In the code, `X` is stored transposed as `matrix` (shape `(4, n)`), so `XᵀX`
appears as `matrix @ matrix.T` and the solution reads:

```python
beta = np.linalg.inv(matrix @ matrix.T) @ matrix @ y
```

## Validation

Forming `(XᵀX)⁻¹` explicitly is convenient but numerically fragile when `XᵀX`
is ill-conditioned. To confirm the implementation is correct, the result is
compared against `np.linalg.lstsq`, which solves the same least-squares problem
with a more stable SVD-based routine:

- `np.allclose(beta, beta_lstsq)` should return `True`
- the maximum coefficient difference is around `1e-12` (float precision)

The condition number of `XᵀX` is also reported. Because the features differ
greatly in scale (latitude and longitude are tens of degrees, elevation is
hundreds of metres), this number is fairly large — standardising the features
beforehand would improve conditioning. It is left unstandardised here to keep
the coefficients in their original, interpretable units.

## Repository structure

```
.
├── normal_equation.py        # load data, solve, validate, write predictions
├── temperature_data.csv      # input dataset (see Data section)
├── requirements.txt
└── README.md
```

## Data

The input file `temperature_data.csv` contains one row per weather station with
the columns:

| index | column        |
|-------|---------------|
| 0     | station id    |
| 1     | station name  |
| 2     | latitude      |
| 3     | longitude     |
| 4     | elevation (m) |
| 5     | temperature   |

Rows with a missing temperature value are skipped during loading.

> Data source: `<add source / link here>`

## How to run

```bash
pip install -r requirements.txt
python normal_equation.py
```

This prints the fitted coefficients and the validation summary, and writes
`predicted_temperature_data.csv` — the original data with an added
`predicted temperature` column.

## Requirements

```
CSV
numpy
```
## Discussion: Alignment with FAIR Data Principles
* Findable: The workflow is published on a public GitHub repository with rich metadata, standardized topic tags and a structured README.md. 
  
* Accessible: All assets—code, datasets, and documentation—are hosted via standard open protocols.
  
* Interoperable: Data structures and code follow non-proprietary, community-standard schemas: CSV for tabular records. Variable (4 parameters) and expected value (temperature) are documented with clear scientific units, allowing seamless integration with external data pipelines.
  
* Reusable: Code and data are released under the permissive MIT License, clearly stating terms for community modification and redistribution. Exact execution environments are locked using environment.yml for version-pinned reproducibility alongside step-by-step documentation.
