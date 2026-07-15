````markdown
# ⚡ Solar-Pumped Hydro Energy Storage (PHES) Optimizer

An interactive optimization framework for designing **small-scale, solar-integrated Pumped Hydro Energy Storage (PHES)** systems. The framework combines a detailed **physics-based simulator** with an **XGBoost machine learning surrogate model** to rapidly identify cost-effective and energy-efficient PHES designs for homes, farms, rural communities, and islands.

---

## 🚀 Key Features

- **Physics-Based Simulator**
  - Simulates an entire **8760-hour (1-year)** operation.
  - Models:
    - Pipe friction losses
    - Evaporation losses
    - Seepage losses
    - Variable pump efficiency
    - Variable turbine efficiency

- **Machine Learning Surrogate**
  - XGBoost model trained using **3200+ simulation samples**
  - Evaluates candidate designs almost instantly
  - Typical optimization time: **under 10 seconds**

- **Dual Optimization Modes**
  - ⚡ **ML Surrogate (Fast)** – Rapid optimization for design exploration
  - 🔬 **Physics Simulator (Slow)** – Higher accuracy using the full simulator

- **Multi-Objective Optimization**
  - NSGA-II evolutionary algorithm
  - Simultaneously:
    - Maximizes round-trip efficiency
    - Minimizes capital cost

- **Interactive Streamlit Web Application**
  - No programming knowledge required
  - User-friendly interface
  - Generates complete optimized designs
  - Produces an estimated shopping list and cost breakdown

- **Real-World Focus**
  - Designed for Sri Lankan locations
  - Tropical climate modelling
  - Supports multiple reservoir types:
    - `new_tank`
    - `excavated`
    - `pond`
    - `river`

---
## 📸 Application Preview

![Home Page](screenshots/interface-1.png)

---

![Optimization Results](screenshots/interface-2.png)

---

![Shopping List](screenshots/interface-2.png)

---

# 📁 Project Structure

```text
PHES-optimization-framework/
│
├── app.py                          # Main Streamlit application
│
├── src/
│   ├── simulator.py                # 8760-hour physics simulator
│   ├── physics.py                  # Physical equations
│   ├── cost_model.py               # Capital cost estimation
│   ├── solar_data_loader.py        # Solar data using PVlib
│   ├── user_inputs.py              # User input model
│   └── constants.py                # Physical constants
│
├── optimization/
│   ├── optimization.py             # NSGA-II using ML surrogate
│   └── optimization_physics.py     # NSGA-II using physics simulator
│
├── scripts/
│   ├── generate_dataset.py         # Dataset generation
│   └── train_surrogate.py          # Train XGBoost model
│
├── models/                         # Trained ML models
├── data/                           # Generated datasets
├── Requirements.txt                # Python dependencies
└── README.md
```

---

# 🛠 Installation

## 1. Clone the Repository

```bash
git clone https://github.com/UserLasa73/PHES-optimization-framework.git
cd PHES-optimization-framework
```

## 2. Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r Requirements.txt
```

---

# ▶️ Usage

Launch the Streamlit application:

```bash
streamlit run app.py
```

---

## Using the Application

### Step 1 – Enter Site Information

Provide:

- Location
- Solar PV capacity (kWp)
- Daily energy demand (kWh/day)
- Required autonomy (days)
- Reservoir type

Optional constraints:

- Maximum reservoir volume
- Budget
- Minimum efficiency

---

### Step 2 – Select Optimization Mode

Choose one of the following:

- **ML Surrogate (Fast)** ⚡
  - Rapid optimization
  - Ideal for exploring multiple design options

- **Physics Simulator (Slow)** 🔬
  - Uses the full 8760-hour simulator
  - Higher accuracy
  - Suitable for validation

---

### Step 3 – Run Optimization

Click **Optimize Design**.

The application will generate:

- ✅ Optimal PHES design
- ✅ Pareto front
- ✅ Performance statistics
- ✅ Estimated shopping list
- ✅ Cost breakdown

---

# 📊 Comparison Test

The following test compares the **ML Surrogate** and the **Physics Simulator** using the same design constraints.

### Test Configuration

| Parameter | Value |
|-----------|------:|
| Location | Vavuniya |
| PV Capacity | 20.0 kWp |
| Daily Load | 20.0 kWh/day |
| Required Autonomy | 0.5 days |
| Reservoir Type | new_tank |
| Maximum Reservoir Volume | 800 m³ |
| Budget | LKR 4,000,000 |
| Efficiency Constraint | 70% |

---

## ⚡ ML Surrogate (Fast)

```text
BEST DESIGN

Reservoir Volume : 491 m³
Head Height      : 44.6 m
Pipe Diameter    : 0.272 m
Pump Power       : 4.6 kW
Turbine Power    : 10.0 kW

Round-trip Efficiency : 77.7%
Estimated Cost        : LKR 3,991,539
```

**Optimization Time:** **< 10 seconds**

---

## 🔬 Physics Simulator (Slow)

```text
BEST DESIGN

Reservoir Volume : 636 m³
Head Height      : 37.2 m
Pipe Diameter    : 0.349 m
Pump Power       : 2.0 kW
Turbine Power    : 6.4 kW

Round-trip Efficiency : 76.8%
Estimated Cost        : LKR 3,998,089
```

**Optimization Time:** Approximately **10 minutes**

---

## 📈 Comparison Summary

| Metric | ML Surrogate | Physics Simulator |
|-------|-------------:|------------------:|
| Reservoir Volume | 491 m³ | 636 m³ |
| Head Height | 44.6 m | 37.2 m |
| Pipe Diameter | 0.272 m | 0.349 m |
| Pump Power | 4.6 kW | 2.0 kW |
| Turbine Power | 10.0 kW | 6.4 kW |
| Round-trip Efficiency | **77.7%** | **76.8%** |
| Estimated Cost | LKR 3,991,539 | LKR 3,998,089 |
| Optimization Time | **< 10 s** | **≈ 40 min** |

---

### Conclusion

The **ML Surrogate** produced a design with **comparable efficiency (77.7% vs. 76.8%)** and a **similar estimated cost** to the full **Physics Simulator**, while reducing optimization time from **approximately 40 minutes to under 10 seconds**. This demonstrates that the XGBoost surrogate model provides a fast and practical alternative for real-time PHES design optimization with minimal compromise in solution quality.

---

# 🤖 Training the Machine Learning Surrogate

To generate a new dataset:

```bash
python scripts/generate_dataset.py
```


Train the XGBoost surrogate model:

```bash
python scripts/train_surrogate.py
```

The trained model will be saved inside the **models/** directory.

---

# 🧰 Technologies Used

- Python
- Streamlit
- XGBoost
- DEAP (NSGA-II)
- PVlib
- NumPy
- Pandas
- Plotly
- SciPy

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 🙏 Acknowledgments

This project was developed as part of a **BSc (Hons) in Information Technology** research project at the **University of Vavuniya, Sri Lanka**.

Special thanks to the developers and maintainers of:

- Streamlit
- XGBoost
- DEAP
- PVlib
- Plotly
- NumPy
- Pandas
- SciPy

whose open-source libraries made this research possible.
````
