# 🚀 PairMorph - Mars Colony Reproduction Simulator

**AI-driven simulation tool that models how a small human colony could sustain healthy, genetically diverse reproduction under Martian conditions.**

## 🌍 Overview

PairMorph creates *synthetic physiological and psychological data* for simulated crew members, applies *evolutionary pairing algorithms* to optimize genetic diversity and minimize fertility/radiation risk, and visualizes the *multi-generation outcomes* (family trees, mutation risk, diversity index).

It's designed to look scientifically credible — not a game, but a research-inspired simulation tool (think: "NASA policy sandbox meets ethical AI").

## 🧠 System Architecture

- **`data.py`** → Generates *realistic physiological data* based on NASA-style distributions (age, radiation tolerance, VO₂ max, fertility index, immune resilience)
- **`simulation.py`** → Computes compatibility, radiation exposure decay, mutation risk, and pairing optimization (genetic algorithms)
- **`visualization.py`** → Interactive visualizations (Plotly) of compatibility, fertility trends, diversity evolution
- **`main.py`** → Streamlit dashboard with NASA-anchored scientific analysis interface

## 🧩 Core Features

1. **Synthetic data, not real genomes** (ethical + flexible)
2. **Radiation-driven fertility decay** and mutation-risk calculation
3. **Multi-objective pairing optimizer** — balances diversity, fertility, and psychological compatibility
4. **Visual reports** for policymakers or educators
5. **Scientific methodology** with NASA research foundations

## 🧬 Technical Stack

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **Plotly** - Interactive visualizations
- **Streamlit** - Web application framework
- **SciPy** - Scientific computing

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd pairmorph.py
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run main.py
   ```

4. **Open your browser** to `http://localhost:8501`

### Basic Usage

1. **Generate Crew**: Click "🔄 Generate New Crew" to create a Mars mission crew
2. **Choose Analysis**: The interface provides comprehensive scientific analysis
3. **Check Compatibility**: Use the compatibility matrix to analyze crew member pairs
4. **Run Simulation**: Set number of generations and click "🚀 Run Simulation"
5. **Analyze Results**: View family trees, diversity evolution, and radiation impact

## 🔬 Scientific Analysis Interface

The application provides comprehensive scientific analysis with:

- **NASA-anchored realism** with detailed analysis
- **Comprehensive compatibility matrix** with weighted factors
- **Multi-generation simulation** with inheritance modeling
- **Radiation impact analysis** with scientific methodology
- **Detailed statistics** and export capabilities

## 🧪 Scientific Methodology

### Radiation Modeling
- **Mars surface radiation**: ~0.6 mSv/day (vs Earth's ~0.1 mSv/day)
- **Exponential fertility decay** model based on radiation biology research
- **Mutation probability** increases with cumulative exposure

### Compatibility Factors
- **Genetic Diversity (30%)**: Blood type compatibility and genetic marker diversity
- **Fertility Match (25%)**: Optimal fertility range assessment
- **Psychological Compatibility (20%)**: Stress resistance, social compatibility, circadian rhythms
- **Radiation Risk (15%)**: Combined radiation tolerance and exposure
- **Age Compatibility (10%)**: Optimal age difference for reproduction

### Inheritance Model
- **Traits inherited** as averages with genetic variation
- **Mutation probability** proportional to radiation exposure
- **Simplified blood type inheritance** (real genetics more complex)

## 📊 Key Visualizations

### Crew Overview Dashboard
- Age distribution and physical characteristics
- Fertility & health metrics
- Psychological trait analysis
- Radiation exposure tracking
- Blood type distribution

### Compatibility Analysis
- Heatmap matrix of all crew member pairs
- Detailed factor breakdown for optimal pairings
- Age difference and fertility level analysis

### Multi-Generation Timeline
- Genetic diversity evolution over time
- Fertility trends and mutation rates
- Population size changes
- Radiation exposure accumulation

### Family Tree
- Visual representation of relationships across generations
- Parent-offspring connections
- Generation-by-generation population tracking

### Radiation Impact Analysis
- Fertility decay over mission duration
- Mutation risk increase with exposure
- Individual tolerance variations

## 🔧 Advanced Features

### Data Export
- **Crew data export** as JSON for external analysis
- **Simulation results export** with detailed generation data
- **Compatibility matrices** for further research

### Customization
- **Adjustable compatibility weights** for different scenarios
- **Variable crew sizes** (30 members typical)
- **Configurable simulation parameters**

### Multi-Generation Simulation
- **Trait inheritance** with genetic variation
- **Radiation accumulation** over time
- **Population dynamics** modeling
- **Diversity maintenance** strategies

## 🧬 Example Scenarios

### Scenario 1: Initial Mars Mission
- **30 crew members** with diverse backgrounds
- **High genetic diversity** for long-term sustainability
- **Optimal age ranges** for reproduction potential

### Scenario 2: Multi-Generation Colony
- **5+ generations** of simulated reproduction
- **Radiation exposure** accumulation over decades
- **Genetic diversity** maintenance strategies

### Scenario 3: Radiation Impact Study
- **Fertility decay** modeling under Mars conditions
- **Mutation risk** assessment for colony health
- **Mitigation strategies** for radiation exposure

## 🎯 Use Cases

### Research & Education
- **Space biology education** for students and researchers
- **Policy simulation** for Mars mission planning
- **Ethics discussion** around space colonization

### Mission Planning
- **Crew selection** optimization for long-term missions
- **Reproduction planning** for sustainable colonies
- **Risk assessment** for radiation exposure

### Public Engagement
- **Science communication** about Mars colonization
- **Interactive demonstrations** of space biology concepts
- **Educational tool** for space exploration topics

## 🔬 Data Sources & Validation

### NASA Research
- **Astronaut selection criteria** and physiological ranges
- **Radiation exposure limits** and health monitoring
- **Mission duration** and environmental factors

### Scientific Literature
- **Radiation biology** research on fertility and mutation rates
- **Population genetics** principles for diversity maintenance
- **Space medicine** studies on long-term spaceflight effects

### Synthetic Data Approach
- **Ethical considerations** - no real genetic data used
- **Realistic distributions** based on scientific research
- **Reproducible results** with seeded random generation

## 🚀 Future Enhancements

### Planned Features
- **Multi-colony simulation** with migration between settlements
- **Resource constraint modeling** (food, water, energy)
- **Advanced inheritance** with more complex genetic traits
- **Ethics dashboard** for policy decision support

### Research Integration
- **Real genetic data** integration (with proper consent)
- **Machine learning** optimization of pairing algorithms
- **Collaborative research** with space agencies
- **Peer-reviewed validation** of simulation models

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for:
- **Code style** and documentation standards
- **Testing requirements** for new features
- **Scientific validation** of new models
- **Ethics considerations** for space colonization research

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **NASA** for astronaut selection criteria and space medicine research
- **Space biology community** for radiation effects research
- **Population genetics researchers** for diversity maintenance principles
- **Streamlit and Plotly** communities for excellent visualization tools

---

**Built with ❤️ for the future of human space exploration**

*"The future of humanity is fundamentally going to bifurcate along two directions: Either we're going to become a multi-planet species and a spacefaring civilization, or we're going to stay stuck on one planet until some eventual extinction event."* - Elon Musk

