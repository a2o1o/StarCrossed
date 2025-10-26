"""
StarCrossed Main Dashboard
Cinematic Mars colony reproduction simulation with scientific rigor
and emotional storytelling.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import List, Dict, Optional
import json
import time

# Import our modules
from data import CrewMember, DataGenerator, create_sample_crew
from simulation import (
    CompatibilityCalculator, PairingOptimizer, MultiGenerationSimulator,
    CompatibilityFactor, GenerationResult
)
from visualization import PairMorphVisualizer, create_summary_statistics
# --- AUTO-GENERATE CREW IF NONE EXISTS ---
from data import DataGenerator

if 'crew' not in st.session_state or len(st.session_state.crew) == 0:
    generator = DataGenerator(seed=42)
    st.session_state.crew = generator.generate_crew(30, [f"Crew Member {i+1}" for i in range(30)])
    st.session_state.population = len(st.session_state.crew)
    st.rerun()




class StarCrossedApp:
    """Main application class for StarCrossed dashboard."""
    
    def __init__(self):
        self.data_generator = DataGenerator(seed=42)
        self.compatibility_calculator = CompatibilityCalculator()
        self.pairing_optimizer = PairingOptimizer(self.compatibility_calculator)
        self.simulator = MultiGenerationSimulator(self.compatibility_calculator)
        self.visualizer = PairMorphVisualizer()
        
        # Initialize session state
        if 'crew' not in st.session_state:
            st.session_state.crew = []
        if 'simulation_results' not in st.session_state:
            st.session_state.simulation_results = []
        if 'current_section' not in st.session_state:
            st.session_state.current_section = 'landing'
        if 'crew_generated' not in st.session_state:
            st.session_state.crew_generated = False
    
    def run(self):
        """Main application entry point."""
        st.set_page_config(
            page_title="StarCrossed — Designing Humanity Beyond Earth",
            page_icon="🔬",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Apply custom CSS
        self.render_custom_css()
        
        # Navigation/routing
        if st.session_state.current_section == 'landing':
            self.render_landing()
        elif st.session_state.current_section == 'crew_setup':
            self.render_crew_setup()
        elif st.session_state.current_section == 'running':
            self.render_simulation_running()
        elif st.session_state.current_section == 'results':
            self.render_results_dashboard()
    
    def render_custom_css(self):
        """Apply StarCrossed brand styling."""
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600&display=swap');
        
        :root {
            --deep-navy: #050510;
            --crimson: #FF3B52;
            --neon-blue: #00E0FF;
            --warm-orange: #FF8850;
            --text-gray: #E0E0E0;
            --card-bg: #0a0b1a;
        }
        
        /* Global */
        .stApp {
            background: var(--deep-navy) !important;
            color: var(--text-gray) !important;
        }
        
        .main .block-container {
            background: var(--deep-navy);
            color: var(--text-gray);
            font-family: 'Inter', sans-serif;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', monospace !important;
            color: var(--neon-blue) !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(135deg, var(--crimson), var(--warm-orange));
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.75rem 2rem;
            font-family: 'Orbitron', monospace;
            font-weight: 700;
            font-size: 1.1rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            box-shadow: 0 0 20px rgba(255, 59, 82, 0.3);
        }
        
        .stButton > button:hover {
            box-shadow: 0 0 30px rgba(255, 59, 82, 0.6);
            transform: translateY(-2px);
        }
        
        /* Cards */
        .mission-card {
            background: var(--card-bg);
            border: 2px solid var(--neon-blue);
            border-radius: 20px;
            padding: 3rem;
            margin: 2rem auto;
            max-width: 900px;
            box-shadow: 0 0 40px rgba(0, 224, 255, 0.2);
            animation: glow 3s ease-in-out infinite;
        }
        
        @keyframes glow {
            0%, 100% { box-shadow: 0 0 40px rgba(0, 224, 255, 0.2); }
            50% { box-shadow: 0 0 60px rgba(0, 224, 255, 0.4); }
        }
        
        .logo-heart {
            font-size: 4rem;
            color: var(--crimson);
            text-align: center;
            margin: 2rem 0;
        }
        
        .tagline {
            font-family: 'Orbitron', monospace;
            font-size: 1.5rem;
            color: var(--text-gray);
            text-align: center;
            margin: 1rem 0;
            letter-spacing: 3px;
        }
        
        .description {
            font-family: 'Inter', sans-serif;
            font-size: 1.1rem;
            color: var(--text-gray);
            text-align: center;
            margin: 2rem 0;
            line-height: 1.8;
            opacity: 0.85;
        }
        
        /* Metrics */
        .metric-container {
            background: var(--card-bg);
            border: 1px solid var(--neon-blue);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-container:hover {
            border-color: var(--crimson);
            box-shadow: 0 0 20px rgba(255, 59, 82, 0.3);
        }
        
        .metric-value {
            font-family: 'Orbitron', monospace;
            font-size: 2.5rem;
            font-weight: 900;
            color: var(--neon-blue);
            margin: 0.5rem 0;
        }
        
        .metric-label {
            font-family: 'Inter', sans-serif;
            font-size: 0.9rem;
            color: var(--text-gray);
            text-transform: uppercase;
            letter-spacing: 1px;
            opacity: 0.7;
        }
        
        /* Section Headers */
        .section-header {
            font-family: 'Orbitron', monospace;
            font-size: 2.5rem;
            font-weight: 900;
            color: var(--neon-blue);
            text-align: center;
            margin: 3rem 0 2rem 0;
            letter-spacing: 4px;
            text-shadow: 0 0 30px rgba(0, 224, 255, 0.5);
        }
        
        /* Loading Animation */
        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: var(--deep-navy);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 9999;
        }
        
        .spinner {
            border: 4px solid var(--card-bg);
            border-top: 4px solid var(--neon-blue);
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: var(--card-bg);
            color: var(--text-gray);
            border: 2px solid var(--neon-blue);
            font-family: 'Orbitron', monospace;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 2px;
            padding: 1rem 2rem;
            margin-right: 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: var(--neon-blue);
            color: var(--deep-navy);
        }
        
        /* DataTable */
        .dataframe {
            background: var(--card-bg);
            color: var(--text-gray);
        }
        
        /* Remove Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """, unsafe_allow_html=True)
    
    def render_landing(self):
        """Cinematic landing section."""
        st.markdown("""
        <div style="text-align: center; padding: 6rem 2rem; min-height: 80vh;">
            <h1 style="font-family: 'Orbitron', monospace; font-size: 5rem; font-weight: 900; 
                       color: #00E0FF; letter-spacing: 8px; margin: 5rem 0 5rem 0; 
                       text-shadow: 0 0 30px rgba(0, 224, 255, 0.6);">
                StarCrossed
            </h1>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🚀 LAUNCH MISSION", key="launch_btn", use_container_width=True):
                if not st.session_state.crew_generated:
                    # Generate initial crew
                    with st.spinner("Initializing mission parameters..."):
                        st.session_state.crew = self.data_generator.generate_initial_mars_crew()
                        st.session_state.crew_generated = True
                        st.session_state.current_section = 'crew_setup'
                else:
                    st.session_state.current_section = 'crew_setup'
                st.rerun()
    
    def render_crew_setup(self):
        """Crew setup and simulation configuration."""
        st.markdown('<div class="section-header">MISSION CONTROL</div>', unsafe_allow_html=True)
        
        # Two-panel layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<h3 style="font-family: \'Orbitron\'; color: #00E0FF; text-align: center;">CREW CONFIGURATION</h3>', unsafe_allow_html=True)
            
            crew_size = st.slider("Crew Size", min_value=10, max_value=50, value=30, step=5)
            
            if st.button("🔄 GENERATE NEW CREW", use_container_width=True):
                with st.spinner("Generating crew..."):
                    # Generate crew based on size
                    generator = DataGenerator(seed=None)
                    names = [f"Crew Member {i+1}" for i in range(crew_size)]
                    st.session_state.crew = generator.generate_crew(crew_size, names[:crew_size])
                st.rerun()
            
            if st.button("➕ ADD CREW MEMBER", use_container_width=True):
                new_member = self.data_generator.generate_crew_member(f"CM_{len(st.session_state.crew)+1:03d}")
                st.session_state.crew.append(new_member)
                st.rerun()
            
            st.markdown(f"### Crew Status: {len(st.session_state.crew)} members")
            
            if st.session_state.crew:
                crew_df = pd.DataFrame([{
                    'Name': m.name,
                    'Age': m.age,
                    'Fertility': f"{m.fertility_index:.2f}",
                    'Radiation Tolerance': f"{m.radiation_tolerance:.2f}",
                    'VO2 Max': f"{m.vo2_max:.1f}"
                } for m in st.session_state.crew[:10]])  # Show first 10
                st.dataframe(crew_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown('<h3 style="font-family: \'Orbitron\'; color: #00E0FF; text-align: center;">SIMULATION CONTROLS</h3>', unsafe_allow_html=True)
            
            num_generations = st.slider("Number of Generations", min_value=1, max_value=10, value=5)
            
            st.markdown("### Mission Ready")
            
            if len(st.session_state.crew) >= 2:
                if st.button("APPROVE & RUN SIMULATION", type="primary", use_container_width=True):
                    st.session_state.current_section = 'running'
                    st.rerun()
            else:
                st.warning("Generate a crew first to run simulation.")
    
    def render_simulation_running(self):
        """Animated simulation running screen."""
        st.markdown("""
        <div style="text-align: center; margin: 10rem 0;">
            <div class="spinner"></div>
            <h2 style="font-family: 'Orbitron'; color: #00E0FF; margin-top: 2rem; letter-spacing: 3px;">
                ANALYZING MULTI-GENERATION OUTCOMES
            </h2>
            <p style="color: #E0E0E0; margin-top: 1rem; font-size: 1.2rem;">
                Modeling Mars colony sustainability...
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Run the simulation
        num_generations = 5  # Default
        with st.spinner("Running simulation..."):
            st.session_state.simulation_results = self.simulator.simulate_generations(
                st.session_state.crew, num_generations
            )
            time.sleep(1)  # Brief pause for effect
            st.session_state.current_section = 'results'
            st.rerun()
    
    def render_results_dashboard(self):
        """Results dashboard with Genetic and Psychological tabs."""
        st.markdown('<div class="section-header">SIMULATION RESULTS</div>', unsafe_allow_html=True)
        
        # Summary metrics
        if st.session_state.simulation_results:
            results = st.session_state.simulation_results
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{results[-1].diversity_index:.3f}</div>
                    <div class="metric-label">Diversity Index</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                avg_fertility = np.mean([r.average_fertility for r in results])
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{avg_fertility:.3f}</div>
                    <div class="metric-label">Avg Fertility</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{len(results)}</div>
                    <div class="metric-label">Generations</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                final_offspring = len(results[-1].offspring) if results[-1].offspring else 0
                st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-value">{final_offspring}</div>
                    <div class="metric-label">Final Population</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Tabbed analysis
                    # --- TAB SETUP ---
            tab1, tab2, tab3 = st.tabs(["GENETIC ANALYSIS", "PSYCHOLOGICAL ANALYSIS", "FINAL RESULTS"])

            with tab1:
                self.render_genetic_analysis()

            with tab2:
                self.render_psychological_analysis()

            with tab3:
                st.markdown('<h3 style="font-family: Orbitron; color: #00E0FF; text-align: center;">FINAL RESULTS</h3>', unsafe_allow_html=True)
                st.markdown("### Multi-Generation Simulation Overview")

                if st.session_state.simulation_results:
                    from visualization import (
                        plot_generational_diversity,
                        plot_population_trend,
                        plot_mutation_trend
                    )

                    # Pull results from the stored simulation
                    results = st.session_state.simulation_results

                    # Plot diversity over generations
                    st.subheader("📈 Genetic Diversity Over Generations")
                    st.plotly_chart(plot_generational_diversity({
                        "generations": [r.generation for r in results],
                        "diversity": [r.diversity_index for r in results]
                    }), use_container_width=True)

                    # Plot population trend
                    st.subheader("👶 Population Trend")
                    st.plotly_chart(plot_population_trend({
                        "generations": [r.generation for r in results],
                        "population": [len(r.offspring) if r.offspring else 0 for r in results]
                    }), use_container_width=True)

                    # Plot mutation risk
                    st.subheader("🧫 Mutation Risk Trend")
                    st.plotly_chart(plot_mutation_trend({
                        "generations": [r.generation for r in results],
                        "mutation_rate": [r.mutation_rate for r in results]
                    }), use_container_width=True)

                    st.caption("Note: This projection is based on simplified generational and radiation models.")
                else:
                    st.info("Run a simulation to generate final results.")

            

            
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("NEW MISSION", use_container_width=True):
                st.session_state.crew = []
                st.session_state.crew_generated = False
                st.session_state.simulation_results = []
                st.session_state.current_section = 'landing'
                st.rerun()
        with col2:
            if st.button("RUN AGAIN", use_container_width=True):
                st.session_state.simulation_results = []
                st.session_state.current_section = 'crew_setup'
                st.rerun()
        with col3:
            if st.button("EXPORT DATA", use_container_width=True):
                self.export_data()
    
    def render_genetic_analysis(self):
        """Genetic analysis tab."""
        if not st.session_state.simulation_results:
            st.info("No simulation results available.")
            return
        
        results = st.session_state.simulation_results
        
        # Diversity Index Over Generations
        st.markdown("### Diversity Index Evolution")
        generations = [r.generation for r in results]
        diversity = [r.diversity_index for r in results]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=generations, y=diversity, mode='lines+markers',
            line=dict(color='#00E0FF', width=3),
            marker=dict(size=12, color='#00E0FF'),
            name='Diversity Index'
        ))
        fig.update_layout(
            xaxis_title="Generation",
            yaxis_title="Diversity Index",
            template="plotly_dark",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0'),
            title_font=dict(color='#00E0FF')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fertility vs Radiation Risk (using crew data)
        st.markdown("### Fertility vs Radiation Risk")
        if st.session_state.crew:
            crew_data = pd.DataFrame([{
                'Fertility': m.fertility_index,
                'Radiation Tolerance': m.radiation_tolerance,
                'Name': m.name,
                'Age': m.age
            } for m in st.session_state.crew])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=crew_data['Fertility'],
                y=crew_data['Radiation Tolerance'],
                mode='markers+text',
                text=crew_data['Name'].str.split().str[-1],
                textposition='top center',
                marker=dict(size=10, color='#FF3B52'),
                name='Crew Members'
            ))
            fig.update_layout(
                xaxis_title="Fertility Index",
                yaxis_title="Radiation Tolerance",
                template="plotly_dark",
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#E0E0E0'),
                title_font=dict(color='#00E0FF')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_psychological_analysis(self):
        """Psychological analysis tab."""
        if len(st.session_state.crew) < 2:
            st.info("Not enough crew members for psychological analysis.")
            return
        
        # Compatibility Matrix
        st.markdown("### Compatibility Matrix")
        compatibility_fig = self.visualizer.create_compatibility_matrix(
            st.session_state.crew[:15], self.compatibility_calculator  # Limit for readability
        )
        st.plotly_chart(compatibility_fig, use_container_width=True)
        
        # Optimal Pairings
        st.markdown("### Optimal Pairings")
        pairs = self.pairing_optimizer.optimize_pairings(st.session_state.crew)
        
        if pairs:
            pair_data = []
            for i, (member1, member2) in enumerate(pairs[:10], 1):
                scores = self.compatibility_calculator.calculate_compatibility(member1, member2)
                overall_score = sum(scores[factor.value] * self.compatibility_calculator.weights[factor] 
                                  for factor in CompatibilityFactor)
                pair_data.append({
                    'Pair': f"{i}. {member1.name.split()[-1]} & {member2.name.split()[-1]}",
                    'Score': f"{overall_score:.3f}",
                    'Age Diff': abs(member1.age - member2.age),
                    'Fertility Avg': f"{(member1.fertility_index + member2.fertility_index)/2:.2f}"
                })
            pair_df = pd.DataFrame(pair_data)
            st.dataframe(pair_df, use_container_width=True, hide_index=True)
    
    def export_data(self):
        """Export simulation data as JSON."""
        if st.session_state.simulation_results:
            data = {
                'crew_size': len(st.session_state.crew),
                'generations': len(st.session_state.simulation_results),
                'results': []
            }
            
            for result in st.session_state.simulation_results:
                data['results'].append({
                    'generation': result.generation,
                    'diversity_index': result.diversity_index,
                    'average_fertility': result.average_fertility,
                    'mutation_rate': result.mutation_rate,
                    'radiation_exposure': result.radiation_exposure,
                    'offspring_count': len(result.offspring)
                })
            
            st.download_button(
                label="DOWNLOAD JSON",
                data=json.dumps(data, indent=2),
                file_name="starcrossed_results.json",
                mime="application/json"
            )


def main():
    """Main application entry point."""
    app = StarCrossedApp()
    app.run()


if __name__ == "__main__":
    main()