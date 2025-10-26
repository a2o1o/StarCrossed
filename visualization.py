"""
PairMorph Visualization Module
Creates interactive visualizations for compatibility analysis, fertility trends,
genetic diversity evolution, and multi-generation outcomes using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st

from data import CrewMember, Gender, BloodType
from simulation import GenerationResult, CompatibilityCalculator, CompatibilityFactor


class PairMorphVisualizer:
    """Creates comprehensive visualizations for PairMorph simulation results."""
    
    def __init__(self):
        self.color_palette = {
            'primary': '#00d4ff',      # Cyan blue
            'secondary': '#ff6b35',    # Orange
            'success': '#00ff88',      # Green
            'warning': '#ffaa00',      # Amber
            'info': '#8b5cf6',         # Purple
            'light': '#fbbf24',        # Yellow
            'dark': '#1e293b',         # Dark blue-gray
            'accent': '#06b6d4',       # Light cyan
            'danger': '#ef4444'        # Red
        }
    
    def create_crew_overview_dashboard(self, crew: List[CrewMember]) -> go.Figure:
        """Create a comprehensive dashboard showing crew member characteristics."""
        # Prepare data
        data = []
        for member in crew:
            data.append({
                'Name': member.name,
                'Age': member.age,
                'Gender': member.gender.value,
                'Blood Type': member.blood_type.value,
                'Height (cm)': member.height_cm,
                'Weight (kg)': member.weight_kg,
                'VO2 Max': member.vo2_max,
                'Fertility Index': member.fertility_index,
                'Radiation Tolerance': member.radiation_tolerance,
                'Immune Resilience': member.immune_resilience,
                'Stress Resistance': member.stress_resistance,
                'Social Compatibility': member.social_compatibility,
                'Radiation Exposure': member.radiation_exposure
            })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Age Distribution', 'Physical Characteristics',
                          'Fertility & Health', 'Psychological Traits',
                          'Radiation Exposure', 'Blood Type Distribution'),
            specs=[[{"type": "histogram"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Age distribution
        fig.add_trace(
            go.Histogram(x=df['Age'], name='Age Distribution', 
                        marker_color=self.color_palette['primary']),
            row=1, col=1
        )
        
        # Physical characteristics scatter
        fig.add_trace(
            go.Scatter(x=df['Height (cm)'], y=df['Weight (kg)'],
                      mode='markers+text', text=df['Name'],
                      textposition='top center', name='Physical Traits',
                      marker=dict(size=df['VO2 Max'], color=df['VO2 Max'],
                                colorscale='Viridis', showscale=True)),
            row=1, col=2
        )
        
        # Fertility & Health
        fig.add_trace(
            go.Scatter(x=df['Fertility Index'], y=df['Immune Resilience'],
                      mode='markers+text', text=df['Name'],
                      textposition='top center', name='Health Traits',
                      marker=dict(size=df['Radiation Tolerance']*20,
                                color=df['Radiation Tolerance'],
                                colorscale='RdYlGn')),
            row=2, col=1
        )
        
        # Psychological traits
        fig.add_trace(
            go.Scatter(x=df['Stress Resistance'], y=df['Social Compatibility'],
                      mode='markers+text', text=df['Name'],
                      textposition='top center', name='Psychological Traits',
                      marker=dict(size=15, color=self.color_palette['info'])),
            row=2, col=2
        )
        
        # Radiation exposure
        fig.add_trace(
            go.Bar(x=df['Name'], y=df['Radiation Exposure'],
                  name='Radiation Exposure', marker_color=self.color_palette['warning']),
            row=3, col=1
        )
        
        # Blood type distribution
        blood_counts = df['Blood Type'].value_counts()
        fig.add_trace(
            go.Pie(labels=blood_counts.index, values=blood_counts.values,
                  name='Blood Types'),
            row=3, col=2
        )
        
        fig.update_layout(
            title_text="Mars Colony Crew Overview Dashboard",
            showlegend=False,
            height=800,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Rajdhani'),
            title_font=dict(color='#00d4ff', family='Orbitron', size=20)
        )
        
        return fig
    
    def create_compatibility_matrix(self, crew: List[CrewMember], 
                                  calculator: CompatibilityCalculator) -> go.Figure:
        """Create a heatmap showing compatibility between all crew members."""
        n = len(crew)
        compatibility_matrix = np.zeros((n, n))
        
        # Calculate compatibility scores
        for i in range(n):
            for j in range(n):
                if i != j:
                    scores = calculator.calculate_compatibility(crew[i], crew[j])
                    weighted_score = sum(scores[factor.value] * calculator.weights[factor] 
                                       for factor in CompatibilityFactor)
                    compatibility_matrix[i][j] = weighted_score
                else:
                    compatibility_matrix[i][j] = 1.0  # Self-compatibility
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=compatibility_matrix,
            x=[member.name for member in crew],
            y=[member.name for member in crew],
            colorscale='RdYlGn',
            zmin=0, zmax=1,
            text=np.round(compatibility_matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Crew Compatibility Matrix",
            xaxis_title="Crew Member",
            yaxis_title="Crew Member",
            width=600,
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', family='Rajdhani'),
            title_font=dict(color='#00d4ff', family='Orbitron', size=18)
        )
        
        return fig
    
    def create_pairing_analysis(self, pairs: List[Tuple[CrewMember, CrewMember]], 
                              calculator: CompatibilityCalculator) -> go.Figure:
        """Create detailed analysis of optimal pairings."""
        pair_data = []
        
        for i, (member1, member2) in enumerate(pairs):
            scores = calculator.calculate_compatibility(member1, member2)
            weighted_score = sum(scores[factor.value] * calculator.weights[factor] 
                               for factor in CompatibilityFactor)
            
            pair_data.append({
                'Pair': f"{member1.name} & {member2.name}",
                'Overall Score': weighted_score,
                'Genetic Diversity': scores['genetic_diversity'],
                'Fertility Match': scores['fertility_match'],
                'Psychological': scores['psychological_compatibility'],
                'Radiation Risk': scores['radiation_risk'],
                'Age Compatibility': scores['age_compatibility'],
                'Age Diff': abs(member1.age - member2.age),
                'Fertility Avg': (member1.fertility_index + member2.fertility_index) / 2
            })
        
        df = pd.DataFrame(pair_data)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Overall Compatibility Scores', 'Detailed Factor Breakdown',
                          'Age Differences', 'Fertility Levels'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Overall scores
        fig.add_trace(
            go.Bar(x=df['Pair'], y=df['Overall Score'],
                  name='Overall Score', marker_color=self.color_palette['primary']),
            row=1, col=1
        )
        
        # Detailed breakdown
        factors = ['Genetic Diversity', 'Fertility Match', 'Psychological', 
                  'Radiation Risk', 'Age Compatibility']
        colors = [self.color_palette['primary'], self.color_palette['secondary'],
                 self.color_palette['success'], self.color_palette['warning'],
                 self.color_palette['info']]
        
        for i, factor in enumerate(factors):
            fig.add_trace(
                go.Bar(x=df['Pair'], y=df[factor], name=factor,
                      marker_color=colors[i], opacity=0.7),
                row=1, col=2
            )
        
        # Age differences
        fig.add_trace(
            go.Bar(x=df['Pair'], y=df['Age Diff'],
                  name='Age Difference', marker_color=self.color_palette['light']),
            row=2, col=1
        )
        
        # Fertility scatter
        fig.add_trace(
            go.Scatter(x=df['Pair'], y=df['Fertility Avg'],
                      mode='markers', name='Avg Fertility',
                      marker=dict(size=15, color=self.color_palette['dark'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Optimal Pairing Analysis",
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_generation_timeline(self, results: List[GenerationResult]) -> go.Figure:
        """Create timeline showing evolution across generations."""
        generations = [r.generation for r in results]
        diversity = [r.diversity_index for r in results]
        fertility = [r.average_fertility for r in results]
        mutation_rate = [r.mutation_rate for r in results]
        radiation = [r.radiation_exposure for r in results]
        population = [len(r.offspring) for r in results]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Genetic Diversity Evolution', 'Fertility Trends',
                          'Mutation Rate & Radiation', 'Population Size'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}]]
        )
        
        # Genetic diversity
        fig.add_trace(
            go.Scatter(x=generations, y=diversity, mode='lines+markers',
                      name='Genetic Diversity', line=dict(color=self.color_palette['primary'])),
            row=1, col=1
        )
        
        # Fertility trends
        fig.add_trace(
            go.Scatter(x=generations, y=fertility, mode='lines+markers',
                      name='Average Fertility', line=dict(color=self.color_palette['secondary'])),
            row=1, col=2
        )
        
        # Mutation rate
        fig.add_trace(
            go.Scatter(x=generations, y=mutation_rate, mode='lines+markers',
                      name='Mutation Rate', line=dict(color=self.color_palette['warning'])),
            row=2, col=1
        )
        
        # Radiation exposure (secondary y-axis)
        fig.add_trace(
            go.Scatter(x=generations, y=radiation, mode='lines+markers',
                      name='Radiation Exposure', line=dict(color=self.color_palette['dark'])),
            row=2, col=1, secondary_y=True
        )
        
        # Population size
        fig.add_trace(
            go.Bar(x=generations, y=population, name='Population Size',
                  marker_color=self.color_palette['success']),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Multi-Generation Evolution Timeline",
            height=600,
            showlegend=True
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Diversity Index", row=1, col=1)
        fig.update_yaxes(title_text="Fertility Index", row=1, col=2)
        fig.update_yaxes(title_text="Mutation Rate", row=2, col=1)
        fig.update_yaxes(title_text="Radiation (mSv)", row=2, col=1, secondary_y=True)
        fig.update_yaxes(title_text="Population Size", row=2, col=2)
        
        return fig
    
    def create_family_tree(self, results: List[GenerationResult]) -> go.Figure:
        """Create a family tree visualization showing relationships across generations."""
        # This is a simplified family tree - in practice, you'd want more sophisticated tree layout
        fig = go.Figure()
        
        y_positions = {}
        x_positions = {}
        
        # Calculate positions for each generation
        for result in results:
            generation = result.generation
            parents = result.parents
            offspring = result.offspring
            
            # Position parents
            for i, (parent1, parent2) in enumerate(parents):
                parent_y = generation * 2
                parent1_x = i * 4
                parent2_x = i * 4 + 1
                
                y_positions[parent1.id] = parent_y
                x_positions[parent1.id] = parent1_x
                y_positions[parent2.id] = parent_y
                x_positions[parent2.id] = parent2_x
                
                # Add parent nodes
                fig.add_trace(go.Scatter(
                    x=[parent1_x], y=[parent_y],
                    mode='markers+text',
                    marker=dict(size=15, color=self.color_palette['primary']),
                    text=[parent1.name.split()[-1]],  # Last name only
                    textposition='middle center',
                    name=f'Generation {generation} Parents',
                    showlegend=False
                ))
            
            # Position offspring
            for i, child in enumerate(offspring):
                child_y = generation * 2 + 1
                child_x = i * 2 + 0.5
                
                y_positions[child.id] = child_y
                x_positions[child.id] = child_x
                
                # Add child nodes
                fig.add_trace(go.Scatter(
                    x=[child_x], y=[child_y],
                    mode='markers+text',
                    marker=dict(size=12, color=self.color_palette['secondary']),
                    text=[f"Gen{generation+1}_{i+1}"],
                    textposition='middle center',
                    name=f'Generation {generation+1} Offspring',
                    showlegend=False
                ))
        
        # Add connecting lines (simplified)
        for result in results:
            generation = result.generation
            parents = result.parents
            offspring = result.offspring
            
            for i, (parent1, parent2) in enumerate(parents):
                if i < len(offspring):
                    child = offspring[i]
                    
                    # Line from parents to child
                    fig.add_trace(go.Scatter(
                        x=[x_positions[parent1.id], x_positions[parent2.id], x_positions[child.id]],
                        y=[y_positions[parent1.id], y_positions[parent2.id], y_positions[child.id]],
                        mode='lines',
                        line=dict(color='gray', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        fig.update_layout(
            title="Mars Colony Family Tree",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=400,
            width=800
        )
        
        return fig
    
    def create_radiation_impact_analysis(self, crew: List[CrewMember]) -> go.Figure:
        """Create analysis of radiation impact on crew health and fertility."""
        from simulation import RadiationModel
        
        radiation_model = RadiationModel()
        
        # Simulate radiation exposure over time
        months = list(range(0, 61, 6))  # 0 to 60 months, every 6 months
        fertility_data = []
        mutation_data = []
        
        for month in months:
            month_fertility = []
            month_mutation = []
            
            for member in crew:
                # Calculate exposure
                member.radiation_exposure = radiation_model.calculate_cumulative_exposure(month)
                member.mission_duration_months = month
                
                # Calculate effects
                fertility = radiation_model.fertility_decay_factor(member)
                mutation = radiation_model.mutation_probability(member)
                
                month_fertility.append(fertility)
                month_mutation.append(mutation)
            
            fertility_data.append(np.mean(month_fertility))
            mutation_data.append(np.mean(month_mutation))
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Fertility Decay Over Time', 'Mutation Risk Increase'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Fertility decay
        fig.add_trace(
            go.Scatter(x=months, y=fertility_data, mode='lines+markers',
                      name='Average Fertility', line=dict(color=self.color_palette['primary'])),
            row=1, col=1
        )
        
        # Mutation risk
        fig.add_trace(
            go.Scatter(x=months, y=mutation_data, mode='lines+markers',
                      name='Mutation Risk', line=dict(color=self.color_palette['warning'])),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Radiation Impact Analysis",
            height=400,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Mission Duration (months)")
        fig.update_yaxes(title_text="Fertility Index", row=1, col=1)
        fig.update_yaxes(title_text="Mutation Probability", row=1, col=2)
        
        return fig


def create_summary_statistics(crew: List[CrewMember], results: List[GenerationResult]) -> Dict:
    """Create summary statistics for the simulation."""
    stats = {
        'initial_population': len(crew),
        'final_population': len(results[-1].offspring) if results else 0,
        'generations_simulated': len(results),
        'average_diversity': np.mean([r.diversity_index for r in results]) if results else 0,
        'average_fertility': np.mean([r.average_fertility for r in results]) if results else 0,
        'total_mutation_rate': np.mean([r.mutation_rate for r in results]) if results else 0,
        'average_radiation_exposure': np.mean([r.radiation_exposure for r in results]) if results else 0
    }
    
    return stats


if __name__ == "__main__":
    # Test the visualization module
    from data import create_sample_crew
    from simulation import CompatibilityCalculator, MultiGenerationSimulator
    
    # Create test data
    crew = create_sample_crew()
    calculator = CompatibilityCalculator()
    simulator = MultiGenerationSimulator(calculator)
    results = simulator.simulate_generations(crew, num_generations=3)
    
    # Test visualizations
    visualizer = PairMorphVisualizer()
    
    print("Creating visualizations...")
    
    # Test each visualization
    overview_fig = visualizer.create_crew_overview_dashboard(crew)
    compatibility_fig = visualizer.create_compatibility_matrix(crew, calculator)
    timeline_fig = visualizer.create_generation_timeline(results)
    radiation_fig = visualizer.create_radiation_impact_analysis(crew)
    
    print("Visualizations created successfully!")
    print(f"Summary statistics: {create_summary_statistics(crew, results)}")

