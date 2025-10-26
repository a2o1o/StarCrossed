"""
PairMorph Simulation Engine
Handles compatibility calculations, radiation exposure modeling, mutation risk assessment,
and multi-objective pairing optimization for Mars colony reproduction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import random
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity
from data import CrewMember, Gender, DataGenerator


class CompatibilityFactor(Enum):
    GENETIC_DIVERSITY = "genetic_diversity"
    FERTILITY_MATCH = "fertility_match"
    PSYCHOLOGICAL_COMPATIBILITY = "psychological_compatibility"
    RADIATION_RISK = "radiation_risk"
    AGE_COMPATIBILITY = "age_compatibility"


@dataclass
class PairingResult:
    """Result of pairing two crew members."""
    member1: CrewMember
    member2: CrewMember
    compatibility_score: float
    fertility_probability: float
    mutation_risk: float
    offspring_traits: Dict[str, float]
    detailed_scores: Dict[str, float]


@dataclass
class GenerationResult:
    """Result of a complete generation simulation."""
    generation: int
    parents: List[Tuple[CrewMember, CrewMember]]
    offspring: List[CrewMember]
    diversity_index: float
    average_fertility: float
    mutation_rate: float
    radiation_exposure: float


class RadiationModel:
    """Models radiation exposure and its effects on fertility and mutation risk."""
    
    def __init__(self):
        # Mars surface radiation: ~0.6 mSv/day (vs Earth's ~0.1 mSv/day)
        self.daily_mars_radiation = 0.6  # mSv/day
        self.earth_baseline = 0.1  # mSv/day
        
    def calculate_cumulative_exposure(self, mission_duration_months: int) -> float:
        """Calculate cumulative radiation exposure."""
        return self.daily_mars_radiation * mission_duration_months * 30
    
    def fertility_decay_factor(self, member: CrewMember) -> float:
        """Calculate fertility decay due to radiation exposure."""
        # Radiation affects fertility exponentially
        exposure_factor = member.radiation_exposure / 1000  # Convert to Sv
        base_fertility = member.fertility_index
        
        # Exponential decay model based on radiation biology research
        decay_rate = 0.1  # 10% decay per Sv
        radiation_effect = np.exp(-decay_rate * exposure_factor)
        
        # Age effect (already in fertility_index)
        return base_fertility * radiation_effect
    
    def mutation_probability(self, member: CrewMember) -> float:
        """Calculate mutation probability based on radiation exposure."""
        # Base mutation rate (very low)
        base_rate = 0.001
        
        # Radiation increases mutation rate
        exposure_factor = member.radiation_exposure / 1000  # Convert to Sv
        radiation_multiplier = 1 + exposure_factor * 2  # 2x increase per Sv
        
        # Individual radiation tolerance affects susceptibility
        tolerance_factor = 1 - member.radiation_tolerance
        
        return base_rate * radiation_multiplier * tolerance_factor


class CompatibilityCalculator:
    """Calculates compatibility between crew members across multiple factors."""
    
    def __init__(self, weights: Optional[Dict[CompatibilityFactor, float]] = None):
        if weights is None:
            self.weights = {
                CompatibilityFactor.GENETIC_DIVERSITY: 0.3,
                CompatibilityFactor.FERTILITY_MATCH: 0.25,
                CompatibilityFactor.PSYCHOLOGICAL_COMPATIBILITY: 0.2,
                CompatibilityFactor.RADIATION_RISK: 0.15,
                CompatibilityFactor.AGE_COMPATIBILITY: 0.1
            }
        else:
            self.weights = weights
    
    def calculate_compatibility(self, member1: CrewMember, member2: CrewMember) -> Dict[str, float]:
        """Calculate comprehensive compatibility between two members."""
        scores = {}
        
        # Genetic diversity (higher is better)
        scores['genetic_diversity'] = self._genetic_diversity_score(member1, member2)
        
        # Fertility match (optimal fertility range)
        scores['fertility_match'] = self._fertility_match_score(member1, member2)
        
        # Psychological compatibility
        scores['psychological_compatibility'] = self._psychological_compatibility_score(member1, member2)
        
        # Radiation risk (lower is better)
        scores['radiation_risk'] = self._radiation_risk_score(member1, member2)
        
        # Age compatibility
        scores['age_compatibility'] = self._age_compatibility_score(member1, member2)
        
        return scores
    
    def _genetic_diversity_score(self, member1: CrewMember, member2: CrewMember) -> float:
        """Calculate genetic diversity score (higher = more diverse)."""
        # Blood type compatibility
        blood_compatibility = self._blood_type_compatibility(member1.blood_type.value, member2.blood_type.value)
        
        # Genetic marker diversity
        marker_distances = []
        for key in member1.genetic_markers:
            if key in member2.genetic_markers:
                distance = abs(member1.genetic_markers[key] - member2.genetic_markers[key])
                marker_distances.append(distance)
        
        marker_diversity = np.mean(marker_distances) if marker_distances else 0.5
        
        # Combine factors
        return (blood_compatibility + marker_diversity) / 2
    
    def _blood_type_compatibility(self, blood1: str, blood2: str) -> float:
        """Calculate blood type compatibility for reproduction."""
        # Simplified compatibility matrix
        compatibility_matrix = {
            ('A+', 'A+'): 0.7, ('A+', 'A-'): 0.8, ('A+', 'B+'): 0.9, ('A+', 'B-'): 0.9,
            ('A+', 'AB+'): 0.8, ('A+', 'AB-'): 0.9, ('A+', 'O+'): 0.6, ('A+', 'O-'): 0.7,
            ('A-', 'A-'): 0.8, ('A-', 'B+'): 0.9, ('A-', 'B-'): 0.9, ('A-', 'AB+'): 0.9,
            ('A-', 'AB-'): 0.9, ('A-', 'O+'): 0.7, ('A-', 'O-'): 0.8,
            ('B+', 'B+'): 0.7, ('B+', 'B-'): 0.8, ('B+', 'AB+'): 0.8, ('B+', 'AB-'): 0.9,
            ('B+', 'O+'): 0.6, ('B+', 'O-'): 0.7,
            ('B-', 'B-'): 0.8, ('B-', 'AB+'): 0.9, ('B-', 'AB-'): 0.9, ('B-', 'O+'): 0.7,
            ('B-', 'O-'): 0.8,
            ('AB+', 'AB+'): 0.7, ('AB+', 'AB-'): 0.8, ('AB+', 'O+'): 0.5, ('AB+', 'O-'): 0.6,
            ('AB-', 'AB-'): 0.8, ('AB-', 'O+'): 0.6, ('AB-', 'O-'): 0.7,
            ('O+', 'O+'): 0.6, ('O+', 'O-'): 0.7, ('O-', 'O-'): 0.8
        }
        
        # Check both orders
        key1 = (blood1, blood2)
        key2 = (blood2, blood1)
        
        return compatibility_matrix.get(key1, compatibility_matrix.get(key2, 0.5))
    
    def _fertility_match_score(self, member1: CrewMember, member2: CrewMember) -> float:
        """Calculate fertility match score (optimal range)."""
        fertility1 = member1.fertility_index
        fertility2 = member2.fertility_index
        
        # Optimal fertility range is 0.6-0.8
        optimal_range = (0.6, 0.8)
        
        # Calculate how close both members are to optimal range
        score1 = 1 - abs(fertility1 - np.mean(optimal_range)) / (optimal_range[1] - optimal_range[0])
        score2 = 1 - abs(fertility2 - np.mean(optimal_range)) / (optimal_range[1] - optimal_range[0])
        
        return (score1 + score2) / 2
    
    def _psychological_compatibility_score(self, member1: CrewMember, member2: CrewMember) -> float:
        """Calculate psychological compatibility."""
        # Stress resistance compatibility
        stress_compat = 1 - abs(member1.stress_resistance - member2.stress_resistance)
        
        # Social compatibility
        social_compat = (member1.social_compatibility + member2.social_compatibility) / 2
        
        # Circadian rhythm compatibility
        circadian_compat = self._circadian_compatibility(member1.circadian_rhythm_type, 
                                                        member2.circadian_rhythm_type)
        
        return (stress_compat + social_compat + circadian_compat) / 3
    
    def _circadian_compatibility(self, rhythm1: str, rhythm2: str) -> float:
        """Calculate circadian rhythm compatibility."""
        compatibility_matrix = {
            ('morning', 'morning'): 0.9,
            ('morning', 'evening'): 0.3,
            ('morning', 'flexible'): 0.7,
            ('evening', 'evening'): 0.9,
            ('evening', 'flexible'): 0.7,
            ('flexible', 'flexible'): 0.8
        }
        
        key = tuple(sorted([rhythm1, rhythm2]))
        return compatibility_matrix.get(key, 0.5)
    
    def _radiation_risk_score(self, member1: CrewMember, member2: CrewMember) -> float:
        """Calculate radiation risk score (lower is better, so invert)."""
        # Higher radiation exposure = higher risk
        total_exposure = member1.radiation_exposure + member2.radiation_exposure
        
        # Lower radiation tolerance = higher risk
        avg_tolerance = (member1.radiation_tolerance + member2.radiation_tolerance) / 2
        
        # Risk increases with exposure and decreases with tolerance
        risk = total_exposure / 1000 * (1 - avg_tolerance)
        
        # Convert to score (lower risk = higher score)
        return max(0, 1 - risk)
    
    def _age_compatibility_score(self, member1: CrewMember, member2: CrewMember) -> float:
        """Calculate age compatibility score."""
        age_diff = abs(member1.age - member2.age)
        
        # Optimal age difference is 2-8 years
        if age_diff <= 2:
            return 0.8
        elif age_diff <= 8:
            return 1.0
        elif age_diff <= 15:
            return 0.7
        else:
            return 0.3


class PairingOptimizer:
    """Optimizes crew member pairings using multi-objective optimization."""
    
    def __init__(self, compatibility_calculator: CompatibilityCalculator):
        self.compatibility_calculator = compatibility_calculator
        self.radiation_model = RadiationModel()
    
    def optimize_pairings(self, crew: List[CrewMember], 
                         max_pairs: Optional[int] = None) -> List[Tuple[CrewMember, CrewMember]]:
        """Find optimal pairings using genetic algorithm approach."""
        if len(crew) < 2:
            return []
        
        # Filter for reproductive age (simplified)
        reproductive_crew = [m for m in crew if 25 <= m.age <= 45]
        
        if len(reproductive_crew) < 2:
            return []
        
        # Generate all possible pairs
        pairs = []
        for i in range(len(reproductive_crew)):
            for j in range(i + 1, len(reproductive_crew)):
                pairs.append((reproductive_crew[i], reproductive_crew[j]))
        
        # Calculate compatibility scores for all pairs
        pair_scores = []
        for pair in pairs:
            scores = self.compatibility_calculator.calculate_compatibility(pair[0], pair[1])
            weighted_score = sum(scores[factor.value] * self.compatibility_calculator.weights[factor] 
                               for factor in CompatibilityFactor)
            pair_scores.append((pair, weighted_score, scores))
        
        # Sort by compatibility score
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select non-overlapping pairs
        selected_pairs = []
        used_members = set()
        
        for (pair, score, scores) in pair_scores:
            if pair[0] not in used_members and pair[1] not in used_members:
                selected_pairs.append(pair)
                used_members.add(pair[0])
                used_members.add(pair[1])
                
                if max_pairs and len(selected_pairs) >= max_pairs:
                    break
        
        return selected_pairs
    
    def simulate_offspring(self, parent1: CrewMember, parent2: CrewMember) -> CrewMember:
        """Simulate offspring from two parents."""
        # Generate offspring traits
        offspring_age = 0  # Newborn
        
        # Gender is random
        offspring_gender = random.choice(list(Gender))
        
        # Blood type inheritance (simplified)
        offspring_blood_type = self._inherit_blood_type(parent1.blood_type, parent2.blood_type)
        
        # Physical traits (average with some variation)
        height_cm = (parent1.height_cm + parent2.height_cm) / 2 + np.random.normal(0, 5)
        weight_kg = (parent1.weight_kg + parent2.weight_kg) / 2 + np.random.normal(0, 3)
        
        # VO2 Max (inherited with variation)
        vo2_max = (parent1.vo2_max + parent2.vo2_max) / 2 + np.random.normal(0, 3)
        vo2_max = max(35, min(65, vo2_max))
        
        # Radiation tolerance (inherited)
        radiation_tolerance = (parent1.radiation_tolerance + parent2.radiation_tolerance) / 2
        radiation_tolerance += np.random.normal(0, 0.05)
        radiation_tolerance = np.clip(radiation_tolerance, 0.3, 0.95)
        
        # Immune resilience (inherited)
        immune_resilience = (parent1.immune_resilience + parent2.immune_resilience) / 2
        immune_resilience += np.random.normal(0, 0.05)
        immune_resilience = np.clip(immune_resilience, 0.4, 0.9)
        
        # Fertility (inherited, but starts high for offspring)
        fertility_base = (parent1.fertility_index + parent2.fertility_index) / 2
        fertility_index = min(0.9, fertility_base + np.random.normal(0, 0.1))
        fertility_index = max(0.3, fertility_index)
        
        # Genetic diversity (inherited)
        genetic_diversity_score = (parent1.genetic_diversity_score + parent2.genetic_diversity_score) / 2
        genetic_diversity_score += np.random.normal(0, 0.1)
        genetic_diversity_score = np.clip(genetic_diversity_score, 0.2, 0.9)
        
        # Psychological traits (inherited)
        stress_resistance = (parent1.stress_resistance + parent2.stress_resistance) / 2
        stress_resistance += np.random.normal(0, 0.05)
        stress_resistance = np.clip(stress_resistance, 0.4, 0.9)
        
        social_compatibility = (parent1.social_compatibility + parent2.social_compatibility) / 2
        social_compatibility += np.random.normal(0, 0.05)
        social_compatibility = np.clip(social_compatibility, 0.5, 0.9)
        
        # Circadian rhythm (inherited with some variation)
        circadian_options = [parent1.circadian_rhythm_type, parent2.circadian_rhythm_type, "flexible"]
        circadian_rhythm_type = random.choice(circadian_options)
        
        # Generate offspring ID
        offspring_id = f"GEN2_{len(parent1.id.split('_')) + 1:03d}"
        offspring_name = f"Offspring of {parent1.name} & {parent2.name}"
        
        return CrewMember(
            id=offspring_id,
            name=offspring_name,
            age=offspring_age,
            gender=offspring_gender,
            blood_type=offspring_blood_type,
            height_cm=height_cm,
            weight_kg=weight_kg,
            vo2_max=vo2_max,
            radiation_tolerance=radiation_tolerance,
            immune_resilience=immune_resilience,
            fertility_index=fertility_index,
            genetic_diversity_score=genetic_diversity_score,
            stress_resistance=stress_resistance,
            social_compatibility=social_compatibility,
            circadian_rhythm_type=circadian_rhythm_type,
            radiation_exposure=0.0,  # Start with no exposure
            mission_duration_months=0
        )
    
    def _inherit_blood_type(self, blood1, blood2):
        """Simplified blood type inheritance."""
        # This is a simplified model - real inheritance is more complex
        from data import BloodType
        
        # For simplicity, randomly inherit from one parent
        return random.choice([blood1, blood2])


class MultiGenerationSimulator:
    """Simulates multiple generations of colony reproduction."""
    
    def __init__(self, compatibility_calculator: CompatibilityCalculator):
        self.compatibility_calculator = compatibility_calculator
        self.pairing_optimizer = PairingOptimizer(compatibility_calculator)
        self.radiation_model = RadiationModel()
        self.data_generator = DataGenerator()
    
    def simulate_generations(self, initial_crew: List[CrewMember], 
                           num_generations: int = 5) -> List[GenerationResult]:
        """Simulate multiple generations of reproduction."""
        results = []
        current_crew = initial_crew.copy()
        
        for generation in range(num_generations):
            print(f"Simulating Generation {generation + 1}...")
            
            # Update radiation exposure for all crew members
            for member in current_crew:
                member.mission_duration_months += 12  # 1 year per generation
                member.radiation_exposure = self.radiation_model.calculate_cumulative_exposure(
                    member.mission_duration_months
                )
            
            # Find optimal pairings
            pairs = self.pairing_optimizer.optimize_pairings(current_crew)
            
            # Generate offspring
            offspring = []
            for parent1, parent2 in pairs:
                # Calculate fertility probability
                fertility_prob = self._calculate_fertility_probability(parent1, parent2)
                
                # Random chance of successful reproduction
                if random.random() < fertility_prob:
                    child = self.pairing_optimizer.simulate_offspring(parent1, parent2)
                    offspring.append(child)
            
            # Calculate generation metrics
            diversity_index = self.data_generator.calculate_genetic_diversity(current_crew + offspring)
            avg_fertility = np.mean([m.fertility_index for m in current_crew])
            mutation_rate = np.mean([self.radiation_model.mutation_probability(m) for m in current_crew])
            avg_radiation = np.mean([m.radiation_exposure for m in current_crew])
            
            # Store results
            result = GenerationResult(
                generation=generation + 1,
                parents=pairs,
                offspring=offspring,
                diversity_index=diversity_index,
                average_fertility=avg_fertility,
                mutation_rate=mutation_rate,
                radiation_exposure=avg_radiation
            )
            results.append(result)
            
            # Update crew for next generation
            current_crew = offspring.copy()
            
            if len(current_crew) < 2:
                print(f"Population too small to continue after generation {generation + 1}")
                break
        
        return results
    
    def _calculate_fertility_probability(self, parent1: CrewMember, parent2: CrewMember) -> float:
        """Calculate probability of successful reproduction."""
        # Base fertility probability
        base_prob = (parent1.fertility_index + parent2.fertility_index) / 2
        
        # Adjust for radiation effects
        fertility1 = self.radiation_model.fertility_decay_factor(parent1)
        fertility2 = self.radiation_model.fertility_decay_factor(parent2)
        
        # Combined fertility probability
        combined_fertility = (fertility1 + fertility2) / 2
        
        # Additional factors
        age_factor = 1.0
        if parent1.age > 40 or parent2.age > 40:
            age_factor = 0.7
        elif parent1.age > 35 or parent2.age > 35:
            age_factor = 0.9
        
        return min(0.8, combined_fertility * age_factor)


if __name__ == "__main__":
    # Test the simulation engine
    from data import create_sample_crew
    
    crew = create_sample_crew()
    print(f"Testing with {len(crew)} crew members")
    
    # Test compatibility calculation
    calculator = CompatibilityCalculator()
    scores = calculator.calculate_compatibility(crew[0], crew[1])
    print(f"Compatibility scores: {scores}")
    
    # Test pairing optimization
    optimizer = PairingOptimizer(calculator)
    pairs = optimizer.optimize_pairings(crew)
    print(f"Found {len(pairs)} optimal pairs")
    
    # Test multi-generation simulation
    simulator = MultiGenerationSimulator(calculator)
    results = simulator.simulate_generations(crew, num_generations=3)
    
    for result in results:
        print(f"Generation {result.generation}: {len(result.offspring)} offspring, "
              f"diversity={result.diversity_index:.3f}, "
              f"fertility={result.average_fertility:.3f}")

