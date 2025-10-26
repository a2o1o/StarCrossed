"""
PairMorph Data Generation Module
Generates realistic physiological and psychological data for simulated crew members
based on NASA-style distributions and Martian environmental constraints.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random


class Gender(Enum):
    MALE = "male"
    FEMALE = "female"
    NON_BINARY = "non_binary"


class BloodType(Enum):
    A_POS = "A+"
    A_NEG = "A-"
    B_POS = "B+"
    B_NEG = "B-"
    AB_POS = "AB+"
    AB_NEG = "AB-"
    O_POS = "O+"
    O_NEG = "O-"


@dataclass
class CrewMember:
    """Individual crew member with physiological and psychological traits."""
    id: str
    name: str
    age: int
    gender: Gender
    blood_type: BloodType
    
    # Physical traits
    height_cm: float
    weight_kg: float
    vo2_max: float  # ml/kg/min
    radiation_tolerance: float  # 0-1 scale
    immune_resilience: float  # 0-1 scale
    
    # Reproductive traits
    fertility_index: float  # 0-1 scale
    genetic_diversity_score: float  # 0-1 scale
    
    # Psychological traits
    stress_resistance: float  # 0-1 scale
    social_compatibility: float  # 0-1 scale
    circadian_rhythm_type: str  # "morning", "evening", "flexible"
    
    # Environmental exposure
    radiation_exposure: float = 0.0  # cumulative mSv
    mission_duration_months: int = 0
    
    # Genetic markers (simplified)
    genetic_markers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.genetic_markers is None:
            self.genetic_markers = self._generate_genetic_markers()
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, CrewMember):
            return self.id == other.id
        return False
    
    def _generate_genetic_markers(self) -> Dict[str, float]:
        """Generate synthetic genetic markers for diversity calculation."""
        return {
            'marker_1': random.uniform(0, 1),
            'marker_2': random.uniform(0, 1),
            'marker_3': random.uniform(0, 1),
            'marker_4': random.uniform(0, 1),
            'marker_5': random.uniform(0, 1),
        }


class DataGenerator:
    """Generates realistic crew member data based on NASA research and Martian constraints."""
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_crew_member(self, member_id: str, name: str = None) -> CrewMember:
        """Generate a single crew member with realistic traits."""
        if name is None:
            name = f"Crew Member {member_id}"
        
        # Age distribution (astronaut age range: 26-55, peak around 35-40)
        age = int(np.random.normal(37, 8))
        age = max(26, min(55, age))
        
        # Gender distribution
        gender = random.choice(list(Gender))
        
        # Blood type distribution (realistic population distribution)
        blood_type_probs = {
            BloodType.O_POS: 0.37, BloodType.A_POS: 0.33, BloodType.B_POS: 0.09,
            BloodType.AB_POS: 0.03, BloodType.O_NEG: 0.07, BloodType.A_NEG: 0.06,
            BloodType.B_NEG: 0.02, BloodType.AB_NEG: 0.01
        }
        # Normalize probabilities to sum to 1
        total_prob = sum(blood_type_probs.values())
        blood_type_probs = {k: v/total_prob for k, v in blood_type_probs.items()}
        blood_type = np.random.choice(list(blood_type_probs.keys()), p=list(blood_type_probs.values()))
        
        # Physical measurements (NASA astronaut ranges)
        height_cm = np.random.normal(170, 10) if gender == Gender.MALE else np.random.normal(160, 8)
        weight_kg = np.random.normal(75, 12) if gender == Gender.MALE else np.random.normal(65, 10)
        
        # VO2 Max (fitness level) - NASA astronauts typically 40-60 ml/kg/min
        vo2_max = np.random.normal(50, 8)
        vo2_max = max(35, min(65, vo2_max))
        
        # Radiation tolerance (inversely related to age, some genetic component)
        base_tolerance = 0.7 + (55 - age) * 0.01  # Better tolerance when younger
        radiation_tolerance = np.clip(np.random.normal(base_tolerance, 0.15), 0.3, 0.95)
        
        # Immune resilience (declines with age, some genetic variation)
        base_immunity = 0.8 - (age - 26) * 0.01  # Declines with age
        immune_resilience = np.clip(np.random.normal(base_immunity, 0.1), 0.4, 0.9)
        
        # Fertility index (age-dependent, gender-specific)
        if gender == Gender.FEMALE:
            # Female fertility peaks around 25-30, declines significantly after 35
            fertility_base = max(0.1, 0.9 - (age - 25) * 0.02)
        else:
            # Male fertility more stable but still age-dependent
            fertility_base = max(0.3, 0.8 - (age - 30) * 0.01)
        
        fertility_index = np.clip(np.random.normal(fertility_base, 0.1), 0.1, 0.95)
        
        # Genetic diversity score (random but influences compatibility)
        genetic_diversity_score = np.random.uniform(0.2, 0.9)
        
        # Psychological traits
        stress_resistance = np.random.uniform(0.4, 0.9)  # Astronauts selected for high stress tolerance
        social_compatibility = np.random.uniform(0.5, 0.9)
        
        # Circadian rhythm type
        circadian_types = ["morning", "evening", "flexible"]
        circadian_probs = [0.3, 0.2, 0.5]  # More flexible types in space
        circadian_rhythm_type = np.random.choice(circadian_types, p=circadian_probs)
        
        return CrewMember(
            id=member_id,
            name=name,
            age=age,
            gender=gender,
            blood_type=blood_type,
            height_cm=height_cm,
            weight_kg=weight_kg,
            vo2_max=vo2_max,
            radiation_tolerance=radiation_tolerance,
            immune_resilience=immune_resilience,
            fertility_index=fertility_index,
            genetic_diversity_score=genetic_diversity_score,
            stress_resistance=stress_resistance,
            social_compatibility=social_compatibility,
            circadian_rhythm_type=circadian_rhythm_type
        )
    
    def generate_crew(self, size: int, names: List[str] = None) -> List[CrewMember]:
        """Generate a crew of specified size."""
        crew = []
        
        if names is None:
            names = [f"Crew Member {i+1}" for i in range(size)]
        
        for i in range(size):
            member_id = f"CM_{i+1:03d}"
            name = names[i] if i < len(names) else f"Crew Member {i+1}"
            crew.append(self.generate_crew_member(member_id, name))
        
        return crew
    
    def generate_initial_mars_crew(self) -> List[CrewMember]:
        """Generate a realistic initial Mars mission crew (30 members)."""
        mars_names = [
            "Dr. Sarah Chen", "Commander Marcus Rodriguez", "Dr. Elena Volkov",
            "Dr. James Kim", "Dr. Aisha Patel", "Dr. David Thompson",
            "Dr. Maria Santos", "Dr. Alex Johnson", "Dr. Lisa Wang",
            "Dr. Michael Brown", "Dr. Priya Sharma", "Dr. Carlos Mendez",
            "Dr. Jennifer Lee", "Dr. Robert Taylor", "Dr. Amara Okafor",
            "Dr. Thomas Anderson", "Dr. Sofia Petrov", "Dr. Kevin Murphy",
            "Dr. Rachel Green", "Dr. Daniel Park", "Dr. Isabella Rossi",
            "Dr. Benjamin Wright", "Dr. Maya Patel", "Dr. Christopher Davis",
            "Dr. Olivia Johnson", "Dr. Nathan Chen", "Dr. Victoria Smith",
            "Dr. Alexander Kumar", "Dr. Emma Wilson", "Dr. Ryan O'Connor"
        ]
        
        crew_size = 30
        return self.generate_crew(crew_size, mars_names[:crew_size])
    
    def calculate_genetic_diversity(self, crew: List[CrewMember]) -> float:
        """Calculate overall genetic diversity of the crew."""
        if len(crew) < 2:
            return 0.0
        
        # Calculate pairwise genetic distances
        distances = []
        for i in range(len(crew)):
            for j in range(i + 1, len(crew)):
                distance = self._genetic_distance(crew[i], crew[j])
                distances.append(distance)
        
        return np.mean(distances) if distances else 0.0
    
    def _genetic_distance(self, member1: CrewMember, member2: CrewMember) -> float:
        """Calculate genetic distance between two crew members."""
        # Use genetic markers and other traits
        marker_distances = []
        for key in member1.genetic_markers:
            if key in member2.genetic_markers:
                distance = abs(member1.genetic_markers[key] - member2.genetic_markers[key])
                marker_distances.append(distance)
        
        # Add diversity from other genetic factors
        blood_type_distance = 0.0 if member1.blood_type == member2.blood_type else 1.0
        
        # Combine distances
        marker_avg = np.mean(marker_distances) if marker_distances else 0.5
        return (marker_avg + blood_type_distance) / 2


def create_sample_crew() -> List[CrewMember]:
    """Create a sample crew for testing and demonstration."""
    generator = DataGenerator(seed=42)
    return generator.generate_initial_mars_crew()


if __name__ == "__main__":
    # Test the data generation
    generator = DataGenerator(seed=42)
    crew = generator.generate_initial_mars_crew()
    
    print(f"Generated {len(crew)} crew members:")
    for member in crew:
        print(f"{member.name} ({member.age}y, {member.gender.value}): "
              f"Fertility={member.fertility_index:.2f}, "
              f"Radiation Tolerance={member.radiation_tolerance:.2f}, "
              f"VO2 Max={member.vo2_max:.1f}")
    
    diversity = generator.calculate_genetic_diversity(crew)
    print(f"\nOverall genetic diversity: {diversity:.3f}")
