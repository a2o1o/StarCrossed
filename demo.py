#!/usr/bin/env python3
"""
PairMorph Demo Script
Demonstrates the core functionality of the Mars colony reproduction simulator.
"""

from data import create_sample_crew
from simulation import CompatibilityCalculator, MultiGenerationSimulator
from visualization import PairMorphVisualizer, create_summary_statistics

def main():
    print("🚀 PairMorph - Mars Colony Reproduction Simulator Demo")
    print("=" * 60)
    
    # Generate initial crew
    print("\n1. Generating Mars Mission Crew...")
    crew = create_sample_crew()
    print(f"   ✅ Generated {len(crew)} crew members")
    
    # Display crew overview
    print("\n2. Crew Overview:")
    for i, member in enumerate(crew, 1):
        print(f"   {i}. {member.name} ({member.age}y, {member.gender.value})")
        print(f"      Fertility: {member.fertility_index:.2f}, "
              f"Radiation Tolerance: {member.radiation_tolerance:.2f}, "
              f"VO2 Max: {member.vo2_max:.1f}")
    
    # Compatibility analysis
    print("\n3. Compatibility Analysis...")
    calculator = CompatibilityCalculator()
    
    # Test compatibility between first two members
    if len(crew) >= 2:
        scores = calculator.calculate_compatibility(crew[0], crew[1])
        overall_score = sum(scores[factor.value] * calculator.weights[factor] 
                          for factor in calculator.weights.keys())
        
        print(f"   Compatibility between {crew[0].name} and {crew[1].name}:")
        print(f"   Overall Score: {overall_score:.3f}")
        for factor, score in scores.items():
            print(f"   - {factor}: {score:.3f}")
    
    # Pairing optimization
    print("\n4. Optimal Pairing Analysis...")
    simulator = MultiGenerationSimulator(calculator)
    pairs = simulator.pairing_optimizer.optimize_pairings(crew)
    
    print(f"   Found {len(pairs)} optimal pairs:")
    for i, (member1, member2) in enumerate(pairs, 1):
        scores = calculator.calculate_compatibility(member1, member2)
        overall_score = sum(scores[factor.value] * calculator.weights[factor] 
                          for factor in calculator.weights.keys())
        print(f"   {i}. {member1.name} & {member2.name} (Score: {overall_score:.3f})")
    
    # Multi-generation simulation
    print("\n5. Multi-Generation Simulation...")
    print("   Running 3-generation simulation...")
    
    results = simulator.simulate_generations(crew, num_generations=3)
    
    print(f"   ✅ Simulated {len(results)} generations")
    
    # Display results
    print("\n6. Simulation Results:")
    for result in results:
        print(f"   Generation {result.generation}:")
        print(f"   - Parents: {len(result.parents)} pairs")
        print(f"   - Offspring: {len(result.offspring)}")
        print(f"   - Diversity Index: {result.diversity_index:.3f}")
        print(f"   - Average Fertility: {result.average_fertility:.3f}")
        print(f"   - Mutation Rate: {result.mutation_rate:.6f}")
        print(f"   - Radiation Exposure: {result.radiation_exposure:.1f} mSv")
    
    # Summary statistics
    print("\n7. Summary Statistics:")
    stats = create_summary_statistics(crew, results)
    print(f"   Initial Population: {stats['initial_population']}")
    print(f"   Final Population: {stats['final_population']}")
    print(f"   Generations Simulated: {stats['generations_simulated']}")
    print(f"   Average Diversity: {stats['average_diversity']:.3f}")
    print(f"   Average Fertility: {stats['average_fertility']:.3f}")
    print(f"   Total Mutation Rate: {stats['total_mutation_rate']:.6f}")
    print(f"   Average Radiation Exposure: {stats['average_radiation_exposure']:.1f} mSv")
    
    print("\n🎉 Demo completed successfully!")
    print("\nTo run the full interactive dashboard:")
    print("   streamlit run main.py")
    print("\nTo explore the code:")
    print("   - data.py: Crew member generation and genetic modeling")
    print("   - simulation.py: Compatibility algorithms and multi-generation simulation")
    print("   - visualization.py: Interactive charts and analysis")
    print("   - main.py: Streamlit dashboard with scientific analysis interface")

if __name__ == "__main__":
    main()

