use crate::models::models::{create_organism_pool, eliminate_unfit_fractions, natural_selection};
use std::{time::Instant};

const FITNESS_THRESHOLD: f64 = 1e-5;
const POPULATION_SIZE: i32 = 100;

pub fn run_tga() {
        // Get start time.
        let start_time = Instant::now();

        // Create initial pool of organisms.
        let mut population = create_organism_pool(POPULATION_SIZE);

        let mut generation = 0;

        // Loop until we find a solution.
        loop {
                // Sort the population by fitness.
                population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

                // The best organisms is the first one.
                let best_organism = &population[0];

                // Check if the best organism has a fitness of 1.0 or less.
                if best_organism.fitness <= FITNESS_THRESHOLD {
                        println!("Found solution in generation {}. The organism is {:?}.", generation, best_organism);
                        println!("The algorithm took {} seconds to run.", start_time.elapsed().as_secs());
                        return
                }

                // Otherwise, we eliminiate the worst half of the population.
                eliminate_unfit_fractions(&mut population);

                // Perform natural selection on the remaining population. 
                // Replace the old population with the new one.
                natural_selection(&mut population);
                generation += 1;
        }
}

// This should sort by fitness for either the force organism or the simple organism from the Organism enum.
