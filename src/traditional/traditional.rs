use crate::{
    models::models::{
        ForceOrganism, Organism, Population, EXE_DIR_PATH, FITNESS_THRESHOLD, NUMBER_ATOMS,
        POPULATION_SIZE,
    },
    utils::utils,
};
use std::time::Instant;

pub fn run_tga() {
    // Get start time.
    let start_time = Instant::now();

    // Create initial pool of organisms.
    // let mut population = create_organism_pool(POPULATION_SIZE);
    //     let mut population = SimpleOrganism::new_population(POPULATION_SIZE);
    let mut population = ForceOrganism::new_population(*POPULATION_SIZE, *NUMBER_ATOMS);
    let mut generation = 0;

    // Loop until we find a solution.
    loop {
        // Sort the population by fitness.
        population.sort_by(|a, b| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap());

        // The best organisms is the first one.
        let best_organism = &population[0];

        // Check if the best organism has a fitness of 1.0 or less.
        if best_organism.get_fitness() <= *FITNESS_THRESHOLD {
            println!("\nYes. The superior fighter is clear.");
            println!(
                "Found solution in generation {}. The organism is {}.",
                generation, best_organism.id
            );
            println!(
                "The algorithm took {} seconds to run.",
                start_time.elapsed().as_secs()
            );
            let best_path = &EXE_DIR_PATH
                .join("best")
                .join(format!("{}", generation))
                .join(format!("{}", best_organism.id));

            best_organism.save_to_file(best_path, true);

            // let mut parameter_file =
            //     std::fs::File::create(EXE_DIR_PATH.join("best").join("parameter.out"))
            //         .expect("Error creating parameter.out file.");

            // Write the parameters to the file.
            // parameter_file
            //     .write_all(
            //         format!(
            //             "Generation: {}\nFitness: {}\nBest Organism: {}\nNumber Unfit: {}\nTournament Size: {}\nMutation Rate: {}\nStrength: {}\n",
            //             generation,
            //             "CONVERGED",
            //             best_organism.id,
            //             utils::evalute_number_failed(&population.clone()),
            //             *TOURNAMENT_SIZE,
            //             *MUTATION_RATE,
            //             *MUTATION_STRENGTH,
            //         )
            //         .as_bytes(),
            //     )
            //     .unwrap();

            return;
        }

        if generation % 10 == 0 {
            println!(
                "Time taken: {:?} | Generation {} | Fitness {} | Best Organism {} | Number Unfit: {}",
                Instant::now().duration_since(start_time),
                generation,
                best_organism.get_fitness(),
                best_organism.id,
                utils::evalute_number_failed(&population.clone())
            );
            let best_path = &EXE_DIR_PATH
                .join("best")
                .join(format!("{}", generation))
                .join(format!("{}", best_organism.id));

            best_organism.save_to_file(best_path, true);

            // Open a parameter.out file and write the parameters to it.
            // let mut parameter_file =
            //     std::fs::File::create(EXE_DIR_PATH.join("best").join("parameter.out"))
            //         .expect("Error creating parameter.out file.");

            // // Write the parameters to the file.

            // parameter_file
            //     .write_all(
            //         format!(
            //             "Generation: {}\nFitness: {}\nBest Organism: {}\nNumber Unfit: {}\nTournament Size: {}\nMutation Rate: {}\nStrength: {}\n",
            //             generation,
            //             best_organism.get_fitness(),
            //             best_organism.id,
            //             utils::evalute_number_failed(&population.clone()),
            //             *TOURNAMENT_SIZE,
            //             *MUTATION_RATE,
            //             *MUTATION_STRENGTH,
            //         )
            //         .as_bytes(),
            //     )
            //     .unwrap();

            // return;
        }

        // Otherwise, we eliminiate the worst half of the population
        population.eliminate_unfit_fraction();

        // Perform natural selection on the remaining population.
        // Replace the old population with the new one.
        population.natural_selection();
        generation += 1;
    }
}

// This should sort by fitness for either the force organism or the simple organism from the Organism enum.
