use crate::models::models::{ForceOrganism, SimpleOrganism, create_organism_pool};

pub fn run_tga() {
        let pool: Vec<SimpleOrganism> = create_organism_pool(100);
        println!("First three organisms in the pool: {:?}", &pool[0..3]);

}

// This should sort by fitness for either the force organism or the simple organism from the Organism enum.
