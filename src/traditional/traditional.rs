use crate::models::models::{ForceOrganism, SimpleOrganism, Organism, create_organism_pool};

pub fn run_tga() {
        let pool: Vec<SimpleOrganism> = create_organism_pool(100);
        println!("First three organisms in the pool: {:?}", &pool[0..3]);

}

// This should sort by fitness for either the force organism or the simple organism from the Organism enum.
pub fn eliminate_unfit_fractions(pool: &mut Vec<SimpleOrganism>) {
        // Ensure the pool is an even number.
        assert_eq!(pool.len() % 2, 0);
        
        // Sort the pool by fitness.__rust_force_expr!
        pool.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        // Remove the top half of the pool (i.e., the least fit).
        pool.drain((pool.len() / 2)..pool.len());
}

#[cfg(test)]
mod tests {
        use super::*;
        #[test]
        fn test_elimination() {
                let mut o1 = SimpleOrganism::new(0, 3);
                o1.fitness = 1.0;
                
                let mut o2 = SimpleOrganism::new(1, 3);
                o2.fitness = 3.0;

                let mut o3 = SimpleOrganism::new(2, 3); 
                o3.fitness = 2.0;

                let mut o4 = SimpleOrganism::new(3, 3); 
                o4.fitness = 5.0;

                let mut pool: Vec<SimpleOrganism> = vec![o1, o2, o3, o4];

                eliminate_unfit_fractions(&mut pool);

                assert_eq!(pool[0].fitness, 1.0);
                assert_eq!(pool[1].fitness, 2.0);
        }
}