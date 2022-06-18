use crate::utils::domain::random_float;
use rand::{Rng};

#[allow(dead_code)] // The DNA Size will always be positive (non-negative) and therefore is u32.
const DNA_SIZE: u32 = 3;
#[allow(dead_code)] // The lower domain will always be 0.0 
const DOMAIN_LOWER: f64 = 0.0;
#[allow(dead_code)] 
const DOMAIN_UPPER: f64 = 10.0;

const TARGET_LIST: &'static [f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
const TOURNAMENT_SIZE: u32 = 3;

// pub enum Organism {
//         ForceOrganism(ForceOrganism),
//         SimpleOrganism(SimpleOrganism),
// }

#[derive(Debug)]
pub struct ForceOrganism {
        pub id: i32,
        pub dna: Vec<Vec<Vec<f64>>>,
        pub fitness: f32
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleOrganism {
        pub id: i32,
        pub dna: Vec<f64>,
        pub fitness: f64
}

impl SimpleOrganism {
        pub fn new(id: i32, dna_size: i32) -> SimpleOrganism {
                let mut o = SimpleOrganism {
                        id: id,
                        dna: Vec::new(),
                        fitness: 0.0
                };
        
                for _ in 0..dna_size {
                        let gene = random_float(DOMAIN_LOWER, DOMAIN_UPPER).try_into().unwrap();
                        o.dna.push(gene);
                }
        
                return o
        }

        pub fn new_debug(id: i32, dna: Vec<f64>) -> SimpleOrganism {
                let mut o = SimpleOrganism {
                        id: id,
                        dna: Vec::new(),
                        fitness: 0.0
                };

                o.dna = dna;
                return o
        }

        pub fn evalute_fitness(&mut self, target: Vec<f64>) {
                assert_eq!(self.dna.len(), target.len());
                let mut sum: f64 = 0.0;
                for i in 0..self.dna.len() {
                        sum += difference_squared(self.dna[i], target[i]);
                }

                self.fitness = sum;
        }

}


// Make the pool of organisms. 
// TODO: Later, this should use the enum to make the pool of organisms.
pub fn create_organism_pool(size: i32) -> Vec<SimpleOrganism> {
        // Initialize the empty pool.
        let mut pool: Vec<SimpleOrganism> = Vec::new();
        // Create the organisms up to the size. 
        for i in 0..size {
                let mut o = SimpleOrganism::new(i, DNA_SIZE.try_into().unwrap());
                o.evalute_fitness(TARGET_LIST.to_vec());
                pool.push(o);
        }
        pool
}


fn difference_squared(a: f64, b: f64) -> f64 {
        let diff = a - b;
        diff * diff
}

pub fn eliminate_unfit_fractions(pool: &mut Vec<SimpleOrganism>) {
        // Ensure the pool is an even number.
        assert_eq!(pool.len() % 2, 0);
        
        // Sort the pool by fitness.__rust_force_expr!
        pool.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        // Remove the top half of the pool (i.e., the least fit).
        // TODO: Make this eliminate different fractions of the pool.
        pool.drain((pool.len() / 2)..pool.len());
}

pub fn natural_selection(pool: &mut Vec<SimpleOrganism>) {
        // Copy the original pool so we don't use the new organisms in the mating.
        let new_pool = pool.clone();
        
        // The number of iterations should be equal to the pool since we cut it in half.
        // That is, we must refill the pool.
        // TODO: Refill different fractions of the pool depending on the above function.
        for _ in 0..pool.len() {
                let mut parents: Vec<SimpleOrganism> = Vec::new();
                for _ in 0..3 {
                        parents.push(tournament_round(&pool.to_vec()));
                }

                // new_pool.push(Mating());
        }
}

pub fn tournament_round(pool: &Vec<SimpleOrganism>) -> SimpleOrganism {
        // The number of iterations should be equal to the pool since we cut it in half.
        // That is, we must refill the pool.
        let mut group: Vec<SimpleOrganism> = Vec::new();
        // Each group will have TOURNAMENT_SIZE members.
        for _ in 0..TOURNAMENT_SIZE {
                let mut rng = rand::thread_rng();
                let index = rng.gen_range(0..pool.len());
                // Is clone necessary? 
                // Maybe there is a better way to do this with some sort of reference?
                group.push(pool[index].clone());
        }
        
        // Sort the group by fitness.
        group.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
        
        // Return the most fit organism.
        group[0].clone()
}

#[cfg(test)]
mod tests {
        use super::*;
        fn generate_test_organism() -> Vec<SimpleOrganism> {
                let mut o1 = SimpleOrganism::new(0, 3);
                o1.fitness = 1.0;
                
                let mut o2 = SimpleOrganism::new(1, 3);
                o2.fitness = 3.0;

                let mut o3 = SimpleOrganism::new(2, 3); 
                o3.fitness = 2.0;

                let mut o4 = SimpleOrganism::new(3, 3); 
                o4.fitness = 5.0;

                vec![o1, o2, o3, o4]
        }

        #[test]
        fn test_evalute_fitness() {
                let mut test_org = SimpleOrganism{id: 0, dna: vec![1.0, 1.0, 1.0], fitness: 0.0};
                
                let target: Vec<f64> = vec![1.0, 2.0, 3.0];

                test_org.evalute_fitness(target);

                println!("Wanted fitness: 5.0, got fitness: {}", test_org.fitness);

                assert_eq!(test_org.fitness, 5.0);

        }
        #[test]
        fn test_create_organism() {
                // Ensure the DNA length is correct.
                let o = SimpleOrganism::new(0, DNA_SIZE.try_into().unwrap());
                assert_eq!(o.dna.len(), DNA_SIZE.try_into().unwrap());

                // Ensure that the bounds are obeyed.
                for dna in o.dna.iter() {
                        assert!(*dna >= DOMAIN_LOWER && *dna <= DOMAIN_UPPER);
                }

                println!("{:?}", o);
        }
        #[test]
        fn test_elimination() {
                let mut pool: Vec<SimpleOrganism> = generate_test_organism();

                eliminate_unfit_fractions(&mut pool);

                assert_eq!(pool[0].fitness, 1.0);
                assert_eq!(pool[1].fitness, 2.0);
        }

        #[test]
        fn test_tournament_round() {
                let pool: Vec<SimpleOrganism> = generate_test_organism()[1..4].to_vec(); 

                let expected = pool[1].clone();

                let best = tournament_round(&pool);

                assert_eq!(best, expected);
        }

}