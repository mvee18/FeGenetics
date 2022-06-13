use crate::utils::domain::random_float;

#[allow(dead_code)] // The DNA Size will always be positive (non-negative) and therefore is u32.
const DNA_SIZE: u32 = 3;
#[allow(dead_code)] // The lower domain will always be 0.0 
const DOMAIN_LOWER: f32 = 0.0;
#[allow(dead_code)] 
const DOMAIN_UPPER: f32 = 10.0;

#[derive(Debug)]
pub struct Organism {
        pub id: i32,
        pub dna: Vec<f32>,
        pub fitness: f32
}

impl Organism {
        pub fn new(id: i32, dna_size: i32) -> Organism {
                let mut o = Organism {
                        id: id,
                        dna: Vec::new(),
                        fitness: 0.0
                };
        
                for _ in 0..dna_size {
                        let gene = random_float(DOMAIN_LOWER, DOMAIN_UPPER);
                        o.dna.push(gene);
                }
        
                return o
        }
}


pub fn create_organism_pool(size: i32) -> Vec<Organism> {
        let mut pool: Vec<Organism> = Vec::new();
        for i in 0..size {
                let o = Organism::new(i, DNA_SIZE.try_into().unwrap());
                pool.push(o);
        }
        pool
}

pub fn evalute_fitness(pool: &mut Vec<Organism>, target: Vec<f32>) {
        // Zero fitness should be the worst, but since we're finding the difference, we'll use the opposite.
        for c in 0..pool.len() {
                // Assert that the organism's dna and the target's dna are the same size.
                assert_eq!(pool[c].dna.len(), target.len());

                let mut fitness = 0.0;
                for (i, dna) in pool[c].dna.iter().enumerate() {
                        fitness += difference_squared(*dna, target[i]);
                }

                pool[c].fitness = fitness;
        }
}

fn difference_squared(a: f32, b: f32) -> f32 {
        let diff = a - b;
        diff * diff
}

#[cfg(test)]
mod tests {
        use super::*;
        #[test]
        fn test_evalute_fitness() {
                let mut pool: Vec<Organism> = Vec::new();
                let test_org = Organism{id: 0, dna: vec![1.0, 1.0, 1.0], fitness: 0.0};
                
                pool.push(test_org);

                let target: Vec<f32> = vec![1.0, 2.0, 3.0];

                evalute_fitness(&mut pool, target);

                println!("Wanted fitness: 5.0, got fitness: {}", pool[0].fitness);

                assert_eq!(pool[0].fitness, 5.0);

        }
        #[test]
        fn test_create_organism() {
                // Ensure the DNA length is correct.
                let o = Organism::new(0, DNA_SIZE.try_into().unwrap());
                assert_eq!(o.dna.len(), DNA_SIZE.try_into().unwrap());

                // Ensure that the bounds are obeyed.
                for dna in o.dna.iter() {
                        assert!(*dna >= DOMAIN_LOWER && *dna <= DOMAIN_UPPER);
                }

                println!("{:?}", o);
        }
}