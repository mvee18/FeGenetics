use crate::utils::domain::random_float;

#[allow(dead_code)] // The DNA Size will always be positive (non-negative) and therefore is u32.
const DNA_SIZE: u32 = 3;
#[allow(dead_code)] // The lower domain will always be 0.0 
const DOMAIN_LOWER: f64 = 0.0;
#[allow(dead_code)] 
const DOMAIN_UPPER: f64 = 10.0;

const TargetList: &'static [f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];

pub enum Organism {
        ForceOrganism(ForceOrganism),
        SimpleOrganism(SimpleOrganism),
}

#[derive(Debug)]
pub struct ForceOrganism {
        pub id: i32,
        pub dna: Vec<Vec<Vec<f64>>>,
        pub fitness: f32
}

#[derive(Debug)]
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

        pub fn new_DEBUG(id: i32, dna: Vec<f64>) -> SimpleOrganism {
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
                o.evalute_fitness(TargetList.to_vec());
                pool.push(o);
        }
        pool
}

fn difference_squared(a: f64, b: f64) -> f64 {
        let diff = a - b;
        diff * diff
}

#[cfg(test)]
mod tests {
        use super::*;
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

}