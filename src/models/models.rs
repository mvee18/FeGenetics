use crate::utils::domain::random_float;
use rand_distr::{Normal, Distribution};
use rand::seq::SliceRandom;
use std::option::Option;
use uuid::Uuid;

#[allow(dead_code)] // The DNA Size will always be positive (non-negative) and therefore is u32.
const DNA_SIZE: u32 = 5;
#[allow(dead_code)] // The lower domain will always be 0.0
const DOMAIN_LOWER: f64 = 0.0;
#[allow(dead_code)]
const DOMAIN_UPPER: f64 = 10.0;


// Parameters for the genetic algorithms
const TARGET_LIST: &'static [f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
const TOURNAMENT_SIZE: u32 = 3;
const MUTATION_RATE: f64 = 0.01;
pub const POPULATION_SIZE: i32 = 100;
pub const FITNESS_THRESHOLD: f64 = 1e-5;

// The Organism is a trait (interface).
pub trait Organism {
        fn new(size: i32) -> Self where Self: Sized;
        fn new_population(pop_size: i32) -> Vec<Self> 
        where Self: Sized {
                let mut population: Vec<Self> = Vec::new();
                for _ in 0..pop_size {
                        let mut organism = Self::new(DNA_SIZE.try_into().unwrap());
                        organism.evaluate_fitness(TARGET_LIST.to_vec());
                        population.push(organism);
                }
                population    
        }
        fn get_fitness(&self) -> f64;
        fn evaluate_fitness(&mut self, target: Vec<f64>);
        fn mutate(&mut self);
}

pub trait Population: Sized  {
        fn natural_selection(&mut self);
        // fn quadratic_mating(&mut self);
}

// Newtype wrapper for Simple Organism population.

impl Population for Vec<SimpleOrganism> {
        fn natural_selection(&mut self) {
                // Copy the original pool so we don't use the new organisms in the mating.
                let new_pool = self.clone();
                // The number of iterations should be equal to the pool since we cut it in half.
                // That is, we must refill the pool.
                // TODO: Refill different fractions of the pool depending on the above function.
                for _ in 0..new_pool.len() {
                        let mut parents: Vec<SimpleOrganism> = Vec::new();
                        for _ in 0..3 {
                                parents.push(tournament_round(&new_pool.to_vec()));
                        }

                        // We push onto the original pool.
                        self.push(quadratic_mating(&mut parents));
                }
        }
}


// impl Population for Vec<ForceOrganism> {
//         fn natural_selection(&mut self) {
//                 organism_natural_selection(&mut self);
//         }
// }

#[derive(Debug, Clone, PartialEq)]
pub struct ForceOrganism {
        pub id: String,
        pub dna: Vec<Vec<Vec<f64>>>,
        pub fitness: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleOrganism {
        pub id: String,
        pub dna: Vec<f64>,
        pub fitness: f64,
}

impl Organism for SimpleOrganism {
        fn new(dna_size: i32) -> SimpleOrganism {
                let id = Uuid::new_v4().to_string();

                let mut o = SimpleOrganism {
                        id: id,
                        dna: Vec::new(),
                        fitness: 0.0,
                };
                for _ in 0..dna_size {
                        let gene = random_float(DOMAIN_LOWER, DOMAIN_UPPER).try_into().unwrap();
                        o.dna.push(gene);
                }

                return o;
        }

        fn evaluate_fitness(&mut self, target: Vec<f64>) {
                assert_eq!(self.dna.len(), target.len());
                let mut sum: f64 = 0.0;
                for i in 0..self.dna.len() {
                        sum += difference_squared(self.dna[i], target[i]);
                }

                // MSE = mean squared error
                self.fitness = sum / (self.dna.len() as f64);
        }

        fn mutate(&mut self) {
                // Generate random float for mutation chance.
                for i in 0..self.dna.len() {
                        if random_float(0.0, 1.0) < MUTATION_RATE {
                                let norm_distr = Normal::new(self.dna[i], 0.1).unwrap();
                                self.dna[i] = norm_distr.sample(&mut rand::thread_rng());
                        }
                }
        }

        fn get_fitness(&self) -> f64 {
                return self.fitness;
        }
}


// Make the pool of organisms.
// TODO: Later, this should use the enum to make the pool of organisms.
// pub fn create_organism_pool(size: i32) -> Vec<SimpleOrganism> {
//         // Initialize the empty pool.
//         let mut pool: Vec<SimpleOrganism> = Vec::new();
//         // Create the organisms up to the size.
//         for _ in 0..size {
//                 let mut o = SimpleOrganism::new(DNA_SIZE.try_into().unwrap());
//                 o.evaluate_fitness(TARGET_LIST.to_vec());
//                 pool.push(o);
//         }
//         pool
// }

fn difference_squared(a: f64, b: f64) -> f64 {
        let diff = a - b;
        diff * diff
}

// pub fn eliminate_unfit_fraction(pool: &mut Vec<SimpleOrganism>) {
//         // Ensure the pool is an even number.
//         assert_eq!(pool.len() % 2, 0);
//         // Sort the pool by fitness.__rust_force_expr!
//         pool.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
//         // Remove the top half of the pool (i.e., the least fit).
//         // TODO: Make this eliminate different fractions of the pool.
//         pool.drain((pool.len() / 2)..pool.len());
// }

pub fn eliminate_unfit_fraction<T>(pool: &mut Vec<T>) 
where T: Organism {
        // Sort the population by fitness.
        pool.sort_by(|a, b| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap());

        pool.drain((pool.len() / 2)..pool.len());
}

// pub fn organism_natural_selection<T>(pool: &mut Vec<T>) where T: Organism, T: Clone {
//         // Copy the original pool so we don't use the new organisms in the mating.
//         let new_pool = pool.clone();
//         // The number of iterations should be equal to the pool since we cut it in half.
//         // That is, we must refill the pool.
//         // TODO: Refill different fractions of the pool depending on the above function.
//         for _ in 0..new_pool.len() {
//                 let mut parents: Vec<T> = Vec::new();
//                 for _ in 0..3 {
//                         parents.push(tournament_round(&new_pool.to_vec()));
//                 }

//                 // We push onto the original pool.
//                 pool.push(quadratic_mating(&mut parents));
//         }
// }

pub fn tournament_round<T>(pool: &Vec<T>) -> T where T: Organism, T: Clone {
        // The number of iterations should be equal to the pool since we cut it in half.
        // That is, we must refill the pool.
        let mut group: Vec<T> = Vec::new();
        make_tournament_group(&mut group, pool);
        // Sort the group by fitness.
        group.sort_by(|a, b| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap());
        // Return the most fit organism.
        group[0].clone()
}

// Separate functionality for testing.
fn make_tournament_group<T>(grp: &mut Vec<T>, pool: &Vec<T>) where T: Organism, T: Clone {
        // Each group will have TOURNAMENT_SIZE members.
        // We don't want a duplicate organism. We will generate a sequence from 0 to the pool size.
        // Then, we will shuffle the sequence and take the first TOURNAMENT_SIZE elements.
        let mut v: Vec<u32> = (0..pool.len().try_into().unwrap()).collect();
        v.shuffle(&mut rand::thread_rng());
        for i in 0..TOURNAMENT_SIZE {
                // Make sure it fits.
                let index = usize::try_from(i).unwrap();

                // Is clone necessary?
                // Maybe there is a better way to do this with some sort of reference?
                grp.push(pool[v[index] as usize].clone());
        }

        // println!("The group is: {:?}\n", grp);
}

// TODO: This test can fail if the same organism is selected twice. We should fix this.


pub fn quadratic_mating(parents: &mut Vec<SimpleOrganism>) -> SimpleOrganism {
        // There should always be three parents.
        assert_eq!(parents.len(), 3);
        // We should initialize the child.
        let mut child = SimpleOrganism{
                id: Uuid::new_v4().to_string(),
                // This makes push much less costly. We know the size of the DNA
                // is going to be the same.
                dna: Vec::with_capacity(DNA_SIZE.try_into().unwrap()),
                fitness: 0.0,
        };

        // We need to sort the vector by fitness.
        parents.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Now, we fit to a quadratic curve. The first parent is the most fit.
        let (p1, p2, p3) = (&parents[0], &parents[1], &parents[2]); 

        // Now, we iterate over the DNA of the first parent.
        for i in 0..p1.dna.len() {
                let a = (1.0 / (p3.dna[i] - p2.dna[i])) * (((p3.fitness - p1.fitness) / (p3.dna[i] - p1.dna[i])) - ((p2.fitness - p1.fitness) / (p2.dna[i] - p1.dna[i])));
                let b = ((p2.fitness - p1.fitness) / (p2.dna[i] - p1.dna[i])) - (a * (p2.dna[i] + p1.dna[i]));

                let critical_point: f64= b / (-2.0 * a);
                let concavity = 2.0*a; 

                // println!("{} {} {} {} {}", p1.dna[i], p2.dna[i], p3.dna[i], critical_point, concavity);

                // If the concavity is positive (minimized), then we should use the critical point.
                if concavity > 0.0 && (critical_point.abs() > DOMAIN_LOWER && critical_point.abs() < DOMAIN_UPPER) {
                        child.dna.push(critical_point);
                } else {
                        // Otherwise, we should use a linear interpolation.
                        let result = linear_interpolation(p1.dna[i], p3.dna[i]);
                        match result {
                                Some(x) => child.dna.push(x),
                                // If None, i.e., the linear interpolation failed, we randomly select one of the parents.
                                None => {
                                        let rand_parent = parents.choose(&mut rand::thread_rng()).unwrap();
                                        child.dna.push(rand_parent.dna[i]);
                                },
                        }
                }

        }

        // Evaluate the fitness of the child.
        child.evaluate_fitness(TARGET_LIST.to_vec());
        child.mutate();
        return child;
}

// We interpolate between the most fit and least fit organisms.
pub fn linear_interpolation(best: f64, worst: f64) -> Option<f64> {
        let mut beta = rand::random::<f64>();

        // Keep a counter for the number of iterations.
        let mut counter = 0;

        // Base case.
        let mut dna = beta * (best - worst) + worst;

        // We need to make sure that the DNA is within the domain.
        // If it is, we recursively call this function while reducing the beta.
        while dna.abs() < DOMAIN_LOWER || dna.abs() > DOMAIN_UPPER {
                beta = beta / 2.0;
                dna = beta * (best - worst) + worst;
                counter += 1;
                // If we have exceeded the number of iterations, then we should return None.
                if counter > 3 {
                        return None;
                }
        }

        return Some(dna);
}


#[cfg(test)]
mod tests {
        use super::*;
        fn generate_test_organism() -> Vec<SimpleOrganism> {
                let mut o1 = SimpleOrganism::new(3);
                o1.fitness = 1.0;
                let mut o2 = SimpleOrganism::new(3);
                o2.fitness = 3.0;

                let mut o3 = SimpleOrganism::new(3);
                o3.fitness = 2.0;

                let mut o4 = SimpleOrganism::new(3);
                o4.fitness = 5.0;

                vec![o1, o2, o3, o4]
        }

        #[test]
        fn test_evalute_fitness() {
                let mut test_org = SimpleOrganism {
                        id: "testOrg".to_string(),
                        dna: vec![1.0, 1.0, 1.0],
                        fitness: 0.0,
                };
                let target: Vec<f64> = vec![1.0, 2.0, 3.0];

                test_org.evaluate_fitness(target);

                // println!("Wanted fitness: 5.0, got fitness: {}", test_org.fitness);

                let wanted = 1.6666667;

                assert!((test_org.fitness - wanted).abs() < 0.00001);
        }
        #[test]
        fn test_create_organism() {
                // Ensure the DNA length is correct.
                let o = SimpleOrganism::new(DNA_SIZE.try_into().unwrap());
                assert_eq!(o.dna.len(), DNA_SIZE.try_into().unwrap());

                // Ensure that the bounds are obeyed.
                for dna in o.dna.iter() {
                        let abs_dna = dna.abs();
                        assert!(abs_dna >= DOMAIN_LOWER && abs_dna <= DOMAIN_UPPER.abs());
                }

                println!("{:?}", o);
        }
        #[test]
        fn test_elimination() {
                let mut pool: Vec<SimpleOrganism> = generate_test_organism();

                eliminate_unfit_fraction(&mut pool);

                assert_eq!(pool[0].fitness, 1.0);
                assert_eq!(pool[1].fitness, 2.0);
        }

        #[test]
        fn test_tournament_group_maker() {
                let mut group: Vec<SimpleOrganism> = Vec::new();
                let pool: Vec<SimpleOrganism> = generate_test_organism()[1..4].to_vec();

                make_tournament_group(&mut group, &pool);

                // Check that there are no duplicate organisms in the group.
                let mut unique_orgs: Vec<SimpleOrganism> = Vec::new();
                for org in group.iter() {
                        if !unique_orgs.contains(&org) {
                                unique_orgs.push(org.clone());
                        }
                }
                
                // Check that the number of unique organisms is equal to the
                // number of organisms in the group.
                assert_eq!(unique_orgs.len(), group.len());
        }

        #[test]
        fn test_tournament_round() {
                let pool: Vec<SimpleOrganism> = generate_test_organism()[1..4].to_vec();

                println!("The pool is: {:?}", pool);

                let expected = pool[1].clone();

                let best = tournament_round(&pool);

                assert_eq!(best, expected);
        }

        fn linear_interpolation_helper(best: f64, worst: f64) {
                let result = linear_interpolation(best, worst);
                match result {
                        Some(x) => {
                                assert!(x >= DOMAIN_LOWER && x <= DOMAIN_UPPER);
                                // println!("The value is: {}", x);
                        },
                        None => {
                                // println!("The parent values are: {}, {}", best, worst);
                                assert!(false)},
                }
        }

        #[test]
        fn test_linear_interpolation() {
                // Scan over the upper and lower bounds.
                // Not sure if this test is comprehensive enough?
                for lower in 0..10 {
                        for upper in 0..10 {
                                linear_interpolation_helper(lower as f64, upper as f64);
                        }
                }

        }

        // This case demonstrates the simplest case where the concavity is positive.
        #[test]
        fn test_quadratic_mating_case_one() {
                        let p1: SimpleOrganism = SimpleOrganism{
                                id: "1".to_string(),
                                dna: vec![-1.0, -1.0, -1.0, -1.0, -1.0],
                                fitness: 1.0,
                        };
                        let p2: SimpleOrganism = SimpleOrganism{
                                id: "2".to_string(),
                                dna: vec![0.0, 0.0, 0.0, 0.0, 0.0],
                                fitness: 0.0,
                        };
                        let p3: SimpleOrganism = SimpleOrganism{
                                id: "3".to_string(),
                                dna: vec![1.0, 1.0, 1.0, 1.0, 1.0],
                                fitness: 1.0,
                        };
                        let mut parents = vec![p1, p2, p3];

                        let child = quadratic_mating(&mut parents);

                        println!("Quadratic fitting: {:?}", child);
                }

}
