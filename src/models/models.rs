use crate::programs::spectro::run_spectro;
use crate::utils::domain::{check_and_fix_domain, determine_number_force_constants, Derivatives};
use crate::utils::domain::{random_float, random_float_mc};
use crate::utils::utils::{create_directory, Target};
use lazy_static::lazy_static;
use rand::seq::SliceRandom;
use rand_distr::{Distribution, Normal};
use std::any::Any;
use std::fs::{read_to_string, File};
use std::io::Write;
use std::option::Option;
use std::path::{Path, PathBuf};
use strum::IntoEnumIterator;
use uuid::Uuid;

#[allow(dead_code)] // The DNA Size will always be positive (non-negative) and therefore is u32.
const DNA_SIZE: u32 = 5;
#[allow(dead_code)] // The lower domain will always be 0.0
const DOMAIN_LOWER: f64 = 0.0;
#[allow(dead_code)]
const DOMAIN_UPPER: f64 = 10.0;

// Parameters for the genetic algorithms
// const TARGET_LIST: &'static [f64] = &[2.0, 4.0, 6.0, 8.0, 10.0];
const TOURNAMENT_SIZE: u32 = 300;
const MUTATION_RATE: f64 = 0.10;
pub const POPULATION_SIZE: i32 = 8000;
pub const FITNESS_THRESHOLD: f64 = 1.0;
pub const NUMBER_ATOMS: i32 = 3;
pub const FORT_FILES: [&'static str; 3] = ["fort.15", "fort.30", "fort.40"];
pub const INITIAL_GUESS: &'static str = "/home/mvee/rust/fegenetics/tests/water_test";

lazy_static! {
    #[derive(Clone, Copy)]
    pub static ref TARGET: Target = Target::initialize(&PathBuf::from("/home/mvee/rust/fegenetics/src/input/target.toml"));
    // This is for the simple organisms.
    // #[derive(Clone, Copy)]
    // pub static ref TARGET: Target = Target::initialize(&PathBuf::from("/home/mvee/rust/fegenetics/tests/simple/target.toml"));
}

// The Organism is a trait (interface).
pub trait Organism {
    fn new(size: i32) -> Self
    where
        Self: Sized;
    fn new_population(pop_size: i32, dna_size: i32) -> Vec<Self>
    where
        Self: Sized,
    {
        let mut population: Vec<Self> = Vec::new();
        for _ in 0..pop_size {
            let mut organism = Self::new(dna_size.try_into().unwrap());
            organism.evaluate_fitness(TARGET);
            population.push(organism);
        }
        population
    }
    fn get_fitness(&self) -> f64;
    fn evaluate_fitness(&mut self, target: TARGET);
    fn mutate(&mut self);
    fn save_to_file(&self, path: &str);
    fn as_any(&self) -> &dyn Any;
}

pub trait Population: Sized {
    fn natural_selection(&mut self);
    // We use Box to allow for different types of organisms. It does mean
    // that we need to implement As Any for the organism.
    fn quadratic_mating(&mut self) -> Box<dyn Organism>;
    fn eliminate_unfit_fraction(&mut self);
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
            let child = parents.quadratic_mating();
            // TODO: Really make sure this is the best way of doing it.
            let result: &SimpleOrganism = child.as_any().downcast_ref().unwrap();
            self.push(result.clone());
        }
    }

    fn quadratic_mating(&mut self) -> Box<dyn Organism> {
        // There should always be three parents.
        assert_eq!(self.len(), 3);
        // We should initialize the child.
        let mut child = SimpleOrganism {
            id: Uuid::new_v4().to_string(),
            // This makes push much less costly. We know the size of the DNA
            // is going to be the same.
            dna: Vec::with_capacity(DNA_SIZE.try_into().unwrap()),
            fitness: 0.0,
        };

        // We need to sort the vector by fitness.
        self.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Now, we fit to a quadratic curve. The first parent is the most fit.
        let (p1, p2, p3) = (&self[0], &self[1], &self[2]);

        // Now, we iterate over the DNA of the first parent.
        for i in 0..p1.dna.len() {
            let a = (1.0 / (p3.dna[i] - p2.dna[i]))
                * (((p3.fitness - p1.fitness) / (p3.dna[i] - p1.dna[i]))
                    - ((p2.fitness - p1.fitness) / (p2.dna[i] - p1.dna[i])));
            let b = ((p2.fitness - p1.fitness) / (p2.dna[i] - p1.dna[i]))
                - (a * (p2.dna[i] + p1.dna[i]));

            let critical_point: f64 = b / (-2.0 * a);
            let concavity = 2.0 * a;

            // println!("{} {} {} {} {}", p1.dna[i], p2.dna[i], p3.dna[i], critical_point, concavity);

            // If the concavity is positive (minimized), then we should use the critical point.
            if concavity > 0.0
                && (critical_point.abs() > DOMAIN_LOWER && critical_point.abs() < DOMAIN_UPPER)
            {
                child.dna.push(critical_point);
            } else {
                // Otherwise, we should use a linear interpolation.
                let result = linear_interpolation(p1.dna[i], p3.dna[i]);
                match result {
                    Some(x) => child.dna.push(x),
                    // If None, i.e., the linear interpolation failed, we randomly select one of the parents.
                    None => {
                        let rand_parent = self.choose(&mut rand::thread_rng()).unwrap();
                        child.dna.push(rand_parent.dna[i]);
                    }
                }
            }
        }

        // Evaluate the fitness of the child.
        child.evaluate_fitness(TARGET);
        child.mutate();
        let result = Box::new(child);
        result
    }

    fn eliminate_unfit_fraction(&mut self) {
        self.sort_by(|a, b| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap());

        self.drain((self.len() / 2)..self.len());
    }
}

impl Population for Vec<ForceOrganism> {
    fn natural_selection(&mut self) {
        assert_eq!(self.len() as i32, POPULATION_SIZE / 2);
        // Copy the original pool so we don't use the new organisms in the mating.
        let new_pool = self.clone();
        // The number of iterations should be equal to the pool since we cut it in half.
        // That is, we must refill the pool.
        // TODO: Refill different fractions of the pool depending on the above function.
        for _ in 0..new_pool.len() {
            let mut parents: Vec<ForceOrganism> = Vec::new();
            for _ in 0..3 {
                parents.push(tournament_round(&new_pool.to_vec()));
            }

            // We push onto the original pool.
            let child = parents.quadratic_mating();
            // TODO: Really make sure this is the best way of doing it.
            let result: &ForceOrganism = child.as_any().downcast_ref().unwrap();
            self.push(result.clone());
        }
    }

    fn quadratic_mating(&mut self) -> Box<dyn Organism> {
        // There should always be three parents.
        assert_eq!(self.len(), 3);
        // We should initialize the child.
        let mut child = ForceOrganism {
            id: Uuid::new_v4().to_string(),
            // This makes push much less costly. We know the size of the DNA
            // is going to be the same as the parent.
            // Here, dna[0] is second derivatives, [1] is the third, and [2] is the fourth.
            dna: vec![
                Vec::with_capacity(self[0].dna[0].len()),
                Vec::with_capacity(self[0].dna[1].len()),
                Vec::with_capacity(self[0].dna[2].len()),
            ],
            harmfitness: 0.0,
            rotfitness: 0.0,
            fundfitness: 0.0,
        };

        // We need to sort the vector by fitness.

        // Now, we fit to a quadratic curve. The first parent is the most fit.

        // Now, we iterate over the three chromosomes.
        for i in 0..3 {
            let (f1, f2, f3);
            match i {
                0 => self.sort_by(|a, b| a.harmfitness.partial_cmp(&b.harmfitness).unwrap()),
                1 => self.sort_by(|a, b| a.rotfitness.partial_cmp(&b.rotfitness).unwrap()),
                2 => self.sort_by(|a, b| a.fundfitness.partial_cmp(&b.fundfitness).unwrap()),
                _ => panic!("Invalid chromosome index."),
            }

            match i {
                0 => {
                    (f1, f2, f3) = (
                        &self[0].harmfitness,
                        &self[1].harmfitness,
                        &self[2].harmfitness,
                    )
                }
                1 => {
                    (f1, f2, f3) = (
                        &self[0].rotfitness,
                        &self[1].rotfitness,
                        &self[2].rotfitness,
                    )
                }
                2 => {
                    (f1, f2, f3) = (
                        &self[0].fundfitness,
                        &self[1].fundfitness,
                        &self[2].fundfitness,
                    )
                }
                _ => panic!("Invalid chromosome index."),
            }

            // a = p1, b = p2, c = p3.
            let (p1, p2, p3) = (&self[0], &self[1], &self[2]);
            let zipped = p1.dna[i]
                .iter()
                .zip(p2.dna[i].iter())
                .zip(p3.dna[i].iter())
                .map(|((a, b), c)| (a, b, c));

            // Now, we perform the quadratic interpolation using a closure.
            for (g1, g2, g3) in zipped {
                let a = (1.0 / (g3 - g2)) * (((f3 - f1) / (g3 - g1)) - ((f2 - f1) / (g2 - g1)));
                let b = ((f2 - f1) / (g2 - g1)) - (a * (g2 + g1));

                let critical_point: f64 = b / (-2.0 * a);
                let concavity = 2.0 * a;

                // println!("{} {} {} {} {}", p1.dna[i], p2.dna[i], p3.dna[i], critical_point, concavity);

                // If the concavity is positive (minimized), then we should use the critical point.
                if concavity > 0.0
                    && (critical_point.abs() > DOMAIN_LOWER && critical_point.abs() < DOMAIN_UPPER)
                {
                    child.dna[i].push(critical_point);
                } else {
                    // Otherwise, we should use a linear interpolation.
                    let result = linear_interpolation(*g1, *g3);
                    match result {
                        Some(x) => child.dna[i].push(x),
                        // If None, i.e., the linear interpolation failed, we randomly select one of the parents.
                        None => {
                            let parents = vec![g1, g2, g3];
                            let rand_parent = parents.choose(&mut rand::thread_rng()).unwrap();
                            child.dna[i].push(**rand_parent);
                        }
                    }
                }
            }
        }
        child.save_to_file("/home/mvee/rust/fegenetics/");
        child.evaluate_fitness(TARGET);
        child.mutate();
        let result = Box::new(child);
        result
    }

    fn eliminate_unfit_fraction(&mut self) {
        self.sort_by(|a, b| a.get_fitness().partial_cmp(&b.get_fitness()).unwrap());

        let drained: Vec<_> = self.drain((self.len() / 2)..self.len()).collect();

        for d in drained {
            // We will delete the directory of the drained organisms.
            let dir = PathBuf::from(format!("/home/mvee/rust/fegenetics/organisms/{}", d.id));
            if dir.exists() {
                std::fs::remove_dir_all(&dir).unwrap();
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForceOrganism {
    pub id: String,
    pub dna: Vec<Vec<f64>>,
    pub harmfitness: f64,
    pub rotfitness: f64,
    pub fundfitness: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SimpleOrganism {
    pub id: String,
    pub dna: Vec<f64>,
    pub fitness: f64,
}

impl Organism for SimpleOrganism {
    fn new(dna_size: i32) -> SimpleOrganism {
        let mut o = SimpleOrganism {
            id: Uuid::new_v4().to_string(),
            dna: Vec::new(),
            fitness: 0.0,
        };
        for _ in 0..dna_size {
            let gene = random_float(DOMAIN_LOWER, DOMAIN_UPPER).try_into().unwrap();
            o.dna.push(gene);
        }

        return o;
    }

    fn evaluate_fitness(&mut self, target: TARGET) {
        assert_eq!(self.dna.len(), target.harm.len());
        let mut sum: f64 = 0.0;
        for i in 0..self.dna.len() {
            sum += difference_squared(self.dna[i], target.harm[i]);
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

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save_to_file(&self, _path: &str) {
        unimplemented!("Will not be implemented for this organism type as it does not need to be saved to a file.")
    }
}

impl Organism for ForceOrganism {
    // Here, size is the number of atoms.
    fn new(n_atoms: i32) -> Self
    where
        Self: Sized,
    {
        if INITIAL_GUESS != "" {
            let o =
                ForceOrganism::initial_guess_mock("/home/mvee/rust/fegenetics/tests/water_test");
            let project_root = env!("CARGO_MANIFEST_DIR");
            o.save_to_file(project_root);
            return o;
        } else {
            let mut o = ForceOrganism {
                id: Uuid::new_v4().to_string(),
                dna: Vec::with_capacity(3),
                harmfitness: 0.0,
                rotfitness: 0.0,
                fundfitness: 0.0,
            };

            for dn in Derivatives::iter() {
                let size = determine_number_force_constants(n_atoms, dn);
                // Make the chromosome that will be pushed onto the DNA.
                let mut chromosome: Vec<f64> = Vec::with_capacity(size.try_into().unwrap());

                for _ in 0..size {
                    let gene = random_float_mc(dn);
                    chromosome.push(gene);
                }

                o.dna.push(chromosome);
            }
            let project_root = env!("CARGO_MANIFEST_DIR");
            o.save_to_file(project_root);
            return o;
        }
    }

    fn get_fitness(&self) -> f64 {
        let fitness = self.harmfitness + self.rotfitness + self.fundfitness;
        return fitness;
    }

    fn evaluate_fitness(&mut self, target: TARGET) {
        let organism_freqs = run_spectro(self.id.clone());
        match organism_freqs {
            Ok(freqs) => {
                // println!("The test freqs are {:?}\n", organism_freqs);

                let mut hfitness = 0.0;
                for (c, freq) in freqs.lxm_freqs.iter().enumerate() {
                    hfitness +=
                        difference_squared(*freq, target.harm[c]) / (target.harm.len() as f64);
                }

                self.harmfitness = hfitness;

                let mut rfitness = 0.0;
                for (c, rot) in freqs.rots[0].iter().enumerate() {
                    rfitness +=
                        difference_squared(*rot, target.rots[c]) / (target.rots.len() as f64);
                }

                self.rotfitness = rfitness;

                let mut ffitness = 0.0;
                for (c, freq) in freqs.fund.iter().enumerate() {
                    ffitness +=
                        difference_squared(*freq, target.fund[c]) / (target.fund.len() as f64);
                }

                self.fundfitness = ffitness;
                return;
            }
            Err(_) => {
                // println!("Organism {} failed to evaluate fitness: {}", self.id, e);
                (self.harmfitness, self.rotfitness, self.fundfitness) =
                    (std::f64::MAX, std::f64::MAX, std::f64::MAX);
                return;
            }
        }
    }

    fn mutate(&mut self) {
        for (i, chromosome) in self.dna.iter_mut().enumerate() {
            for gene in chromosome.iter_mut() {
                if gene == &0.0 {
                    continue;
                }
                if random_float(0.0, 1.0).abs() < MUTATION_RATE {
                    let norm_distr = Normal::new(*gene, 0.01).unwrap();
                    *gene = norm_distr.sample(&mut rand::thread_rng());

                    check_and_fix_domain(Derivatives::from_index(i), gene)
                }
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn save_to_file(&self, path: &str) {
        // Make organisms directory if it doesn't exist.
        let mut organisms_dir = PathBuf::from(path);

        organisms_dir.push("organisms");
        create_directory(&organisms_dir);
        // println!(
        //     "The organisms directory is: {:?}",
        //     organisms_dir.canonicalize().unwrap().display()
        // );

        // Make each organism's subdirectory. Should be unique due to UUID.
        // Push the organism's subdirectory onto the organisms directory.
        organisms_dir.push(&self.id);
        create_directory(&organisms_dir);
        // println!(
        //     "The organism's directory is {:?}",
        //     organisms_dir.canonicalize().unwrap().display()
        // );

        // Make three separate files for each chromosome.
        for i in 0..self.dna.len() {
            let mut file = File::create(organisms_dir.join(format!("{}", FORT_FILES[i])))
                .expect("Could not create file.");
            let header = format!("{:>5}{:>5}", NUMBER_ATOMS, self.dna[i].len());
            file.write(header.as_bytes())
                .expect("Could not write to file.");
            for j in 0..self.dna[i].len() {
                if j % 3 == 0 {
                    file.write("\n".as_bytes())
                        .expect("Could not write to file.");
                }
                let line = format!("{:>20.10}", self.dna[i][j]);
                file.write_all(line.as_bytes())
                    .expect("Could not write to file.");
            }
            file.write_all("\n".as_bytes())
                .expect("Could not write to file.");
        }
    }
}

impl ForceOrganism {
    fn read_from_file(path: &str) -> Self {
        let mut o = ForceOrganism::newchild();

        let parent_dir = Path::new(path);

        // Closure that will open each file, skip the header, and turn the data into a Vec<f64>.
        let read_file = |file_path: &Path| {
            let raw_data = read_to_string(file_path).expect("Could not read file.");
            let mut data = raw_data
                .split_whitespace()
                .map(|x| x.parse::<f64>().unwrap())
                .collect::<Vec<f64>>();
            // Remove the first two elements.
            data.drain(0..2);

            return data;
        };

        for fort_file in FORT_FILES.iter() {
            let file_path = parent_dir.join(fort_file);
            let data = read_file(&file_path);
            o.dna.push(data);
        }

        o
    }

    pub fn initial_guess_mock(path: &str) -> Self {
        let mut organism = ForceOrganism::read_from_file(path);
        // We will iterate over each gene and mutate it slightly.
        for chromosome in organism.dna.iter_mut() {
            for gene in chromosome.iter_mut() {
                if (*gene - 0.0).abs() < 0.0000001 {
                    continue;
                }
                *gene += random_float(-0.1, 0.1);
            }
        }

        organism
    }

    pub fn newchild() -> Self {
        ForceOrganism {
            id: Uuid::new_v4().to_string(),
            dna: Vec::with_capacity(3),
            harmfitness: 0.0,
            rotfitness: 0.0,
            fundfitness: 0.0,
        }
    }
}

fn difference_squared(a: f64, b: f64) -> f64 {
    let diff = a - b;
    diff * diff
}

pub fn tournament_round<T>(pool: &Vec<T>) -> T
where
    T: Organism,
    T: Clone,
{
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
fn make_tournament_group<T>(grp: &mut Vec<T>, pool: &Vec<T>)
where
    T: Organism,
    T: Clone,
{
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
    fn generate_water_organism() -> ForceOrganism {
        let chr1 = vec![
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.3745760748,
            0.2350376513,
            0.0000000000,
            -0.3433795128,
            -0.2037203150,
            0.0000000000,
            -0.0311965620,
            -0.0313173362,
            0.0000000000,
            0.2350376513,
            0.2182499600,
            0.0000000000,
            -0.2663549875,
            -0.2298918472,
            0.0000000000,
            0.0313173362,
            0.0116418872,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.3433795128,
            -0.2663549875,
            0.0000000000,
            0.6867590256,
            0.0000000000,
            0.0000000000,
            -0.3433795128,
            0.2663549875,
            0.0000000000,
            -0.2037203150,
            -0.2298918472,
            0.0000000000,
            0.0000000000,
            0.4597836944,
            0.0000000000,
            0.2037203150,
            -0.2298918472,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.0311965620,
            0.0313173362,
            0.0000000000,
            -0.3433795128,
            0.2037203150,
            0.0000000000,
            0.3745760748,
            -0.2350376513,
            0.0000000000,
            -0.0313173362,
            0.0116418872,
            0.0000000000,
            0.2663549875,
            -0.2298918472,
            0.0000000000,
            -0.2350376513,
            0.2182499600,
        ];

        let chr2 = vec![
            0.0000000000,
            0.2366970402,
            0.0000000000,
            -0.8023486570,
            0.1803923884,
            0.0000000000,
            -0.8916746917,
            0.0000000000,
            -0.6030781439,
            -0.1713984776,
            0.0000000000,
            -0.2116956863,
            0.0000000000,
            -0.1965752075,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.1834885248,
            0.2070609203,
            0.0000000000,
            -0.2399028478,
            0.0000000000,
            0.7473572427,
            0.0000000000,
            0.9121769066,
            0.6314110182,
            0.2399028478,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.7473572427,
            -0.9162194568,
            0.0000000000,
            0.0000000000,
            -0.1746952820,
            0.0000000000,
            0.8752150270,
            0.0000000000,
            0.6078101990,
            0.1814388426,
            0.2070609203,
            0.0000000000,
            0.0000000000,
            -0.4141218405,
            0.0000000000,
            -0.9162194568,
            -0.6314110182,
            0.0000000000,
            1.8324389136,
            0.0000000000,
            -0.5842093798,
            -0.2015195727,
            0.0000000000,
            0.0000000000,
            0.4030391454,
            0.0000000000,
            -0.0250013539,
            0.0000000000,
            0.0161828191,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0282071615,
            -0.0104857128,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.2399028478,
            0.0000000000,
            -0.0323656382,
            0.0000000000,
            0.0000000000,
            0.2070609203,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.0032058076,
            -0.0056971064,
            0.0000000000,
            0.2399028478,
            -0.1746952820,
            0.0000000000,
            0.0032058076,
            0.0000000000,
            0.0549914143,
            0.0000000000,
            -0.0205022149,
            -0.0283328743,
            -0.0282071615,
            0.0000000000,
            0.0000000000,
            -0.1834885248,
            0.0000000000,
            -0.0000000000,
            0.0040425502,
            0.0000000000,
            0.7473572427,
            0.0000000000,
            0.0410044298,
            0.0236008192,
            0.0000000000,
            -0.9162194568,
            0.5842093798,
            0.0250013539,
            0.0000000000,
            0.0000000000,
            0.2116956863,
            0.0000000000,
            0.0000000000,
            -0.2366970402,
            0.0000000000,
            -0.0549914143,
            0.0164596647,
            0.0000000000,
            -0.7473572427,
            0.8752150270,
            0.0000000000,
            0.8023486570,
            -0.0056971064,
            0.0000000000,
            0.0164596647,
            0.0000000000,
            -0.0047320551,
            -0.0100403650,
            -0.0104857128,
            0.0000000000,
            0.0000000000,
            0.2070609203,
            0.0000000000,
            0.0040425502,
            0.0000000000,
            0.0000000000,
            -0.9162194568,
            0.0000000000,
            -0.0236008192,
            0.0200807300,
            0.0000000000,
            0.6314110182,
            -0.2015195727,
            0.0161828191,
            0.0000000000,
            0.0000000000,
            -0.1965752075,
            0.0000000000,
            0.0000000000,
            0.1803923884,
            0.0000000000,
            -0.0205022149,
            0.0047320551,
            0.0000000000,
            0.9121769066,
            -0.6078101990,
            0.0000000000,
            -0.8916746917,
            0.0000000000,
            0.0283328743,
            -0.0100403650,
            0.0000000000,
            -0.6314110182,
            0.1814388426,
            0.0000000000,
            0.6030781439,
            -0.1713984776,
        ];

        let chr3 = vec![
            0.4917693553,
            0.0000000000,
            -0.8472107534,
            0.0000000000,
            0.7651216393,
            0.0000000000,
            -0.7526888893,
            0.0000000000,
            2.3991494570,
            -0.4503365745,
            0.0000000000,
            2.1525387586,
            0.0000000000,
            0.6150354723,
            -0.8297467338,
            -0.4874319956,
            0.0000000000,
            0.8031210523,
            0.0000000000,
            0.0000000000,
            0.7563757678,
            0.0000000000,
            0.4793312845,
            0.0000000000,
            0.0000000000,
            0.5114520329,
            0.0000000000,
            -0.7882960233,
            0.0000000000,
            -0.7381194455,
            -0.5175258798,
            -0.5594921076,
            0.0000000000,
            0.0000000000,
            1.1189842152,
            0.0000000000,
            0.8296201180,
            0.0000000000,
            -0.6910320131,
            0.7797071915,
            0.0000000000,
            -2.4080762726,
            0.0000000000,
            -2.2139803677,
            -0.5883829693,
            0.0000000000,
            -0.8215871595,
            0.0000000000,
            -0.7847822928,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.8252282376,
            0.7847822928,
            0.0000000000,
            -0.8412941546,
            0.0000000000,
            0.6703851096,
            0.0000000000,
            2.4092089940,
            2.2313371440,
            0.8252282376,
            0.0000000000,
            0.0000000000,
            -1.6504564751,
            0.0000000000,
            -0.6290913024,
            -2.4083340618,
            0.0000000000,
            1.2581826048,
            0.0000000000,
            0.7358207899,
            0.0000000000,
            -2.3933630164,
            0.4612334050,
            0.0000000000,
            -2.1550432297,
            0.0000000000,
            -0.6389084180,
            0.8518688313,
            0.0000000000,
            -0.7381194455,
            0.0000000000,
            -0.4978188847,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.6914565982,
            0.5545010801,
            0.0000000000,
            -0.7847822928,
            0.0000000000,
            2.4100839262,
            0.0000000000,
            2.2155270535,
            0.5937730997,
            0.7847822928,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -2.4100839262,
            -2.2344305157,
            0.0000000000,
            0.0000000000,
            -0.4813301207,
            0.0000000000,
            2.1134628681,
            0.0000000000,
            0.6812641791,
            -0.8738871340,
            0.5545010801,
            0.0000000000,
            0.0000000000,
            -1.1090021602,
            0.0000000000,
            -2.2344305157,
            -0.5937730997,
            0.0000000000,
            4.4688610315,
            0.0000000000,
            -0.7687552586,
            0.9179237393,
            0.0000000000,
            0.0000000000,
            -1.8358474786,
            -0.0043373597,
            0.0000000000,
            0.0440897011,
            0.0000000000,
            0.0000000000,
            -0.0036868785,
            0.0000000000,
            -0.0289947100,
            0.0000000000,
            0.0000000000,
            -0.0240200373,
            0.0000000000,
            -0.0148250291,
            0.0000000000,
            -0.0182563223,
            0.0381945953,
            0.0480400747,
            0.0000000000,
            0.0000000000,
            -0.5594921076,
            0.0000000000,
            -0.0080329585,
            0.0000000000,
            0.0050751014,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.0036410781,
            0.0000000000,
            0.0000000000,
            0.0160659170,
            0.0000000000,
            0.0000000000,
            0.8252282376,
            0.0000000000,
            0.0000000000,
            0.0022986556,
            0.0000000000,
            0.0365854797,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0466628473,
            -0.0566821954,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.7847822928,
            0.0000000000,
            -0.0731709594,
            0.0000000000,
            0.0000000000,
            0.5545010801,
            0.0000000000,
            0.0000000000,
            0.0283573970,
            0.0000000000,
            -0.0292646720,
            0.0000000000,
            0.0219432008,
            -0.0091998853,
            -0.0240200373,
            0.0000000000,
            0.0000000000,
            0.5114520329,
            0.0000000000,
            0.0116740366,
            -0.0050751014,
            0.0000000000,
            -0.8412941546,
            0.0000000000,
            -0.0489615029,
            0.0200967157,
            0.0000000000,
            0.7847822928,
            -0.4813301207,
            -0.0043373597,
            0.0000000000,
            0.0000000000,
            -0.4874319956,
            0.0000000000,
            0.0000000000,
            0.4917693553,
            0.0000000000,
            0.0175906354,
            0.0000000000,
            -0.0740896262,
            -0.0270183021,
            0.0000000000,
            0.0089268156,
            0.0000000000,
            0.0614416091,
            -0.0266525030,
            0.0000000000,
            0.0184661072,
            0.0000000000,
            0.0284065250,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.0369322143,
            -0.0466628473,
            0.0000000000,
            0.0116740366,
            0.0000000000,
            0.0206469036,
            0.0000000000,
            -0.0011327214,
            -0.0173567764,
            -0.0036410781,
            0.0000000000,
            0.0000000000,
            0.8252282376,
            0.0000000000,
            -0.0412938071,
            -0.0008749322,
            0.0000000000,
            -0.6290913024,
            0.0489615029,
            0.0000000000,
            -0.0167209099,
            0.0000000000,
            -0.0604838238,
            0.0451353184,
            -0.0466628473,
            0.0000000000,
            0.0000000000,
            -0.6914565982,
            0.0000000000,
            -0.0000000000,
            0.0189034622,
            0.0000000000,
            2.4100839262,
            0.0000000000,
            0.1209676476,
            -0.0874910794,
            0.0000000000,
            -2.2344305157,
            0.7687552586,
            0.0000000000,
            -0.0360567426,
            0.0000000000,
            -0.0013882229,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0184661072,
            0.0182563223,
            0.0000000000,
            -0.0080329585,
            0.0000000000,
            0.0000000000,
            -0.8215871595,
            0.0000000000,
            -0.0022986556,
            0.0000000000,
            0.0000000000,
            0.7381194455,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0175906354,
            -0.0168680994,
            0.0000000000,
            0.8296201180,
            -0.7358207899,
            0.0000000000,
            -0.0292646720,
            0.0000000000,
            0.0534427226,
            0.0000000000,
            -0.0077940942,
            -0.0440848327,
            -0.0148250291,
            0.0000000000,
            0.0000000000,
            -0.7882960233,
            0.0000000000,
            0.0206469036,
            0.0020076536,
            0.0000000000,
            0.6703851096,
            0.0000000000,
            0.0167209099,
            0.0415803616,
            0.0000000000,
            -2.4100839262,
            2.1134628681,
            0.0440897011,
            0.0000000000,
            0.0000000000,
            0.8031210523,
            0.0000000000,
            0.0000000000,
            -0.8472107534,
            0.0000000000,
            -0.0740896262,
            0.0057864407,
            0.0000000000,
            -0.6910320131,
            2.3933630164,
            0.0000000000,
            0.7651216393,
            0.0000000000,
            0.0168680994,
            0.0000000000,
            -0.0057864407,
            -0.0108968304,
            0.0000000000,
            0.0025044711,
            0.0000000000,
            0.0238729457,
            -0.0221220976,
            0.0000000000,
            -0.0182563223,
            0.0000000000,
            0.0184876002,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0466628473,
            -0.0369752003,
            0.0000000000,
            0.0050751014,
            0.0000000000,
            -0.0020076536,
            0.0000000000,
            -0.0015466858,
            -0.0053901304,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.7847822928,
            0.0000000000,
            0.0008749322,
            0.0030933717,
            0.0000000000,
            2.4083340618,
            0.0200967157,
            0.0000000000,
            0.0415803616,
            0.0000000000,
            -0.0423557611,
            0.0220183027,
            -0.0566821954,
            0.0000000000,
            0.0000000000,
            0.5545010801,
            0.0000000000,
            0.0189034622,
            0.0000000000,
            0.0000000000,
            -2.2344305157,
            0.0000000000,
            0.0874910794,
            -0.0440366053,
            0.0000000000,
            0.5937730997,
            0.9179237393,
            0.0000000000,
            0.0013882229,
            0.0000000000,
            -0.0075907698,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            -0.0284065250,
            0.0184876002,
            0.0000000000,
            -0.0050751014,
            0.0000000000,
            0.0000000000,
            0.7847822928,
            0.0000000000,
            0.0365854797,
            0.0000000000,
            0.0000000000,
            -0.4978188847,
            0.0000000000,
            0.0000000000,
            0.0000000000,
            0.0270183021,
            -0.0108968304,
            0.0000000000,
            -0.7797071915,
            0.4612334050,
            0.0000000000,
            -0.0219432008,
            0.0000000000,
            0.0077940942,
            0.0000000000,
            -0.0009577853,
            -0.0184828154,
            0.0182563223,
            0.0000000000,
            0.0000000000,
            0.7381194455,
            0.0000000000,
            0.0011327214,
            -0.0015466858,
            0.0000000000,
            -2.4092089940,
            0.0000000000,
            -0.0604838238,
            0.0423557611,
            0.0000000000,
            2.2155270535,
            -0.6812641791,
            0.0036868785,
            0.0000000000,
            0.0000000000,
            -0.7563757678,
            0.0000000000,
            0.0000000000,
            0.7526888893,
            0.0000000000,
            -0.0089268156,
            0.0025044711,
            0.0000000000,
            2.4080762726,
            -2.1550432297,
            0.0000000000,
            -2.3991494570,
            -0.0091998853,
            0.0000000000,
            -0.0440848327,
            0.0000000000,
            0.0184828154,
            0.0001037949,
            0.0381945953,
            0.0000000000,
            0.0000000000,
            -0.5175258798,
            0.0000000000,
            -0.0173567764,
            0.0053901304,
            0.0000000000,
            2.2313371440,
            0.0000000000,
            -0.0451353184,
            0.0220183027,
            0.0000000000,
            -0.5937730997,
            -0.8738871340,
            -0.0289947100,
            0.0000000000,
            0.0000000000,
            0.4793312845,
            0.0000000000,
            0.0000000000,
            -0.4503365745,
            0.0000000000,
            0.0614416091,
            -0.0238729457,
            0.0000000000,
            -2.2139803677,
            0.6389084180,
            0.0000000000,
            2.1525387586,
            0.0000000000,
            0.0266525030,
            -0.0221220976,
            0.0000000000,
            0.5883829693,
            0.8518688313,
            0.0000000000,
            -0.6150354723,
            -0.8297467338,
        ];

        ForceOrganism {
            id: "water_test".to_string(),
            dna: vec![chr1, chr2, chr3],
            harmfitness: 0.0,
            rotfitness: 0.0,
            fundfitness: 0.0,
        }
    }

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
            dna: vec![3943.0, 3833.0, 1651.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0],
            fitness: 0.0,
        };

        test_org.evaluate_fitness(TARGET);

        let wanted = 0.226769;
        println!("got fitness: {}", test_org.fitness);

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

        pool.eliminate_unfit_fraction();

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
            }
            None => {
                // println!("The parent values are: {}, {}", best, worst);
                assert!(false)
            }
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
        let p1: SimpleOrganism = SimpleOrganism {
            id: "1".to_string(),
            dna: vec![-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            fitness: 1.0,
        };
        let p2: SimpleOrganism = SimpleOrganism {
            id: "2".to_string(),
            dna: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],

            fitness: 0.0,
        };
        let p3: SimpleOrganism = SimpleOrganism {
            id: "3".to_string(),
            dna: vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            fitness: 1.0,
        };
        let mut parents = vec![p1, p2, p3];

        let child = parents.quadratic_mating();
        let result = child.as_any().downcast_ref::<SimpleOrganism>().unwrap();

        println!("Quadratic fitting: {:?}", result);
    }

    #[test]
    // Water test case.
    fn test_create_force_organism() {
        use crate::utils::domain::{FOURTH_DOMAIN, SECOND_DOMAIN, THIRD_DOMAIN};
        let test_org = ForceOrganism::new(3);

        println!("{:?}", test_org);

        assert_eq!(test_org.dna.len(), 3);
        assert_eq!(test_org.dna[0].len(), 81);
        assert_eq!(test_org.dna[1].len(), 165);
        assert_eq!(test_org.dna[2].len(), 495);

        //Verify that each element of the dna is within the domain.
        for (c, dna) in test_org.dna.iter().enumerate() {
            match c {
                0 => {
                    for x in dna.iter() {
                        assert!(x.abs() <= SECOND_DOMAIN);
                    }
                }
                1 => {
                    for x in dna.iter() {
                        assert!(x.abs() <= THIRD_DOMAIN);
                    }
                }
                2 => {
                    for x in dna.iter() {
                        assert!(x.abs() <= FOURTH_DOMAIN);
                    }
                }
                _ => {
                    assert!(false);
                }
            }
        }
    }

    #[test]
    fn test_save_force() {
        use std::fs;
        use std::path::Path;
        // Check if ./organisms/water_test exists. If so, delete it.
        // Path to cargo.toml
        let cargo_toml_path = Path::new(env!("CARGO_MANIFEST_DIR"));
        let organism_path = cargo_toml_path.join("organisms").join("water_test");
        if organism_path.exists() {
            fs::remove_dir_all(&organism_path).unwrap();
        }

        let o = generate_water_organism();

        o.save_to_file(cargo_toml_path.to_str().unwrap());

        // Read the file back in and compare the contents.
        let fort15 = fs::read_to_string(organism_path.join("fort.15")).unwrap();
        let fort30 = fs::read_to_string(organism_path.join("fort.30")).unwrap();
        let fort40 = fs::read_to_string(organism_path.join("fort.40")).unwrap();

        let test_file_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("water_test");
        let fort15_test = fs::read_to_string(test_file_dir.join("fort.15")).unwrap();
        let fort30_test = fs::read_to_string(test_file_dir.join("fort.30")).unwrap();
        let fort40_test = fs::read_to_string(test_file_dir.join("fort.40")).unwrap();

        assert_eq!(fort15, fort15_test);
        assert_eq!(fort30, fort30_test);
        assert_eq!(fort40, fort40_test);
    }

    #[test]
    fn test_water_evaluate() {
        let mut o = generate_water_organism();
        o.evaluate_fitness(TARGET);
        println!("Fitness = {}", o.get_fitness());
        assert!(o.harmfitness - 2.333333e-06 < 0.000001);
        assert_eq!(o.rotfitness, 0.0);
        assert_eq!(o.fundfitness, 0.0);
    }

    #[test]
    fn test_read_from_file() {
        let cargo_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let organism_path = cargo_dir.join("tests").join("water_test");
        let o = ForceOrganism::read_from_file(organism_path.to_str().unwrap());

        let wanted_organism = generate_water_organism();

        for (i, chromosome) in wanted_organism.dna.iter().enumerate() {
            for (j, gene) in chromosome.iter().enumerate() {
                assert_eq!(o.dna[i][j], *gene);
            }
        }
    }
}
