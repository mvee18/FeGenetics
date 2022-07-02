// This function accepts a PathBuf and creates a new directory if it doesn't exist.
use serde_derive::{Deserialize, Serialize};
use std::io::Read;
use std::path::PathBuf;

pub fn create_directory(path: &PathBuf) {
    if !path.exists() {
        match std::fs::create_dir_all(path) {
            Ok(_) => {}
            Err(e) => {
                panic!("Error creating directory: {}", e);
            }
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Target {
    pub harm: Vec<f64>,
    pub rots: Vec<f64>,
    pub fund: Vec<f64>,
    pub number_atoms: i32,
    pub population_size: i32,
    pub tournament_size: u32,
    pub mutation_rate: f64,
    pub mutation_strength: f64,
    pub fitness_threshold: f64,
    pub initial_guess: String,
    pub spectro_path: String,
    pub spectro_in_path: String,
}

impl Target {
    pub fn initialize(path: &PathBuf) -> Target {
        read_input_toml(path)
    }
}

pub fn read_input_toml(path: &PathBuf) -> Target {
    let mut file = std::fs::File::open(path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    toml::from_str(&contents).unwrap()
}

// Get executable path's directory and return as str.
pub fn get_executable_path() -> PathBuf {
    let mut path = std::env::current_exe().unwrap();
    path.pop();
    path
}

pub fn get_target_path() -> PathBuf {
    let mut path = get_executable_path();
    path.push("target.toml");
    path
}

//Tests
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_parse_toml() {
        let input_path = PathBuf::from("/home/mvee/rust/fegenetics/src/input/target.toml");
        let test_target = Target::initialize(&input_path);
        assert_eq!(test_target.mutation_rate, 0.20);
        assert_eq!(test_target.mutation_strength, 5E-9);
        assert_eq!(test_target.fitness_threshold, 1.0);
    }
}
