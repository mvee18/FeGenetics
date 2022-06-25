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

#[derive(Serialize, Deserialize)]
pub struct Target {
    pub harm: Vec<f64>,
    pub rots: Vec<f64>,
    pub fund: Vec<f64>,
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
