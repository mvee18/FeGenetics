// This function accepts a PathBuf and creates a new directory if it doesn't exist.
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
