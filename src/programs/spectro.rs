use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

const SPECTRO_PATH: &str = "src/input/spectro";
const SPECTRO_IN_PATH: &str = "src/input/spectro.in";

pub fn run_spectro(organism_id: String) -> String {
    // Organism path is organisms/<organism_id>.
    let organism_path = PathBuf::from(format!("organisms/{}", organism_id))
        .canonicalize()
        .unwrap();

    // Get absolute path to SPECTRO_PATH.
    let spectro_path = PathBuf::from(SPECTRO_PATH).canonicalize().unwrap();

    // Get absolute path to SPECTRO_IN_PATH.
    let spectro_in_path = PathBuf::from(SPECTRO_IN_PATH).canonicalize().unwrap();

    // Get current working directory.
    let cwd = std::env::current_dir().unwrap();

    // Change working directory to the organism path.
    std::env::set_current_dir(&organism_path).unwrap();

    // let file_contents = std::fs::read_to_string(spectro_in_path).unwrap();
    let input_file = std::fs::File::open(spectro_in_path).unwrap();

    println!(
        "The file paths are as follows: {} {}",
        spectro_path.display(),
        organism_path.display()
    );

    // Run spectro from spectro path and pipe in the input file.
    let command = Command::new(&spectro_path)
        .stdin(input_file)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();

    let output = command.wait_with_output().unwrap();

    if output.status.success() {
        // Change working directory back to the original working directory.
        std::env::set_current_dir(&cwd).unwrap();
        return String::from_utf8_lossy(&output.stdout).to_string();
    } else {
        std::env::set_current_dir(&cwd).unwrap();
        panic!("Error running spectro in {}", organism_path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_spectro() {
        let organism_id = "test_organism";
        run_spectro(organism_id.to_string());
    }
}
