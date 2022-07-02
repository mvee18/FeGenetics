use std::error::Error;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::thread;
use summarize::Summary;

use crate::models::models::EXE_DIR_PATH;

const SPECTRO_PATH: &str = "/home/mvee/rust/fegenetics/src/input/spectro";
const SPECTRO_IN_PATH: &str = "/home/mvee/rust/fegenetics/src/input/spectro.in";

/*
Incase we need to do this ourselves and not use summarize.
pub struct Spectro {
    harm: Vec<f64>,
    rots: Vec<f64>,
    fund: Vec<f64>,
}

enum State {
    LXM,
    Rot,
    Fund,
    Done,
    None,
}
*/

pub fn run_spectro(organism_id: String) -> Result<Summary, Box<dyn Error>> {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        // Organism path is organisms/<organism_id>.
        let organism_path: PathBuf;
        if organism_id == "test_organism" {
            organism_path = PathBuf::from("/home/mvee/rust/fegenetics/src/input/test_organism")
                .canonicalize()
                .unwrap();
        } else if organism_id == "bad_organism" {
            organism_path = PathBuf::from("/home/mvee/rust/fegenetics/tests/bad_organism")
                .canonicalize()
                .unwrap();
        } else {
            // println!("Trying organism {}", organism_id);
            organism_path = EXE_DIR_PATH
                .join("organisms")
                .join(organism_id)
                .canonicalize()
                .unwrap();
        }

        // Get absolute path to SPECTRO_PATH.
        let spectro_path = PathBuf::from(SPECTRO_PATH).canonicalize().unwrap();

        // Get absolute path to SPECTRO_IN_PATH.
        let spectro_in_path = PathBuf::from(SPECTRO_IN_PATH).canonicalize().unwrap();

        // Get current working directory.
        // let cwd = std::env::current_dir().unwrap();

        // println!("cwd: {:?}\n\n", cwd);

        // Change working directory to the organism path.
        // std::env::set_current_dir(&organism_path).unwrap();

        // let file_contents = std::fs::read_to_string(spectro_in_path).unwrap();
        let input_file = std::fs::File::open(spectro_in_path).unwrap();

        // println!(
        //     "The file paths are as follows: {} {}",
        //     spectro_path.display(),
        //     organism_path.display()
        // );

        // Run spectro from spectro path and pipe in the input file.
        let command = Command::new(&spectro_path)
            .stdin(input_file)
            .stdout(Stdio::piped())
            .current_dir(&organism_path)
            .spawn()
            .unwrap();

        let output = command.wait_with_output().unwrap();
        // std::env::set_current_dir(&cwd).unwrap();

        if output.status.success() {
            // Change working directory back to the original working directory.
            tx.send(output.stdout).unwrap();
        } else {
            panic!("Error running spectro in {}", organism_path.display());
        }
    });

    // Unwrap the output from the channel. Then, parse the output.
    let res = parse_spectro(rx.recv().unwrap().to_vec());
    res
}

pub fn parse_spectro(output: Vec<u8>) -> Result<Summary, Box<dyn std::error::Error>> {
    let result = summarize::Summary::new(output);
    result
}

/*
pub fn run_spectro_local(path: String) {
    let (tx, rx) = mpsc::channel();

    thread::spawn(move || {
        // Organism path is organisms/<organism_id>.
        // println!("Trying organism {}", organism_id);
        let organism_path = PathBuf::from(path).canonicalize().unwrap();

        // Get absolute path to SPECTRO_PATH.
        let spectro_path = PathBuf::from(SPECTRO_PATH).canonicalize().unwrap();

        // Get absolute path to SPECTRO_IN_PATH.
        let spectro_in_path = PathBuf::from(SPECTRO_IN_PATH).canonicalize().unwrap();

        // Get current working directory.
        let cwd = std::env::current_dir().unwrap();

        // println!("cwd: {:?}\n\n", cwd);

        // Change working directory to the organism path.
        std::env::set_current_dir(&organism_path).unwrap();

        // let file_contents = std::fs::read_to_string(spectro_in_path).unwrap();
        let input_file = std::fs::File::open(spectro_in_path).unwrap();

        // println!(
        //     "The file paths are as follows: {} {}",
        //     spectro_path.display(),
        //     organism_path.display()
        // );

        // Run spectro from spectro path and pipe in the input file.
        let command = Command::new(&spectro_path)
            .stdin(input_file)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();

        let output = command.wait_with_output().unwrap();
        std::env::set_current_dir(&cwd).unwrap();

        if output.status.success() {
            // Change working directory back to the original working directory.
            tx.send(output.stdout).unwrap();
        } else {
            panic!("Error running spectro in {}", organism_path.display());
        }
    });

    // Unwrap the output from the channel. Then, parse the output.
    parse_spectro_local(rx.recv().unwrap().to_vec());
}

pub fn parse_spectro_local(spectro_output: Vec<u8>) {
    // Convert the output to a string.
    let output_str = String::from_utf8(spectro_output).unwrap();
    let mut state = State::None;
    // Print line by line.
    for line in output_str.lines() {}
}
*/

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_spectro() {
        let organism_id = "test_organism";
        let result = run_spectro(organism_id.to_string()).unwrap();
        println!("{:?}", result);

        let expected_harm = vec![3943.976, 3833.989, 1651.332];
        let expected_lxm = vec![
            3943.98, 3833.99, 1651.33, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00,
        ];
        let expected_fund = vec![3753.156, 3656.489, 1598.834];
        let expected_rots = vec![14.5054957, 9.2636424, 27.6557350];

        assert_eq!(result.harm, expected_harm);
        assert_eq!(result.lxm_freqs, expected_lxm);
        assert_eq!(result.fund, expected_fund);
        assert_eq!(result.rots[0], expected_rots);
    }

    #[test]
    fn test_bad_organism() {
        let organism_id = "bad_organism";
        let result = run_spectro(organism_id.to_string());
        assert!(result.is_err());
    }

    // #[test]
    // fn test_local_spectro() {
    //     let organism_id = "/home/mvee/rust/fegenetics/tests/h2co/4th/";
    //     run_spectro_local(organism_id.to_string());
    // }
}
