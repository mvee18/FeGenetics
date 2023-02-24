// We are going to use brent's spectro program.
use crate::models::models::{EXE_DIR_PATH, SPECTRO_IN_PATH};
use lazy_static::__Deref;
use spectro::{Output, Spectro};
use std::path::PathBuf;
use std::{error::Error, path::Path};

pub fn run_spectro(organism_id: String) -> Result<Output, Box<dyn Error>> {
    // let mut s = spectro::Spectro::new();
    // s.run();
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

    let spectro = Spectro::load(SPECTRO_IN_PATH.deref());
    // let infile = Path::new(SPECTRO_IN_PATH.deref());
    // let dir = infile.parent().unwrap_or_else(|| Path::new("."));
    let (g, _) = spectro.run_files(
        organism_path.join("fort.15"),
        organism_path.join("fort.30"),
        organism_path.join("fort.40"),
    );
    // (g, spectro)
    // spectro.write_output(&mut std::io::stdout(), &g)?;
    println!("Finished spectro...");
    Ok(g)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_spectro() {
        let test_infile = "src/input/test_organism/spectro.in";

        let spectro = Spectro::load(test_infile);
        let infile = Path::new(test_infile);
        let dir = infile.parent().unwrap_or_else(|| Path::new("."));
        let (g, _) = spectro.run_files(
            dir.join("fort.15"),
            dir.join("fort.30"),
            dir.join("fort.40"),
        );
        // (g, spectro)
        spectro.write_output(&mut std::io::stdout(), &g).unwrap();
        // println!("g: {:?}", g);
        println!("rots: {:?}", g.rots[0]);
    }
}
