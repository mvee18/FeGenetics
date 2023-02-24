mod models;
mod programs;
mod traditional;
mod utils;
use traditional::traditional::run_tga;

fn main() {
    // run_spectro(1234.to_string());
    run_tga();
}
