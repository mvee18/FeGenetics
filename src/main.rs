mod utils;
mod models;
mod traditional;

fn main() {
    use utils::domain::random_float;
    println!("{}", random_float(0.0, 10.0));

    use traditional::traditional::run_tga;
    run_tga();
}
