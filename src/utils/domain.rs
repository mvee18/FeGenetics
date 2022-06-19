use rand::Rng;

pub fn random_float(low: f64, high: f64) -> f64 {
        let mut rng = rand::thread_rng();
        let sign = rand::random::<bool>();

        let n: f64 = rng.gen_range(low..=high);
        if sign {
            n
        } else {
            -n
        }
}

