use rand::Rng;

pub fn random_float(low: f32, high: f32) -> f32 {
        let mut rng = rand::thread_rng();
        let n: f32 = rng.gen_range(low..=high);
        n
}