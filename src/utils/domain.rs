use rand::Rng;
use strum_macros::EnumIter;

pub const SECOND_DOMAIN: f64 = 1.0;
pub const THIRD_DOMAIN: f64 = 3.0;
pub const FOURTH_DOMAIN: f64 = 10.0;

#[derive(EnumIter, Copy, Clone, Debug)]
pub enum Derivatives {
    Second,
    Third,
    Fourth,
}

pub fn random_float(min: f64, max: f64) -> f64 {
    let mut rng = rand::thread_rng();
    let sign = rand::random::<bool>();

    let n = rng.gen_range(min..=max);

    if sign {
        n
    } else {
        -n
    }
}

pub fn random_float_mc(dn: Derivatives) -> f64 {
    let n;

    let mut rng = rand::thread_rng();
    let sign = rand::random::<bool>();

    match dn {
        Derivatives::Second => {
            n = rng.gen_range(0.0..=SECOND_DOMAIN);
        }
        Derivatives::Third => {
            n = rng.gen_range(0.0..=THIRD_DOMAIN);
        }
        Derivatives::Fourth => {
            n = rng.gen_range(0.0..=FOURTH_DOMAIN);
        }
    }

    if sign {
        n
    } else {
        -n
    }
}

// Why no C like syntax for this :(
pub fn determine_number_force_constants(n_atoms: i32, dn: Derivatives) -> i32 {
    match dn {
        Derivatives::Second => (n_atoms * n_atoms) as i32 * 3 * 3,
        Derivatives::Third => {
            let mut c = 0;
            let mut i = 0;
            while i <= (n_atoms * 3) - 1 {
                let mut j = 0;
                while j <= i {
                    let mut k = 0;
                    while k <= j {
                        c += 1;
                        k += 1;
                    }
                    j += 1;
                }
                i += 1;
            }
            c
        }

        Derivatives::Fourth => {
            let mut c = 0;
            let mut i = 0;
            while i <= (n_atoms * 3) - 1 {
                let mut j = 0;
                while j <= i {
                    let mut k = 0;
                    while k <= j {
                        let mut l = 0;
                        while l <= k {
                            c += 1;
                            l += 1;
                        }
                        k += 1;
                    }
                    j += 1;
                }
                i += 1;
            }

            c
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_number_force_constants() {
        assert_eq!(
            determine_number_force_constants(3, Derivatives::Fourth),
            495
        );
        assert_eq!(determine_number_force_constants(3, Derivatives::Third), 165);
        assert_eq!(
            determine_number_force_constants(6, Derivatives::Third),
            1140
        );
    }
}
