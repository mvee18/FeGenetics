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

impl Derivatives {
    // This function will take 0 and return Second, 1 and return Third, etc.
    pub fn from_index(index: usize) -> Derivatives {
        match index {
            0 => Derivatives::Second,
            1 => Derivatives::Third,
            2 => Derivatives::Fourth,
            _ => panic!("Invalid index"),
        }
    }
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

pub fn check_and_fix_domain(dn: Derivatives, gene: &mut f64) {
    while !check_normal_bounds(dn, gene) {
        *gene = random_float_mc(dn);
    }
    // We need to reduce the gene by a random amount to ensure it is within the
    // domain.
}

pub fn check_normal_bounds(dn: Derivatives, n: &f64) -> bool {
    match dn {
        Derivatives::Second => {
            if n.abs() <= SECOND_DOMAIN {
                true
            } else {
                false
            }
        }
        Derivatives::Third => {
            if n.abs() <= THIRD_DOMAIN {
                true
            } else {
                false
            }
        }
        Derivatives::Fourth => {
            if n.abs() <= FOURTH_DOMAIN {
                true
            } else {
                false
            }
        }
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
