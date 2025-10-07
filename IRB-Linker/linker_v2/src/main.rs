// Author: Noel Garcia
// Cargo.toml: add `rand = "0.8"`
// Rust 2021 edition recommended.

use rand::Rng;
use std::sync::atomic::{AtomicI8, Ordering};
use std::thread;
use std::time::Duration;

/// A global cell counter (safe, atomic).
static CELLS: AtomicI8 = AtomicI8::new(10);

#[derive(Copy, Clone)]
struct Slider; // unit-like struct; named with Rust style

#[derive(Debug, Clone, Copy)]
struct Cargo {
    tel: i16,
    rec: i32,
}

struct DataPack {
    datum: Vec<i8>,
}

impl Cargo {
    fn send(&self) -> u32 {
        // Avoid division/modulo by zero: if tel == 0 return 0
        let x = self.tel as i32;
        if x == 0 {
            return 0;
        }
        let x0 = self.rec % x;
        (x0 as u32)
    }

    /// Mutates the global CELLS atomically and picks randomly between -1 and 1
    pub fn pipeclean(&self) {
        let dat = DataPack {
            datum: vec![-1i8, 1i8],
        };
        let vec = dat.datum; // move

        let cells_val = CELLS.load(Ordering::Relaxed) as f64;

        // compute probability in range [0.0, 1.0], clamp as safety
        let raw_prob = (-(cells_val / 126.0) + 1.0).clamp(0.0, 1.0);

        let mut rng = rand::thread_rng();
        let wrb: bool = rng.gen_bool(raw_prob);

        let idx_usize: usize = if wrb { 1 } else { 0 };

        let delta = vec[idx_usize] as i16;
        let _z = self.tel + delta; // kept from original, currently unused

        // Update CELLS safely
        if CELLS.load(Ordering::Relaxed) > 0 {
            // fetch_add uses i8, so convert
            CELLS.fetch_add(vec[idx_usize] as i8, Ordering::Relaxed);
        } else {
            CELLS.store(1, Ordering::Relaxed);
        }
    }

    pub fn receive(&self) -> i32 {
        self.rec + self.tel as i32
    }
}

pub fn main_redirect(_sli: Slider) -> bool {
    let mut rng = rand::thread_rng();
    let _len: i32 = rng.gen_range(-100..=100);
    help_wanted()
}

fn help_wanted() -> bool {
    let mut rng = rand::thread_rng();
    rng.gen_bool(0.50) // 50/50
}

pub fn logic_redirect_db(oldcargo: Cargo) -> Cargo {
    let mut c = Cargo { tel: 20, rec: 50 };

    let _y = oldcargo.send();
    println!("{}", c.receive());

    c.pipeclean();

    println!("{}", c.send());
    c
}

pub fn logic_redirect_pf() {
    // Count from 0..CELLS as i8 and print as i32 for readability
    let cells_val = CELLS.load(Ordering::Relaxed);
    // Ensure non-negative loop bound; if negative treat as 0
    let count = if cells_val > 0 {
        cells_val as i32
    } else {
        0
    };
    let mut x = 0;
    for _ in 0..count {
        x += 1;
    }
    println!("{}", x);
}

fn main() {
    main_ad();
}

pub fn main_ad() {
    let mut oldcargo = Cargo { tel: 20, rec: 50 };

    let y = Slider; // unit-like instantiation
    // no need to clone a unit; reuse `y` directly

    loop {
        let x = main_redirect(y);
        if x {
            println!("dbregtel/");
            oldcargo = logic_redirect_db(oldcargo);
        } else {
            println!("oldcargo/");
            println!("{}", oldcargo.receive());
            logic_redirect_pf();
            println!("{}", oldcargo.send());
        }

        thread::sleep(Duration::from_millis(250));
    }
}
