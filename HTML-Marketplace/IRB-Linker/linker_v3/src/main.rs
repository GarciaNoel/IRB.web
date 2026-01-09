// ==========================================================
//  Inker Rust Browser — Lingua Machina Edition
//  Author: Noel Garcia  |  Edition: 2021  |  rand = "0.8"
// ==========================================================

use rand::Rng;
use std::sync::atomic::{AtomicI8, Ordering};
use std::thread;
use std::time::Duration;

/// Global cell counter (safe, atomic)
static CELLS: AtomicI8 = AtomicI8::new(10);

// ───────────────────────── CORE STRUCTURES ─────────────────────────

#[derive(Copy, Clone)]
struct Slider;                     // Environmental selector

#[derive(Debug, Clone, Copy)]
struct Cargo {
    tel: i16,
    rec: i32,
}

struct DataPack {
    datum: Vec<i8>,
}

// ───────────────────────── IMPLEMENTATIONS ─────────────────────────

impl Cargo {
    fn send(&self) -> u32 {
        let x = self.tel as i32;
        if x == 0 { return 0; }
        let x0 = self.rec % x;
        x0 as u32
    }

    /// Mutates the global CELLS atomically, randomly ±1
    pub fn pipeclean(&self) {
        let dat = DataPack { datum: vec![-1i8, 1i8] };
        let vec = dat.datum;

        let cells_val = CELLS.load(Ordering::Relaxed) as f64;
        let raw_prob = (-(cells_val / 126.0) + 1.0).clamp(0.0, 1.0);

        let mut rng = rand::thread_rng();
        let wrb: bool = rng.gen_bool(raw_prob);

        let idx_usize: usize = if wrb { 1 } else { 0 };
        let delta = vec[idx_usize] as i16;
        let _z = self.tel + delta;

        if CELLS.load(Ordering::Relaxed) > 0 {
            CELLS.fetch_add(vec[idx_usize] as i8, Ordering::Relaxed);
        } else {
            CELLS.store(1, Ordering::Relaxed);
        }
    }

    pub fn receive(&self) -> i32 {
        self.rec + self.tel as i32
    }
}

// ───────────────────────── UTILITIES ─────────────────────────

pub fn main_redirect(_sli: Slider) -> bool {
    let mut rng = rand::thread_rng();
    let _len: i32 = rng.gen_range(-100..=100);
    help_wanted()
}

fn help_wanted() -> bool {
    let mut rng = rand::thread_rng();
    rng.gen_bool(0.50)
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
    let cells_val = CELLS.load(Ordering::Relaxed);
    let count = if cells_val > 0 { cells_val as i32 } else { 0 };
    let mut x = 0;
    for _ in 0..count { x += 1; }
    println!("{}", x);
}

// ───────────────────────── LINGUA MACHINA EXTENSIONS ─────────────────────────

/// Heartbeat of the machine
struct Pulse { ticks: u64 }

impl Pulse {
    pub fn new() -> Self { Pulse { ticks: 0 } }

    pub fn beat(&mut self) {
        self.ticks += 1;
        if self.ticks % 10 == 0 {
            println!("[pulse/{}]", self.ticks);
        }
    }
}

/// Gear — transforms and bridges systems
struct Gear { rotations: u16 }

impl Gear {
    pub fn rotate(&mut self, steps: i8) {
        let new_val = (self.rotations as i32 + steps as i32)
            .clamp(0, 65535) as u16;
        self.rotations = new_val;
        println!("[gear:{:?}]", self.rotations);
    }
}

/// Smudge — entropy controller of the protocol
struct Smudge { entropy: f64 }

impl Smudge {
    pub fn new() -> Self { Smudge { entropy: 0.5 } }

    pub fn distort(&mut self) -> bool {
        let mut rng = rand::thread_rng();
        self.entropy = (self.entropy + rng.gen_range(-0.05..0.05))
            .clamp(0.0, 1.0);
        let flip = rng.gen_bool(self.entropy);
        println!("[smudge:{:.2} flip={}]", self.entropy, flip);
        flip
    }
}

// ───────────────────────── MAIN AD LOOP ─────────────────────────

pub fn main_ad() {
    let mut oldcargo = Cargo { tel: 20, rec: 50 };
    let y = Slider;

    let mut pulse = Pulse::new();
    let mut gear = Gear { rotations: 0 };
    let mut smudge = Smudge::new();

    loop {
        pulse.beat();
        let decision = main_redirect(y);

        let chaos = smudge.distort();
        if chaos { gear.rotate(1); }

        if decision {
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

// ───────────────────────── ENTRY POINT ─────────────────────────

fn main() {
    main_ad();
}
