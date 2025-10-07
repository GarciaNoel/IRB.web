// ==========================================================
//  Inker Rust Browser — Lingua Machina + Resonator
//  Author: Noel Garcia
// ==========================================================

use rand::Rng;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI8, Ordering};
use std::thread;
use std::time::Duration;

/// Global cell counter (safe, atomic)
static CELLS: AtomicI8 = AtomicI8::new(10);

// ───────────────────────── CORE STRUCTURES ─────────────────────────

#[derive(Copy, Clone)]
struct Slider;

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

    pub fn pipeclean(&mut self) {
        let dat = DataPack { datum: vec![-1i8, 1i8] };
        let vec = dat.datum;

        let cells_val = CELLS.load(Ordering::Relaxed) as f64;
        let raw_prob = (-(cells_val / 126.0) + 1.0).clamp(0.0, 1.0);

        let mut rng = rand::thread_rng();
        let wrb: bool = rng.gen_bool(raw_prob);

        let idx_usize: usize = if wrb { 1 } else { 0 };
        let delta = vec[idx_usize] as i16;
        self.tel += delta;
        self.rec += delta as i32;

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

pub fn logic_redirect_db(cargo: &mut Cargo) {
    println!("{}", cargo.receive());
    cargo.pipeclean();
    println!("{}", cargo.send());
}

pub fn logic_redirect_pf() {
    let cells_val = CELLS.load(Ordering::Relaxed);
    let count = if cells_val > 0 { cells_val as i32 } else { 0 };
    let mut x = 0;
    for _ in 0..count { x += 1; }
    println!("{}", x);
}

// ───────────────────────── LINGUA MACHINA EXTENSIONS ─────────────────────────

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

struct Gear { rotations: u16 }

impl Gear {
    pub fn rotate(&mut self, steps: i8) {
        let new_val = (self.rotations as i32 + steps as i32)
            .clamp(0, 65535) as u16;
        self.rotations = new_val;
        println!("[gear:{:?}]", self.rotations);
    }
}

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

// ───────────────────────── SMUDGE RESONATOR ─────────────────────────

struct Resonator {
    mood: i8,
    genre_gate: &'static str,
}

impl Resonator {
    fn new() -> Self {
        Resonator { mood: 0, genre_gate: "NULL" }
    }

    fn interpret(&mut self, input: i32) {
        self.mood = (input % 7) as i8;
        self.genre_gate = match self.mood {
            0 => "XOR",       // Hyperpop
            1 => "NAND",      // Trap Metal
            2 => "NULL",      // Vaporwave
            3 => "NOT",       // Noise Rap
            4 => "AND",       // Alt Nostalgia
            _ => "SMUDGE",    // Undefined entropy
        };
        println!("[resonator] mood={} gate={}", self.mood, self.genre_gate);
    }

    fn echo(&self) {
        println!("Executing genre logic: {}", self.genre_gate);
        match self.genre_gate {
            "XOR" => println!("Dual-state chaos triggered."),
            "NAND" => println!("Resistance gate activated."),
            "NULL" => println!("Ambient residue returned."),
            "NOT" => println!("Inversion protocol engaged."),
            "AND" => println!("Sentiment stack resolved."),
            _ => println!("Entropy overflow."),
        }
    }
}

// ───────────────────────── LINKED THREADS ─────────────────────────

fn smudge_engine(shared: Arc<Mutex<Cargo>>, pulse: &mut Pulse, gear: &mut Gear, smudge: &mut Smudge) {
    pulse.beat();
    if smudge.distort() {
        gear.rotate(1);
    }
    let mut cargo = shared.lock().unwrap();
    logic_redirect_db(&mut cargo);
}

fn resonator_compiler(shared: Arc<Mutex<Cargo>>, res: &mut Resonator) {
    let cargo = shared.lock().unwrap();
    let input = cargo.receive();
    res.interpret(input);
    res.echo();
}

// ───────────────────────── ENTRY POINT ─────────────────────────

fn main() {
    let shared_cargo = Arc::new(Mutex::new(Cargo { tel: 20, rec: 50 }));
    let y = Slider;

    let mut pulse = Pulse::new();
    let mut gear = Gear { rotations: 0 };
    let mut smudge = Smudge::new();
    let mut res = Resonator::new();

    let smudge_clone = Arc::clone(&shared_cargo);
    let resonator_clone = Arc::clone(&shared_cargo);

    thread::spawn(move || {
        loop {
            smudge_engine(smudge_clone.clone(), &mut pulse, &mut gear, &mut smudge);
            thread::sleep(Duration::from_millis(500));
        }
    });

    thread::spawn(move || {
        loop {
            resonator_compiler(resonator_clone.clone(), &mut res);
            thread::sleep(Duration::from_millis(700));
        }
    });

    loop {
        let decision = main_redirect(y);
        if decision {
            println!("dbregtel/");
        } else {
            println!("oldcargo/");
            logic_redirect_pf();
        }
        thread::sleep(Duration::from_secs(10));
    }
}
