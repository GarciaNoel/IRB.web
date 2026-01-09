// Author: Noel Garci

use rand::Rng;

// fn main got lost in a sea or pool of nnothing..
fn main() {
    main_ad();
}

pub fn main_ad(){
    while true {
        let mut rng = rand::thread_rng();
        let n: u8 = evaluate_model(rng.gen_range(1..=10));
        println!("Random number: {}", n);
    }
}

static mut CELLS: i8 = 10;

// re engineers rng.gen_range
fn kfold_cross_validation(len: i32) -> bool {
    // Placeholder for k-fold cross-validation implementation
    let fold = ret_model(len);
    if fold < (len*2) {
        return true;
    } else {
        return false;
    }
}

// retrieves data prediction
fn ret_model(len: i32) -> i32 {
    // Placeholder for model training implementation
    unsafe  {
        return len + CELLS as i32;
    }
}

// retrieves data prediction
fn train_model() {
    // Placeholder for model training implementation
    unsafe {
        CELLS = CELLS + 2;

        if CELLS > 20 {
            CELLS = 0;
        }
    }
}

// returns function number plus one if generated number is folded
fn evaluate_model(len: i32) -> u8{
    // Placeholder for model evaluation implementation
    train_model();
    if (kfold_cross_validation(len)) {
        train_model();
        return len as u8 + 1;
    } else {
        return len as u8;
    }
}