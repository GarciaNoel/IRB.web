// Author: Noel Garcia

use rand::Rng;

fn main() {
    while true {
        let x = main_redirect();
        if x {
            logic_redirect();
        }    
    }
}

pub fn main_redirect() -> bool {
    let mut rng = rand::thread_rng();
    let len: i32 = rng.gen_range(-100..=100);
    return help_wanted(len);
}

fn help_wanted (len: i32) -> bool {
    let mut rng = rand::thread_rng();
    let wrb: bool = rng.gen_bool(0.50);
    return wrb
}

pub fn logic_redirect() {
    let x = 1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1;
}