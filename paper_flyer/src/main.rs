// Author: Noel Garcia

use rand::Rng;

#[derive(Copy, Clone)]
struct slider;

fn main() {
    let y = slider;
    let yy = y.clone();

    while true {
        let x = main_redirect(yy);
        if x {
            logic_redirect();
        }    
    }
}

pub fn main_redirect(sli: slider) -> bool {
    let mut rng = rand::thread_rng();
    let len: i32 = rng.gen_range(-100..=100);
    return help_wanted(len);
}

fn help_wanted (len: i32) -> bool {
    let mut rng = rand::thread_rng();
    let wrb: bool = rng.gen_bool(0.50);
    return wrb
}

//HERE PUT FOR LOOP LOGIC FOR ADDING W/ STATS [TODO]
pub fn logic_redirect() {
    let x = 1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1+1;
}