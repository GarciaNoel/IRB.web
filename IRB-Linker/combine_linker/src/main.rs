// Author: Noel Garci

use rand::Rng;

// fn main got lost in a sea or pool of nnothing..
fn main() {
    main_ad();
}

pub fn main_ad(){
    let mut oldcargo = cargo {
        tel: 20,
        rec: 50,
    };

    let y = slider;
    let yy = y.clone();

    while true {
        let x = main_redirect(yy);
        if x {
            println!("{}","dbregtel/");
            oldcargo = logic_redirect_db(oldcargo);
        } else {
            println!("{}","oldcargo/");
            println!("{}",oldcargo.receive());
            logic_redirect_pf();
            println!("{}",oldcargo.send());
        }
    }

}

static mut CELLS: i8 = 10;

#[derive(Copy, Clone)]
struct slider;

struct cargo {
    tel: i16,
    rec: i32,
}

struct data_pack {
    datum: Vec<i8>,
}

impl cargo {
    fn send(&self) -> u32 {
        let x = self.tel as i32;
        let x0 = self.rec % x;
        let x1 = x0 as u32;
        return x1;
    }

    pub fn pipeclean(&self) {
        unsafe {
            let dat = data_pack {
                datum: vec![-1,1],
            };
            
            let vec = dat.datum.clone();
            
            let mut rng = rand::thread_rng();
            let wrb: bool = rng.gen_bool(-(CELLS as f64 / 126.0)+1.0);

            let mut idx = 0;
            if wrb { idx = 1;}

            let z = self.tel + (vec[idx]) as i16;
        
            if CELLS > 0 {
                CELLS += vec[idx] as i8;
            } else {
                CELLS = 1;
            }
        }
    }

    pub fn receive(&self) -> i32 {
        let y = self.rec + self.tel as i32;
        return y;
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

pub fn logic_redirect_db(oldcargo: cargo) -> cargo {
    let c = cargo {
        tel: 20,
        rec: 50,
    };

    let _y = oldcargo.send();
    println!("{}",c.receive());
    
    c.pipeclean();

    println!("{}",c.send());
    return c;
}

pub fn logic_redirect_pf() {
    unsafe {
        let mut x = 0;
        for i in 0..CELLS {
            x += 1;
        }
        println!("{}",x);
    }
}
