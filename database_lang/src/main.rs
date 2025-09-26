// Author: Noel Garci

use rand::Rng;

// fn main got lost in a sea or pool of nnothing..
fn main() {
    main_ad();
}

struct cargo {
    tel: i16,
    rec: i32,
}

struct data_pack {
    datum: Vec<u8>,
}

impl cargo {
    fn send(&self) -> u32 {
        let x = self.tel as i32;
        let x0 = self.rec % x;
        let x1 = x0 as u32;
        return x1;
    }

    pub fn pipeclean(&self) {
        let dat = data_pack {
            datum: vec![1,2,3,4,5,6,7,8,9,0],
        };
        
        let vec = dat.datum.clone();

        let z = self.tel + (vec[0]) as i16;
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

pub fn logic_redirect(oldcargo: cargo) -> cargo {
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

#[derive(Copy, Clone)]
struct slider;

pub fn main_ad(){
    let mut oldcargo = cargo {
        tel: 10,
        rec: 20,
    };

    let y = slider;
    let yy = y.clone();

    while true {
        let x = main_redirect(yy);
        if x {
            oldcargo = logic_redirect(oldcargo);
        }    
    }

}
