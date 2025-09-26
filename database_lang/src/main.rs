// Author: Noel Garcia

use rand::seq::SliceRandom;
use rand::thread_rng;

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
                
        let mut rng = thread_rng();
       
        if let Some(random_element) = dat.datum..choose(&mut rng) {
            println!("Random element from Vec<u8>: {}", random_element);
        } else {
            println!("The vector is empty.");
        }

        let z = self.tel + (random_element*100);
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

    let y = oldcargo.send();
    println!("{}",c.receive(););
    
    c.pipeclean();

    println!("{}",c.send());
}

struct slider {

}

pub fn main_ad(){
    let oldcargo = cargo {
        tel: 10,
        rec: 20,
    };

    let y = slider {};

    while true {
        let x = main_redirect(slider);
        if x {
            oldcargo = logic_redirect(oldcargo);
        }    
    }

}
