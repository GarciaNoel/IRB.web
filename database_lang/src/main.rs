// Author: Noel Garcia

// fn main got lost in a sea or pool of nnothing..
fn main() {
    main_ad();
}

struct cargo {
    tel: i16,
    rec: i32,
}

impl cargo {
    fn send(&self) -> u32 {
        let x = self.tel as i32;
        let x0 = self.rec % x;
        let x1 = x0 as u32;
        return x1;
    }
}


// redirects main basically forever must be updated for main args..
pub fn main_ad(){
    let c = cargo {
        tel: 20,
        rec: 50,
    };

    println!("{}",c.send());
}


// now you want all data to essentially only be strings

struct data_pack {

}